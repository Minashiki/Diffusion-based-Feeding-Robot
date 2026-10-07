"""Audited 50k EMA leading 10/20/50-step comparison on early validation conditions."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

import diffusers
import numpy as np
import torch
from torch.utils.data import default_collate

from acquire_diagnosis import direction_metrics, sample_from_noise, verify_evidence
from diagnose_sampling import action_metrics, check_checkpoint, rollout_metrics, spacing_override
from prefix_rollout import noise_seed_at_tick, run_prefix_policy
from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies import dit
from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.runtime import precision, setup, to_device
from feedingrobot.sim.model import ROOT, load_json


COUNTS = (10, 20, 50)
SEEDS = (0, 1, 2)


def inference_config(config, count):
    result = deepcopy(config)
    result['diffusion']['inference_steps'] = count
    return result


def settings(config, count):
    _, scheduler = dit.schedulers(config)
    scheduler.set_timesteps(count)
    alpha = scheduler.alphas_cumprod[scheduler.timesteps[0]]
    return dict(config=dict(scheduler.config), timesteps=scheduler.timesteps.tolist(), inference_steps=count,
                eta=0., first_step_epsilon_error_gain=float(((1-alpha)/alpha).sqrt()))


def select_candidate(summaries):
    qualified = []
    for count in (20, 50):
        for seed in SEEDS:
            baseline, candidate = summaries['10'][str(seed)]['metrics'], summaries[str(count)][str(seed)]['metrics']
            if (not candidate['finite'] or not baseline['finite']
                    or candidate['angular']['vector_rmse'] > baseline['angular']['vector_rmse']*.9
                    or candidate['linear']['vector_rmse'] > baseline['linear']['vector_rmse']*1.05
                    or any(candidate[k]['clipped_fraction'] > baseline[k]['clipped_fraction'] for k in ('linear', 'angular'))):
                break
        else:
            qualified.append(count)
    return min(qualified, key=lambda count: np.mean([summaries[str(count)][str(s)]['metrics']['angular']['vector_rmse'] for s in SEEDS])) if qualified else None


def noise_for(cases, seed, horizon, device):
    return torch.stack([torch.randn((horizon, 6), device=device,
        generator=torch.Generator(device=device).manual_seed(noise_seed_at_tick(seed, c['tick']))) for c in cases])


@torch.no_grad()
def epsilon_probe(model, config, normalization, device, items, cases, truth, mask, limits, output):
    # Known forward noise permits epsilon-error measurement, never a policy input.
    archives, summaries = {}, {}
    _, scheduler = dit.schedulers(config)
    mean = torch.as_tensor(normalization['action_mean'], device=device)
    scale = torch.as_tensor(normalization['action_std'], device=device)
    for timestep in (20, 50, 90):
        errors, reconstructions = [], []
        for seed in SEEDS:
            seed_errors, seed_reconstructed = [], []
            for start in range(0, len(items), 8):
                batch = to_device(default_collate(items[start:start+8]), device)
                noise = noise_for(cases[start:start+8], seed, model.horizon, device)
                t = torch.full((len(noise),), timestep, dtype=torch.long, device=device)
                noisy = scheduler.add_noise(batch['actions'], noise, t)
                estimate = model.denoiser(noisy, t, model.encoder(batch))
                alpha = scheduler.alphas_cumprod[timestep].to(device)
                reconstructed = (noisy-(1-alpha).sqrt()*estimate)/alpha.sqrt()
                seed_errors.append((estimate-noise).float().cpu().numpy())
                seed_reconstructed.append((reconstructed*scale+mean).float().cpu().numpy())
            errors.append(np.concatenate(seed_errors))
            reconstructions.append(np.concatenate(seed_reconstructed))
        errors, reconstructed = np.stack(errors), np.stack(reconstructions)
        finite = bool(np.isfinite(errors).all() and np.isfinite(reconstructed).all())
        repeated_mask = np.tile(mask, (3, 1))
        selected = errors.reshape(-1, model.horizon, 6)[repeated_mask]
        summaries[str(timestep)] = dict(finite=finite, normalized_epsilon_component_rmse=float(np.sqrt(np.mean(selected**2))),
            first_step_epsilon_error_gain=float(((1-scheduler.alphas_cumprod[timestep])/scheduler.alphas_cumprod[timestep]).sqrt()),
            reconstruction=action_metrics(reconstructed.reshape(-1, model.horizon, 6), np.tile(truth, (3, 1, 1)), repeated_mask, limits))
        archives[f'epsilon_error_t{timestep}'] = errors
        archives[f'reconstructed_t{timestep}'] = reconstructed
        print(f'known-noise probe timestep={timestep}', flush=True)
    np.savez_compressed(output / 'epsilon_probe.npz', **archives)
    return dict(by_timestep=summaries, action_mask='all valid in denoiser; labels mask only statistics',
        limitation='Known forward-noise reconstruction on teacher conditions is not free-running policy success or a causal diagnosis.')


@torch.no_grad()
def compare(model, config, normalization, device, dataset, previous, output, limits):
    episodes = sorted({c['episode'] for c in previous['cases']})
    cases = [dict(index=i, episode=str(dataset.episodes[e][0]), tick=int(dataset.arrays(e)['action_ticks'][start]))
        for i, (e, start, end, phase) in enumerate(dataset.windows)
        if str(dataset.episodes[e][0]) in episodes and phase == 1
        and int(dataset.arrays(e)['action_ticks'][start]) in range(50, 5000, 200)]
    items = [dataset[c['index']] for c in cases]
    labels = default_collate(items)
    truth = labels['actions'].numpy()*np.array(normalization['action_std'])+np.array(normalization['action_mean'])
    mask = labels['action_mask'].numpy().copy()
    mask[:, 4:] = False
    archives = dict(truth=truth, legal_prefix_mask=mask)
    summaries, noise_hashes = {}, {}
    for count in COUNTS:
        summaries[str(count)], noise_hashes[str(count)] = {}, {}
        revised = inference_config(config, count)
        for seed in SEEDS:
            predictions, hashes = [], []
            for start in range(0, len(items), 8):
                batch = to_device(default_collate(items[start:start+8]), device)
                noise = noise_for(cases[start:start+8], seed, model.horizon, device)
                hashes.extend(hashlib.sha256(n.cpu().numpy().tobytes()).hexdigest() for n in noise)
                predictions.append(sample_from_noise(model, revised, batch, normalization, noise).float().cpu().numpy())
            prediction = np.concatenate(predictions)
            archives[f'leading_{count}_seed_{seed}'] = prediction
            noise_hashes[str(count)][str(seed)] = hashes
            time_bins = {}
            for lo, hi in ((0, 2000), (2000, 5000)):
                indices = np.array([i for i,c in enumerate(cases) if lo <= c['tick'] < hi])
                p,t,m = prediction[indices],truth[indices],mask[indices]
                time_bins[f'{lo/1000:g}-{hi/1000:g}s'] = dict(metrics=action_metrics(p,t,m,limits), direction=direction_metrics(p[m],t[m]))
            summaries[str(count)][str(seed)] = dict(metrics=action_metrics(prediction,truth,mask,limits),
                direction=direction_metrics(prediction[mask],truth[mask]), by_time=time_bins,
                by_episode={episode: action_metrics(prediction[idx],truth[idx],mask[idx],limits)
                    for episode in episodes for idx in [np.array([i for i,c in enumerate(cases) if c['episode']==episode])]})
            print(f'leading steps={count} seed={seed}: {len(cases)} windows', flush=True)
    np.savez_compressed(output / 'samples.npz', **archives)
    assert noise_hashes['10'] == noise_hashes['20'] == noise_hashes['50']
    finite = all(np.isfinite(value).all() for key,value in archives.items() if key.startswith('leading_'))
    candidate = select_candidate(summaries) if finite else None
    result = dict(status='completed' if finite else 'nonfinite_samples', all_samples_finite=finite, cases=cases,
        by_steps_and_seed=summaries, initial_noise_sha256=noise_hashes, noise_pairing_verified=True,
        candidate_steps=candidate, gate='Every seed: angular RMSE <=90% of 10-step; linear RMSE <=105%; no higher clipping fraction.',
        acceptance_status='not_evaluated', physics=[])
    if finite and candidate is None:
        result['epsilon_probe'] = epsilon_probe(model,config,normalization,device,items,cases,truth,mask,limits,output)
    if candidate is not None:
        revised = inference_config(config,candidate)
        for episode in episodes:
            rollout = run_prefix_policy(model,revised,normalization,episode,device,execute_steps=4)
            rollout['predicted_velocity'] = rollout_metrics(rollout,limits)
            write_json(output / f'{Path(episode).name}_leading_{candidate}.json',rollout)
            result['physics'].append({k: rollout[k] for k in ('source_episode','pickup','success','failure_reason','simulated_s','contact_peak_n','wrist_peak_n')})
            print(f'physical steps={candidate}: pickup={rollout["pickup"]} reason={rollout["failure_reason"]}',flush=True)
            if not rollout['pickup'] or rollout['failure_reason'] in ('invalid_command','nonfinite_state'):
                break
    return result


def run(args):
    checkpoint_path, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
    parent = check_checkpoint(checkpoint)
    checkpoint_sha256 = sha256(checkpoint_path)
    previous = verify_evidence(args.previous_report,checkpoint_sha256)
    if previous['mode'] != 'dense_acquire':
        raise ValueError('Requires the audited dense ACQUIRE report')
    config,normalization = checkpoint['config'],checkpoint['normalization']
    device,hardware = setup(config,args.device)
    model = dit.ActionDiT(config).to(device)
    model.load_state_dict(checkpoint['ema'])
    model.eval().requires_grad_(False)
    dataset = ActionWindows(ROOT/config['dataset'],'validation',model.horizon,normalization)
    robot = load_json('configs/robots/panda.json')
    limits = (robot['linear_speed_limit'],robot['angular_speed_limit'])
    output.mkdir(parents=True,exist_ok=False)
    snapshot = output/'tool_snapshot'
    snapshot.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file():
            shutil.copyfile(file,snapshot/file.name)
    with spacing_override('leading'):
        samplers = {str(count): settings(config,count) for count in COUNTS}
    provenance = dict(schema_version=1,mode='step_sensitivity',diagnostic=True,checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,training_step=50000,weights='ema',parent=parent,source_check='passed',
        original_source_hashes=checkpoint['source_hashes'],training_config=config,normalization=normalization,
        hardware=hardware,precision='bf16_autocast' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32',
        diffusers_version=diffusers.__version__,samplers=samplers,noise_seeds=list(SEEDS),
        previous_report=str(Path(args.previous_report).resolve()),previous_report_sha256=sha256(args.previous_report),
        tool_sha256={p.name:sha256(p) for p in snapshot.iterdir()},optimizer_updates=0,formal_test_run=False,
        dp_v1='not_frozen',m5_status='incomplete',scope='three_normal_validation_scenes_early_teacher_conditions',
        limitation='Leading counts change the initial timestep; effects cannot be attributed solely to number of steps.')
    write_json(output/'provenance.json',provenance)
    try:
        with spacing_override('leading'),precision(device):
            result = compare(model,config,normalization,device,dataset,previous,output,limits)
    except Exception as error:
        write_json(output/'report.json',dict(provenance,status='error',error=f'{type(error).__name__}: {error}'))
        raise
    report = dict(provenance,**result,source_unchanged=input_hashes()==checkpoint['source_hashes'])
    report['evidence_sha256'] = {str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='report.json'}
    write_json(output/'report.json',report)
    if not report['source_unchanged']:
        raise ValueError('Original runtime inputs changed')
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--previous-report',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = run(args)
    print('report:',Path(args.output).resolve()/'report.json',flush=True)
    raise SystemExit(0 if report['status']=='completed' else 1)
