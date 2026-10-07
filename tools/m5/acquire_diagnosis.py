"""Dense ACQUIRE diagnostics followed by a matched 1/4-step execution comparison."""

import argparse
import hashlib
from pathlib import Path
import shutil

import diffusers
import numpy as np
import torch
from torch.utils.data import default_collate

from diagnose_sampling import check_checkpoint, action_metrics, rollout_metrics, sampler_settings, spacing_override
from prefix_rollout import noise_seed_at_tick, run_prefix_policy
from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies import dit, evaluation
from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.runtime import precision, setup, to_device
from feedingrobot.sim.model import ROOT, load_json


def verify_evidence(path, checkpoint_sha256):
    path = Path(path).resolve()
    report = read_json(path)
    if (report.get('checkpoint_sha256') != checkpoint_sha256 or not report.get('source_unchanged')
            or report.get('diagnostic') is not True or report.get('weights') != 'ema'):
        raise ValueError('Input evidence does not match the audited 50k EMA checkpoint')
    for relative, expected in report['evidence_sha256'].items():
        if sha256(path.parent / relative) != expected:
            raise ValueError(f'Changed evidence: {relative}')
    return report


@torch.no_grad()
def sample_from_noise(model, config, batch, normalization, noise):
    # Same denoising steps as dit.sample_actions, with explicitly paired initial noise.
    _, scheduler = dit.schedulers(config)
    scheduler.set_timesteps(config['diffusion']['inference_steps'], device=batch['states'].device)
    condition = model.encoder(batch)
    actions = noise.clone()
    for step in scheduler.timesteps:
        prediction = model.denoiser(actions, step.expand(len(actions)), condition)
        actions = scheduler.step(prediction, step, actions, eta=0).prev_sample
    mean = torch.as_tensor(normalization['action_mean'], device=actions.device)
    scale = torch.as_tensor(normalization['action_std'], device=actions.device)
    return actions * scale + mean


def condition_item(record):
    result = {}
    for key in ('states', 'history', 'interaction', 'state_mask', 'history_mask'):
        value = np.asarray(record[key], dtype=bool if key.endswith('_mask') else np.float32)
        result[key] = torch.as_tensor(value)
    result['phase'] = torch.tensor(record['phase_index'], dtype=torch.long)
    return result


def angle_degrees(a, b):
    relative = np.asarray(a).reshape(3, 3) @ np.asarray(b).reshape(3, 3).T
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))


def pose_reference(observations, ticks, teacher_arrays, fields):
    teacher_ticks = np.asarray(teacher_arrays['action_ticks'])
    teacher = np.asarray(teacher_arrays['action_observations'])
    eligible = (teacher_ticks <= 15000) & (teacher_arrays['action_phases'] == 1)
    path_ticks, path = teacher_ticks[eligible], teacher[eligible]
    rows = []
    for observation, tick in zip(observations, ticks):
        if tick > 15000:
            continue
        index = np.searchsorted(teacher_ticks, tick)
        if index == len(teacher_ticks) or teacher_ticks[index] != tick:
            continue
        reference = teacher[index]
        position = observation[fields['tcp_position']]
        nearest = int(np.linalg.norm(path[:, fields['tcp_position']] - position, axis=1).argmin())
        rows.append(dict(tick=int(tick), time_reference_position_mm=float(np.linalg.norm(position-reference[fields['tcp_position']])*1000),
            time_reference_angle_deg=angle_degrees(observation[fields['tcp_rotation']], reference[fields['tcp_rotation']]),
            nearest_position_teacher_tick=int(path_ticks[nearest]),
            nearest_path_position_mm=float(np.linalg.norm(position-path[nearest, fields['tcp_position']])*1000),
            nearest_path_angle_deg=angle_degrees(observation[fields['tcp_rotation']], path[nearest, fields['tcp_rotation']])))
    return dict(rows=rows, limitation='Time/path reference differences are not a known target pose for off-demonstration states.')


def direction_metrics(prediction, truth):
    result = {}
    for name, columns, active in (('linear', slice(0, 3), .003), ('angular', slice(3, 6), .1)):
        reference, estimate = truth[..., columns], prediction[..., columns]
        norms = np.linalg.norm(reference, axis=-1)
        selected = norms >= active
        if not selected.any():
            result[name] = dict(active_actions=0, reference_speed_threshold=active)
            continue
        dot = np.sum(estimate[selected]*reference[selected], axis=-1)
        ratio = dot/norms[selected]**2
        cosine = dot/np.maximum(norms[selected]*np.linalg.norm(estimate[selected], axis=-1), 1e-12)
        result[name] = dict(active_actions=int(selected.sum()), reference_speed_threshold=active,
                           projection_ratio_p50=float(np.median(ratio)), cosine_p50=float(np.median(cosine)))
    return result


@torch.no_grad()
def dense(model, config, normalization, device, dataset, previous, previous_path, output, limits):
    episodes = [Path(p) for p in previous['completed_scenes']]
    conditions = read_json(Path(previous_path).parent / 'conditions.json')
    episode_indices = {str(path): i for i, (path, _) in enumerate(dataset.episodes)}
    windows = {(e, int(dataset.arrays(e)['action_ticks'][start])): i
               for i, (e, start, end, phase) in enumerate(dataset.windows) if phase == 1 and int(dataset.arrays(e)['action_ticks'][start]) <= 15000}
    items, cases, truth, masks, references = [], [], [], [], {}
    mean, scale = np.array(normalization['action_mean']), np.array(normalization['action_std'])
    for episode in episodes:
        e = episode_indices[str(episode)]
        live = {r['tick']: r for r in conditions[episode.name] if r['phase'] == 'ACQUIRE' and r['tick'] <= 15000}
        arrays = dataset.arrays(e)
        raw_arrays = dict(arrays, action_observations=arrays['action_observations'],
                          action_phases=np.load(episode / 'action_phases.npy', mmap_mode='r'))
        normalized = np.array([r['states'][-1][:108] for r in live.values()])
        raw = normalized * np.array(normalization['observation_std']) + np.array(normalization['observation_mean'])
        references[episode.name] = pose_reference(raw, list(live), raw_arrays, dataset.fields)
        for tick in range(50, 15001, 200):
            if (e, tick) not in windows:
                continue
            teacher = dataset[windows[e, tick]]
            labels = teacher['actions'].numpy() * scale + mean
            condition_keys = ('states', 'history', 'phase', 'interaction', 'state_mask', 'history_mask')
            for domain in ('teacher', 'policy'):
                if domain == 'policy' and tick not in live:
                    continue
                items.append({k: teacher[k] for k in condition_keys} if domain == 'teacher' else condition_item(live[tick]))
                cases.append(dict(episode=str(episode), tick=tick, domain=domain))
                truth.append(labels)
                mask = teacher['action_mask'].numpy().copy()
                mask[4:] = False
                masks.append(mask)
    truth, masks = np.stack(truth), np.stack(masks)
    samples, initial_hashes = [], {}
    with spacing_override('leading'), precision(device):
        for seed in (0, 1, 2):
            predictions, hashes = [], []
            for start in range(0, len(items), 8):
                batch = to_device(default_collate(items[start:start+8]), device)
                noise = torch.stack([torch.randn((model.horizon, 6), device=device,
                    generator=torch.Generator(device=device).manual_seed(noise_seed_at_tick(seed, c['tick'])))
                    for c in cases[start:start+8]])
                hashes.extend(hashlib.sha256(n.cpu().numpy().tobytes()).hexdigest() for n in noise)
                predictions.append(sample_from_noise(model, config, batch, normalization, noise).float().cpu().numpy())
            samples.append(np.concatenate(predictions))
            initial_hashes[str(seed)] = hashes
            print(f'dense ACQUIRE seed={seed}: {len(items)} condition windows', flush=True)
    prediction = np.stack(samples)
    finite = bool(np.isfinite(prediction).all())
    np.savez_compressed(output / 'samples.npz', prediction=prediction, same_time_teacher_reference=truth, legal_prefix_mask=masks)
    summaries = {}
    if finite:
        for domain in ('teacher', 'policy'):
            summaries[domain] = {}
            for lo, hi in ((0, 2000), (2000, 5000), (5000, 8000), (8000, 15001)):
                indices = np.array([i for i, c in enumerate(cases) if c['domain'] == domain and lo <= c['tick'] < hi])
                if not len(indices):
                    continue
                p = prediction[:, indices].reshape(-1, model.horizon, 6)
                t, m = np.tile(truth[indices], (3, 1, 1)), np.tile(masks[indices], (3, 1))
                summaries[domain][f'{lo/1000:g}-{hi/1000:g}s'] = dict(windows=len(indices),
                    metrics=action_metrics(p, t, m, limits), direction=direction_metrics(p[m], t[m]))
    return dict(status='completed' if finite else 'nonfinite_samples', all_samples_finite=finite,
                scope='dense_acquire_offline', interval_ticks=200, noise_seeds=[0, 1, 2], cases=cases,
                initial_noise_sha256=initial_hashes, by_domain_and_time=summaries, pose_reference=references,
                teacher_domain='Legal demonstration commands are labels only on demonstration conditions.',
                policy_domain='Same-time teacher commands are references, not ground-truth actions for policy states.',
                inputs='Past observations only; no label action, future mask, seed, or tick fed into encoder.')


def cadence(model, config, normalization, device, dataset, dense_report, output, limits):
    if not dense_report['all_samples_finite']:
        return dict(status='nonfinite_samples', physics_skipped=True)
    episodes = [p for p, m in dataset.episodes if not m['scenario'].get('recover', False)][:3]
    results = {1: [], 4: []}
    with spacing_override('leading'):
        for episode in episodes:
            for execute_steps in (4, 1):
                result = run_prefix_policy(model, config, normalization, episode, device, execute_steps=execute_steps)
                result['predicted_velocity'] = rollout_metrics(result, limits)
                e = next(i for i, (p, _) in enumerate(dataset.episodes) if p == episode)
                arrays = dict(dataset.arrays(e), action_phases=np.load(episode / 'action_phases.npy', mmap_mode='r'))
                result['pose_reference'] = pose_reference(np.asarray([r['observation'] for r in result['trace']]),
                    [r['tick'] for r in result['trace']], arrays, dataset.fields)
                write_json(output / f'{episode.name}_prefix_{execute_steps}.json', result)
                results[execute_steps].append(result)
                print(f'cadence {episode.name} prefix={execute_steps}: pickup={result["pickup"]} success={result["success"]} reason={result["failure_reason"]} simulated_s={result["simulated_s"]:.3f}', flush=True)
                if result['failure_reason'] in ('invalid_command', 'nonfinite_state'):
                    return dict(status='unsafe_or_nonfinite', stopped_after=str(episode), execute_steps=execute_steps)
            # Complete the first matched case before expanding to the remaining two.
    arms = {}
    for steps, rows in results.items():
        arms[str(steps)] = dict(**evaluation.summarize(rows, config), pickup_successes=sum(r['pickup'] for r in rows),
            inference_calls=sum(len(r['inference_s']) for r in rows),
            inference_total_s=sum(sum(r['inference_s']) for r in rows),
            wall_total_s=sum(r['wall_s'] for r in rows),
            outcomes=[{k: r[k] for k in ('source_episode', 'pickup', 'success', 'failure_reason', 'simulated_s', 'wall_s')} for r in rows])
    return dict(status='completed', scope='three_normal_scenes_cadence_diagnostic', arms=arms,
        noise_pairing='base_seed=0; seed=base_seed*1000003+tick//50; identical initial noise at common planning ticks',
        horizon=16, execute_steps=[4, 1], replan_ticks=[200, 50],
        acceptance_status='not_evaluated', realtime_status='not_verified', teacher_in_execution=False)


def run(args):
    checkpoint_path, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    parent = check_checkpoint(checkpoint)
    checkpoint_sha256 = sha256(checkpoint_path)
    previous = verify_evidence(args.previous_report, checkpoint_sha256)
    if previous['mode'] != ('pickup' if args.mode == 'dense' else 'dense_acquire'):
        raise ValueError('dense requires the prior pickup report; cadence requires the dense report')
    config, normalization = checkpoint['config'], checkpoint['normalization']
    device, hardware = setup(config, args.device)
    model = dit.ActionDiT(config).to(device)
    model.load_state_dict(checkpoint['ema'])
    model.eval().requires_grad_(False)
    dataset = ActionWindows(ROOT / config['dataset'], 'validation', model.horizon, normalization)
    robot = load_json('configs/robots/panda.json')
    limits = (robot['linear_speed_limit'], robot['angular_speed_limit'])
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'tool_snapshot'
    snapshot.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file():
            shutil.copyfile(file, snapshot / file.name)
    provenance = dict(schema_version=1, mode='dense_acquire' if args.mode == 'dense' else 'cadence', diagnostic=True,
        checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256, training_step=50000, weights='ema',
        parent=parent, source_check='passed', original_source_hashes=checkpoint['source_hashes'], hardware=hardware,
        precision='bf16_autocast' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32',
        diffusers_version=diffusers.__version__,
        sampler=sampler_settings(config, 'leading'), training_config=config, normalization=normalization,
        original_evaluator_sha256=sha256(ROOT / 'src/feedingrobot/policies/evaluation.py'),
        tool_sha256={p.name: sha256(p) for p in snapshot.iterdir()},
        previous_report=str(Path(args.previous_report).resolve()), previous_report_sha256=sha256(args.previous_report),
        optimizer_updates=0, formal_test_run=False, dp_v1='not_frozen', m5_status='incomplete')
    write_json(output / 'provenance.json', provenance)
    try:
        if args.mode == 'dense':
            result = dense(model, config, normalization, device, dataset, previous, args.previous_report, output, limits)
        else:
            result = cadence(model, config, normalization, device, dataset, previous, output, limits)
    except Exception as error:
        write_json(output / 'report.json', dict(provenance, status='error', error=f'{type(error).__name__}: {error}'))
        raise
    report = dict(provenance, **result, source_unchanged=input_hashes() == checkpoint['source_hashes'])
    report['evidence_sha256'] = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file() and p.name != 'report.json'}
    write_json(output / 'report.json', report)
    if not report['source_unchanged']:
        raise ValueError('Original runtime inputs changed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dense', 'cadence'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--previous-report', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args)
    print('report:', Path(args.output).resolve() / 'report.json', flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
