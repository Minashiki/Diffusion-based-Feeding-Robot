"""Read-only, paired diffusion diagnostics on expert validation observations."""

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

# Bound native libraries before importing Torch/NumPy when launched as a script.
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:6])

import diffusers
from diffusers import DDIMScheduler
import numpy as np
import torch
from torch.utils.data import default_collate

from feedingrobot.data.episodes import input_hashes
from feedingrobot.policies import dit
from feedingrobot.policies.audit import sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.models import check_normalization, create_model, model_name
from feedingrobot.policies.normalization import denormalize_actions
from feedingrobot.policies.runtime import setup, to_device
from feedingrobot.sim.events import PHASES
from feedingrobot.sim.model import ROOT, load_json


def paired_noise(count, horizon, seed):
    """Generate once on CPU, independently of sampler, conditions and batch size."""
    return torch.randn((count, horizon, 6), generator=torch.Generator().manual_seed(seed))


def noise_hash(noise):
    return hashlib.sha256(noise.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def clean_and_epsilon(prediction, noisy, timesteps, scheduler):
    alpha = scheduler.alphas_cumprod.to(device=noisy.device, dtype=noisy.dtype)[timesteps][:, None, None]
    if scheduler.config.prediction_type == 'epsilon':
        return (noisy - (1-alpha).sqrt() * prediction) / alpha.sqrt(), prediction
    if scheduler.config.prediction_type == 'sample':
        return prediction, (noisy - alpha.sqrt() * prediction) / (1-alpha).sqrt()
    raise ValueError('Diagnostics support epsilon and sample prediction only')


def action_metrics(prediction, truth, mask, limits):
    prediction, truth = np.asarray(prediction, dtype=float)[mask], np.asarray(truth, dtype=float)[mask]
    result = dict(actions=len(prediction), finite=bool(np.isfinite(prediction).all() and np.isfinite(truth).all()))
    if not len(prediction) or not result['finite']:
        return result
    for name, columns, limit in (('linear', slice(0, 3), limits[0]), ('angular', slice(3, 6), limits[1])):
        p, target = prediction[:, columns], truth[:, columns]
        speed, target_speed = np.linalg.norm(p, axis=-1), np.linalg.norm(target, axis=-1)
        direction_valid = (speed > 1e-8) & (target_speed > 1e-8)
        cosine = np.sum(p[direction_valid] * target[direction_valid], axis=-1) / (
            speed[direction_valid] * target_speed[direction_valid])
        result[name] = dict(vector_rmse=float(np.sqrt(np.mean(np.sum((p-target)**2, axis=-1)))),
                            component_rmse=np.sqrt(np.mean((p-target)**2, axis=0)).tolist(),
                            norm_p50=float(np.median(speed)), norm_p95=float(np.quantile(speed, .95)),
                            norm_max=float(speed.max()), over_limit_fraction=float(np.mean(speed > limit)),
                            limit=float(limit), direction_count=int(direction_valid.sum()),
                            direction_cosine_mean=float(cosine.mean()) if len(cosine) else None)
    return result


def grouped_metrics(prediction, truth, mask, cases, limits):
    groups = {'all': np.arange(len(cases))}
    for field in ('phase', 'kind', 'group_id'):
        for value in sorted({case[field] for case in cases}):
            groups[f'{field}:{value}'] = np.array([i for i, case in enumerate(cases) if case[field] == value])
    result = {}
    for name, indices in groups.items():
        legal = mask[indices]
        first = legal.copy()
        first[:, 1:] = False
        prefix = legal.copy()
        prefix[:, 4:] = False
        result[name] = dict(windows=len(indices), first_action=action_metrics(prediction[indices], truth[indices], first, limits),
                           first_four=action_metrics(prediction[indices], truth[indices], prefix, limits),
                           legal_horizon=action_metrics(prediction[indices], truth[indices], legal, limits))
    return result


def select_cases(dataset, count, seed, mode='random', phase_filter=None):
    """Round-robin phases and episodes; select fixed unique legal windows."""
    rng = np.random.default_rng(seed)
    pools = {}
    for phase, episodes in sorted(dataset.by_phase.items()):
        if phase_filter is not None and PHASES[phase] != phase_filter:
            continue
        queues = [([min(indices, key=lambda index: dataset.windows[index][1])] if mode == 'first'
                   else list(rng.permutation(indices))) for _, indices in sorted(episodes.items())]
        rng.shuffle(queues)
        pools[phase] = [queue[position] for position in range(max(map(len, queues)))
                        for queue in queues if position < len(queue)]
    chosen, available = [], sum(map(len, pools.values()))
    if not available:
        raise ValueError('No legal action windows match the selected phase')
    while len(chosen) < min(count, available):
        for phase in pools:
            if pools[phase] and len(chosen) < count:
                chosen.append(int(pools[phase].pop(0)))
    cases = []
    for index in chosen:
        episode, start, end, phase = dataset.windows[index]
        path, metadata = dataset.episodes[episode]
        cases.append(dict(index=index, episode=str(path), group_id=str(metadata['group_id']),
                          phase=PHASES[phase], kind='recovery' if metadata['scenario'].get('recover', False) else 'normal',
                          tick=int(dataset.arrays(episode)['action_ticks'][start]), legal_length=end-start))
    return cases


def condition_order(count, mode, seed):
    indices = np.arange(count)
    if mode == 'shuffle':
        if count < 2:
            raise ValueError('Condition shuffling requires at least two windows')
        permutation = np.random.default_rng(seed).permutation(count)
        indices[permutation] = np.roll(permutation, 1)
    elif mode not in ('original', 'ignore'):
        raise ValueError(f'Unknown condition ablation: {mode}')
    return indices


def encode_condition(model, batch, mode):
    condition = model.encoder(batch)
    if mode == 'ignore':
        condition = dit.Condition(torch.zeros_like(condition.global_features), torch.zeros_like(condition.tokens),
                                  torch.ones_like(condition.valid))
    return condition


@contextmanager
def compute_precision(device, precision):
    if precision == 'bf16':
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 diagnostics require a CUDA device supporting BF16')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            yield
    else:
        backends = (torch.backends.cuda.matmul, torch.backends.cudnn, torch.backends.cudnn.conv)
        previous = [backend.fp32_precision for backend in backends]
        try:
            for backend in backends:
                backend.fp32_precision = 'ieee'
            with torch.autocast(device.type, enabled=False):
                yield
        finally:
            for backend, value in zip(backends, previous):
                backend.fp32_precision = value


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def sample_chain(model, scheduler, condition, initial_noise, action_mask=None):
    actions = initial_noise.clone()
    for timestep in scheduler.timesteps:
        steps = timestep.expand(len(actions))
        prediction = model.denoiser(actions, steps, condition, action_mask)
        actions = scheduler.step(prediction.float(), timestep, actions, eta=0).prev_sample
        if action_mask is not None:
            actions = actions * action_mask[..., None]
    return actions


@torch.no_grad()
def diagnose(model, config, normalization, dataset, cases, device, args, limits):
    labels = default_collate([dataset[case['index']] for case in cases])
    truth = denormalize_actions(labels['actions'], normalization).numpy()
    mask = labels['action_mask'].numpy()
    ddpm, original_ddim = dit.schedulers(config)
    if ddpm.config.prediction_type not in ('epsilon', 'sample'):
        raise ValueError('Diagnostics support epsilon and sample prediction only')
    records = dict(forward_noised=[], sampled=[], samples=[])
    for seed in args.seeds:
        noise = paired_noise(len(cases), model.horizon, seed)
        hashes = [noise_hash(window) for window in noise]
        for mode in args.conditions:
            order = condition_order(len(cases), mode, args.selection_seed)
            for timestep in args.timesteps:
                started = time.perf_counter()
                clean_predictions, epsilon_predictions, target_predictions = [], [], []
                for start in range(0, len(cases), args.batch_size):
                    end = min(start + args.batch_size, len(cases))
                    batch = to_device({key: value[start:end] for key, value in labels.items()}, device)
                    donor = to_device({key: value[order[start:end]] for key, value in labels.items()}, device)
                    epsilon = noise[start:end].to(device)
                    steps = torch.full((end-start,), timestep, dtype=torch.long, device=device)
                    noisy = ddpm.add_noise(batch['actions'], epsilon, steps) * batch['action_mask'][..., None]
                    with compute_precision(device, args.precision):
                        predicted = model.denoiser(noisy, steps, encode_condition(model, donor, mode), batch['action_mask'])
                    clean, eps = clean_and_epsilon(predicted.float(), noisy, steps, ddpm)
                    clean_predictions.append(denormalize_actions(clean, normalization).cpu().numpy())
                    epsilon_predictions.append(eps.cpu().numpy())
                    target_predictions.append(predicted.float().cpu().numpy())
                synchronize(device)
                alpha = float(ddpm.alphas_cumprod[timestep])
                eps_error = (np.concatenate(epsilon_predictions).astype(float) - noise.numpy())[mask]
                target = noise.numpy() if ddpm.config.prediction_type == 'epsilon' else labels['actions'].numpy()
                target_error = (np.concatenate(target_predictions).astype(float) - target)[mask]
                finite = bool(np.isfinite(eps_error).all() and np.isfinite(target_error).all())
                records['forward_noised'].append(dict(seed=seed, condition=mode, timestep=timestep,
                    alpha_bar=alpha, epsilon_to_x0_gain=float(np.sqrt((1-alpha)/alpha)),
                    known_noise_sha256=hashes, model_target=ddpm.config.prediction_type, finite=finite,
                    model_target_mse=float(np.mean(target_error**2)) if finite else None,
                    epsilon_component_rmse=np.sqrt(np.mean(eps_error**2, axis=0)).tolist() if finite else None,
                    epsilon_mse=float(np.mean(eps_error**2)) if finite else None, elapsed_seconds=time.perf_counter()-started,
                    clean_action_metrics=grouped_metrics(np.concatenate(clean_predictions), truth, mask, cases, limits)))
            for spacing, steps_count in args.samplers:
                scheduler = DDIMScheduler.from_config(original_ddim.config, timestep_spacing=spacing)
                scheduler.set_timesteps(steps_count, device=device)
                started, predictions = time.perf_counter(), []
                for start in range(0, len(cases), args.batch_size):
                    end = min(start + args.batch_size, len(cases))
                    donor = to_device({key: value[order[start:end]] for key, value in labels.items()}, device)
                    action_mask = labels['action_mask'][start:end].to(device) if args.sampling_mask == 'teacher' else None
                    with compute_precision(device, args.precision):
                        predicted = sample_chain(model, scheduler, encode_condition(model, donor, mode),
                                                 noise[start:end].to(device), action_mask)
                    predictions.append(denormalize_actions(predicted, normalization).float().cpu().numpy())
                synchronize(device)
                prediction = np.concatenate(predictions)
                common = dict(seed=seed, condition=mode, spacing=spacing, steps=steps_count)
                records['sampled'].append(dict(**common, timesteps=scheduler.timesteps.tolist(),
                    initial_noise_sha256=hashes, condition_source_indices=order.tolist(),
                    all_generated_values_finite=bool(np.isfinite(prediction).all()),
                    elapsed_seconds=time.perf_counter()-started,
                    metrics=grouped_metrics(prediction, truth, mask, cases, limits)))
                for index, case in enumerate(cases):
                    records['samples'].append(dict(**common, case_index=index,
                        metrics=grouped_metrics(prediction[index:index+1], truth[index:index+1], mask[index:index+1], [case], limits)['all']))
                print(f'{mode} seed={seed} {spacing}:{steps_count} completed', flush=True)
    return records


def parse_csv(value, convert=str):
    return [convert(item) for item in value.split(',')]


def run(args):
    started = time.perf_counter()
    checkpoint_path, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    config = checkpoint['config']
    check_normalization(config, checkpoint['normalization'])
    normalization = copy.deepcopy(checkpoint['normalization'])
    normalization['observation_timing'] = args.observation_timing
    if checkpoint.get('model_name', model_name(config)) != model_name(config):
        raise ValueError('Checkpoint model name differs from its configuration')
    total_steps = config['diffusion']['train_steps']
    args.timesteps = (args.timesteps if args.timesteps is not None else
                      sorted({0, total_steps//4, total_steps//2, 3*total_steps//4, 9*total_steps//10, total_steps-1}))
    if (args.samples < 1 or args.batch_size < 1 or not args.seeds or not args.conditions
            or any(not 0 <= t < total_steps for t in args.timesteps)
            or any(spacing not in ('leading', 'trailing') or not 1 <= steps <= total_steps for spacing, steps in args.samplers)):
        raise ValueError('Invalid sample count, batch size, seed, timestep or sampler selection')
    runtime_config = copy.deepcopy(config)
    runtime_config['training'].update(cpu_budget=min(6, config['training']['cpu_budget']),
                                      torch_threads=min(6, config['training']['torch_threads']))
    device, hardware = setup(runtime_config, args.device)
    model = create_model(config).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint[args.weights])
    dataset = ActionWindows(ROOT / config['dataset'], 'validation', model.horizon, normalization)
    cases = select_cases(dataset, args.samples, args.selection_seed, args.case_mode, args.phase)
    robot = load_json(f'configs/robots/{config["robot_id"]}.json')
    limits = (robot['linear_speed_limit'], robot['angular_speed_limit'])
    output.mkdir(parents=True, exist_ok=False)
    result = diagnose(model, config, normalization, dataset, cases, device, args, limits)
    samples = result.pop('samples')
    (output / 'samples.json').write_text(json.dumps(samples, indent=2, allow_nan=False) + '\n')
    finite = (all(row['finite'] for row in result['forward_noised'])
              and all(row['all_generated_values_finite'] for row in result['sampled']))
    report = dict(schema_version=1, diagnostic=True, status='completed' if finite else 'nonfinite_predictions',
                  all_predictions_finite=finite, checkpoint=str(checkpoint_path),
                  checkpoint_sha256=sha256(checkpoint_path), training_step=checkpoint['step'], weights=args.weights,
                  model=model_name(config), training_config=config, normalization=normalization,
                  checkpoint_normalization=checkpoint['normalization'],
                  normalization_override=dict(observation_timing=args.observation_timing),
                  checkpoint_source_hashes=checkpoint.get('source_hashes'), tool_sha256=sha256(__file__),
                  current_source_hashes=input_hashes(),
                  source_binding_check='not_performed; historical source hashes are recorded, not claimed current',
                  split='validation', cases=cases, batch_size=args.batch_size, noise_seeds=args.seeds,
                  case_mode=args.case_mode, phase_filter=args.phase, selection_seed=args.selection_seed,
                  prediction_type=config['diffusion'].get('prediction_type', 'epsilon'), precision=args.precision,
                  hardware=hardware, diffusers_version=diffusers.__version__, optimizer_updates=0,
                  control_runs=0, sampling_action_mask=args.sampling_mask,
                  metric_action_mask='teacher legality; first four and whole legal horizon',
                  rmse_definition='vector RMSE = sqrt(mean(sum squared error over 3 physical components))',
                  epsilon_units='dimensionless diffusion noise; physical RMSE applies only to clean actions',
                  noise_pairing='identical CPU initial noise per window and seed across samplers/conditions/batches',
                  forward_noised_scope='expert actions with known noise; no epsilon ground truth claimed for generated chains',
                  samples_file='samples.json', samples_sha256=sha256(output / 'samples.json'),
                  elapsed_seconds=time.perf_counter()-started, **result)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--weights', choices=('ema', 'model'), default='ema')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--selection-seed', type=int, default=123)
    parser.add_argument('--case-mode', choices=('random', 'first'), default='random')
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--observation-timing', choices=('recorded', 'pre_action'), default='recorded')
    parser.add_argument('--seeds', type=lambda value: parse_csv(value, int), default=[0, 1])
    parser.add_argument('--timesteps', type=lambda value: parse_csv(value, int))
    parser.add_argument('--samplers', type=lambda value: [(name, int(steps)) for name, steps in
                        (item.split(':') for item in value.split(','))],
                        default=[('trailing', 10), ('trailing', 100), ('leading', 10)])
    parser.add_argument('--conditions', type=parse_csv, default=['original'])
    parser.add_argument('--sampling-mask', choices=('none', 'teacher'), default='none')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
