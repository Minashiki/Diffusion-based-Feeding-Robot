"""Audited 50k EMA sampling diagnostics; never trains, tests, or freezes a policy."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

import diffusers
from diffusers import DDIMScheduler
import numpy as np
import torch
from torch.utils.data import default_collate

from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies import dit, evaluation
from feedingrobot.policies.audit import checkpoint_check, parent_check, read_json, sha256
from feedingrobot.policies.data import ActionWindows, features
from feedingrobot.policies.runtime import precision, setup, to_device
from feedingrobot.sim.events import PHASES
from feedingrobot.sim.model import ROOT, load_json


SEEDS = (0, 1, 2)
BATCH_SIZE = 8


@contextmanager
def spacing_override(spacing):
    original = dit.schedulers

    def revised(config):
        ddpm, ddim = original(config)
        return ddpm, DDIMScheduler.from_config(ddim.config, timestep_spacing=spacing)

    dit.schedulers = revised
    try:
        yield
    finally:
        dit.schedulers = original


def sampler_settings(config, spacing):
    with spacing_override(spacing):
        _, scheduler = dit.schedulers(config)
    scheduler.set_timesteps(10)
    alpha = scheduler.alphas_cumprod[scheduler.timesteps[0]]
    return dict(config=dict(scheduler.config), timesteps=scheduler.timesteps.tolist(),
                inference_steps=10, eta=0., first_step_epsilon_error_gain=float(((1-alpha)/alpha).sqrt()))


def check_checkpoint(checkpoint):
    config = checkpoint['config']
    parent = parent_check(config)
    checkpoint_check(checkpoint, config, parent)
    if checkpoint.get('diagnostic') or checkpoint['step'] != 50000:
        raise ValueError('These diagnostics require the formal 50000-step checkpoint')
    if (config['diffusion'] != dict(train_steps=100, inference_steps=10, beta_schedule='squaredcos_cap_v2')
            or config['training']['seed'] != 0):
        raise ValueError('Expected the original 100/10-step cosine configuration and seed 0')
    return parent


def selected_windows(dataset):
    selected, seen = [], set()
    for index, (episode, start, end, phase) in enumerate(dataset.windows):
        metadata = dataset.episodes[episode][1]
        recovery = bool(metadata['scenario'].get('recover', False))
        if (recovery and PHASES[phase] != 'RECOVER') or (episode, phase) in seen:
            continue
        seen.add((episode, phase))
        selected.append(dict(index=index, episode=str(dataset.episodes[episode][0]),
                             phase=PHASES[phase], kind='recovery' if recovery else 'normal',
                             tick=int(dataset.arrays(episode)['action_ticks'][start]), legal_length=end-start))
    return selected


def action_metrics(prediction, truth, mask, limits):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    prediction, truth = prediction[mask], truth[mask]
    if not len(prediction):
        return dict(actions=0)
    if not np.isfinite(prediction).all():
        return dict(actions=len(prediction), finite=False)
    result = dict(actions=len(prediction), finite=True)
    for name, columns, limit in (('linear', slice(0, 3), limits[0]), ('angular', slice(3, 6), limits[1])):
        speeds = np.linalg.norm(prediction[:, columns], axis=-1)
        error = prediction[:, columns] - truth[:, columns]
        result[name] = dict(vector_rmse=float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))),
                            p50=float(np.median(speeds)), p90=float(np.quantile(speeds, .90)),
                            p99=float(np.quantile(speeds, .99)), maximum=float(speeds.max()),
                            clipped_fraction=float(np.mean(speeds > limit)), limit=limit)
    return result


def grouped_metrics(prediction, truth, mask, cases, limits):
    groups = {'all': np.arange(len(cases))}
    for field in ('phase', 'kind'):
        for value in sorted({c[field] for c in cases}):
            groups[f'{field}:{value}'] = np.array([i for i, c in enumerate(cases) if c[field] == value])
    result = {}
    for name, indices in groups.items():
        selected = mask[indices].copy()
        prefix = selected.copy()
        prefix[:, 4:] = False
        result[name] = dict(windows=len(indices),
                           legal_prefix=action_metrics(prediction[indices], truth[indices], prefix, limits),
                           legal_horizon=action_metrics(prediction[indices], truth[indices], selected, limits))
    return result


@torch.no_grad()
def offline(model, config, normalization, device, output, dataset, cases, limits):
    items = [dataset[c['index']] for c in cases]
    labels = default_collate(items)
    mean, scale = np.array(normalization['action_mean']), np.array(normalization['action_std'])
    truth = labels['actions'].numpy() * scale + mean
    mask = labels['action_mask'].numpy()
    archives = dict(truth=truth, action_mask=mask)
    summaries, finite = {}, True
    for spacing in ('trailing', 'leading'):
        summaries[spacing] = {}
        for seed in SEEDS:
            predictions, noise_hashes = [], []
            with spacing_override(spacing), precision(device):
                for start in range(0, len(items), BATCH_SIZE):
                    batch = to_device(default_collate(items[start:start+BATCH_SIZE]), device)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    noise = torch.randn((len(batch['states']), model.horizon, 6), device=device, generator=generator)
                    noise_hashes.extend(hashlib.sha256(n.cpu().numpy().tobytes()).hexdigest() for n in noise)
                    generator.manual_seed(seed)
                    actions = dit.sample_actions(model, config, batch, normalization, generator)
                    predictions.append(actions.float().cpu().numpy())
                    print(f'offline {spacing} seed={seed}: {min(start+BATCH_SIZE, len(items))}/{len(items)}', flush=True)
            prediction = np.concatenate(predictions)
            finite &= bool(np.isfinite(prediction).all())
            archives[f'{spacing}_{seed}'] = prediction
            summaries[spacing][str(seed)] = dict(initial_noise_sha256=noise_hashes,
                metrics=grouped_metrics(prediction, truth, mask, cases, limits))
    np.savez_compressed(output / 'samples.npz', **archives,
                        **{k: v.numpy() for k, v in labels.items() if k not in ('actions', 'action_mask')})
    pooled = {}
    for spacing in summaries:
        pooled[spacing] = grouped_metrics(np.concatenate([archives[f'{spacing}_{s}'] for s in SEEDS]),
            np.tile(truth, (len(SEEDS), 1, 1)), np.tile(mask, (len(SEEDS), 1)), cases * len(SEEDS), limits)
    for seed in SEEDS:
        assert summaries['trailing'][str(seed)]['initial_noise_sha256'] == summaries['leading'][str(seed)]['initial_noise_sha256']
    return dict(status='completed' if finite else 'nonfinite_samples', all_samples_finite=finite,
                windows=len(cases), episodes=len({c['episode'] for c in cases}), noise_seeds=list(SEEDS),
                batch_size=BATCH_SIZE, noise_pairing='same seed and batch; per-window initial-noise SHA256 recorded',
                inference_action_mask='all valid; label mask used only in metrics',
                rmse_definition='sqrt(mean(sum((prediction-label)^2 over the 3 vector components)))',
                cases=cases, by_seed=summaries, pooled=pooled)


def rollout_metrics(result, limits):
    actions = np.asarray([t['predicted'] for t in result['trace']], dtype=float).reshape(-1, 6)
    metrics = action_metrics(actions, np.zeros_like(actions), np.ones(len(actions), bool), limits)
    for component in ('linear', 'angular'):
        if component in metrics:
            metrics[component].pop('vector_rmse')
    return metrics


def history_baseline(checkpoint_path, episode, limits):
    path = checkpoint_path.parent / 'validation_50000' / f'{episode.name}.json'
    if not path.exists():
        return dict(status='not_available')
    result = read_json(path)
    return dict(status='historical_reference_only', path=str(path), sha256=sha256(path),
                pickup=result['pickup'], success=result['success'], failure_reason=result['failure_reason'],
                predicted_velocity=rollout_metrics(result, limits),
                limitation='Historical CUDA/autocast baseline is not precision-matched to the CPU FP32 candidate.')


def condition_records(replans, captured):
    return [dict(tick=t['tick'], phase=t['phase'],
                 **{('phase_index' if k == 'phase' else k): v[0] for k, v in batch.items()})
            for t, batch in zip(replans, captured)]


def rollouts(model, config, normalization, device, output, dataset, mode, limits, checkpoint_path):
    episodes = [path for path, m in dataset.episodes if mode == 'validation' or not m['scenario'].get('recover', False)]
    if mode == 'pickup':
        episodes = episodes[:3]
    results, conditions = [], {}
    with spacing_override('leading'):
        for episode in episodes:
            captured = []
            def capture(module, args):
                captured.append({k: v.detach().cpu().numpy() for k, v in args[0].items()})
            handle = model.encoder.register_forward_pre_hook(capture)
            try:
                result = evaluation.run_policy(model, config, normalization, episode, device, viewer=False)
            finally:
                handle.remove()
            result['predicted_velocity'] = rollout_metrics(result, limits)
            if mode == 'pickup':
                result['historical_trailing'] = history_baseline(checkpoint_path, episode, limits)
            write_json(output / f'{episode.name}.json', result)
            replans = [t for t in result['trace'] if t['queue_age_ticks'] == 0]
            conditions[episode.name] = condition_records(replans, captured)
            results.append(result)
            print(f'{mode} {len(results)}/{len(episodes)} {episode.name}: pickup={result["pickup"]} success={result["success"]} reason={result["failure_reason"]}', flush=True)
            if result['failure_reason'] == 'invalid_command':
                break
    write_json(output / 'conditions.json', conditions)
    return dict(**evaluation.summarize(results, config), episodes=len(results), split='validation',
                noise_seed=0, teacher_in_execution=False, headless=True,
                scope='pickup_probe' if mode == 'pickup' else 'full_validation_diagnostic',
                scene_list=[str(p) for p in episodes], completed_scenes=[r['source_episode'] for r in results],
                pickup_successes=sum(r['pickup'] for r in results),
                nonfinite_or_invalid_command=any(r['failure_reason'] == 'invalid_command' for r in results))


def read_offline_evidence(path, checkpoint_sha256, source_hashes):
    path = Path(path).resolve()
    report = read_json(path)
    if (report.get('mode') != 'offline' or report.get('diagnostic') is not True
            or report.get('checkpoint_sha256') != checkpoint_sha256
            or report.get('original_source_hashes') != source_hashes
            or report.get('source_unchanged') is not True):
        raise ValueError('Offline evidence must match this audited checkpoint and original source')
    for relative, expected in report['evidence_sha256'].items():
        if sha256(path.parent / relative) != expected:
            raise ValueError(f'Offline evidence changed: {relative}')
    return report


def run(args):
    path, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    parent = check_checkpoint(checkpoint)
    config, normalization = checkpoint['config'], checkpoint['normalization']
    device, hardware = setup(config, args.device)
    model = dit.ActionDiT(config).to(device)
    model.load_state_dict(checkpoint['ema'])
    model.eval().requires_grad_(False)
    dataset = ActionWindows(ROOT / config['dataset'], 'validation', model.horizon, normalization)
    cases = selected_windows(dataset)
    robot = load_json('configs/robots/panda.json')
    limits = (robot['linear_speed_limit'], robot['angular_speed_limit'])
    checkpoint_sha256 = sha256(path)
    prior = None if args.mode == 'offline' else read_offline_evidence(
        args.offline_report, checkpoint_sha256, checkpoint['source_hashes'])
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'tool_snapshot'
    snapshot.mkdir()
    tool_files = [p for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()]
    for file in tool_files:
        shutil.copyfile(file, snapshot / file.name)
    provenance = dict(schema_version=1, diagnostic=True, mode=args.mode, checkpoint=str(path),
        checkpoint_sha256=checkpoint_sha256, training_step=checkpoint['step'], weights='ema',
        source_check='passed', original_source_hashes=checkpoint['source_hashes'], parent=parent,
        training_config=config, normalization=normalization,
        tool_sha256={p.name: sha256(snapshot / p.name) for p in tool_files},
        diffusers_version=diffusers.__version__, hardware=hardware,
        precision='bf16_autocast' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32',
        samplers={s: sampler_settings(config, s) for s in ('trailing', 'leading')},
        optimizer_updates=0, formal_test_run=False, dp_v1='not_frozen', m5_status='incomplete')
    if prior is not None:
        provenance.update(offline_report=str(Path(args.offline_report).resolve()),
                          offline_report_sha256=sha256(args.offline_report))
    write_json(output / 'provenance.json', provenance)
    try:
        if args.mode == 'offline':
            result = offline(model, config, normalization, device, output, dataset, cases, limits)
        elif not prior['all_samples_finite']:
            result = dict(status='nonfinite_samples', episodes=0, physics_skipped=True)
        else:
            # A per-run finite check precedes physics, including manual validation runs.
            batch = to_device(default_collate([dataset[c['index']] for c in cases[:BATCH_SIZE]]), device)
            with spacing_override('leading'), precision(device):
                sample = dit.sample_actions(model, config, batch, normalization,
                    torch.Generator(device=device).manual_seed(0))
            if not torch.isfinite(sample).all():
                result = dict(status='nonfinite_samples', episodes=0, physics_skipped=True)
            else:
                result = rollouts(model, config, normalization, device, output, dataset, args.mode, limits, path)
                if args.mode == 'pickup' and result['pickup_successes'] == 0 and not result['nonfinite_or_invalid_command']:
                    from followup import diagnose_failure, scale_improved
                    if scale_improved(prior):
                        result['followup'] = diagnose_failure(model, config, normalization, device,
                            output, dataset, cases, limits, read_json(output / 'conditions.json'))
    except Exception as error:
        write_json(output / 'report.json', dict(**provenance, status='error', error=f'{type(error).__name__}: {error}'))
        raise
    unchanged = input_hashes() == checkpoint['source_hashes']
    result.update(source_unchanged=unchanged)
    report = dict(provenance, **result)
    report['evidence_sha256'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                               if p.is_file() and p.name != 'report.json'}
    write_json(output / 'report.json', report)
    if not unchanged:
        raise ValueError('Original runtime inputs changed during diagnostics')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('offline', 'pickup', 'validation'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--offline-report', help='Required before pickup/validation; audited offline report.json')
    args = parser.parse_args()
    if args.mode != 'offline' and not args.offline_report:
        parser.error('pickup/validation requires --offline-report')
    report = run(args)
    print(f'report: {Path(args.output).resolve() / "report.json"}', flush=True)
    return 0 if report['status'] in ('completed', 'passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
