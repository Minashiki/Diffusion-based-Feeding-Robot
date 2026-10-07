"""Offline-only mask and sampled-condition coverage checks after failed pickup."""

from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import default_collate

from feedingrobot.data.episodes import write_json
from feedingrobot.policies import dit
from feedingrobot.policies.data import ActionWindows, features
from feedingrobot.policies.runtime import precision, to_device
from feedingrobot.sim.events import PHASES
from feedingrobot.sim.model import ROOT

from diagnose_sampling import BATCH_SIZE, grouped_metrics, spacing_override


def scale_improved(report):
    a = report['pooled']['trailing']['all']['legal_prefix']
    b = report['pooled']['leading']['all']['legal_prefix']
    return report['all_samples_finite'] and all(
        b[k]['p50'] <= b[k]['limit'] and b[k]['clipped_fraction'] < a[k]['clipped_fraction']
        for k in ('linear', 'angular'))


@contextmanager
def label_mask_override(model, mask):
    original = model.denoiser
    def masked(actions, steps, condition, action_mask=None):
        return original(actions, steps, condition, mask if action_mask is None else action_mask)
    model.denoiser = masked
    try:
        yield
    finally:
        del model.denoiser


@torch.no_grad()
def mask_comparison(model, config, normalization, device, dataset, limits):
    short = {}
    for i, (episode, start, end, phase) in enumerate(dataset.windows):
        if end-start < model.horizon:
            first, _ = short.get((episode, phase), (i, i))
            short[episode, phase] = first, i
    cases = []
    for indices in short.values():
        for i in dict.fromkeys(indices):
            episode, start, end, phase = dataset.windows[i]
            path, metadata = dataset.episodes[episode]
            cases.append(dict(index=i, episode=str(path), phase=PHASES[phase],
                kind='recovery' if metadata['scenario'].get('recover', False) else 'normal',
                tick=int(dataset.arrays(episode)['action_ticks'][start]), legal_length=end-start))
    if not cases:
        return dict(status='no_short_windows')
    items = [dataset[c['index']] for c in cases]
    labels = default_collate(items)
    truth = labels['actions'].numpy() * np.array(normalization['action_std']) + np.array(normalization['action_mean'])
    predictions = {False: [], True: []}
    with spacing_override('leading'), precision(device):
        for start in range(0, len(items), BATCH_SIZE):
            batch = to_device(default_collate(items[start:start+BATCH_SIZE]), device)
            for masked in (False, True):
                generator = torch.Generator(device=device).manual_seed(0)
                if masked:
                    with label_mask_override(model, batch['action_mask']):
                        sample = dit.sample_actions(model, config, batch, normalization, generator)
                else:
                    sample = dit.sample_actions(model, config, batch, normalization, generator)
                predictions[masked].append(sample.float().cpu().numpy())
    return dict(status='completed', diagnostic_only=True, noise_seed=0, cases=cases,
        selection='first/last short legal window per validation episode and phase, including ACQUIRE',
        all_valid=grouped_metrics(np.concatenate(predictions[False]), truth, labels['action_mask'].numpy(), cases, limits),
        label_mask=grouped_metrics(np.concatenate(predictions[True]), truth, labels['action_mask'].numpy(), cases, limits),
        limitation='Uses future teacher segment lengths offline; never inject this mask into physical inference.')


def condition_consistency(dataset, cases):
    maximum, masks_equal = 0., True
    for case in cases:
        episode, start, _, _ = dataset.windows[case['index']]
        a = dataset.arrays(episode)
        tick = int(a['action_ticks'][start])
        # Recreate the runtime's causal, bounded observation buffer on the same recorded observations.
        eligible = np.flatnonzero((a['feature_ticks'] <= tick) & a['feature_valid'])
        ticks = a['feature_ticks'][eligible]
        eligible = eligible[np.r_[ticks[1:] != ticks[:-1], True]][-32:]
        rebuilt = features(a['feature_ticks'][eligible], a['feature_values'][eligible],
                           a['feature_valid'][eligible], tick, dataset.fields, dataset.normalization)
        expected = dataset[case['index']]
        for key, value in rebuilt.items():
            if key.endswith('_mask'):
                masks_equal &= bool(np.array_equal(value, expected[key].numpy()))
            else:
                maximum = max(maximum, float(np.max(np.abs(value - expected[key].numpy()))))
    return dict(windows=len(cases), maximum_absolute_difference=maximum, masks_equal=masks_equal,
                same_features_function=True, status='matched' if masks_equal and maximum <= 1e-6 else 'different',
                limitation='Tests the assembler on identical observations; closed-loop observations can still drift.')


def vector(batch):
    return np.concatenate([np.asarray(batch[k]).reshape(-1) for k in
                           ('states', 'history', 'interaction', 'state_mask', 'history_mask')])


def representatives(dataset, phases):
    samples = []
    for phase, episodes in sorted(dataset.by_phase.items()):
        if PHASES[phase] not in phases:
            continue
        for episode, indices in sorted(episodes.items()):
            for index in dict.fromkeys((indices[0], indices[len(indices)//2], indices[-1])):
                _, start, _, _ = dataset.windows[index]
                samples.append(dict(phase=PHASES[phase], episode=str(dataset.episodes[episode][0]),
                    tick=int(dataset.arrays(episode)['action_ticks'][start]),
                    value=vector({k: v.numpy() for k, v in dataset[index].items()})))
    return samples


def coverage(config, normalization, validation, conditions):
    phases = {r['phase'] for rows in conditions.values() for r in rows}
    train = ActionWindows(ROOT / config['dataset'], 'train', config['model']['horizon'], normalization)
    train_samples, validation_samples = representatives(train, phases), representatives(validation, phases)
    report = {}
    for phase in sorted(phases):
        candidates = [r for r in train_samples if r['phase'] == phase]
        if not candidates:
            report[phase] = dict(status='no_train_labels')
            continue
        matrix = np.stack([r['value'] for r in candidates])
        def nearest(value):
            distances = np.sqrt(np.mean((matrix - value)**2, axis=1))
            i = int(distances.argmin())
            return dict(normalized_condition_rms_distance=float(distances[i]),
                        nearest_train={k: v for k, v in candidates[i].items() if k != 'value'})
        reference = [nearest(r['value'])['normalized_condition_rms_distance']
                     for r in validation_samples if r['phase'] == phase]
        failed = [dict(episode=episode, tick=r['tick'], **nearest(vector(r)))
                  for episode, rows in conditions.items() for r in rows if r['phase'] == phase]
        distances = [r['normalized_condition_rms_distance'] for r in failed]
        report[phase] = dict(sampled_train_windows=len(candidates), sampled_validation_windows=len(reference),
            failed_policy_conditions=len(failed), validation_distance_p95=float(np.quantile(reference, .95)) if reference else None,
            policy_distance_p50=float(np.median(distances)), policy_distance_p95=float(np.quantile(distances, .95)),
            nearest_by_policy_condition=failed)
    return dict(by_phase=report, sampling='first/middle/last legal window per episode and entered phase',
                coverage_gap='not_proven',
                limitation='Distances use sampled normalized conditions, not exhaustive demonstration coverage or a calibrated OOD threshold.')


def diagnose_failure(model, config, normalization, device, output, dataset, cases, limits, conditions):
    print('Pickup still failed: checking offline masks, causal feature assembly, and sampled train coverage...', flush=True)
    report = dict(diagnostic=True, trigger='scale_improved_but_pickup_0_of_3',
        mask_comparison=mask_comparison(model, config, normalization, device, dataset, limits),
        condition_consistency=condition_consistency(dataset, cases),
        demonstration_coverage=coverage(config, normalization, dataset, conditions),
        recommendation='Inspect the evidence before deciding on training or data changes; no optimizer updates performed.')
    write_json(output / 'followup.json', report)
    return dict(path=str(output / 'followup.json'), mask_comparison='completed',
                condition_consistency=report['condition_consistency']['status'], demonstration_coverage='sampled_not_proven')
