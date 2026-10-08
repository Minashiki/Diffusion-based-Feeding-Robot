"""Versioned raw, Min-Max and legacy Z-score action representations."""

import json

import numpy as np
import torch


def normalize_actions(actions, normalization):
    if normalization.get('kind') == 'none':
        return actions
    if normalization.get('kind') == 'minmax':
        return (actions - np.asarray(normalization['action_offset'])) / np.asarray(normalization['action_scale'])
    return (actions - np.asarray(normalization['action_mean'])) / np.asarray(normalization['action_std'])


def denormalize_actions(actions, normalization):
    if normalization.get('kind') == 'none':
        return actions
    names = ('action_offset', 'action_scale') if normalization.get('kind') == 'minmax' else ('action_mean', 'action_std')
    offset, scale = (normalization[name] for name in names)
    if isinstance(actions, torch.Tensor):
        offset = torch.as_tensor(offset, device=actions.device)
        scale = torch.as_tensor(scale, device=actions.device)
    else:
        offset, scale = np.asarray(offset), np.asarray(scale)
    return actions * scale + offset


def no_normalization(dataset):
    """Record raw input semantics without fitting any data statistics."""
    return dict(schema_version=3, kind='none', count=len(dataset),
                observation_schema=json.loads(json.dumps(dataset.schema)))


def update_ranges(batch, report):
    """Count held-out range excursions without modifying model inputs."""
    for name, mask in (('states', 'state_mask'), ('history', 'history_mask'), ('actions', 'action_mask')):
        if name not in batch:
            continue
        rows = batch[name][batch[mask]].detach().float().cpu().numpy()
        if not len(rows):
            continue
        stats = report.setdefault(name, dict(rows=0, below_count=[0]*rows.shape[-1],
                                            above_count=[0]*rows.shape[-1], max_deviation=[0.]*rows.shape[-1]))
        stats['rows'] += len(rows)
        stats['below_count'] = (np.asarray(stats['below_count']) + (rows < 0).sum(0)).tolist()
        stats['above_count'] = (np.asarray(stats['above_count']) + (rows > 1).sum(0)).tolist()
        stats['max_deviation'] = np.maximum(stats['max_deviation'], np.maximum(-rows, rows-1).clip(0).max(0)).tolist()
        stats['below_fraction'] = (np.asarray(stats['below_count']) / stats['rows']).tolist()
        stats['above_fraction'] = (np.asarray(stats['above_count']) / stats['rows']).tolist()


def fit_minmax(dataset):
    if dataset.split != 'train' or dataset.normalization is not None:
        raise ValueError('Min-Max fitting requires raw train data only')
    width = sum(size for _, size, _ in dataset.schema['fields'])
    columns = np.r_[tuple(np.arange(dataset.fields[name].start, dataset.fields[name].stop)
                         for name in ('compensated_wrench', 'q', 'dq', 'tcp_twist_world'))]
    sizes = dict(observation=width, velocity=12, state_age=1, history_age=1, action=6)
    low = {name: np.full(size, np.inf) for name, size in sizes.items()}
    high = {name: np.full(size, -np.inf) for name, size in sizes.items()}
    for index in range(len(dataset)):
        sample = dataset[index]
        states = sample['states'].numpy()[sample['state_mask'].numpy()]
        history = sample['history'].numpy()[sample['history_mask'].numpy()]
        values = dict(observation=states[:, :width], velocity=states[:, width:width+12],
                      state_age=states[:, -2:-1], history_age=history[:, -2:-1],
                      action=sample['actions'].numpy()[sample['action_mask'].numpy()])
        for name, rows in values.items():
            if len(rows):
                low[name] = np.minimum(low[name], rows.min(0))
                high[name] = np.maximum(high[name], rows.max(0))
        if len(history):
            low['observation'][columns] = np.minimum(low['observation'][columns], history[:, :len(columns)].min(0))
            high['observation'][columns] = np.maximum(high['observation'][columns], history[:, :len(columns)].max(0))
    result = dict(schema_version=2, kind='minmax', source_split='train', count=len(dataset),
                  observation_schema=json.loads(json.dumps(dataset.schema)), clipping=False)
    for name in sizes:
        if not (np.isfinite(low[name]).all() and np.isfinite(high[name]).all()):
            raise ValueError(f'No finite valid training values for {name}')
        scale = high[name] - low[name]
        result[f'{name}_offset'] = low[name].tolist()
        result[f'{name}_scale'] = np.where(scale > 1e-8, scale, 1.).tolist()
    for name in ('stage', 'interaction', 'execution_status'):
        field = dataset.fields[name]
        result['observation_offset'][field] = [0.] * (field.stop-field.start)
        result['observation_scale'][field] = [1.] * (field.stop-field.start)
    return result
