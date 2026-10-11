"""V4/V5 action limits and compatible historical dataset representations."""

import json

import numpy as np
import torch


def normalize_actions(actions, normalization):
    if normalization.get('kind') == 'dp_limits':
        from feedingrobot.policies.dp_normalizer import linear_transform
        return linear_transform(actions, normalization['params_dict']['action'])
    if normalization.get('kind') == 'none':
        return actions
    if normalization.get('kind') == 'limits':
        offset, scale = _limits_parameters(actions, normalization)
        return (actions - offset) / scale
    return (actions - np.asarray(normalization['action_mean'])) / np.asarray(normalization['action_std'])


def denormalize_actions(actions, normalization):
    if normalization.get('kind') == 'dp_limits':
        from feedingrobot.policies.dp_normalizer import linear_transform
        return linear_transform(actions, normalization['params_dict']['action'], forward=False)
    if normalization.get('kind') == 'none':
        return actions
    if normalization.get('kind') == 'limits':
        offset, scale = _limits_parameters(actions, normalization)
        return actions * scale + offset
    offset, scale = (normalization[name] for name in ('action_mean', 'action_std'))
    if isinstance(actions, torch.Tensor):
        offset = torch.as_tensor(offset, device=actions.device)
        scale = torch.as_tensor(scale, device=actions.device)
    else:
        offset, scale = np.asarray(offset), np.asarray(scale)
    return actions * scale + offset


def _limits_parameters(actions, normalization):
    values = (normalization['action_offset'], normalization['action_scale'])
    if isinstance(actions, torch.Tensor):
        return tuple(torch.as_tensor(value, dtype=actions.dtype, device=actions.device) for value in values)
    return tuple(np.asarray(value) for value in values)


def fit_limits(dataset):
    """Fit action limits from valid raw training windows; keep other inputs raw."""
    if dataset.split != 'train' or dataset.normalization is not None:
        raise ValueError('Limits fitting requires raw train data only')
    from feedingrobot.policies.data import ActionWindows
    from feedingrobot.policies.corpus import TrainingWindows
    readers = ((dataset,) if isinstance(dataset, ActionWindows) else
               tuple(reader for reader in (dataset.base, dataset.extension) if reader is not None)
               if isinstance(dataset, TrainingWindows) and not dataset.extra else ())
    low, high = np.full(6, np.inf), np.full(6, -np.inf)
    if readers:
        for reader in readers:
            if reader.split != 'train' or reader.normalization is not None:
                raise ValueError('Limits fitting requires raw train data only')
            starts = [[] for _ in reader.episodes]
            for episode, start, _, _ in reader.windows:
                starts[episode].append(start)
            for (directory, _), indices in zip(reader.episodes, starts):
                if indices:
                    # Every legal action starts a window. Match __getitem__'s
                    # FP32 labels, without duplicating future rows or features.
                    rows = np.load(directory / 'actions.npy', mmap_mode='r')[indices].astype(np.float32)
                    low = np.minimum(low, rows.min(0))
                    high = np.maximum(high, rows.max(0))
    else:
        for index in range(len(dataset)):
            sample = dataset[index]
            rows = sample['actions'][sample['action_mask']].detach().cpu().numpy()
            if len(rows):
                low = np.minimum(low, rows.min(0))
                high = np.maximum(high, rows.max(0))
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError('No finite valid training values for action')
    ranges = high - low
    constant = ranges < 1e-4
    # Match Diffusion Policy: constant channels become zero; near-constant
    # channels retain their small changes without amplifying them.
    offset = np.where(constant, low, (low + high) / 2)
    scale = np.where(constant, 1., ranges / 2)
    return dict(schema_version=4, kind='limits', source_split='train', count=len(dataset),
                observation_schema=json.loads(json.dumps(dataset.schema)), clipping=False,
                action_min=low.tolist(), action_max=high.tolist(), action_constant_mask=constant.tolist(),
                action_offset=offset.tolist(), action_scale=scale.tolist())
