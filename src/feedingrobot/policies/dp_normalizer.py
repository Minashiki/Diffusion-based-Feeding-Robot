"""DP/DP3 FP32 per-channel limits, fitted on valid pre-action train inputs."""

from collections import OrderedDict
import copy
import json

import numpy as np
import torch

from feedingrobot.policies.data import ActionWindows, features, field_slices

HISTORY_FIELDS = ('compensated_wrench', 'q', 'dq', 'tcp_twist_world')
IDENTITY_FIELDS = ('stage', 'interaction', 'execution_status')


@torch.no_grad()
def fit_linear_limits(data):
    """DP/DP3 _fit: last_n_dims=1, FP32, limits [-1,1], fit_offset=True."""
    data = torch.as_tensor(data).to(dtype=torch.float32).reshape(-1, data.shape[-1])
    if not len(data) or not torch.isfinite(data).all():
        raise ValueError('No finite valid training values')
    input_min = data.min(axis=0).values
    input_max = data.max(axis=0).values
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)
    input_range = input_max - input_min
    ignore_dim = input_range < 1e-4
    input_range[ignore_dim] = 2.
    scale = 2. / input_range
    offset = -1. - scale * input_min
    offset[ignore_dim] = -input_min[ignore_dim]
    return dict(scale=scale.tolist(), offset=offset.tolist(), input_stats=dict(
                min=input_min.tolist(), max=input_max.tolist(),
                mean=input_mean.tolist(), std=input_std.tolist()))


def linear_transform(value, params, forward=True):
    """Use the upstream FP32 operation order; keep the caller's array container."""
    tensor = isinstance(value, torch.Tensor)
    value = (value.to(dtype=torch.float32) if tensor else
             torch.from_numpy(np.array(value, dtype=np.float32, copy=True)))
    scale = torch.as_tensor(params['scale'], dtype=torch.float32, device=value.device)
    offset = torch.as_tensor(params['offset'], dtype=torch.float32, device=value.device)
    shape = value.shape
    value = value.reshape(-1, len(scale))
    value = value * scale + offset if forward else (value - offset) / scale
    value = value.reshape(shape)
    return value if tensor else value.numpy()


def _query_indices(arrays, queries):
    available = np.flatnonzero(arrays['feature_valid'])
    positions = np.searchsorted(arrays['feature_ticks'][available], queries, side='right') - 1
    valid = positions >= 0
    return available[positions[valid]], queries[valid]


def fit_dp_limits(dataset):
    """Fit raw source rows once; history shares each physical observation channel."""
    from feedingrobot.policies.corpus import TrainingWindows
    if dataset.split != 'train' or dataset.normalization is not None:
        raise ValueError('DP limits fitting requires raw train data only')
    readers = ((dataset,) if isinstance(dataset, ActionWindows) else
               tuple(reader for reader in (dataset.base, dataset.extension) if reader is not None)
               if isinstance(dataset, TrainingWindows) and not dataset.extra else ())
    if not readers:
        raise ValueError('DP limits fitting requires raw action-window datasets')
    fields = dataset.fields
    width = sum(size for _, size, _ in dataset.schema['fields'])
    rows = {name: [] for name in fields}
    rows.update(action=[], velocity=[], state_age=[], history_age=[])
    for original in readers:
        if original.split != 'train' or original.normalization is not None:
            raise ValueError('DP limits fitting requires raw train data only')
        # Fit with exactly the same causal input semantics as training/inference,
        # without changing the caller or any frozen episode files.
        reader = copy.copy(original)
        reader.cache = OrderedDict()
        reader.normalization = dict(kind='none', observation_timing='pre_action')
        starts = [[] for _ in reader.episodes]
        for episode, start, _, _ in reader.windows:
            starts[episode].append(start)
        for episode, indices in enumerate(starts):
            if not indices:
                continue
            arrays = reader.arrays(episode)
            anchors = arrays['action_ticks'][indices]
            state_indices, state_queries = _query_indices(arrays, (anchors[:, None] + [-50, 0]).reshape(-1))
            history_indices, history_queries = _query_indices(arrays, (anchors[:, None] + np.arange(-180, 1, 20)).reshape(-1))
            for name, field in fields.items():
                selected = np.unique(np.r_[state_indices, history_indices] if name in HISTORY_FIELDS else state_indices)
                rows[name].append(arrays['feature_values'][selected, field].astype(np.float32))
            rows['action'].append(arrays['actions'][indices].astype(np.float32))
            rows['state_age'].append(((state_queries - arrays['feature_ticks'][state_indices]) * .001).astype(np.float32)[:, None])
            rows['history_age'].append(((history_queries - arrays['feature_ticks'][history_indices]) * .001).astype(np.float32)[:, None])
            velocities = []
            for tick in anchors:
                sample = features(arrays['feature_ticks'], arrays['feature_values'], arrays['feature_valid'], int(tick), fields)
                if sample['state_mask'].any():
                    velocities.append(sample['states'][sample['state_mask']][0, width:width+12])
            rows['velocity'].append(np.asarray(velocities, dtype=np.float32))
    fitted = {name: fit_linear_limits(np.concatenate(values)) for name, values in rows.items()}
    observation = dict(scale=[], offset=[], input_stats={name: [] for name in ('min', 'max', 'mean', 'std')})
    for name, field in fields.items():
        params = fitted[name]
        size = field.stop - field.start
        observation['scale'].extend([1.] * size if name in IDENTITY_FIELDS else params['scale'])
        observation['offset'].extend([0.] * size if name in IDENTITY_FIELDS else params['offset'])
        for stat in observation['input_stats']:
            observation['input_stats'][stat].extend(params['input_stats'][stat])
    return dict(schema_version=5, kind='dp_limits', source_split='train', count=len(dataset),
                observation_schema=json.loads(json.dumps(dataset.schema)), observation_timing='pre_action',
                clipping=False, fit_rows={name: sum(len(value) for value in values) for name, values in rows.items()},
                params_dict=dict(observation=observation, **{name: fitted[name] for name in
                                 ('action', 'velocity', 'state_age', 'history_age')}))


def check_dp_limits(normalization):
    if (normalization.get('schema_version') != 5 or normalization.get('source_split') != 'train'
            or normalization.get('observation_timing') != 'pre_action' or normalization.get('clipping') is not False):
        raise ValueError('V5 requires versioned train-only pre-action DP normalization')
    fields = field_slices(normalization['observation_schema'])
    width = sum(field.stop-field.start for field in fields.values())
    for name, size in dict(observation=width, velocity=12, state_age=1, history_age=1, action=6).items():
        params = normalization['params_dict'][name]
        stats = {stat: np.asarray(params['input_stats'][stat], dtype=np.float32) for stat in ('min', 'max', 'mean', 'std')}
        scale, offset = (np.asarray(params[part], dtype=np.float32) for part in ('scale', 'offset'))
        if (any(value.shape != (size,) or not np.isfinite(value).all() for value in [scale, offset, *stats.values()])
                or np.any(scale <= 0) or np.any(stats['max'] < stats['min']) or np.any(stats['std'] < 0)):
            raise ValueError(f'Invalid V5 {name} normalization')
        ranges = stats['max'] - stats['min']
        constant = ranges < 1e-4
        ranges[constant] = 2.
        expected_scale = 2. / ranges
        expected_offset = -1. - expected_scale * stats['min']
        expected_offset[constant] = -stats['min'][constant]
        if name == 'observation':
            for field in IDENTITY_FIELDS:
                expected_scale[fields[field]], expected_offset[fields[field]] = 1., 0.
        if not (np.array_equal(scale, expected_scale) and np.array_equal(offset, expected_offset)):
            raise ValueError(f'V5 {name} normalization differs from fitted DP limits')
