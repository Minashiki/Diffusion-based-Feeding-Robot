"""Explicit selection for the supported V4 and V5 policies."""

import numpy as np

from feedingrobot.policies.audit import read_json
from feedingrobot.policies.dit_v4 import MinaDiTV4
from feedingrobot.policies.dit_v5 import MinaDiTV5
from feedingrobot.policies.dp_normalizer import check_dp_limits
from feedingrobot.sim.model import ROOT

MODEL_NAMES = ('MinaDiTV4', 'MinaDiTV5')
DEFAULT_MODEL = 'MinaDiTV5'
CONFIGS = {'MinaDiTV4': 'configs/minadit_v4.json',
           'MinaDiTV5': 'configs/minadit_v5.json'}


def model_name(config):
    name = config['model'].get('name')
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model: {name}')
    return name


def resolve_config(path=None, name=None):
    if name is not None and name not in MODEL_NAMES:
        raise ValueError(f'Unknown model: {name}')
    config = read_json(ROOT / (path or CONFIGS[name or DEFAULT_MODEL]))
    if name is not None and model_name(config) != name:
        raise ValueError('Selected model and configuration differ')
    return config


def create_model(config):
    return {'MinaDiTV4': MinaDiTV4, 'MinaDiTV5': MinaDiTV5}[model_name(config)](config)


def check_normalization(config, normalization):
    expected = {'MinaDiTV4': 'limits', 'MinaDiTV5': 'dp_limits'}[model_name(config)]
    if normalization.get('kind') != expected:
        raise ValueError('Model and normalization kind differ')
    if expected == 'dp_limits':
        check_dp_limits(normalization)
    if expected == 'limits':
        if normalization.get('schema_version') != 4 or normalization.get('source_split') != 'train':
            raise ValueError('V4 requires versioned train-only action normalization')
        if normalization.get('observation_timing') != 'pre_action':
            raise ValueError('V4 normalization requires pre-action observation timing')
        for part in ('offset', 'scale'):
            value = np.asarray(normalization[f'action_{part}'])
            if (value.shape != (6,) or not np.isfinite(value).all()
                    or (part == 'scale' and np.any(value <= 0))):
                raise ValueError('Invalid V4 action normalization')
        low, high = (np.asarray(normalization.get(f'action_{part}', [])) for part in ('min', 'max'))
        constant = np.asarray(normalization.get('action_constant_mask', []))
        if (low.shape != (6,) or high.shape != (6,) or constant.shape != (6,)
                or constant.dtype != np.dtype(bool) or not np.isfinite(low).all()
                or not np.isfinite(high).all() or np.any(high < low)
                or normalization.get('clipping') is not False):
            raise ValueError('Invalid V4 action normalization limits')
        ranges = high - low
        expected_constant = ranges < 1e-4
        if (not np.array_equal(constant, expected_constant)
                or not np.allclose(normalization['action_offset'], np.where(constant, low, (low+high)/2), rtol=0, atol=1e-12)
                or not np.allclose(normalization['action_scale'], np.where(constant, 1., ranges/2), rtol=0, atol=1e-12)):
            raise ValueError('V4 action normalization differs from fitted limits')
