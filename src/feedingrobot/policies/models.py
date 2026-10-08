"""Explicit model selection; unnamed historical configurations remain V1."""

import numpy as np

from feedingrobot.policies.audit import read_json
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.policies.dit_v2 import MinaDiTV2
from feedingrobot.policies.dit_v3 import MinaDiTV3
from feedingrobot.sim.model import ROOT

MODEL_NAMES = ('MinaDiTv1', 'MinaDiTV2', 'MinaDiTV3')
DEFAULT_MODEL = 'MinaDiTV3'
CONFIGS = {'MinaDiTv1': 'configs/dp_dit.json', 'MinaDiTV2': 'configs/minadit_v2.json',
           'MinaDiTV3': 'configs/minadit_v3.json'}
MinaDiTv1 = ActionDiT


def model_name(config):
    name = config['model'].get('name', 'MinaDiTv1')
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
    return {'MinaDiTv1': MinaDiTv1, 'MinaDiTV2': MinaDiTV2, 'MinaDiTV3': MinaDiTV3}[model_name(config)](config)


def check_normalization(config, normalization):
    expected = {'MinaDiTv1': 'zscore', 'MinaDiTV2': 'minmax', 'MinaDiTV3': 'none'}[model_name(config)]
    if normalization.get('kind', 'zscore') != expected:
        raise ValueError('Model and normalization kind differ')
    if expected == 'none' and normalization.get('schema_version') != 3:
        raise ValueError('V3 requires versioned no-normalization metadata')
    if expected == 'minmax':
        if normalization.get('schema_version') != 2 or normalization.get('source_split') != 'train':
            raise ValueError('V2 requires versioned train-only Min-Max normalization')
        width = sum(size for _, size, _ in normalization['observation_schema']['fields'])
        for name, size in dict(observation=width, velocity=12, state_age=1, history_age=1, action=6).items():
            offset, scale = (np.asarray(normalization[f'{name}_{part}']) for part in ('offset', 'scale'))
            if (offset.shape != (size,) or scale.shape != (size,) or not np.isfinite(offset).all()
                    or not np.isfinite(scale).all() or np.any(scale <= 0)):
                raise ValueError(f'Invalid {name} Min-Max normalization')
