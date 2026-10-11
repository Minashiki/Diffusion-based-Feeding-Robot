"""V5 selection, trailing/100 FP32 sampling and fitted-parameter resume."""

import copy
import json

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from test_m5 import compute_budget, config, dataset, episode
from feedingrobot.envs.feeding import EXECUTION_STATES
from feedingrobot.policies import audit, training
from feedingrobot.policies.dit import noise_loss, sample_actions, schedulers
from feedingrobot.policies.dit_v5 import MinaDiTV5
from feedingrobot.policies.dp_normalizer import fit_dp_limits
from feedingrobot.policies.models import check_normalization, create_model, resolve_config
from feedingrobot.policies.normalization import denormalize_actions
from feedingrobot.policies.runtime import precision


def v5(config):
    settings = copy.deepcopy(config)
    settings['model'].update(name='MinaDiTV5', depth=6)
    settings['diffusion'].update(inference_steps=100, timestep_spacing='trailing',
                                 prediction_type='sample', clip_sample=True)
    settings['training']['precision'] = 'fp32'
    return settings


def add_commands(dataset):
    for path in dataset.directory.glob('*/*/manifest.json'):
        for name in ('observations', 'action_observations'):
            values = np.load(path.parent / f'{name}.npy')
            field = dataset.fields['execution_status']
            values[:, field] = 0
            values[:, field.start + EXECUTION_STATES.index('idle')] = 1
            np.save(path.parent / f'{name}.npy', values)
        (path.parent / 'commands.json').write_text(json.dumps([]))


def test_v5_config_changes_only_version_and_sampler(config):
    previous = resolve_config(name='MinaDiTV4')
    expected = copy.deepcopy(previous)
    expected['model']['name'] = 'MinaDiTV5'
    expected['diffusion'].update(inference_steps=100, timestep_spacing='trailing')
    assert resolve_config(name='MinaDiTV5') == expected
    assert resolve_config()['model']['name'] == 'MinaDiTV5'
    assert isinstance(create_model(v5(config)), MinaDiTV5)
    with pytest.raises(ValueError, match='differ'):
        resolve_config('configs/minadit_v4.json', 'MinaDiTV5')
    for name in ('MinaDiTv1', 'MinaDiTV2', 'MinaDiTV3'):
        with pytest.raises(ValueError, match='Unknown model'):
            resolve_config(name=name)


def test_v5_requires_its_versioned_train_normalization(config, dataset):
    add_commands(dataset)
    normalization = fit_dp_limits(dataset)
    check_normalization(v5(config), normalization)
    assert normalization['schema_version'] == 5 and normalization['kind'] == 'dp_limits'
    for key, value in (('schema_version', 4), ('source_split', 'validation'),
                       ('observation_timing', 'recorded'), ('clipping', True), ('kind', 'limits')):
        with pytest.raises(ValueError):
            check_normalization(v5(config), dict(normalization, **{key: value}))
    with pytest.raises(ValueError, match='normalization'):
        check_normalization(resolve_config(name='MinaDiTV4'), normalization)


def test_v5_sampling_uses_all_trailing_steps_and_returns_physical_actions(config, dataset):
    add_commands(dataset)
    settings = v5(config)
    normalization = fit_dp_limits(dataset)
    dataset.normalization = normalization
    batch = default_collate([dataset[0]])
    observed = []
    class Oracle:
        horizon = 16
        def encoder(self, batch):
            assert not torch.is_autocast_enabled('cpu')
            return None
        def denoiser(self, actions, steps, condition):
            observed.append((int(steps[0]), actions.dtype))
            return torch.full_like(actions, 2.)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = sample_actions(Oracle(), settings, batch, normalization, torch.Generator().manual_seed(7))
        assert torch.is_autocast_enabled('cpu')
    assert observed == [(step, torch.float32) for step in range(99, -1, -1)]
    expected = denormalize_actions(torch.ones_like(actual), normalization)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_v5_fp32_training_overrides_outer_autocast(config, dataset):
    add_commands(dataset)
    settings = v5(config)
    dataset.normalization = fit_dp_limits(dataset)
    batch = default_collate([dataset[0]])
    model = create_model(settings)
    observed = []
    handle = model.output.register_forward_hook(lambda module, inputs, output: observed.append(output.dtype))
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with precision(torch.device('cpu'), settings):
                scheduler, _ = schedulers(settings)
                loss = noise_loss(model, scheduler, batch)
            assert torch.is_autocast_enabled('cpu')
        assert observed == [torch.float32] and loss.dtype == torch.float32
        loss.backward()
    finally:
        handle.remove()


def test_v5_resume_reuses_fit_and_matches_continuous(config, dataset, tmp_path, monkeypatch):
    episode(dataset.directory, split='validation')
    add_commands(dataset)
    settings = v5(config)
    settings['dataset'] = str(dataset.directory)
    settings['training'].update(validation_every=2, validation_batches=1, validation_action_samples=2)
    parent = dict(parent_sha256='parent', dataset_binding='data')
    monkeypatch.setattr(training, 'parent_check', lambda _: parent)
    monkeypatch.setattr(training, 'input_hashes', lambda: {'code': 'fixed'})
    monkeypatch.setattr(audit, 'input_hashes', lambda: {'code': 'fixed'})
    def setup(config, device):
        torch.manual_seed(config['training']['seed'])
        torch.set_num_threads(1)
        return torch.device('cpu'), {'device': 'cpu'}
    monkeypatch.setattr(training, 'setup', setup)
    continuous, resumed = tmp_path / 'continuous', tmp_path / 'resumed'
    training.train(settings, continuous, device='cpu', updates=4, headless=True)
    training.train(settings, resumed, device='cpu', updates=2, headless=True)
    def refit(*args):
        raise AssertionError('Resume must reuse checkpoint normalization')
    monkeypatch.setattr(training, 'fit_dp_limits', refit)
    training.train(settings, resumed, device='cpu', updates=4, resume=resumed / 'last.pt', headless=True)
    a, b = (torch.load(path / 'last.pt', weights_only=False) for path in (continuous, resumed))
    assert a['normalization'] == b['normalization']
    assert a['normalization']['kind'] == 'dp_limits' and a['normalization']['schema_version'] == 5
    assert a['normalization']['observation_timing'] == 'pre_action'
    assert a['normalization']['source_binding']['m4_dataset_binding'] == 'data'
    assert a['model_name'] == b['model_name'] == 'MinaDiTV5'
    records = [json.loads(row) for row in (continuous / 'metrics.jsonl').read_text().splitlines()]
    assert np.isfinite(records[-1]['validation_actions']['first_four']['linear']['vector_rmse'])
    for section in ('model', 'ema'):
        for name in a[section]:
            torch.testing.assert_close(a[section][name], b[section][name], rtol=0, atol=0)
