"""Normalized epsilon/sample targets, physical sampling and exact V4 resume."""

import copy
import json

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from test_m5 import compute_budget, config, dataset, episode
from feedingrobot.policies import audit, training
from feedingrobot.policies.dit import noise_loss, sample_actions, schedulers
from feedingrobot.policies.dit_v4 import MinaDiTV4
from feedingrobot.policies.models import check_normalization, create_model, resolve_config
from feedingrobot.policies.normalization import denormalize_actions, fit_limits
from feedingrobot.policies.runtime import precision
from feedingrobot.envs.feeding import EXECUTION_STATES


def v4(config, target='sample'):
    settings = copy.deepcopy(config)
    settings['model'].update(name='MinaDiTV4', depth=6)
    settings['diffusion'].update(prediction_type=target, clip_sample=True)
    settings['training']['precision'] = 'fp32'
    return settings


def test_v4_explicit_version_and_fitted_limits(config, dataset):
    assert resolve_config()['model']['name'] == 'MinaDiTV5'
    assert resolve_config(name='MinaDiTV4')['diffusion']['prediction_type'] == 'sample'
    assert isinstance(create_model(v4(config)), MinaDiTV4)
    norm = fit_limits(dataset)
    norm['observation_timing'] = 'pre_action'
    check_normalization(v4(config), norm)
    for key, value in (('action_scale', [1.] * 6), ('action_min', []),
                       ('action_constant_mask', [False] * 6), ('clipping', True)):
        wrong = dict(norm, **{key: value})
        with pytest.raises(ValueError, match='normalization'):
            check_normalization(v4(config), wrong)


@pytest.mark.parametrize('name', ['MinaDiTV4', 'MinaDiTV5'])
def test_shared_architecture_padding_and_future_independence(config, dataset, name):
    with torch.device('meta'):
        full = create_model(resolve_config(name=name))
    assert sum(p.numel() for p in full.parameters()) == 41007462
    settings = v4(config)
    settings['model']['name'] = name
    model = create_model(settings).eval()
    assert len(model.blocks) == 6
    assert not any(key.startswith('action_projection.') for key in model.state_dict())
    actions = torch.randn(2, 16, 6)
    padded = model.action_projection(actions)
    torch.testing.assert_close(padded[..., :6], actions, rtol=0, atol=0)
    assert torch.count_nonzero(padded[..., 6:]) == 0
    torch.nn.init.normal_(model.output.weight, std=.02)
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.05)
    batch = default_collate([dataset[0], dataset[-1]])
    condition = model.encoder(batch)
    steps = torch.tensor([10, 70])
    before = model.denoiser(actions, steps, condition)
    future = actions.clone()
    future[:, 8:] += torch.randn_like(future[:, 8:]) * 1000
    after = model.denoiser(future, steps, condition)
    torch.testing.assert_close(before[:, :8], after[:, :8], rtol=0, atol=0)
    past = actions.clone()
    past[:, 0] += torch.randn_like(past[:, 0]) * 100
    assert not torch.allclose(before[:, 6], model.denoiser(past, steps, condition)[:, 6])


@pytest.mark.parametrize('target', ['epsilon', 'sample'])
def test_v4_loss_target_and_padding(config, dataset, target):
    settings = v4(config, target)
    norm = fit_limits(dataset)
    original = dataset[0]
    dataset.normalization = norm
    scaled = dataset[0]
    for name in ('states', 'history', 'phase', 'interaction'):
        torch.testing.assert_close(scaled[name], original[name], rtol=0, atol=0)
    batch = default_collate([scaled])
    model = create_model(settings)
    scheduler, _ = schedulers(settings)
    noise = torch.full_like(batch['actions'], .3)
    steps = torch.tensor([99])
    loss = noise_loss(model, scheduler, batch, noise=noise, timesteps=steps)
    expected = noise if target == 'epsilon' else batch['actions']
    mask = batch['action_mask'][..., None]
    torch.testing.assert_close(loss, (expected.square() * mask).sum() / (mask.sum()*6))
    changed = dict(batch, actions=batch['actions'].clone())
    changed['actions'][~batch['action_mask']] = 10000
    torch.testing.assert_close(loss, noise_loss(model, scheduler, changed, noise=noise, timesteps=steps))


def test_v4_sample_clips_clean_action_then_returns_physical_units(config, dataset):
    settings = v4(config)
    norm = fit_limits(dataset)
    batch = default_collate([dataset[0]])
    class Oracle:
        horizon = 16
        def encoder(self, batch):
            return None
        def denoiser(self, actions, steps, condition):
            return torch.full_like(actions, 2.)
    actual = sample_actions(Oracle(), settings, batch, norm, torch.Generator().manual_seed(7))
    expected = denormalize_actions(torch.ones_like(actual), norm)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    variable = ~np.array(norm['action_constant_mask'])
    np.testing.assert_allclose(actual[0, 0].numpy()[variable], np.array(norm['action_max'])[variable], atol=1e-7)


def test_v4_fp32_training_overrides_outer_autocast(config, dataset):
    settings = v4(config)
    model = create_model(settings)
    batch = default_collate([dataset[0]])
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


def test_v4_resume_matches_continuous(config, dataset, tmp_path, monkeypatch):
    episode(dataset.directory, split='validation')
    for path in dataset.directory.glob('*/*/manifest.json'):
        for name in ('observations', 'action_observations'):
            values = np.load(path.parent/f'{name}.npy')
            field = dataset.fields['execution_status']
            values[:, field] = 0
            values[:, field.start + EXECUTION_STATES.index('idle')] = 1
            np.save(path.parent/f'{name}.npy', values)
        (path.parent/'commands.json').write_text(json.dumps([]))
    settings = v4(config)
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
    training.train(settings, resumed, device='cpu', updates=4, resume=resumed / 'last.pt', headless=True)
    a, b = (torch.load(path/'last.pt', weights_only=False) for path in (continuous, resumed))
    assert a['normalization'] == b['normalization']
    assert a['normalization']['kind'] == 'limits' and a['config']['diffusion']['prediction_type'] == 'sample'
    assert a['normalization']['observation_timing'] == 'pre_action'
    records = [json.loads(row) for row in (continuous/'metrics.jsonl').read_text().splitlines()]
    assert 'validation_actions' in records[-1] and np.isfinite(records[-1]['validation_actions']['first_four']['linear']['vector_rmse'])
    for section in ('model', 'ema'):
        for name in a[section]:
            torch.testing.assert_close(a[section][name], b[section][name], rtol=0, atol=0)
