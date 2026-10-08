"""MinaDiTV3 raw physical inputs, strict FP32 inference and exact resume."""

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from test_m5 import compute_budget, config, dataset, episode
from feedingrobot.policies import audit, training
from feedingrobot.policies.data import fit_normalization
from feedingrobot.policies.dit import noise_loss, sample_actions, schedulers
from feedingrobot.policies.dit_v2 import MinaDiTV2
from feedingrobot.policies.dit_v3 import MinaDiTV3
from feedingrobot.policies.models import check_normalization, create_model, model_name, resolve_config
from feedingrobot.policies.normalization import denormalize_actions, fit_minmax, no_normalization, normalize_actions


def v3(config):
    result = copy.deepcopy(config)
    result['model'].update(name='MinaDiTV3', depth=6)
    return result


def enable_branches(model):
    torch.nn.init.normal_(model.output.weight, std=.02)
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.05)


def test_v3_default_and_explicit_version_selection(config):
    default = resolve_config()
    assert model_name(default) == 'MinaDiTV3'
    previous = resolve_config(name='MinaDiTV2')
    assert default['dataset'] == previous['dataset']
    assert default['model'] == dict(previous['model'], name='MinaDiTV3')
    assert isinstance(create_model(default), MinaDiTV3)
    assert isinstance(create_model(previous), MinaDiTV2)
    assert model_name(resolve_config(name='MinaDiTv1')) == 'MinaDiTv1'
    with pytest.raises(ValueError, match='differ'):
        resolve_config('configs/minadit_v2.json', 'MinaDiTV3')
    with torch.device('meta'):
        full = create_model(default)
    assert sum(p.numel() for p in full.parameters()) == 41007462
    settings = v3(config)
    settings['model']['depth'] = 12
    with pytest.raises(ValueError, match='six blocks'):
        create_model(settings)


def test_v3_inputs_and_action_units_are_unchanged(dataset):
    original = [dataset[i] for i in range(len(dataset))]
    normalization = no_normalization(dataset)
    assert normalization['schema_version'] == 3 and normalization['kind'] == 'none'
    assert normalization['count'] == len(dataset)
    assert normalization['observation_schema'] == audit.read_json(dataset.directory / 'normalization.json')['observation_schema']
    dataset.normalization = normalization
    for index, raw in enumerate(original):
        current = dataset[index]
        for name in raw:
            torch.testing.assert_close(current[name], raw[name], rtol=0, atol=0)
        for name, mask in (('states', 'state_mask'), ('history', 'history_mask'), ('actions', 'action_mask')):
            assert torch.count_nonzero(current[name][~current[mask]]) == 0
        episode_index, start, end, _ = dataset.windows[index]
        physical = dataset.arrays(episode_index)['actions'][start:end].astype(np.float32)
        np.testing.assert_array_equal(current['actions'][current['action_mask']].numpy(), physical)
    physical = np.array([[-3., .01, -.1, .2, -.5, 10.]], dtype=np.float64)
    for convert in (normalize_actions, denormalize_actions):
        np.testing.assert_array_equal(convert(physical, normalization), physical)
        tensor = torch.as_tensor(physical, dtype=torch.float32)
        result = convert(tensor, normalization)
        assert result.dtype == tensor.dtype
        torch.testing.assert_close(result, tensor, rtol=0, atol=0)


def test_v3_rejects_scaled_or_unversioned_normalization(config, dataset):
    settings = v3(config)
    raw = no_normalization(dataset)
    check_normalization(settings, raw)
    for incompatible in (fit_minmax(dataset), fit_normalization(dataset), dict(raw, schema_version=2)):
        with pytest.raises(ValueError, match='normalization'):
            check_normalization(settings, incompatible)
    for name in ('MinaDiTV2', 'MinaDiTv1'):
        with pytest.raises(ValueError, match='normalization'):
            check_normalization(resolve_config(name=name), raw)


def test_v3_raw_loss_gradients_masks_and_ddim(config, dataset):
    settings = v3(config)
    normalization = no_normalization(dataset)
    dataset.normalization = normalization
    batch = default_collate([dataset[0], dataset[-1]])
    model = create_model(settings)
    enable_branches(model)
    scheduler, _ = schedulers(settings)
    noise = torch.randn_like(batch['actions'])
    steps = torch.tensor([50, 99])
    loss = noise_loss(model, scheduler, batch, noise=noise, timesteps=steps)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].self_attention.in_proj_weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    changed = dict(batch, actions=batch['actions'].clone())
    changed['actions'][~batch['action_mask']] = 10000
    torch.testing.assert_close(loss, noise_loss(model, scheduler, changed, noise=noise, timesteps=steps), rtol=0, atol=0)
    model.eval()
    sampled = sample_actions(model, settings, batch, normalization, torch.Generator().manual_seed(7))
    repeated = sample_actions(model, settings, batch, normalization, torch.Generator().manual_seed(7))
    assert sampled.shape == (2, 16, 6) and sampled.dtype == torch.float32
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(sampled, repeated, rtol=0, atol=0)


@pytest.mark.parametrize('device_name', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_v3_sampling_overrides_outer_bf16_and_restores_precision(config, dataset, device_name):
    device = torch.device(device_name)
    settings = v3(config)
    batch = {name: value.to(device) for name, value in default_collate([dataset[0], dataset[-1]]).items()}
    model = create_model(settings).to(device).eval()
    enable_branches(model)
    normalization = no_normalization(dataset)
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    previous = matmul.fp32_precision, cudnn.fp32_precision, cudnn.conv.fp32_precision
    observed = []
    def record(module, inputs, output):
        observed.append(output.dtype)
        assert output.dtype == torch.float32
        assert not torch.is_autocast_enabled(device_name)
        assert matmul.fp32_precision == cudnn.fp32_precision == 'ieee'
        assert cudnn.conv.fp32_precision == 'ieee'
    handles = [module.register_forward_hook(record) for module in
               (model.encoder.state[0], model.encoder.conv1, model.output)]
    try:
        matmul.fp32_precision = cudnn.fp32_precision = 'tf32'
        cudnn.conv.fp32_precision = 'tf32'
        ordinary = sample_actions(model, settings, batch, normalization, torch.Generator(device=device).manual_seed(17))
        with torch.autocast(device_name, dtype=torch.bfloat16):
            mixed = sample_actions(model, settings, batch, normalization, torch.Generator(device=device).manual_seed(17))
            assert torch.is_autocast_enabled(device_name)
        assert observed and mixed.dtype == torch.float32
        torch.testing.assert_close(mixed, ordinary, rtol=0, atol=0)
        assert matmul.fp32_precision == cudnn.fp32_precision == 'tf32'
        assert cudnn.conv.fp32_precision == 'tf32'
    finally:
        for handle in handles:
            handle.remove()
        matmul.fp32_precision, cudnn.fp32_precision, cudnn.conv.fp32_precision = previous


def test_v3_failed_sampling_restores_precision(config, dataset, monkeypatch):
    settings = v3(config)
    model = create_model(settings)
    batch = default_collate([dataset[0]])
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    previous = matmul.fp32_precision, cudnn.fp32_precision, cudnn.conv.fp32_precision
    def fail(batch):
        assert not torch.is_autocast_enabled('cpu')
        assert matmul.fp32_precision == cudnn.fp32_precision == 'ieee'
        assert cudnn.conv.fp32_precision == 'ieee'
        raise RuntimeError('sampling failed')
    monkeypatch.setattr(model.encoder, 'forward', fail)
    try:
        matmul.fp32_precision = cudnn.fp32_precision = 'tf32'
        cudnn.conv.fp32_precision = 'tf32'
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with pytest.raises(RuntimeError, match='sampling failed'):
                sample_actions(model, settings, batch, no_normalization(dataset))
            assert torch.is_autocast_enabled('cpu')
        assert matmul.fp32_precision == cudnn.fp32_precision == 'tf32'
        assert cudnn.conv.fp32_precision == 'tf32'
    finally:
        matmul.fp32_precision, cudnn.fp32_precision, cudnn.conv.fp32_precision = previous


def test_v3_resume_matches_uninterrupted_training_and_rejects_other_versions(config, dataset, tmp_path, monkeypatch):
    episode(dataset.directory, split='validation')
    episode(dataset.directory, recover=True, split='validation')
    settings = v3(config)
    settings['dataset'] = str(dataset.directory)
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
    a, b = (torch.load(path / 'last.pt', weights_only=False) for path in (continuous, resumed))
    assert a['model_name'] == b['model_name'] == 'MinaDiTV3'
    assert a['normalization'] == b['normalization']
    assert a['normalization']['kind'] == 'none' and a['normalization']['schema_version'] == 3
    for section in ('model', 'ema'):
        for name in a[section]:
            torch.testing.assert_close(a[section][name], b[section][name], rtol=0, atol=0)
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    for index, state in a['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b['optimizer']['state'][index][name], rtol=0, atol=0)
    for name in ('MinaDiTV2', 'MinaDiTv1'):
        wrong = copy.deepcopy(settings)
        wrong['model']['name'] = name
        with pytest.raises(ValueError, match='Checkpoint'):
            training.train(wrong, resumed, device='cpu', updates=5, resume=resumed / 'last.pt', headless=True)
    b['normalization']['kind'] = 'minmax'
    with pytest.raises(ValueError, match='normalization'):
        audit.checkpoint_check(b, settings, parent)
