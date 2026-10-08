"""MinaDiT selection, causal denoising, train scaling and exact resume."""

import copy
import importlib.util
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from test_m5 import compute_budget, config, dataset, episode
from feedingrobot.policies import audit, training
from feedingrobot.policies.data import ActionWindows, features, fit_normalization
from feedingrobot.policies.dit import ActionDiT, noise_loss, sample_actions, schedulers
from feedingrobot.policies.dit_v2 import MinaDiTV2
from feedingrobot.policies.legacy import RUNTIME, verify_runtime
from feedingrobot.policies.models import check_normalization, create_model, model_name, resolve_config
from feedingrobot.policies.normalization import denormalize_actions, fit_minmax, normalize_actions, update_ranges


def v2(config):
    result = copy.deepcopy(config)
    result['model'].update(name='MinaDiTV2', depth=6)
    return result


def enable_branches(model):
    torch.nn.init.normal_(model.output.weight, std=.02)
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.05)


def test_default_selection_and_explicit_conflicts(config):
    default = resolve_config()
    assert model_name(default) == 'MinaDiTV3'
    assert default['dataset'] == resolve_config(name='MinaDiTv1')['dataset']
    assert 'correction_report' not in default and 'correction_audit' not in default
    assert model_name(resolve_config(name='MinaDiTv1')) == 'MinaDiTv1'
    assert isinstance(create_model(config), ActionDiT)
    assert isinstance(create_model(v2(config)), MinaDiTV2)
    with pytest.raises(ValueError, match='differ'):
        resolve_config('configs/dp_dit.json', 'MinaDiTV2')
    wrong = v2(config)
    wrong['model']['depth'] = 12
    with pytest.raises(ValueError, match='six blocks'):
        create_model(wrong)


def test_historical_eval_routes_and_rejects_model_mismatch(config, tmp_path, monkeypatch):
    from feedingrobot.policies import legacy
    from feedingrobot.scripts.eval_dp import evaluate_checkpoint
    path = tmp_path / 'legacy.pt'
    source = {'historical': 'record'}
    torch.save(dict(schema_version=1, config=config, source_hashes=source), path)
    output = tmp_path / 'evaluation'
    calls = []
    def run(entry, arguments, *, expected):
        calls.append((entry, arguments, expected))
        output.mkdir()
        (output / 'report.json').write_text('{"status":"failed"}')
        return 1
    monkeypatch.setattr(legacy, 'run', run)
    report = evaluate_checkpoint(path, 'validation', output, 'cpu', True, 'MinaDiTv1')
    assert report['status'] == 'failed'
    assert calls[0][0] == 'feedingrobot.scripts.eval_dp' and calls[0][2] == source
    assert '--headless' in calls[0][1]
    with pytest.raises(ValueError, match='checkpoint differ'):
        evaluate_checkpoint(path, 'validation', output, 'cpu', True, 'MinaDiTV2')
    assert len(calls) == 1


def test_v1_snapshot_parameters_and_predictions_remain_exact(config, dataset):
    verify_runtime()
    spec = importlib.util.spec_from_file_location('minadit_legacy_dit', RUNTIME / 'src/feedingrobot/policies/dit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(23)
    original = module.ActionDiT(config)
    torch.manual_seed(23)
    current = create_model(config)
    assert original.state_dict().keys() == current.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
    enable_branches(current)
    original.load_state_dict(current.state_dict())
    batch = default_collate([dataset[0], dataset[-1]])
    actions = torch.randn_like(batch['actions'])
    steps = torch.tensor([10, 50])
    torch.testing.assert_close(original.denoiser(actions, steps, original.encoder(batch), batch['action_mask']),
                               current.denoiser(actions, steps, current.encoder(batch), batch['action_mask']),
                               rtol=0, atol=0)
    norm = fit_normalization(dataset)
    before = module.sample_actions(original, config, batch, norm, torch.Generator().manual_seed(4))
    after = sample_actions(current, config, batch, norm, torch.Generator().manual_seed(4))
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_v2_padding_parameter_count_and_future_independence(config, dataset):
    settings = resolve_config(name='MinaDiTV2')
    with torch.device('meta'):
        full = create_model(settings)
    assert sum(p.numel() for p in full.parameters()) == 41007462
    model = create_model(v2(config)).eval()
    assert len(model.blocks) == 6
    assert not any(name.startswith('action_projection.') for name in model.state_dict())
    actions = torch.randn(2, 16, 6)
    padded = model.action_projection(actions)
    torch.testing.assert_close(padded[..., :6], actions, rtol=0, atol=0)
    assert torch.count_nonzero(padded[..., 6:]) == 0
    enable_branches(model)
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


def test_minmax_train_range_inverse_constants_and_categories(dataset):
    norm = fit_minmax(dataset)
    assert norm['kind'] == 'minmax' and norm['source_split'] == 'train'
    raw = [dataset[i] for i in range(len(dataset))]
    dataset.normalization = norm
    for i, original in enumerate(raw):
        scaled = dataset[i]
        for name, mask in (('states', 'state_mask'), ('history', 'history_mask'), ('actions', 'action_mask')):
            legal = scaled[name][scaled[mask]]
            assert legal.min() >= -1e-6 and legal.max() <= 1+1e-6
            assert torch.count_nonzero(scaled[name][~scaled[mask]]) == 0
        for name in ('stage', 'interaction', 'execution_status'):
            field = dataset.fields[name]
            torch.testing.assert_close(original['states'][:, field], scaled['states'][:, field], rtol=0, atol=0)
        for name in ('phase', 'interaction', 'state_mask', 'history_mask', 'action_mask'):
            torch.testing.assert_close(original[name], scaled[name], rtol=0, atol=0)
        decoded = denormalize_actions(scaled['actions'].numpy(), norm)
        np.testing.assert_allclose(decoded[scaled['action_mask']], original['actions'][original['action_mask']], atol=1e-8)
    assert norm['action_scale'][0] == 1 and norm['action_offset'][0] == 0
    arbitrary = np.array([[-3., .01, -.1, .2, -.5, .6]])
    np.testing.assert_allclose(denormalize_actions(normalize_actions(arbitrary, norm), norm), arbitrary, atol=1e-12)


def test_minmax_rejects_nontrain_and_does_not_clip_or_use_future(dataset):
    norm = fit_minmax(dataset)
    episode(dataset.directory, split='validation')
    assert fit_minmax(dataset) == norm
    validation = ActionWindows(dataset.directory, 'validation')
    with pytest.raises(ValueError, match='train'):
        fit_minmax(validation)
    a = dataset.arrays(0)
    tick = int(a['action_ticks'][3])
    values = a['feature_values'].copy()
    before = features(a['feature_ticks'], values, a['feature_valid'], tick, dataset.fields, norm)
    values[a['feature_ticks'] > tick] = 10000
    after = features(a['feature_ticks'], values, a['feature_valid'], tick, dataset.fields, norm)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    outside = normalize_actions(np.array(norm['action_offset']) + np.array(norm['action_scale']) * 2, norm)
    np.testing.assert_allclose(outside, 2.)
    dataset.normalization = norm
    batch = default_collate([dataset[0], dataset[-1]])
    batch['actions'][batch['action_mask']] = 2
    unchanged = batch['actions'].clone()
    ranges = {}
    update_ranges(batch, ranges)
    assert ranges['actions']['above_fraction'] == [1.]*6
    assert ranges['actions']['max_deviation'] == [1.]*6
    torch.testing.assert_close(batch['actions'], unchanged, rtol=0, atol=0)
    online = {name: value for name, value in batch.items() if name not in ('actions', 'action_mask')}
    online_ranges = {}
    update_ranges(online, online_ranges)
    assert set(online_ranges) == {'states', 'history'}


def test_v2_noise_gradients_short_masks_sampling_and_scaler_guard(config, dataset):
    config = v2(config)
    norm = fit_minmax(dataset)
    dataset.normalization = norm
    batch = default_collate([dataset[0], dataset[-1]])
    model = create_model(config)
    enable_branches(model)
    scheduler, _ = schedulers(config)
    noise = torch.randn_like(batch['actions']) * 10
    steps = torch.tensor([50, 99])
    assert scheduler.add_noise(batch['actions'], noise, steps).abs().max() > 1
    loss = noise_loss(model, scheduler, batch, noise=noise, timesteps=steps)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].self_attention.in_proj_weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    changed = dict(batch, actions=batch['actions'].clone())
    changed['actions'][~batch['action_mask']] = 10000
    torch.testing.assert_close(loss, noise_loss(model, scheduler, changed, noise=noise, timesteps=steps), rtol=0, atol=0)
    sampled = sample_actions(model, config, batch, norm, torch.Generator().manual_seed(7))
    assert sampled.shape == (2, 16, 6) and torch.isfinite(sampled).all()
    with pytest.raises(ValueError, match='normalization'):
        check_normalization(config, fit_normalization(ActionWindows(dataset.directory, 'train')))
    with pytest.raises(ValueError, match='normalization'):
        check_normalization(resolve_config(name='MinaDiTv1'), norm)


def test_v2_resume_matches_uninterrupted_training(config, dataset, tmp_path, monkeypatch):
    episode(dataset.directory, split='validation')
    episode(dataset.directory, recover=True, split='validation')
    config = v2(config)
    config['dataset'] = str(dataset.directory)
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
    training.train(config, continuous, device='cpu', updates=4, headless=True)
    training.train(config, resumed, device='cpu', updates=2, headless=True)
    training.train(config, resumed, device='cpu', updates=4, resume=resumed / 'last.pt', headless=True)
    a, b = (torch.load(path / 'last.pt', weights_only=False) for path in (continuous, resumed))
    assert a['model_name'] == b['model_name'] == 'MinaDiTV2'
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
        torch.testing.assert_close(a['ema'][name], b['ema'][name], rtol=0, atol=0)
    wrong = copy.deepcopy(config)
    wrong['model'].pop('name')
    with pytest.raises(ValueError, match='Checkpoint'):
        training.train(wrong, resumed, device='cpu', updates=5, resume=resumed / 'last.pt', headless=True)
    b['normalization']['kind'] = 'zscore'
    with pytest.raises(ValueError, match='normalization'):
        audit.checkpoint_check(b, config, parent)
