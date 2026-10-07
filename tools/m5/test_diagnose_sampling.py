"""Regression checks for the independent diagnostics, outside the training hash set."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import diagnose_sampling as diagnostic
import followup
from feedingrobot.policies import audit, dit


class SmallModel:
    horizon = 16

    def __init__(self):
        self.inputs = []

    def encoder(self, batch):
        return None

    def denoiser(self, actions, steps, condition, action_mask=None):
        self.inputs.append((steps[0].item(), actions.clone(), action_mask))
        return actions * .1 if action_mask is None else actions * .1 * action_mask[..., None]


@pytest.fixture
def config():
    return dict(diffusion=dict(train_steps=100, inference_steps=10, beta_schedule='squaredcos_cap_v2'),
                training=dict(seed=0))


def sample(model, config, seed=7):
    return dit.sample_actions(model, config, dict(states=torch.zeros(2, 2, 122)),
        dict(action_mean=[0.]*6, action_std=[1.]*6), torch.Generator().manual_seed(seed))


def test_trailing_is_exactly_the_original_sampler(config):
    model = SmallModel()
    expected = sample(model, config)
    original = dit.schedulers
    train_config = dict(original(config)[0].config)
    with diagnostic.spacing_override('trailing'):
        actual = sample(model, config)
        assert dict(dit.schedulers(config)[0].config) == train_config
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dit.schedulers is original


def test_spacing_pairs_noise_but_changes_timesteps(config):
    model = SmallModel()
    with diagnostic.spacing_override('trailing'):
        trailing = sample(model, config)
    first = model.inputs[0]
    model.inputs.clear()
    with diagnostic.spacing_override('leading'):
        leading = sample(model, config)
    second = model.inputs[0]
    assert (first[0], second[0]) == (99, 90)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert first[2] is second[2] is None
    assert not torch.equal(trailing, leading)
    assert diagnostic.sampler_settings(config, 'leading')['timesteps'] == list(range(90, -1, -10))


def test_exception_restores_spacing_and_mask(config):
    original = dit.schedulers
    model = SmallModel()
    original_method = model.denoiser
    with pytest.raises(RuntimeError):
        with diagnostic.spacing_override('leading'), followup.label_mask_override(model, torch.ones(2, 16, dtype=torch.bool)):
            raise RuntimeError('interrupted')
    assert dit.schedulers is original
    assert model.denoiser == original_method


@pytest.mark.parametrize('change', ['source', 'data', 'config', 'step', 'diagnostic'])
def test_checkpoint_rejection_is_not_bypassed(config, monkeypatch, change):
    parent = dict(parent_sha256='parent', dataset_binding='data')
    checkpoint = dict(schema_version=1, config=config, parent=parent, source_hashes={'physics': 'old'},
                      step=50000, diagnostic=False)
    monkeypatch.setattr(diagnostic, 'parent_check', lambda _: parent)
    monkeypatch.setattr(audit, 'input_hashes', lambda: {'physics': 'old'})
    assert diagnostic.check_checkpoint(checkpoint) == parent
    changed = copy.deepcopy(checkpoint)
    if change == 'source':
        changed['source_hashes']['physics'] = 'forged'
    elif change == 'data':
        changed['parent']['dataset_binding'] = 'other'
    elif change == 'config':
        changed['config']['diffusion']['inference_steps'] = 50
    elif change == 'step':
        changed['step'] = 10000
    else:
        changed['diagnostic'] = True
    with pytest.raises(ValueError):
        diagnostic.check_checkpoint(changed)


def test_metrics_ignore_padding_and_use_vector_rmse():
    prediction = np.ones((1, 16, 6)) * .01
    truth, mask = np.zeros_like(prediction), np.zeros((1, 16), bool)
    mask[:, :2] = True
    prediction[:, 2:] = np.nan
    metrics = diagnostic.action_metrics(prediction, truth, mask, (.05, .5))
    assert metrics['finite'] and metrics['actions'] == 2
    assert metrics['linear']['vector_rmse'] == pytest.approx(np.sqrt(3) * .01)
    assert metrics['linear']['clipped_fraction'] == 0
    mask[:, 2] = True
    assert not diagnostic.action_metrics(prediction, truth, mask, (.05, .5))['finite']


def test_selection_is_first_legal_window_per_episode_phase():
    class Dataset:
        episodes = [(Path('normal'), {'scenario': {}}), (Path('recovery'), {'scenario': {'recover': True}})]
        windows = [(0, 0, 16, 1), (0, 1, 17, 1), (0, 2, 3, 3), (1, 0, 8, 7), (1, 1, 8, 7), (1, 8, 9, 3)]
        def arrays(self, episode):
            return {'action_ticks': np.arange(20)*50}
    selected = diagnostic.selected_windows(Dataset())
    assert [r['index'] for r in selected] == [0, 2, 3]
    assert [r['legal_length'] for r in selected] == [16, 1, 8]


def test_offline_evidence_is_bound_and_nonfinite_flag_is_preserved(tmp_path):
    sample_file = tmp_path / 'samples.npz'
    sample_file.write_bytes(b'samples')
    report = dict(mode='offline', diagnostic=True, checkpoint_sha256='checkpoint',
        original_source_hashes={'code': 'original'}, source_unchanged=True, all_samples_finite=False,
        evidence_sha256={'samples.npz': audit.sha256(sample_file)})
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    loaded = diagnostic.read_offline_evidence(path, 'checkpoint', {'code': 'original'})
    assert not loaded['all_samples_finite']
    with pytest.raises(ValueError):
        diagnostic.read_offline_evidence(path, 'other', {'code': 'original'})
    sample_file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        diagnostic.read_offline_evidence(path, 'checkpoint', {'code': 'original'})


@pytest.mark.parametrize('argument', ['--freeze', '--mode=test'])
def test_cli_cannot_test_or_freeze(monkeypatch, argument):
    monkeypatch.setattr(sys, 'argv', ['diagnose_sampling.py', '--mode', 'offline', '--checkpoint', '50k.pt',
                                     '--device', 'cpu', '--output', '/tmp/unused_m5_test', argument])
    with pytest.raises(SystemExit) as error:
        diagnostic.main()
    assert error.value.code == 2


def test_rollout_requires_prior_offline_evidence(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['diagnose_sampling.py', '--mode', 'pickup', '--checkpoint', '50k.pt',
                                     '--device', 'cpu', '--output', '/tmp/unused_m5_test'])
    with pytest.raises(SystemExit) as error:
        diagnostic.main()
    assert error.value.code == 2


def test_tool_files_are_not_added_to_original_training_hashes():
    assert not any(name.startswith('tools/m5/') for name in diagnostic.input_hashes())


def test_velocity_statistics_do_not_claim_teacher_error():
    result = {'trace': [{'predicted': [.01, 0., 0., 0., .1, 0.]}]}
    metrics = diagnostic.rollout_metrics(result, (.05, .5))
    assert 'vector_rmse' not in metrics['linear']


def test_nonfinite_offline_result_really_prevents_physics(tmp_path, config, monkeypatch):
    checkpoint_path = tmp_path / 'step_50000.pt'
    checkpoint = dict(config=dict(config, dataset='unused'), normalization={}, ema={}, step=50000,
                      source_hashes={})
    torch.save(checkpoint, checkpoint_path)
    class Model:
        horizon = 16
        def to(self, device):
            return self
        def load_state_dict(self, state):
            pass
        def eval(self):
            return self
        def requires_grad_(self, enabled):
            return self
    monkeypatch.setattr(diagnostic, 'check_checkpoint', lambda _: {'status': 'passed'})
    monkeypatch.setattr(diagnostic, 'setup', lambda *a: (torch.device('cpu'), {}))
    monkeypatch.setattr(dit, 'ActionDiT', lambda _: Model())
    monkeypatch.setattr(diagnostic, 'ActionWindows', lambda *a: SimpleNamespace(windows=[], episodes=[]))
    monkeypatch.setattr(diagnostic, 'read_offline_evidence', lambda *a: {'all_samples_finite': False})
    monkeypatch.setattr(diagnostic, 'input_hashes', lambda: {})
    def forbidden(*args, **kwargs):
        pytest.fail('A nonfinite offline result reached physics')
    monkeypatch.setattr(diagnostic, 'rollouts', forbidden)
    prior = tmp_path / 'offline.json'
    prior.write_text('{}')
    output = tmp_path / 'pickup'
    report = diagnostic.run(SimpleNamespace(checkpoint=str(checkpoint_path), output=str(output),
        mode='pickup', device='cpu', offline_report=str(prior)))
    assert report['physics_skipped'] and report['status'] == 'nonfinite_samples'
    assert report['dp_v1'] == 'not_frozen' and report['formal_test_run'] is False
    assert not list(output.glob('*.pt')) and not (output / 'freeze_manifest.json').exists()


def test_short_window_mask_ablation_is_only_temporary(config):
    model = SmallModel()
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :2] = True
    with followup.label_mask_override(model, mask):
        sample(model, config)
    assert all(torch.equal(entry[2], mask) for entry in model.inputs)
    model.inputs.clear()
    sample(model, config)
    assert all(entry[2] is None for entry in model.inputs)


def test_recorded_condition_keeps_environment_and_model_phase():
    batch = dict(phase=np.array([1]), states=np.zeros((1, 2, 122)))
    record = diagnostic.condition_records([dict(tick=50, phase='ACQUIRE')], [batch])[0]
    assert record['phase'] == 'ACQUIRE' and record['phase_index'] == 1
    assert record['states'].shape == (2, 122)


def test_followup_includes_acquire_tail_and_recovery_short_windows(config):
    class Dataset:
        episodes = [(Path('normal'), {'scenario': {}}), (Path('recovery'), {'scenario': {'recover': True}})]
        windows = [(0, 0, 16, 1), (0, 1, 16, 1), (0, 15, 16, 1), (0, 16, 17, 3),
                   (1, 0, 8, 7), (1, 7, 8, 7)]
        def arrays(self, episode):
            return {'action_ticks': np.arange(20)*50}
        def __getitem__(self, index):
            _, start, end, _ = self.windows[index]
            mask = torch.arange(16) < end-start
            return dict(states=torch.zeros(2, 122), actions=torch.zeros(16, 6), action_mask=mask)
    result = followup.mask_comparison(SmallModel(), config, dict(action_mean=[0.]*6, action_std=[1.]*6),
                                      torch.device('cpu'), Dataset(), (.05, .5))
    assert [c['index'] for c in result['cases']] == [1, 2, 3, 4, 5]
    assert result['all_valid']['phase:ACQUIRE']['windows'] == 2
    assert result['label_mask']['phase:RECOVER']['windows'] == 2
