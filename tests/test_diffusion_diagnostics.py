"""Oracle reconstruction, teacher masks and paired-noise offline diagnostics."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler, DDIMScheduler

from tools.m5.diffusion_diagnostics import run as diagnostics


@pytest.fixture
def normalization():
    return dict(kind='limits', schema_version=4, source_split='train', action_offset=[0.]*6,
                action_scale=[1.]*6, action_min=[-1.]*6, action_max=[1.]*6,
                action_constant_mask=[False]*6, clipping=False, observation_timing='pre_action')


@pytest.mark.parametrize('prediction_type', ['epsilon', 'sample'])
def test_known_noise_reconstructs_expert_actions(prediction_type):
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                              prediction_type=prediction_type, clip_sample=False)
    clean = torch.randn(3, 4, 6, dtype=torch.float64)
    noise = torch.randn_like(clean)
    steps = torch.tensor([0, 50, 99])
    noisy = scheduler.add_noise(clean, noise, steps)
    target = noise if prediction_type == 'epsilon' else clean
    recovered, epsilon = diagnostics.clean_and_epsilon(target, noisy, steps, scheduler)
    torch.testing.assert_close(recovered, clean, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(epsilon, noise, rtol=1e-10, atol=1e-10)


def test_masked_physical_metrics_and_direction():
    truth = np.zeros((1, 5, 6))
    prediction = truth.copy()
    truth[0, 0, 0] = .02
    prediction[0, 0, 0] = -.06
    prediction[0, 1, 3] = .6
    prediction[0, 2:] = np.nan
    mask = np.array([[True, True, False, False, False]])
    metrics = diagnostics.action_metrics(prediction, truth, mask, (.05, .5))
    assert metrics['finite'] and metrics['actions'] == 2
    assert metrics['linear']['vector_rmse'] == pytest.approx(.08 / np.sqrt(2))
    assert metrics['linear']['over_limit_fraction'] == .5
    assert metrics['linear']['direction_count'] == 1
    assert metrics['linear']['direction_cosine_mean'] == -1
    assert metrics['angular']['over_limit_fraction'] == .5
    assert metrics['angular']['direction_cosine_mean'] is None


def test_first_four_metrics_do_not_include_later_legal_actions():
    prediction, truth = np.zeros((1, 6, 6)), np.zeros((1, 6, 6))
    prediction[:, 4:, 0] = 1
    cases = [dict(phase='ACQUIRE', kind='normal', group_id='case')]
    metrics = diagnostics.grouped_metrics(prediction, truth, np.ones((1, 6), bool), cases, (.05, .5))['all']
    assert metrics['first_four']['linear']['vector_rmse'] == 0
    assert metrics['legal_horizon']['linear']['vector_rmse'] == pytest.approx(np.sqrt(2/6))


def test_paired_noise_does_not_depend_on_batch_or_condition_order():
    noise = diagnostics.paired_noise(5, 4, 7)
    reordered = diagnostics.condition_order(5, 'shuffle', 123)
    assert sorted(reordered) == list(range(5))
    assert np.all(reordered != np.arange(5))
    for batch_size in (1, 2, 5):
        batched = torch.cat([noise[start:start+batch_size] for start in range(0, 5, batch_size)])
        assert diagnostics.noise_hash(batched) == diagnostics.noise_hash(diagnostics.paired_noise(5, 4, 7))
    assert not torch.equal(noise, diagnostics.paired_noise(5, 4, 8))


@pytest.mark.parametrize('prediction_type', ['epsilon', 'sample'])
def test_oracle_sampler_and_deployment_mask(prediction_type):
    scheduler = DDIMScheduler(num_train_timesteps=20, beta_schedule='squaredcos_cap_v2',
                             prediction_type=prediction_type, clip_sample=False, timestep_spacing='trailing')
    scheduler.set_timesteps(10)
    clean = torch.full((2, 4, 6), .03)
    masks_seen = []

    def denoiser(noisy, steps, condition, action_mask):
        masks_seen.append(action_mask)
        alpha = scheduler.alphas_cumprod[steps][:, None, None]
        return clean if prediction_type == 'sample' else (noisy-alpha.sqrt()*clean)/(1-alpha).sqrt()

    model = SimpleNamespace(denoiser=denoiser)
    noise = diagnostics.paired_noise(2, 4, 0)
    result = diagnostics.sample_chain(model, scheduler, None, noise)
    torch.testing.assert_close(result, clean, atol=1e-6, rtol=1e-5)
    assert all(mask is None for mask in masks_seen)
    teacher_mask = torch.tensor([[True, True, False, False]]).expand(2, -1)
    masked = diagnostics.sample_chain(model, scheduler, None, noise, teacher_mask)
    assert all(torch.equal(mask, teacher_mask) for mask in masks_seen[10:])
    assert not masked[:, 2:].any()
    torch.testing.assert_close(noise, diagnostics.paired_noise(2, 4, 0), rtol=0, atol=0)


def test_ignore_encoded_condition_is_actually_zero():
    condition = diagnostics.dit.Condition(torch.ones(2, 3), torch.ones(2, 4, 3), torch.zeros(2, 4, dtype=torch.bool))
    model = SimpleNamespace(encoder=lambda _: condition)
    ignored = diagnostics.encode_condition(model, {}, 'ignore')
    assert not ignored.global_features.any() and not ignored.tokens.any()
    assert ignored.valid.all()
    assert condition.global_features.all()


@pytest.mark.parametrize('nonfinite', [False, True])
def test_end_to_end_report_is_read_only_and_uses_checkpoint_metadata(tmp_path, monkeypatch, nonfinite, normalization):
    config = dict(robot_id='panda', dataset='dataset', model=dict(name='MinaDiTV4'),
                  diffusion=dict(train_steps=100, inference_steps=10, beta_schedule='squaredcos_cap_v2'),
                  training=dict(cpu_budget=6, torch_threads=2, seed=0))
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(dict(config=config, normalization=normalization, ema={}, step=5), checkpoint)
    original_bytes = checkpoint.read_bytes()

    class Model:
        horizon = 4
        def to(self, _):
            return self
        def eval(self):
            return self
        def requires_grad_(self, _):
            return self
        def load_state_dict(self, _):
            pass
        def encoder(self, batch):
            size = len(batch['states'])
            return diagnostics.dit.Condition(torch.zeros(size, 3), torch.zeros(size, 1, 3), torch.ones(size, 1, dtype=torch.bool))
        def denoiser(self, actions, steps, condition, action_mask=None):
            return torch.full_like(actions, float('nan')) if nonfinite else actions

    class Dataset:
        by_phase = {1: {0: [0]}, 3: {1: [1]}}
        windows = [(0, 0, 4, 1), (1, 0, 2, 3)]
        episodes = [(Path('normal'), dict(group_id='a', scenario={})),
                    (Path('recovery'), dict(group_id='b', scenario=dict(recover=True)))]
        def __len__(self):
            return 2
        def arrays(self, _):
            return dict(action_ticks=[0])
        def __getitem__(self, index):
            return dict(states=torch.zeros(2, 3), actions=torch.full((4, 6), .01),
                        action_mask=torch.tensor([True, True, index == 0, index == 0]))

    monkeypatch.setattr(diagnostics, 'create_model', lambda _: Model())
    monkeypatch.setattr(diagnostics, 'ActionWindows', lambda *a: Dataset())
    monkeypatch.setattr(diagnostics, 'setup', lambda *a: (torch.device('cpu'), {}))
    args = SimpleNamespace(checkpoint=str(checkpoint), output=str(tmp_path/'report'), device='cpu', weights='ema',
        samples=2, batch_size=1, seeds=[0], conditions=['original', 'ignore', 'shuffle'], selection_seed=123,
        timesteps=[0, 99], samplers=[('trailing', 10), ('leading', 10)], precision='fp32', sampling_mask='none',
        case_mode='random', phase=None, observation_timing='recorded')
    report = diagnostics.run(args)
    assert report['status'] == ('nonfinite_predictions' if nonfinite else 'completed')
    assert report['optimizer_updates'] == report['control_runs'] == 0
    if nonfinite:
        assert all(row['epsilon_mse'] is None for row in report['forward_noised'])
    assert report['training_config'] == config and report['normalization']['kind'] == 'limits'
    assert checkpoint.read_bytes() == original_bytes
    assert len(report['forward_noised']) == 6 and len(report['sampled']) == 6
    hashes = [row['initial_noise_sha256'] for row in report['sampled']]
    assert all(value == hashes[0] for value in hashes)
    assert len(json.loads((tmp_path/'report'/'samples.json').read_text())) == 12
    assert json.loads((tmp_path/'report'/'report.json').read_text())['split'] == 'validation'
    with pytest.raises(FileExistsError):
        diagnostics.run(args)


def test_first_window_selection_is_phase_filtered_and_episode_balanced():
    class Dataset:
        windows = [(0, 5, 9, 1), (0, 0, 4, 1), (0, 7, 9, 3),
                   (1, 8, 12, 1), (1, 2, 6, 1)]
        by_phase = {1: {0: [0, 1], 1: [3, 4]}, 3: {0: [2]}}
        episodes = [(Path('first'), dict(group_id='a', scenario={})),
                    (Path('second'), dict(group_id='b', scenario={}))]
        def arrays(self, _):
            return dict(action_ticks=np.arange(20)*50)
    a = diagnostics.select_cases(Dataset(), 8, 123, 'first', 'ACQUIRE')
    b = diagnostics.select_cases(Dataset(), 8, 123, 'first', 'ACQUIRE')
    assert a == b
    assert sorted(case['index'] for case in a) == [1, 4]
    assert all(case['phase'] == 'ACQUIRE' for case in a)
    with pytest.raises(ValueError, match='selected phase'):
        diagnostics.select_cases(Dataset(), 8, 123, 'first', 'RECOVER')


def test_timing_override_preserves_checkpoint_and_pairing(tmp_path, monkeypatch, normalization):
    name = 'MinaDiTV4'
    config = dict(robot_id='panda', dataset='dataset', model=dict(name=name),
                  diffusion=dict(train_steps=100, inference_steps=10, beta_schedule='squaredcos_cap_v2'),
                  training=dict(cpu_budget=6, torch_threads=2, seed=0))
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(config=config, normalization=normalization, ema={}, step=5, model_name=name), path)
    original_bytes = path.read_bytes()
    captured = []

    class Model:
        horizon = 4
        def to(self, _):
            return self
        def eval(self):
            return self
        def requires_grad_(self, _):
            return self
        def load_state_dict(self, _):
            pass

    class Dataset:
        by_phase = {1: {0: [0]}}
        windows = [(0, 0, 4, 1)]
        episodes = [(Path('first'), dict(group_id='a', scenario={}))]
        def __init__(self, directory, split, horizon, effective):
            captured.append(effective)
        def arrays(self, _):
            return dict(action_ticks=[0])

    def diagnose(model, config, effective, dataset, cases, device, args, limits):
        return dict(forward_noised=[dict(finite=True)], sampled=[dict(all_generated_values_finite=True)],
                    samples=[dict(noise=diagnostics.noise_hash(diagnostics.paired_noise(len(cases), model.horizon, 0)))])

    monkeypatch.setattr(diagnostics, 'create_model', lambda _: Model())
    monkeypatch.setattr(diagnostics, 'ActionWindows', Dataset)
    monkeypatch.setattr(diagnostics, 'setup', lambda *a: (torch.device('cpu'), {}))
    monkeypatch.setattr(diagnostics, 'diagnose', diagnose)
    args = SimpleNamespace(checkpoint=str(path), output='', device='cpu', weights='ema', samples=1, batch_size=1,
        seeds=[0], conditions=['original'], selection_seed=123, timesteps=[99], samplers=[('trailing', 10)],
        precision='fp32', sampling_mask='none', case_mode='first', phase='ACQUIRE', observation_timing='recorded')
    reports = []
    for timing in ('recorded', 'pre_action'):
        args.observation_timing, args.output = timing, str(tmp_path/timing)
        reports.append(diagnostics.run(args))
    assert reports[0]['cases'] == reports[1]['cases']
    assert (tmp_path/'recorded'/'samples.json').read_bytes() == (tmp_path/'pre_action'/'samples.json').read_bytes()
    assert captured[0]['observation_timing'] == 'recorded' and captured[1]['observation_timing'] == 'pre_action'
    assert reports[1]['normalization'] == dict(normalization, observation_timing='pre_action')
    assert reports[1]['checkpoint_normalization'] == normalization
    assert path.read_bytes() == original_bytes
    assert reports[1]['model'] == name
