"""Noise, pose-reference, and original-evaluator regressions for ACQUIRE probes."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import acquire_diagnosis as acquire
import prefix_rollout
from diagnose_sampling import spacing_override
from test_diagnose_sampling import SmallModel
from feedingrobot.policies import dit, evaluation
from feedingrobot.policies.audit import read_json
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT


@pytest.fixture(scope='module', autouse=True)
def cpu_budget():
    setup(read_json(ROOT / 'configs/dp_dit.json'), 'cpu')


def test_explicit_noise_sampler_matches_original():
    config = read_json(ROOT / 'configs/dp_dit.json')
    model = SmallModel()
    batch = dict(states=torch.zeros(2, 2, 122))
    norm = dict(action_mean=[.01]*6, action_std=[.1]*6)
    noise = torch.randn((2, 16, 6), generator=torch.Generator().manual_seed(9))
    with spacing_override('leading'):
        original = dit.sample_actions(model, config, batch, norm, torch.Generator().manual_seed(9))
        explicit = acquire.sample_from_noise(model, config, batch, norm, noise)
    torch.testing.assert_close(original, explicit, rtol=0, atol=0)


def test_noise_is_paired_at_common_planning_ticks():
    grids = {steps: list(range(50, 1051, steps*50)) for steps in (1, 4)}
    for tick in grids[4]:
        assert tick in grids[1]
        a = torch.randn((1, 16, 6), generator=torch.Generator().manual_seed(prefix_rollout.noise_seed_at_tick(0, tick)))
        b = torch.randn((16, 6), generator=torch.Generator().manual_seed(prefix_rollout.noise_seed_at_tick(0, tick)))
        torch.testing.assert_close(a[0], b, rtol=0, atol=0)
    assert prefix_rollout.noise_seed_at_tick(0, 50) != prefix_rollout.noise_seed_at_tick(1, 50)


def test_nearest_path_distinguishes_lag_from_same_time_pose_difference():
    rotation = np.eye(3).reshape(-1)
    observations = np.array([np.r_[[x, 0., 0.], rotation] for x in (0., .01, .02)])
    teacher = dict(action_ticks=np.array([0, 50, 100]), action_observations=observations, action_phases=np.ones(3))
    fields = dict(tcp_position=slice(0, 3), tcp_rotation=slice(3, 12))
    row = acquire.pose_reference(observations[1:2], [100], teacher, fields)['rows'][0]
    assert row['time_reference_position_mm'] == pytest.approx(10.)
    assert row['nearest_position_teacher_tick'] == 50
    assert row['nearest_path_position_mm'] == pytest.approx(0.)


def test_direction_metrics_do_not_divide_by_quiescent_angular_labels():
    truth = np.array([[.01, 0, 0, 0, .5, 0], [0, 0, 0, 0, 0, 0]])
    metrics = acquire.direction_metrics(truth*.8, truth)
    assert metrics['angular']['active_actions'] == 1
    assert metrics['angular']['projection_ratio_p50'] == pytest.approx(.8)
    assert metrics['angular']['cosine_p50'] == pytest.approx(1.)


def native_prefix(steps, paired_noise):
    config = read_json(ROOT / 'configs/dp_dit.json')
    norm = dict(action_mean=[0.]*6, action_std=[.0005]*6, observation_mean=[0.]*108,
                observation_std=[1.]*108, velocity_mean=[0.]*12, velocity_std=[1.]*12)
    episode = sorted((ROOT / config['dataset'] / 'validation').glob('normal_*/manifest.json'))[0].parent
    model = SmallModel()
    with spacing_override('leading'):
        result = prefix_rollout.run_prefix_policy(model, config, norm, episode, torch.device('cpu'),
            execute_steps=steps, paired_noise=paired_noise, max_ticks=300)
    return result, model


def test_four_step_sequential_path_matches_original_real_physics():
    config = read_json(ROOT / 'configs/dp_dit.json')
    norm = dict(action_mean=[0.]*6, action_std=[.0005]*6, observation_mean=[0.]*108,
                observation_std=[1.]*108, velocity_mean=[0.]*12, velocity_std=[1.]*12)
    episode = sorted((ROOT / config['dataset'] / 'validation').glob('normal_*/manifest.json'))[0].parent
    model = SmallModel()
    with spacing_override('leading'):
        original = evaluation.run_policy(model, config, norm, episode, torch.device('cpu'), max_ticks=300)
    actual, revised_model = native_prefix(4, False)
    assert actual['events'] == original['events']
    assert actual['failure_reason'] == original['failure_reason']
    assert actual['simulated_s'] == original['simulated_s']
    assert len(actual['trace']) == len(original['trace'])
    for before, after in zip(original['trace'], actual['trace']):
        for key in before:
            if isinstance(before[key], list):
                np.testing.assert_allclose(before[key], after[key], rtol=0, atol=1e-12)
            else:
                assert before[key] == after[key]
    for a, b in zip(model.inputs, revised_model.inputs):
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)


def test_one_step_replans_every_action_and_discards_the_tail():
    result, model = native_prefix(1, True)
    assert [r['tick'] for r in result['plans']] == [50, 100, 150, 200, 250]
    assert all(r['queue_age_ticks'] == 0 for r in result['trace'])
    assert result['replan_ticks'] == 50 and len(model.inputs) == 50


def test_invalid_prefix_is_rejected_before_physics():
    with pytest.raises(ValueError, match='only 1 or 4'):
        prefix_rollout.run_prefix_policy(None, None, None, None, None, execute_steps=2)
