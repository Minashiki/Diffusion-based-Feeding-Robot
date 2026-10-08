"""Safety rejection and physical initial-state compatibility."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('start_pose_run', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def success():
    return dict(success=True, accepted_normal=True, truncated=False, time_s=57.,
                max_episode_s=60., terminated=True, failure_reason=None, scenario={'recover': False},
                events=[{'name': name} for name in ('pickup', 'delivery', 'success')])


def replay():
    return dict(status='passed', max_physics_reference_error=0., max_observation_error=0.,
                events_equal=True, result_equal=True)


@pytest.mark.parametrize('change,expected', [
    ({'truncated': True}, 'time_limit'),
    ({'time_s': 60.001}, 'time_limit'),
    ({'success': False, 'failure_reason': 'contact_limit'}, 'contact_limit'),
    ({'events': [{'name': 'success'}]}, 'task_failed'),
    ({'scenario': {'recover': True}}, 'task_failed'),
])
def test_failed_tasks_never_accepted(change, expected):
    result = success()
    result.update(change)
    assert run.rejection(result, replay()) == expected


@pytest.mark.parametrize('change', [
    {'status': 'failed'}, {'max_physics_reference_error': 1e-9},
    {'max_observation_error': 1e-6}, {'events_equal': False}, {'result_equal': False},
])
def test_replay_failure_excludes_even_successful_task(change):
    check = replay()
    check.update(change)
    assert run.rejection(success(), check) == 'replay_failed'


def test_stratification_and_determinism():
    for cell in run.CELLS:
        low, high = run.cell_bounds(cell)
        assert run.cell_for((low + high) / 2) == cell
    assert len(run.CELLS) * run.CONFIG['per_cell'] == 200
    assert run.cell_for([.6, -.1, .2]) is None
    assert run.cell_for([np.nan, -.1, .2]) is None


def test_empty_historical_interrupted_manifest_and_diagnostic_seed_conflict(tmp_path, monkeypatch):
    historical = tmp_path / 'outputs/single_bean/v1/interrupted/old_attempt'
    historical.mkdir(parents=True)
    (historical / 'manifest.json').touch()
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'CELLS', [run.CELLS[0]])
    monkeypatch.setattr(run, 'recipe', lambda *args, **kwargs: (999, {}, {}, 'train_group'))
    assert run.historical_ids({}) == [str((historical / 'manifest.json').relative_to(tmp_path))]
    (historical / 'report.json').write_text('{"rejected":[{"seed":999}]}')
    with pytest.raises(ValueError, match='Historical seed/group conflict'):
        run.historical_ids({})


def test_rejections_physically_outside_training_root(tmp_path):
    dataset, output = tmp_path / 'data', tmp_path / 'evidence'
    output.mkdir()
    report = dict(accepted=[], rejected=[])
    for reason in ('time_limit', 'contact_limit', 'replay_failed', 'ik_unreachable',
                   'path_collision', 'joint_margin', 'settling_failed'):
        pending = output / 'pending' / reason
        pending.mkdir(parents=True)
        (pending / 'manifest.json').write_text('{}')
        run.record(dict(name=reason, status='rejected', rejection_reason=reason, cell_index=0,
                        cell=[0, 0, 0], seed=1), dataset, output, report)
        assert (output / 'rejected' / reason / 'manifest.json').exists()
    assert list(dataset.glob('*/*/manifest.json')) == []
    assert report['accepted'] == []
    assert report['accepted_count'] == 0


@pytest.mark.parametrize('location', ['pending', 'train', 'not_started'])
def test_interrupted_attempts_quarantined_and_seed_not_reused(tmp_path, location):
    dataset, output = tmp_path / 'data', tmp_path / 'evidence'
    (output / 'pending').mkdir(parents=True)
    row = dict(name='normal_123', seed=123, cell_index=0, cell=[0, 0, 0], attempt=0)
    report = dict(accepted=[], rejected=[], active=[row])
    if location != 'not_started':
        root = output / 'pending' if location == 'pending' else dataset / 'train'
        (root / row['name']).mkdir(parents=True)
        (root / row['name'] / 'manifest.json').write_text('{}')
    run.recover_interrupted(dataset, output, report)
    assert report['active'] == [] and report['accepted'] == []
    assert report['rejected'][0]['seed'] == 123
    assert report['rejected'][0]['rejection_reason'] == 'interrupted'
    assert not (dataset / 'train' / row['name']).exists()
    assert (output / 'rejected' / row['name'] / 'rejection.json').exists()


def test_real_initial_state_roundtrip_and_safety_rejections(monkeypatch):
    from feedingrobot.envs import FeedingGymEnv
    from feedingrobot.scripts.validate_m1 import restore_numerical_state
    cfg = run.read_json(run.ROOT / 'configs/collect.json')
    seed, scenario, parameters, _ = run.recipe(cfg, 'train', 10000, recover=False)
    target = np.array([.475, -.229, .1525])
    cell = run.cell_for(target)
    state, pose = run.prepare(seed, scenario, parameters, cfg, target, cell)
    env = FeedingGymEnv('panda')
    try:
        env.reset(seed=seed, options={'scenario': scenario})
        restore_numerical_state(env.task, state)
        assert np.array_equal(env.task.get_state()['physics'], state['physics'])
        assert env.task.tick == 0 and env.task.data.time == 0 and env.task.logic.events == []
        pairs = run.collision_pairs(env.task)
        run.check_pose(env.task, target, np.eye(3), cell, pairs)
        assert pose['joint_margin_rad'] >= .02 and pose['initial_clearance_m'] >= .005
        monkeypatch.setattr(run, 'clearance', lambda *args: .004)
        with pytest.raises(ValueError, match='initial_collision'):
            run.check_pose(env.task, target, np.eye(3), cell, pairs)
        monkeypatch.undo()
        env.task.data.qpos[env.task.index.qpos[0]] = env.task.model.jnt_range[env.task.index.joints[0], 0]
        with pytest.raises(ValueError, match='joint_margin'):
            run.check_pose(env.task, target, np.eye(3), cell, pairs)
    finally:
        env.close()


def test_unreachable_and_colliding_ik():
    from feedingrobot.envs import FeedingGymEnv
    env = FeedingGymEnv('panda')
    try:
        configuration = run.mink.Configuration(env.task.model)
        configuration.update(env.task.data.qpos.copy())
        pairs = run.collision_pairs(env.task)
        with pytest.raises(ValueError, match='ik_unreachable|path_collision'):
            run.solve(env.task, configuration, [2., 0., .2], np.eye(3), pairs)
    finally:
        env.close()


def test_settling_failure_rejected_and_reset_config_restored(monkeypatch):
    env = run.FeedingGymEnv('panda')
    original_q = copy.deepcopy(env.task.robot_config['reset_q'])
    original_reset = env.reset
    calls = 0
    def reset(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('natural settling failed')
        return original_reset(*args, **kwargs)
    monkeypatch.setattr(env, 'reset', reset)
    monkeypatch.setattr(run, 'FeedingGymEnv', lambda _: env)
    cfg = run.read_json(run.ROOT / 'configs/collect.json')
    seed, scenario, parameters, _ = run.recipe(cfg, 'train', 10000, recover=False)
    target = [.475, -.229, .1525]
    with pytest.raises(ValueError, match='settling_failed'):
        run.prepare(seed, scenario, parameters, cfg, np.array(target), run.cell_for(target))
    assert env.task.robot_config['reset_q'] == original_q
