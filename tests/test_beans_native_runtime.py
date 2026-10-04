"""M1-B contracts: failures remain failures; control checks use empty reset."""

import copy

import mujoco
import numpy as np
import pytest

from feedingrobot.sim.beans import bean_diagnostics, place_beans, spawn_clearance
from feedingrobot.sim.contacts import read_contacts
from feedingrobot.sim.task import FeedingTask
from feedingrobot.scripts.validate_m1b import parameters_match, replay_initial


@pytest.fixture(scope='module', params=['panda', 'ur5e'])
def task(request):
    return FeedingTask(request.param)


def test_seeded_spawn_and_repeated_reset(task):
    first = copy.deepcopy(task.reset_diagnostics)
    task.reset(seed=0)
    np.testing.assert_array_equal(first['initial_qpos'], task.reset_diagnostics['initial_qpos'])
    np.testing.assert_allclose(first['final']['linear_speed_m_s'], task.reset_diagnostics['final']['linear_speed_m_s'], atol=1e-12)
    assert task.reset_diagnostics['spawn_clearance_m'] >= .0005 - 1e-9
    positions = task.reset_diagnostics['initial_qpos'][task.index.bean_qpos[:, :3]]
    local = positions - task.scene_config['bowl_frame_position_m']
    np.testing.assert_allclose(local[0], task.scene_config["beans"]["reset_position_bowl_m"], atol=1e-12)
    task.reset(preset='empty')
    place_beans(task, 'beans_in_bowl', 1)
    assert spawn_clearance(task, 'beans_in_bowl') >= .0005 - 1e-9
    np.testing.assert_array_equal(first['initial_qpos'][task.index.bean_qpos], task.data.qpos[task.index.bean_qpos])
    np.testing.assert_array_equal(task.data.qvel[task.index.bean_dofs], 0)


def test_timeout_is_explicit_and_velocities_are_preserved(task):
    task.reset(preset='empty')
    old = task.bean_acceptance['max_settle_s']
    task.bean_acceptance['max_settle_s'] = .002
    try:
        state = task.reset()
        assert state['terminated'] and state['failure_reason'] == 'bean_reset_settling_failed'
        assert task.reset_diagnostics['status'] == 'failed'
        assert task.reset_diagnostics['bean_settle_s'] == .002
        assert np.linalg.norm(task.data.qvel[task.index.bean_dofs]) > 0
        with pytest.raises(RuntimeError, match='reset required'):
            task.step_physics()
    finally:
        task.bean_acceptance['max_settle_s'] = old
    task.reset(preset='empty')
    assert not task.terminated and task.failure_reason is None


def test_empty_clears_execution_and_snapshot_arrays(task):
    task.reset(preset='empty')
    task.external_wrench = np.ones((3, 3))
    task.adapter.stop('injected', fault=True)
    task.monitor.update([dict(group1='food', group2='spoon', force_n=6.)], .001, .001)
    task.reset(preset='empty')
    assert task.external_wrench is None and task.adapter.fault is None and task.adapter.command is None
    assert task.monitor.peak_n == 0 and task.tick == 0 and task.data.time == 0
    assert not task.data.xfrc_applied.any() and not task.data.qfrc_applied.any()
    state = task.snapshot()
    assert 'food_position' not in state
    assert state['bean_ids'] == list(task.index.bean_ids)
    for key, shape in [('bean_positions', (1, 3)), ('bean_quaternions', (1, 4)),
                       ('bean_linear_velocities_world', (1, 3)), ('bean_angular_velocities_world', (1, 3))]:
        assert state[key].shape == shape
        state[key][:] = 99
        assert not np.any(task.snapshot()[key] == 99)
    assert 'food_relative_world' not in task.provider.observe()['policy_obs']
    assert task.provider.observe()['policy_obs']['bean_relative_world'].shape == (1, 3)


def test_world_velocity_and_readonly_geometry(task):
    task.reset(preset='empty')
    task.data.qvel[task.index.bean_dofs[0]] = [.1, .2, .3, .4, .5, .6]
    mujoco.mj_forward(task.model, task.data)
    expected = np.zeros(6)
    mujoco.mj_objectVelocity(task.model, task.data, mujoco.mjtObj.mjOBJ_BODY,
                            int(task.index.bean_bodies[0]), expected, 0)
    state = task.snapshot()
    np.testing.assert_array_equal(state['bean_linear_velocities_world'][0], expected[3:])
    np.testing.assert_array_equal(state['bean_angular_velocities_world'][0], expected[:3])
    before = task.data.qpos.copy(), task.data.qvel.copy()
    bean_diagnostics(task)
    np.testing.assert_array_equal(before[0], task.data.qpos)
    np.testing.assert_array_equal(before[1], task.data.qvel)


def test_finite_bowl_boundary_and_no_distance_only_support(task):
    task.reset(preset='empty')
    origin = np.array(task.scene_config['bowl_frame_position_m'])
    for offset in ([.12, 0, .02], [0, 0, .1], [0, 0, -.03]):
        task.data.qpos[task.index.bean_qpos[0]] = np.r_[origin + offset, [1, 0, 0, 0]]
        mujoco.mj_forward(task.model, task.data)
        task.contacts = read_contacts(task.model, task.data, task.index)
        task.applied_contacts = []
        diag = bean_diagnostics(task)
        assert not diag['in_bowl'][0] and not diag['bowl_supported'][0]
    task.data.qpos[task.index.bean_qpos[0]] = np.r_[origin + [0, 0, .01], [1, 0, 0, 0]]
    mujoco.mj_forward(task.model, task.data)
    task.contacts = []
    diag = bean_diagnostics(task)
    assert diag['in_bowl'][0] and not diag['bowl_supported'][0]


def test_support_chain_and_handle_exclusion(two_bean_task):
    task = two_bean_task
    task.reset(preset='empty')
    geoms = task.index.bean_collision_geoms
    bottom = task.model.geom('collision_bowl_fast_bottom_disk').id
    task.data.qpos[task.index.bean_qpos[0, :3]] = [.45, -.18, -.0116]
    task.data.qpos[task.index.bean_qpos[1, :3]] = [.45, -.18, -.0036]
    mujoco.mj_forward(task.model, task.data)
    row = dict(force_on_geom2_world=np.array([0., 0., .005]), distance=0., bean1_id=None, bean2_id='bean_000')
    task.contacts = [dict(row, geom1=bottom, geom2=int(geoms[0])),
                     dict(row, geom1=int(geoms[0]), geom2=int(geoms[1]), bean1_id='bean_000', bean2_id='bean_001')]
    diag = bean_diagnostics(task)
    assert diag['bowl_supported'][:2].all()
    task.contacts = [dict(row, geom1=task.index.handle_geoms[0], geom2=int(geoms[0]))]
    assert not bean_diagnostics(task)['spoon_supported'][0]


@pytest.mark.parametrize('position,quat,inside', [
    ([.424175858567567, -.1768602305806861, -.010548815064459418],
     [.03566980058072457, .958722793463676, -.029665598356785144, .2805320354146199], True),
    ([.43059798035223473, -.1624639860151046, -.010182441477566861],
     [-.28763827357984995, -.4033632718480294, -.8496075032944488, .18091264422872216], True),
    ([.4264270008093231, -.18043104123629944, -.01199854376466694],
     [-.38055655400223737, .0530739930414062, -.12493839653894778, .9147405411037927], False),
    ([.498, -.18, .0094], [1, 0, 0, 0], False),
])
def test_finite_wall_lower_edge_regression(task, position, quat, inside):
    task.reset(preset='empty')
    task.data.qpos[task.index.bean_qpos[0]] = np.r_[position, quat]
    mujoco.mj_forward(task.model, task.data)
    task.contacts = read_contacts(task.model, task.data, task.index)
    task.applied_contacts = []
    before = task.data.qpos.copy(), task.data.qvel.copy()
    diag = bean_diagnostics(task)
    assert bool(diag['in_bowl'][0]) == inside
    np.testing.assert_array_equal(task.data.qpos, before[0])
    np.testing.assert_array_equal(task.data.qvel, before[1])


def test_complete_state_and_schema_rejection(task):
    task.reset(preset='empty')
    state = task.get_state()
    task.step_physics()
    expected = task.data.qpos.copy(), task.data.qvel.copy()
    task.set_state(state)
    task.step_physics()
    np.testing.assert_array_equal(task.data.qpos, expected[0])
    np.testing.assert_array_equal(task.data.qvel, expected[1])
    old = copy.deepcopy(state)
    old['signature']['schema_version'] = 2
    with pytest.raises(ValueError, match='Incompatible'):
        task.set_state(old)
    for preset in ['food_on_plate', 'food_on_spoon']:
        with pytest.raises(ValueError):
            task.reset(preset=preset)
    with pytest.raises(ValueError, match='Single-food'):
        task.reset(scenario={'food_mass_kg': .005})


def test_task_mode_is_not_silently_migrated():
    with pytest.raises(NotImplementedError, match='M3/M4'):
        FeedingTask(task_mode=True)


def test_single_spawn_clearance_has_no_bean_pairs(task):
    task.reset(preset='empty')
    place_beans(task, 'beans_in_bowl', 9)
    assert len(task.index.bean_ids) == 1
    assert np.isfinite(spawn_clearance(task, 'beans_in_bowl'))
    assert spawn_clearance(task, 'beans_in_bowl') >= .0005 - 1e-9


def test_old_fifteen_bean_snapshot_is_incompatible(task, monkeypatch):
    from feedingrobot.sim import model
    original = model.load_json
    def old_scene(path):
        config = original(path)
        if path == 'configs/scene.json':
            config['beans']['count'] = 15
        return config
    monkeypatch.setattr(model, 'load_json', old_scene)
    previous = FeedingTask(task.robot_id)
    assert len(previous.index.bean_ids) == 15
    old_state = previous.get_state()
    assert old_state['signature']['schema_version'] == 3
    task.reset(preset='empty')
    before = task.get_state()['physics'].copy()
    with pytest.raises(ValueError, match='Incompatible'):
        task.set_state(old_state)
    np.testing.assert_array_equal(task.get_state()['physics'], before)


@pytest.mark.parametrize('field,value', [('condim', 3), ('friction', [.25, .25, .0001, .0001, .002]),
                                       ('solref', [.008, 1]), ('solimp', [.96, .995, .001, .5, 2]),
                                       ('includemargin', .0001)])
def test_contact_parameter_mismatch_is_rejected(field, value):
    expected = dict(condim=6, friction=[.25, .0001, .0001], solref=[.002, 1], solimp=[.99, .999, .001, .5, 2])
    row = dict(condim=6, friction=[.25, .25, .0001, .0001, .0001], solref=expected['solref'],
               solimp=expected['solimp'], includemargin=0., bean1_id='bean_000', bean2_id=None)
    assert parameters_match([row], expected)
    row[field] = value
    assert not parameters_match([row], expected)
    assert not parameters_match([], expected)


def test_bean_pair_margin_is_sum():
    expected = dict(condim=6, friction=[.25, .0001, .0001], solref=[.002, 1],
                    solimp=[.99, .999, .001, .5, 2], margin=.00005, gap=0.)
    row = dict(condim=6, friction=[.25, .25, .0001, .0001, .0001], solref=expected['solref'],
               solimp=expected['solimp'], includemargin=.00005, bean1_id='bean_000', bean2_id=None)
    assert parameters_match([row], expected)
    row['bean2_id'] = 'bean_001'
    assert not parameters_match([row], expected)
    row['includemargin'] = .0001
    assert parameters_match([row], expected)


def test_numerical_replay_preserves_initial_state_and_time_grid(task):
    task.reset(seed=2)
    reference = copy.deepcopy(task.reset_diagnostics)
    old_dt = task.dt
    old_timeout = task.bean_acceptance['max_settle_s']
    try:
        task.model.opt.timestep = task.dt = .0005
        task.bean_acceptance['max_settle_s'] = .02
        replay_initial(task, reference['initial_physics_state'], 'beans_in_bowl', 2)
        assert task.tick * task.dt == pytest.approx(task.data.time)
        task._settle_beans('beans_in_bowl')
        metrics = task.reset_diagnostics
        np.testing.assert_array_equal(metrics['initial_physics_state'], reference['initial_physics_state'])
        np.testing.assert_allclose([row['time_s'] for row in metrics['trajectory']], [0, .01, .02], atol=1e-12)
        assert metrics['status'] == 'failed' and task.terminated
        assert metrics['low_speed_window_s'] <= metrics['stable_window_s']
        assert metrics['max_penetration_m'] == 0
        assert metrics['penetration_peak'] is None
        assert np.linalg.norm(task.data.qvel[task.index.bean_dofs]) > 0
    finally:
        task.model.opt.timestep = task.dt = old_dt
        task.bean_acceptance['max_settle_s'] = old_timeout
        task.reset(preset='empty')
