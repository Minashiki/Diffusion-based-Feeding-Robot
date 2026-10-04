"""Adversarial M1-C evidence and reporting contracts."""

import numpy as np
import pytest
import mujoco

from feedingrobot.sim.beans import bean_diagnostics, spoon_frame_state
from feedingrobot.sim.contacts import ContactMonitor, read_contacts
from feedingrobot.sim.task import FeedingTask
from feedingrobot.sim.model import load_json
from feedingrobot.scripts.validate_m1c import pickup_eligible, update_window, sweep_path


@pytest.fixture(scope='module', params=['panda', 'ur5e'])
def task(request):
    return FeedingTask(request.param)


def test_nearby_airborne_and_handle_are_not_head_support(task):
    task.reset(preset='empty')
    rotation = task.data.site_xmat[task.index.tcp].reshape(3, 3)
    tcp = task.data.site_xpos[task.index.tcp].copy()
    for local in ([0., 0., .1], [-.08, 0., .005]):
        task.data.qpos[task.index.bean_qpos[0, :3]] = tcp + rotation @ local
        mujoco.mj_forward(task.model, task.data)
        task.contacts = read_contacts(task.model, task.data, task.index)
        before = task.get_state()['physics'].copy()
        diag = bean_diagnostics(task, spoon_frame=True)
        assert not diag['in_spoon_head'][0] and not diag['spoon_supported'][0]
        np.testing.assert_array_equal(task.get_state()['physics'], before)


def test_airborne_neighbour_inside_head_region_is_not_support(task):
    task.reset(preset='empty')
    rotation = task.data.site_xmat[task.index.tcp].reshape(3,3)
    task.data.qpos[task.index.bean_qpos[0,:3]] = task.data.site_xpos[task.index.tcp] + rotation @ [0.,0.,.02]
    mujoco.mj_forward(task.model, task.data)
    task.contacts = read_contacts(task.model, task.data, task.index)
    diag = bean_diagnostics(task, spoon_frame=True)
    assert diag['in_spoon_head'][0] and not diag['spoon_supported'][0]


def test_relative_velocity_removes_moving_frame_transport(task):
    task.reset(preset='empty')
    state = spoon_frame_state(task)
    np.testing.assert_allclose(state['bean_positions_tcp'],
        (task.data.xpos[task.index.bean_bodies]-task.data.site_xpos[task.index.tcp])
        @ task.data.site_xmat[task.index.tcp].reshape(3,3))
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(task.model, task.data, mujoco.mjtObj.mjOBJ_SITE, task.index.tcp, velocity, 0)
    delta = task.data.xpos[task.index.bean_bodies]-task.data.site_xpos[task.index.tcp]
    rotation = task.data.site_xmat[task.index.tcp].reshape(3,3)
    np.testing.assert_allclose(state['bean_linear_velocities_tcp'],
        (-velocity[3:] - np.cross(velocity[:3], delta)) @ rotation, atol=1e-10)


def test_support_chain_needs_a_real_head_root(two_bean_task):
    task = two_bean_task
    task.reset(preset='beans_on_spoon')
    geoms = task.index.bean_collision_geoms
    task.data.qpos[task.index.bean_qpos[1, :3]] = task.data.xpos[task.index.bean_bodies[0]] + [0., 0., .008]
    mujoco.mj_forward(task.model, task.data)
    root = next(row for row in task.contacts if row['bean1_id'] == 'bean_000' or row['bean2_id'] == 'bean_000')
    edge = dict(geom1=int(geoms[0]), geom2=int(geoms[1]), bean1_id='bean_000', bean2_id='bean_001',
                force_on_geom2_world=np.array([0.,0.,.005]), distance=0.)
    task.contacts = [root, edge]
    diag = bean_diagnostics(task)
    assert diag['spoon_supported'][:2].all()
    assert diag['direct_support']['spoon'] == ['bean_000']
    task.contacts = [edge]
    assert not bean_diagnostics(task)['spoon_supported'].any()


@pytest.mark.parametrize('field', ['in_bowl', 'bowl_supported', 'spoon_supported', 'in_spoon_head'])
def test_pickup_rejects_dual_support_and_geometric_false_positives(field):
    diag = dict(in_bowl=np.array([False]), bowl_supported=np.array([False]),
                spoon_supported=np.array([True]), in_spoon_head=np.array([True]))
    assert pickup_eligible(np.array([True]), diag, np.array([[0.,0.,.1]]), .05, np.array([.004]))[0]
    diag[field] = ~diag[field]
    assert not pickup_eligible(np.array([True]), diag, np.array([[0.,0.,.1]]), .05, np.array([.004]))[0]


def test_pickup_requires_initial_bowl_support_and_full_rim_clearance():
    diag = dict(in_bowl=np.array([False]), bowl_supported=np.array([False]),
                spoon_supported=np.array([True]), in_spoon_head=np.array([True]))
    assert not pickup_eligible(np.array([False]), diag, np.array([[0.,0.,.1]]), .05, np.array([.004]))[0]
    assert not pickup_eligible(np.array([True]), diag, np.array([[0.,0.,.052]]), .05, np.array([.004]))[0]


def test_interruption_resets_pickup_and_joint_drop_windows():
    window = np.array([.499, .1])
    np.testing.assert_allclose(update_window(window, [False, True], .001), [0., .101])
    lost, outside = True, False
    assert update_window(.099, lost and outside, .001) == 0.
    assert update_window(.099, lost and True, .001) == pytest.approx(.1)


def test_multiple_beans_cannot_bypass_semantic_force_limit():
    monitor = ContactMonitor(5.)
    rows = [dict(group1='food', group2='spoon', force_n=3., bean1_id=f'bean_{i:03d}') for i in range(2)]
    assert monitor.update(rows, .001, .001)
    assert monitor.peak_n == 6.
    assert monitor.pair_impulses['food|spoon'] == pytest.approx(.006)


def test_provider_exposes_readonly_spoon_frame_without_schema_change(task):
    task.reset(preset='beans_on_spoon')
    before = task.get_state()
    diag = task.provider.observe()['oracle_info']['beans']
    assert before['signature']['schema_version'] == 3
    for field in ('bean_positions_tcp', 'bean_linear_velocities_tcp', 'bean_angular_velocities_tcp'):
        assert diag[field].shape == (1,3)
    assert diag['in_spoon_head'].shape == (1,)
    assert diag['direct_support']['spoon'] == ['bean_000']
    np.testing.assert_array_equal(task.get_state()['physics'], before['physics'])


def test_sweep_uses_the_same_bowl_trajectory_for_both_robots():
    cfg = load_json('configs/acceptance.json')
    paths = [sweep_path(FeedingTask(robot), cfg) for robot in ('panda', 'ur5e')]
    for left, right in zip(*paths):
        assert left[0] == right[0] and left[3] == right[3]
        np.testing.assert_allclose(left[1], right[1], rtol=0, atol=1e-9)
        np.testing.assert_allclose(left[2], right[2], rtol=0, atol=1e-9)
    assert len(paths[0]) == len(paths[1])
