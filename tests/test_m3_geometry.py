"""New collision geometry must determine support, not the old TCP centre box."""

import mujoco
import numpy as np
import pytest

from feedingrobot.sim.events import evidence
from feedingrobot.sim.model import named_id
from feedingrobot.sim.task import FeedingTask


@pytest.fixture
def task():
    task = FeedingTask(task_mode=True)
    task.reset(preset="beans_on_spoon")
    return task


def place(task, site, local):
    rotation = task.data.site_xmat[site].reshape(3, 3)
    address = task.index.bean_qpos[0, 0]
    task.data.qpos[address:address + 3] = task.data.site_xpos[site] + rotation @ local
    mujoco.mju_mat2Quat(task.data.qpos[address + 3:address + 7], rotation.ravel())
    mujoco.mj_forward(task.model, task.data)


def test_side_load_inside_old_support_box_is_not_support(task):
    place(task, task.index.tcp, [0, 0, .01])
    rotation = task.data.site_xmat[task.index.tcp].reshape(3, 3)
    task.contacts = [dict(geom1=task.index.scoop_geoms[0], geom2=int(task.index.bean_collision_geoms[0]),
                         group1="spoon", group2="food", force_n=.03,
                         force_on_geom2_world=rotation[:, 1] * .03, distance=0.)]
    assert not evidence(task)["supported"]


def test_centre_over_rounded_rim_without_collision_support(task):
    place(task, task.index.tcp, [.019, .012, .02])
    rotation = task.data.site_xmat[task.index.tcp].reshape(3, 3)
    task.contacts = [dict(geom1=task.index.scoop_geoms[0], geom2=int(task.index.bean_collision_geoms[0]),
                         group1="spoon", group2="food", force_n=.03,
                         force_on_geom2_world=rotation[:, 2] * .03, distance=0.)]
    assert not evidence(task)["supported"]


def test_off_bowl_requires_entire_ellipsoid_above_rim(task):
    from feedingrobot.sim.events import ellipsoid_bounds
    bean = int(task.index.bean_collision_geoms[0])
    address = task.index.bean_qpos[0, 0]
    evidence(task)
    task.data.qpos[address:address+3] = [0.45, -.18, task._event_rim + .003]
    mujoco.mj_forward(task.model, task.data)
    task.contacts = []
    lo, hi = ellipsoid_bounds(task.model, task.data, bean, np.zeros(3), np.eye(3))
    assert lo[2] < task._event_rim < hi[2]
    assert not evidence(task)["off_bowl"]


def test_bowl_contact_prevents_pickup_even_above_rim(task):
    address = task.index.bean_qpos[0, 0]
    task.data.qpos[address:address+3] = [.45, -.18, .2]
    mujoco.mj_forward(task.model, task.data)
    task.contacts = [dict(geom1=int(task.index.bean_collision_geoms[0]), geom2=task.index.bowl_geoms[0],
                         group1="food", group2="bowl", force_n=.03, distance=.001)]
    assert not evidence(task)["off_bowl"]


def test_snapshot_rejects_previous_event_rules(task):
    state = task.get_state()
    del state["signature"]["event_rules_version"]
    with pytest.raises(ValueError, match="Incompatible"):
        task.set_state(state)


def test_reset_updates_ik_configuration_and_constraint_ranges(task):
    task.reset(scenario={"recover": True})
    jaw = task.index.head_joints[-1]
    task.data.qpos[task.model.jnt_qposadr[jaw]] = -.2
    mujoco.mj_forward(task.model, task.data)
    task.adapter.reference.update(task.data.qpos.copy())
    task.adapter.reference.check_limits()
    task.reset()
    task.data.qpos[task.model.jnt_qposadr[jaw]] = -.2
    task.adapter.reference.update(task.data.qpos.copy())
    import mink
    with pytest.raises(mink.exceptions.NotWithinConfigurationLimits):
        task.adapter.reference.check_limits()


def test_ellipsoid_bounds_use_projection_radius_not_box(task):
    from feedingrobot.sim.events import ellipsoid_bounds
    bean = int(task.index.bean_collision_geoms[0])
    rotation = np.array([[2**-.5, -2**-.5, 0.], [2**-.5, 2**-.5, 0.], [0., 0., 1.]])
    task.data.geom_xmat[bean] = rotation.ravel()
    lo, hi = ellipsoid_bounds(task.model, task.data, bean, task.data.geom_xpos[bean], np.eye(3))
    axes = task.model.geom_size[bean]
    expected = np.sqrt((axes[0]**2 + axes[1]**2)/2)
    assert hi[0] == pytest.approx(expected)
    assert lo[0] == pytest.approx(-expected)
    assert hi[0] < (axes[0]+axes[1])/2**.5


def test_fast_bean_cannot_confirm_pickup(task):
    e = evidence(task)
    assert e['supported'] and e['off_bowl']
    task.data.qvel[task.index.bean_dofs[0, :3]] = [.002, 0., 0.]
    mujoco.mj_forward(task.model, task.data)
    assert not evidence(task)['pickup_eligible']
