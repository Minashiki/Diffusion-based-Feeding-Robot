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
    task.reset(preset="food_on_spoon")
    return task


def place(task, site, local):
    rotation = task.data.site_xmat[site].reshape(3, 3)
    address = task.index.food_qpos
    task.data.qpos[address:address + 3] = task.data.site_xpos[site] + rotation @ local
    mujoco.mju_mat2Quat(task.data.qpos[address + 3:address + 7], rotation.ravel())
    mujoco.mj_forward(task.model, task.data)


def test_side_load_inside_old_support_box_is_not_support(task):
    place(task, task.index.tcp, [0, 0, .01])
    rotation = task.data.site_xmat[task.index.tcp].reshape(3, 3)
    task.contacts = [dict(geom1=task.index.scoop_geoms[0], geom2=task.food_geom,
                         group1="spoon", group2="food", force_n=.03,
                         force_on_geom2_world=rotation[:, 1] * .03, distance=0.)]
    assert not evidence(task)["supported"]


def test_centre_over_rounded_rim_without_collision_support(task):
    place(task, task.index.tcp, [.019, .012, .02])
    rotation = task.data.site_xmat[task.index.tcp].reshape(3, 3)
    task.contacts = [dict(geom1=task.index.scoop_geoms[0], geom2=task.food_geom,
                         group1="spoon", group2="food", force_n=.03,
                         force_on_geom2_world=rotation[:, 2] * .03, distance=0.)]
    assert not evidence(task)["supported"]


def test_off_plate_uses_plate_normal_not_world_height(task):
    plate = named_id(task.model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
    # A tilted plate: this food is high in world z but still straddles its plane.
    angle = -.35
    task.model.body_quat[task.model.site_bodyid[plate]] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    mujoco.mj_forward(task.model, task.data)
    place(task, plate, [.05, 0., .0065])
    task.contacts = []
    assert not evidence(task)["off_plate"]


def test_plate_contact_prevents_pickup_even_above_plane(task):
    plate = named_id(task.model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
    place(task, plate, [0., 0., .03])
    task.contacts = [dict(geom1=task.food_geom, geom2=task.index.plate_geoms[0],
                         group1="food", group2="plate", force_n=.03, distance=.001)]
    assert not evidence(task)["off_plate"]


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
