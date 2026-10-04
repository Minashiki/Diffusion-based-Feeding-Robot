import mujoco
import numpy as np
import pytest

from feedingrobot.sim.model import RobotIndex
from feedingrobot.sim.task import FeedingTask


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_physical_obstruction_release_cannot_resume(robot, tmp_path, monkeypatch):
    task = FeedingTask(robot)
    task.reset(preset="empty")
    state = task.snapshot()
    rotation = state["tcp_rotation"]
    from feedingrobot.sim import task as task_module
    original_loader = task_module.load_model
    def with_obstacle(*args, **kwargs):
        model, _, config, scene = original_loader(*args, **kwargs)
        xml = tmp_path / "blocked.xml"
        mujoco.mj_saveLastXML(str(xml), model)
        spec = mujoco.MjSpec.from_file(str(xml))
        table = spec.geom("table")
        table.pos = state["tcp_position"] + rotation[:, 0] * .05
        table.size = [.01, .06, .06]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        table.quat = quat
        model = spec.compile()
        return model, RobotIndex(model, config), config, scene
    # Compile the blocker before the trial, including its collision broadphase.
    monkeypatch.setattr(task_module, "load_model", with_obstacle)
    task = FeedingTask(robot)
    task.reset(preset="empty")
    base_r = task.data.site_xmat[task.index.base].reshape(3, 3)
    task.adapter.set_twist(np.r_[base_r.T @ rotation[:, 0] * .05, np.zeros(3)], 0, 2.)
    maximum = 0.
    for _ in range(2000):
        state = task.step_physics()
        maximum = max(maximum, float(np.linalg.norm(task.adapter.target.translation() - state["tcp_position"])))
        if task.terminated or task.adapter.fault:
            break
    assert task.terminated or task.adapter.fault == "blocked"
    assert task.monitor.peak_n > 0
    assert maximum <= task.robot_config["reference_position_limit"] + .001
    assert task.adapter.command is None
    obstacle = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    task.model.geom_contype[obstacle] = 0
    task.model.geom_conaffinity[obstacle] = 0
    if task.terminated:
        with pytest.raises(RuntimeError):
            task.step_physics()
    else:
        goal = task.data.ctrl[task.index.actuators].copy()
        for _ in range(100):
            task.step_physics()
        np.testing.assert_array_equal(goal, task.data.ctrl[task.index.actuators])
    task.reset(preset="empty")
    assert task.adapter.command is None and task.adapter.fault is None


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_nonfinite_solver_output(robot, monkeypatch):
    import mink
    task = FeedingTask(robot)
    task.reset(preset="empty")
    monkeypatch.setattr(mink, "solve_ik", lambda *args, **kwargs: np.full(task.model.nv, np.nan))
    task.adapter.set_twist([.01, 0, 0, 0, 0, 0], 0, 1)
    task.step_physics()
    assert task.adapter.fault == "ik_failure" and task.adapter.command is None
    assert np.isfinite(task.data.ctrl).all()


def test_incomplete_robot_config_fails_at_load():
    task = FeedingTask()
    config = dict(task.robot_config, actuators=["missing"] * task.index.n)
    with pytest.raises(ValueError, match="Missing"):
        RobotIndex(task.model, config)


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_reference_cannot_accumulate_when_feedback_is_stuck(robot):
    task = FeedingTask(robot)
    task.reset(preset="empty")
    task.adapter.set_twist([.05, 0, 0, 0, 0, 0], 0, 10)
    for _ in range(1500):
        # An independently stuck feedback state exercises the reference guard,
        # without the contact cutoff ending this synthetic adapter test first.
        task.adapter.update(task.dt)
        deviation = np.linalg.norm(task.adapter.target.translation() - task.data.site_xpos[task.index.tcp])
        assert deviation <= task.robot_config["reference_position_limit"] + 1e-10
        if task.adapter.fault:
            break
    assert task.adapter.fault == "blocked"
    assert task.adapter.command is None
    ctrl = task.data.ctrl.copy()
    for _ in range(100):
        task.adapter.update(task.dt)
    np.testing.assert_array_equal(task.data.ctrl, ctrl)
