import mujoco
import numpy as np
import pytest

from feedingrobot.sim.model import ROOT
from feedingrobot.sim.task import FeedingTask


@pytest.fixture(params=["panda", "ur5e"])
def task(request):
    return FeedingTask(request.param)


def test_import_is_this_project():
    import feedingrobot
    from pathlib import Path
    assert Path(feedingrobot.__file__).resolve() == ROOT / "src/feedingrobot/__init__.py"


def test_model_contract(task):
    index, model = task.index, task.model
    assert index.n == {"panda": 7, "ur5e": 6}[task.robot_id]
    assert np.all(model.actuator_forcelimited[index.actuators])
    assert np.all(model.body_gravcomp[list(index.arm_bodies)] == 0)
    assert set(index.dofs).isdisjoint(task.adapter.frozen_dofs)
    assert len(index.dofs) + len(task.adapter.frozen_dofs) == model.nv
    assert abs(index.tool_mass - .035) < 1e-9
    assert model.body_mass[index.tool_body] == 0
    assert len(index.spoon_geoms) == 145 and len(index.scoop_geoms) == 130
    assert len(index.handle_geoms) == 15 and len(index.plate_geoms) == 17
    assert model.npair == 145


def test_adapter_never_advances_or_teleports(task):
    task.reset(preset="empty")
    before, time = task.data.qpos.copy(), task.data.time
    task.adapter.set_twist([.01, 0, 0, 0, 0, .02], time, time + 1)
    task.adapter.update(task.dt)
    np.testing.assert_array_equal(task.data.qpos, before)
    assert task.data.time == time
    assert not task.adapter.fault
    np.testing.assert_allclose(task.adapter.last_ik_velocity[task.adapter.frozen_dofs], 0, atol=1e-7)


def test_expiry_stop_and_timestamp_contract(task):
    task.reset(preset="empty")
    task.adapter.set_twist([.01, 0, 0, 0, 0, 0], 0, .02)
    for _ in range(22):
        task.step_physics()
    assert task.adapter.status == "expired"
    assert task.adapter.command is None
    targets = task.data.ctrl[task.index.actuators].copy()
    for _ in range(30):
        task.step_physics()
    np.testing.assert_array_equal(targets, task.data.ctrl[task.index.actuators])
    now = task.data.time
    with pytest.raises(ValueError):
        task.adapter.set_twist(np.zeros(6), now + 1, now + 2)
    task.adapter.set_twist(np.zeros(6), now, now + 1)
    with pytest.raises(ValueError):
        task.adapter.set_twist(np.zeros(6), now - .01, now + 1)
    task.adapter.stop()
    assert task.adapter.command is None


def test_solver_failure_cancels_old_command(task, monkeypatch):
    import mink
    def fail(*args, **kwargs):
        raise mink.exceptions.NoSolutionFound("injected")
    task.adapter.set_twist([.01, 0, 0, 0, 0, 0], task.data.time, task.data.time + 1)
    monkeypatch.setattr(mink, "solve_ik", fail)
    task.step_physics()
    assert task.adapter.fault == "ik_failure"
    assert task.adapter.command is None
    with pytest.raises(RuntimeError):
        task.adapter.set_twist(np.zeros(6), task.data.time, task.data.time + 1)
    task.reset()
    assert task.adapter.fault is None


def test_invalid_command_and_nonfinite_state(task):
    with pytest.raises(ValueError):
        task.adapter.set_twist([np.nan] * 6, 0, 1)
    assert task.adapter.command is None
    task.reset()
    task.data.qvel[task.index.dofs[0]] = np.inf
    result = task.step_physics()
    assert result["terminated"] and result["failure_reason"] == "nonfinite_state"
    with pytest.raises(RuntimeError):
        task.step_physics()
    task.reset()
    assert np.isfinite(task.snapshot()["dq"]).all()


def test_force_limit_terminates_without_more_physics(task):
    task.reset(preset="empty")
    task.set_external_wrench([0, 0, 12], [0, 0, 0], task.data.site_xpos[task.index.tcp])
    state = task.step_physics()
    assert state["terminated"] and state["failure_reason"] == "contact_limit"
    time = task.data.time
    with pytest.raises(RuntimeError):
        task.step_physics()
    assert task.data.time == time


def test_observation_partition_and_copy(task):
    observation = task.provider.observe()
    assert set(observation) == {"policy_obs", "oracle_info", "scenario_state"}
    obs = observation["policy_obs"]
    assert obs["q"].shape == (task.index.n,)
    assert "future_events" not in obs and "contacts" not in obs and "seed" not in obs
    obs["q"][:] = 99
    assert not np.any(task.data.qpos[task.index.qpos] == 99)


def test_quaternion_and_base_twist(task):
    import mink
    for axis in np.eye(3):
        rotation = mink.SO3.exp(axis * np.pi / 2)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.as_matrix().ravel())
        np.testing.assert_allclose(mink.SO3(quat).as_matrix(), rotation.as_matrix(), atol=1e-12)
    task.reset(preset="empty")
    origin = task.adapter.target.translation().copy()
    task.adapter.set_twist([.01, 0, 0, 0, 0, 0], 0, 1)
    task.adapter.update(task.dt)
    base_r = task.data.site_xmat[task.index.base].reshape(3, 3)
    delta = base_r.T @ (task.adapter.target.translation() - origin)
    assert delta[0] > 0
    np.testing.assert_allclose(delta[1:], 0, atol=1e-12)


def test_physical_substep_peak_is_not_averaged():
    from feedingrobot.sim.contacts import ContactMonitor
    monitor = ContactMonitor(5)
    row = dict(group1="spoon", group2="mouth", force_n=6.)
    assert monitor.update([row], .001, .001)
    for _ in range(19):
        assert not monitor.update([], .001, .002)
    assert monitor.peak_n == 6 and monitor.over_limit_s == .001
    assert monitor.impulse_ns == .006
