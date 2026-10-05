import json

import mink
import numpy as np
import pytest

from feedingrobot.data.episodes import annotate, load_episode
from feedingrobot.data.recipes import recipe
from feedingrobot.data.replay import replay_episode
from feedingrobot.data.rollout import run_episode
from feedingrobot.control.adapter import RobotAdapter
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.experts.feasibility import check_waypoints
from feedingrobot.scripts.collect import check_gate, dataset_statistics
from feedingrobot.sim.model import load_json
from feedingrobot.sim.events import evidence
from feedingrobot.sim.task import FeedingTask


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_teacher_units_bounds_and_no_physics_writes(robot):
    env = FeedingGymEnv(robot)
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({}, geometry=teacher_geometry(env.task))
    before = env.task.get_state()
    obs = env.task.provider.observe()["policy_obs"]
    action = teacher.act(obs)
    assert action.shape == (6,) and np.isfinite(action).all()
    assert np.linalg.norm(action[:3]) <= env.task.robot_config["linear_speed_limit"]
    assert np.linalg.norm(action[3:]) <= env.task.robot_config["angular_speed_limit"]
    np.testing.assert_array_equal(before["physics"], env.task.get_state()["physics"])
    teacher.reset({}, geometry=teacher_geometry(env.task))
    obs["future_events"] = [{"time": 0., "jaw": "close"}]
    obs["scenario_state"] = {"seed": -1}
    np.testing.assert_array_equal(action, teacher.act(obs))
    env.close()


def test_acquisition_tracks_measured_single_bean_waypoints():
    env = FeedingGymEnv()
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({}, geometry=teacher_geometry(env.task))
    obs = env.task.provider.observe()["policy_obs"]
    for index, (name, target, rotation, speed) in enumerate(teacher.path):
        teacher.act(obs)
        assert teacher.stage == name and teacher.part == index
        np.testing.assert_array_equal(teacher.target_position, target)
        obs["tcp_position"], obs["tcp_rotation"] = target.copy(), rotation.copy()
    env.close()


def test_scoop_endpoint_is_bounded_when_bean_is_pushed_forward():
    env = FeedingGymEnv()
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({}, geometry=teacher_geometry(env.task))
    obs = env.task.provider.observe()["policy_obs"]
    teacher.part = 3
    teacher.act(obs)
    target = teacher.target_position.copy()
    obs["bean_relative_world"] += np.array([.1, 0., 0.])
    obs["stage"] = "TRANSPORT"
    teacher.act(obs)
    np.testing.assert_array_equal(teacher.target_position, target)
    assert teacher.part == 3 and not teacher.pickup_lift_complete
    env.close()


def test_pickup_phase_finishes_seating_and_waits_one_second():
    env = FeedingGymEnv()
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({}, geometry=teacher_geometry(env.task))
    obs = env.task.provider.observe()["policy_obs"]
    teacher.part = len(teacher.path)-1
    obs["stage"] = "TRANSPORT"
    obs["interaction"][0] = 1.
    teacher.act(obs)
    assert not teacher.pickup_lift_complete and teacher.stage == "wall_seat_level"
    obs["tcp_position"], obs["tcp_rotation"] = (x.copy() for x in teacher.path[-1][1:3])
    obs["time"] = 1.
    teacher.act(obs)
    assert teacher.stage == "pickup_hold" and not teacher.pickup_lift_complete
    np.testing.assert_array_equal(teacher.act(dict(obs, time=1.99)), np.zeros(6))
    teacher.act(dict(obs, time=2.))
    assert not teacher.pickup_lift_complete
    start = teacher.parameters["earliest_transport_start_s"]
    teacher.act(dict(obs, time=start))
    assert teacher.pickup_lift_complete and teacher.stage == "transport"
    teacher.reset({}, geometry=teacher_geometry(env.task))
    assert teacher.pickup_hold_start is None and not teacher.pickup_lift_complete
    teacher.part = len(teacher.path)-1
    teacher.act(dict(obs,time=start-.5))
    teacher.act(dict(obs,time=start))
    assert not teacher.pickup_lift_complete
    teacher.act(dict(obs,time=start+.5))
    assert teacher.pickup_lift_complete
    env.close()


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_phase_hold_preserves_servo_offset_but_fault_stop_reanchors(robot):
    env = FeedingGymEnv(robot)
    env.reset(seed=0)
    task = env.task
    task.adapter.set_twist([.01, 0., 0., 0., 0., 0.], 0., .2)
    for _ in range(50):
        task.step_physics()
    ctrl = task.data.ctrl[task.index.actuators].copy()
    reference = task.adapter.reference.q.copy()
    target = task.adapter.target.wxyz_xyz.copy()
    velocity = task.adapter.velocity.copy()
    physics = task.get_state()["physics"].copy()
    actual_q, actual_dq, time = task.data.qpos.copy(), task.data.qvel.copy(), task.data.time
    assert not np.array_equal(ctrl, task.data.qpos[task.index.qpos])
    task.adapter.stop(hold_reference=True)
    np.testing.assert_array_equal(task.data.ctrl[task.index.actuators], ctrl)
    np.testing.assert_array_equal(task.adapter.reference.q, reference)
    np.testing.assert_array_equal(task.adapter.target.wxyz_xyz, target)
    np.testing.assert_array_equal(task.adapter.velocity, velocity)
    assert task.adapter.command is None
    np.testing.assert_array_equal(task.get_state()["physics"], physics)
    for _ in range(25):
        previous = task.adapter.velocity.copy()
        task.step_physics()
        assert np.linalg.norm(task.adapter.velocity[:3] - previous[:3]) <= task.robot_config["linear_acceleration_limit"] * task.dt + 1e-12
        assert task.adapter.command is None
    np.testing.assert_allclose(task.adapter.velocity, 0., atol=1e-12)
    assert np.linalg.norm(task.adapter.target.translation() - target[4:]) <= np.linalg.norm(velocity[:3])**2 / (2 * task.robot_config["linear_acceleration_limit"]) + 1e-5
    actual_q, actual_dq, time = task.data.qpos.copy(), task.data.qvel.copy(), task.data.time
    task.adapter.stop()
    np.testing.assert_array_equal(task.data.ctrl[task.index.actuators], task.data.qpos[task.index.qpos])
    task.adapter.set_twist([.01, 0., 0., 0., 0., 0.], task.data.time, task.data.time + .2)
    task.adapter.stop("blocked", fault=True, hold_reference=True)
    assert task.adapter.fault == "blocked" and task.adapter.command is None
    np.testing.assert_array_equal(task.data.ctrl[task.index.actuators], task.data.qpos[task.index.qpos])
    np.testing.assert_array_equal(task.data.qpos, actual_q)
    np.testing.assert_array_equal(task.data.qvel, actual_dq)
    assert task.data.time == time
    env.close()


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_scenario_seed_reset_snapshot_and_policy_partition(robot):
    env = FeedingGymEnv(robot)
    nominal_mass = env.task.model.body_mass[env.task.index.bean_bodies[0]]
    seed, scenario, parameters, group = recipe(load_json("configs/collect.json"), "train", 0, recover=True)
    obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    state = env.get_state()
    result = env.step(np.zeros(6))
    env.set_state(state)
    replay = env.step(np.zeros(6))
    np.testing.assert_array_equal(result[0], replay[0])
    assert result[1:4] == replay[1:4]
    other = FeedingGymEnv(robot)
    other.reset(seed=seed, options={"scenario": scenario})
    other.set_state(state)
    np.testing.assert_array_equal(other.step(np.zeros(6))[0], result[0])
    assert not {"parameters", "future_events", "closure_start", "food_mass_kg"}.intersection(env.task.provider.observe()["policy_obs"])
    env.reset(seed=0)
    assert env.task.model.body_mass[env.task.index.bean_bodies[0]] == nominal_mass
    assert env.task.model.jnt_range[env.task.index.head_joints[-1], 0] == -.05
    env.close()
    other.close()


@pytest.mark.parametrize("scenario", [{"food_mass_kg": 0}, {"food_offset_m": [1, 0]},
                                      {"head_phase_rad": float("nan")}, {"recover": 1}, {"unknown": 0}])
def test_invalid_scenario_rejected(scenario):
    env = FeedingGymEnv()
    with pytest.raises(ValueError):
        env.reset(options={"scenario": scenario})
    env.close()


@pytest.mark.parametrize("robot,dt", [("panda", .001), ("panda", .0005), ("ur5e", .001)])
def test_integer_grids_stage_mask_mmap_and_physical_replay(tmp_path, robot, dt):
    config = load_json("configs/collect.json")
    result = run_episode(robot, 0, config, tmp_path / robot, timestep=dt, max_episode_s=.12)
    assert result["truncated"] and not result["success"] and not result["accepted_normal"]
    manifest, a = load_episode(tmp_path / robot)
    ticks = round(.05 / dt)
    np.testing.assert_array_equal(a["action_ticks"], [0, ticks, 2*ticks])
    assert not a["action_mask"][0] and not a["action_mask"][-1]
    assert a["action_mask"][1]
    assert a["action_end_ticks"][0] == 1
    assert isinstance(a["physics"], np.memmap) and isinstance(a["observations"], np.memmap)
    np.testing.assert_array_equal(a["observation_ticks"], np.arange(0, round(.12/dt)+1, round(.02/dt)))
    assert replay_episode(tmp_path / robot)["max_physics_reference_error"] == 0
    commands = json.loads((tmp_path / robot / "commands.json").read_text())
    assert any(c.get("hold_reference") for c in commands)
    assert any(c["kind"] == "stop" and "hold_reference" not in c for c in commands)


def test_corrupt_episode_rejected(tmp_path):
    run_episode("panda", 0, load_json("configs/collect.json"), tmp_path / "episode", max_episode_s=.02)
    path = tmp_path / "episode" / "commands.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_episode(path.parent)


@pytest.mark.parametrize("fault_tick", [100, 109])
def test_replay_adapter_fault_without_advancing_tick(tmp_path, monkeypatch, fault_tick):
    config = load_json("configs/collect.json")
    original = RobotAdapter.update
    def blocked_reference(self, dt):
        if round(self.data.time / dt) == fault_tick:
            self.target = mink.SE3.from_rotation_and_translation(self.target.rotation(),
                self.data.site_xpos[self.index.tcp] + np.array([self.config["reference_position_limit"], 0., 0.]))
        return original(self, dt)
    monkeypatch.setattr(RobotAdapter, "update", blocked_reference)
    result = run_episode("ur5e", 0, config, tmp_path / "fault", max_episode_s=.2)
    assert result["failure_reason"] == "blocked"
    _, arrays = load_episode(tmp_path / "fault")
    assert arrays["physics"][-1, 0] == arrays["physics"][-2, 0]
    assert arrays["physics"][-1, 0] == pytest.approx(fault_tick * .001)
    assert replay_episode(tmp_path / "fault")["max_observation_error"] == 0


def test_split_recipes_are_deterministic_and_disjoint():
    config = load_json("configs/collect.json")
    seen = set()
    for split in ("train", "validation", "test", "acceptance"):
        for recovery in (False, True):
            item = recipe(config, split, 0, recover=recovery)
            assert item == recipe(config, split, 0, recover=recovery)
            assert item[0] not in seen
            seen.add(item[0])


def test_recovery_requires_observed_completed_transition():
    events = [dict(name="phase", time=1., phase="RECOVER", previous="APPROACH"),
              dict(name="phase", time=2., phase="WAIT_READY", previous="RECOVER"),
              dict(name="failure", time=3., reason="food_dropped")]
    segments = annotate(events)
    assert segments[0]["recovery_valid"]
    events[1] = dict(name="failure", time=2., reason="food_dropped")
    assert not annotate(events)[0]["recovery_valid"]


def test_collection_refuses_partial_or_changed_gate():
    config = load_json("configs/collect.json")
    with pytest.raises(ValueError, match="100-seed"):
        check_gate(dict(teacher_gate="passed", robot_id="panda", baseline=dict(attempts=10, successes=10)), config, "panda")


def test_independent_mink_check_does_not_move_simulator():
    env = FeedingGymEnv()
    env.reset(seed=0)
    state = env.task.snapshot()
    before = env.task.get_state()
    checks = check_waypoints(env.task, [("current", state["tcp_position"], state["tcp_rotation"])])
    assert checks[0]["reachable"]
    np.testing.assert_array_equal(before["physics"], env.task.get_state()["physics"])
    env.close()


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_immediate_snapshot_restores_boundary_wrench(robot):
    env = FeedingGymEnv(robot)
    env.reset(seed=7)
    env.step(np.array([.1, 0, 0, 0, 0, 0]))
    state = env.get_state()
    expected = env.observe_policy()
    env.set_state(state)
    np.testing.assert_array_equal(env.observe_policy(), expected)
    env.close()


def test_recipe_groups_never_cross_splits():
    config = load_json("configs/collect.json")
    groups = {}
    for split in ("train", "validation", "test", "acceptance"):
        for index in range(40):
            seed, scenario, parameters, group = recipe(config, split, index)
            assert group not in groups or groups[group] == split
            groups[group] = split


def test_dataset_rejects_group_leakage(tmp_path):
    import json
    for split in ("train", "validation"):
        directory = tmp_path / split / "episode"
        run_episode("panda", 0, load_json("configs/collect.json"), directory, max_episode_s=.02, split=split)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["group_id"] = "same-layout-material-driver-group"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaked across splits"):
        dataset_statistics(tmp_path)


def test_waypoint_stop_before_grid_action_replays_observation_order(tmp_path, monkeypatch):
    original = Teacher.act
    def stop_at_grid(self, obs):
        command = original(self, obs)
        self.stop_requested |= np.isclose(obs['time'], .05)
        return command
    monkeypatch.setattr(Teacher, 'act', stop_at_grid)
    run_episode('panda', 0, load_json('configs/collect.json'), tmp_path/'episode', max_episode_s=.12)
    commands = json.loads((tmp_path/'episode/commands.json').read_text())
    assert any(c['tick'] == 50 and c['kind'] == 'stop' and not c['after_physics'] for c in commands)
    assert replay_episode(tmp_path/'episode')['max_observation_error'] == 0
