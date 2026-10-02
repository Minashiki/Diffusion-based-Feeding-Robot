import json
import queue
from types import SimpleNamespace

import mink
import numpy as np
import pytest

from feedingrobot.data.episodes import load_episode
from feedingrobot.data.rollout import run_episode
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.sim.model import load_json
from feedingrobot.sim.observer_viewer import ObserverViewer


def test_loaded_spoon_tilts_only_after_pickup_and_clearance():
    env = FeedingGymEnv()
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({"carry_pitch_rad": -.2}, geometry=teacher_geometry(env.task))
    obs = env.task.provider.observe()["policy_obs"]
    teacher.act(obs)
    teacher.capture_roll_complete = True
    teacher.part = 4  # Isolate the controller; this is not physical pickup evidence.
    teacher.capture_position = obs["tcp_position"].copy()
    obs["tcp_position"] = obs["tcp_position"].copy()
    obs["tcp_position"][2] = .06
    obs["stage"] = "ACQUIRE"
    teacher.act(obs)
    scoop_rotation = teacher.target_rotation.copy()
    assert teacher.carry_part == 0
    obs["stage"] = "TRANSPORT"
    obs["tcp_position"][2] = teacher.geometry["plate_position"][2] + teacher.parameters["tilt_clearance_m"] - .001
    teacher.act(obs)
    np.testing.assert_array_equal(teacher.target_rotation, scoop_rotation)
    obs["tcp_position"][2] += .002
    teacher.act(obs)
    loaded_rotation = teacher.target_rotation.copy()
    assert teacher.carry_part == 1
    assert np.linalg.norm(mink.SO3.from_matrix(loaded_rotation @ scoop_rotation.T).log()) > .05
    obs["tcp_position"][2] -= .002
    teacher.act(obs)
    np.testing.assert_array_equal(teacher.target_rotation, loaded_rotation)
    np.testing.assert_allclose(teacher.target_position[2], teacher.geometry["plate_position"][2] + teacher.parameters["lift_clearance_m"], atol=1e-12)
    env.close()


def test_supported_pickup_continues_roll_before_transport():
    env = FeedingGymEnv()
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({}, geometry=teacher_geometry(env.task))
    obs = env.task.provider.observe()["policy_obs"]
    teacher.act(obs)
    teacher.part = 3
    obs["tcp_position"] = np.array([.483, -.187, .0134])
    obs["stage"] = "TRANSPORT"
    obs["interaction"] = np.array([1., 0., 0., 0.])
    obs["food_relative_world"] = np.array([.02, 0., .006])
    command = teacher.act(obs)
    assert teacher.part == 4
    assert not teacher.capture_roll_complete and not teacher.pickup_lift_complete
    np.testing.assert_array_equal(teacher.capture_position, obs["tcp_position"])
    assert np.linalg.norm(command[:3]) <= teacher.parameters["linear_speed_m_s"] + 1e-12
    env.close()


def test_transport_segments_do_not_reverse_or_follow_tcp_drift():
    env = FeedingGymEnv()
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({}, geometry=teacher_geometry(env.task))
    obs = env.task.provider.observe()["policy_obs"]
    teacher.act(obs)
    teacher.pickup_lift_complete = True
    obs["stage"] = "TRANSPORT"
    obs["interaction"] = np.array([1., 0., 0., 0.])
    teacher.act(obs)
    height_target, loaded_rotation = teacher.target_position.copy(), teacher.target_rotation.copy()
    assert teacher.carry_part == 2
    obs["tcp_position"] = obs["tcp_position"] + np.array([.003, -.002, .01])
    teacher.act(obs)
    np.testing.assert_array_equal(teacher.target_position, height_target)
    np.testing.assert_array_equal(teacher.target_rotation, loaded_rotation)
    obs["tcp_position"], obs["tcp_rotation"] = height_target.copy(), loaded_rotation.copy()
    obs["food_relative_world"] = loaded_rotation @ np.array([.022, 0., .006])
    teacher.act(obs)
    assert teacher.carry_part == 2
    obs["food_relative_world"] = loaded_rotation @ np.array([.005, 0., .006])
    teacher.act(obs)
    assert teacher.carry_part == 3
    obs["tcp_position"][2] -= .02
    obs["food_relative_world"] = obs["tcp_rotation"] @ np.array([.022, 0., .006])
    mouth = obs["tcp_position"] + obs["mouth_relative_world"]
    teacher.act(obs)
    assert teacher.carry_part == 3
    np.testing.assert_allclose(teacher.target_position,
                               mouth - obs["mouth_rotation"][:, 0] * teacher.geometry["task_config"]["wait_offset_m"])
    np.testing.assert_array_equal(teacher.target_rotation, obs["mouth_rotation"])
    teacher.reset({}, geometry=teacher_geometry(env.task))
    assert teacher.carry_part == 0 and teacher.carry_position is None
    env.close()


def test_observer_refresh_is_wall_clock_limited_and_queue_never_blocks(monkeypatch):
    env = FeedingGymEnv()
    env.reset(seed=0)
    before = env.task.get_state()["physics"].copy()
    display = ObserverViewer(env.task.model, env.task.data)
    display.process = SimpleNamespace(pid=1, exitcode=None, is_alive=lambda: True)
    display.frames, display.messages = queue.Queue(maxsize=1), queue.Queue()
    times = iter([0., .01, .04, .08])
    monkeypatch.setattr("feedingrobot.sim.observer_viewer.time.monotonic", lambda: next(times))
    display.update()
    assert display.sent_frames == 1
    display.update()
    assert display.sent_frames == 1
    display.update()  # A full queue drops this frame immediately.
    assert display.sent_frames == 1
    display.frames.get_nowait()
    display.update()
    assert display.sent_frames == 2
    np.testing.assert_array_equal(env.task.get_state()["physics"], before)
    env.close()


def test_unavailable_observer_is_reported_and_close_is_idempotent(monkeypatch):
    env = FeedingGymEnv()
    env.reset(seed=0)
    def unavailable(_):
        raise OSError("desktop unavailable")
    monkeypatch.setattr("feedingrobot.sim.observer_viewer.multiprocessing.get_context", unavailable)
    display = ObserverViewer(env.task.model, env.task.data)
    result = display.start()
    assert result["status"] == "unavailable" and "desktop unavailable" in result["reason"]
    assert display.close() == result
    display.update()
    env.close()


@pytest.mark.parametrize("condition", ["window_closed", "unavailable"])
def test_observer_failure_or_close_preserves_commands_and_physics(tmp_path, monkeypatch, condition):
    class Display:
        def __init__(self, model, data):
            self.report = dict(status="running", physics_isolated=True, max_fps=30)

        def start(self):
            if condition == "unavailable":
                self.report.update(status="unavailable", reason="display_unavailable")
            return self.report.copy()

        def update(self, **kwargs):
            if condition == "window_closed":
                self.report.update(status="closed", reason="window_closed")

        def close(self):
            return self.report.copy()

    monkeypatch.setattr("feedingrobot.sim.observer_viewer.ObserverViewer", Display)
    config = load_json("configs/collect.json")
    headless = run_episode("panda", 0, config, tmp_path / "headless", max_episode_s=.12)
    visible = run_episode("panda", 0, config, tmp_path / "visible", max_episode_s=.12, viewer=True)
    assert visible["truncated"] and visible["time_s"] == headless["time_s"]
    assert visible["events"] == headless["events"]
    assert visible["visualization"]["reason"] == ("window_closed" if condition == "window_closed"
                                                  else "display_unavailable")
    _, first = load_episode(tmp_path / "headless")
    _, second = load_episode(tmp_path / "visible")
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert json.loads((tmp_path / "headless/commands.json").read_text()) == json.loads(
        (tmp_path / "visible/commands.json").read_text())
