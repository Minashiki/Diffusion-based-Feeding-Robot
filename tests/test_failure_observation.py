import importlib.util
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from feedingrobot.data.episodes import load_episode
from feedingrobot.data.replay import replay_episode
from feedingrobot.data.rollout import run_episode
from feedingrobot.control.adapter import RobotAdapter
from feedingrobot.sim.model import load_json
from feedingrobot.sim.task import FeedingTask


spec = importlib.util.spec_from_file_location("observe_m4", Path(__file__).parents[1] / "tools/observe_m4.py")
observe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observe)


def test_archived_mesh_paths_require_recorded_bytes(tmp_path, monkeypatch):
    from feedingrobot.sim import model
    archive, workspace = tmp_path/"archive", tmp_path/"workspace"
    relative = Path("assets/mesh.obj")
    recorded = workspace/relative
    recorded.parent.mkdir(parents=True)
    recorded.write_bytes(b"recorded mesh")
    manifest = dict(input_hashes={str(relative): hashlib.sha256(recorded.read_bytes()).hexdigest()})
    def merge(root, path):
        ET.SubElement(root.find("asset"), "mesh", file=str(archive/relative))
        return "world"
    monkeypatch.setattr(model,"_merge_asset",merge)
    with observe.recorded_asset_paths(archive,workspace,manifest):
        root = ET.fromstring("<mujoco><asset/></mujoco>")
        assert model._merge_asset(root,None)=="world"
        assert root.find("asset/mesh").get("file")==str(recorded)
        recorded.write_bytes(b"changed mesh")
        with pytest.raises(ValueError,match="missing or changed"):
            model._merge_asset(ET.fromstring("<mujoco><asset/></mujoco>"),None)
    assert model._merge_asset is merge


class Display:
    status, reason = "running", None

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.report = dict(status=self.status, reason=self.reason)

    def start(self):
        return self.report.copy()

    def update(self, **kwargs):
        pass

    def close(self):
        return self.report.copy()


@pytest.fixture
def display(monkeypatch):
    monkeypatch.setattr("feedingrobot.sim.observer_viewer.ObserverViewer", Display)
    monkeypatch.setattr(observe.time, "sleep", lambda duration: None)


def fail_at_tick(monkeypatch, original_reason="food_dropped", tail_reason=None, phase_change=False):
    original_step = getattr(FeedingTask.step_physics, "unwrapped", FeedingTask.step_physics)

    def step(task, **kwargs):
        result = original_step(task, **kwargs)
        if not kwargs.get("_settling", False):
            reason = original_reason if task.tick == 53 else tail_reason if task.tick == 54 else None
            if reason:
                task.logic.emit("failure", task.data.time, reason=reason)
                task.logic.failure_reason = reason
                task._terminate(reason)
                result = task.snapshot()
            if phase_change and task.tick == 54:
                task.logic.switch("WAIT_READY", task.data.time)
        return result

    step.unwrapped = original_step
    monkeypatch.setattr(FeedingTask, "step_physics", step)


def run(directory):
    return run_episode("panda", 0, load_json("configs/collect.json"), directory, max_episode_s=.12)


def test_cli_default_and_fractional_seconds():
    assert observe.parse_args(["--episode", "local"]).post_failure_seconds == 0
    args = observe.parse_args(["--case", "1", "--cases-file", "cases.json", "--viewer",
                               "--post-failure-seconds", "2.5"])
    assert args.post_failure_seconds == 2.5


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_seconds(value):
    with pytest.raises(SystemExit):
        observe.parse_args(["--episode", "local", "--viewer", f"--post-failure-seconds={value}"])


@pytest.mark.parametrize("arguments", [
    ["--episode", "local", "--post-failure-seconds", "2"],
    ["--case", "1"],
    ["--episode", "local", "--cases-file", "cases.json"],
    ["--episode", "local", "--case", "1", "--cases-file", "cases.json"],
])
def test_cli_rejects_incomplete_or_conflicting_inputs(arguments):
    with pytest.raises(SystemExit):
        observe.parse_args(arguments)


@pytest.mark.parametrize("duration", [0., .0505])
def test_original_prefix_is_unchanged_and_replays_exactly(tmp_path, monkeypatch, display, duration):
    fail_at_tick(monkeypatch)
    baseline = run(tmp_path / "baseline")
    with observe.Observation(True, duration) as observer:
        visible = run(tmp_path / "visible")
    _, first = load_episode(tmp_path / "baseline")
    _, second = load_episode(tmp_path / "visible")
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert baseline["events"] == visible["events"]
    assert baseline["time_s"] == visible["time_s"]
    assert visible["time_s"] == pytest.approx(.053)
    assert baseline["failure_reason"] == visible["failure_reason"] == "food_dropped"
    assert (tmp_path / "baseline/commands.json").read_bytes() == (tmp_path / "visible/commands.json").read_bytes()
    assert replay_episode(tmp_path / "visible")["max_physics_reference_error"] == 0
    if not duration:
        assert observer.capture is None
        assert observer.continue_failure(visible, tmp_path / "branch")["reason"] == "disabled"
    else:
        pre_stop = observer.capture["state"]["adapter"]
        assert pre_stop["command"] is not None and pre_stop["fault"] is None
        assert observer.task.adapter.fault == "food_dropped" and observer.task.adapter.command is None
        teacher = observer.capture["teacher"]
        assert teacher.last_time == pytest.approx(.05) and teacher.part == observer.teacher.part
        report = observer.continue_failure(visible, tmp_path / "branch")
        assert report["reason"] == "duration_reached" and report["elapsed_seconds"] == pytest.approx(.051)
        rows = [json.loads(row) for row in (tmp_path / "branch/actions.jsonl").read_text().splitlines()]
        assert [row["tick"] for row in rows if row["kind"] == "twist"] == [100]
        observations = [json.loads(row) for row in (tmp_path / "branch/observations.jsonl").read_text().splitlines()]
        assert [row["tick"] for row in observations] == [53, 60, 80, 100, 104]
        assert observer.task.failure_reason == "food_dropped" and observer.task.tick == 53


def test_replay_reconstructs_teacher_and_pre_stop_state(tmp_path, monkeypatch, display):
    fail_at_tick(monkeypatch)
    with observe.Observation(True, .002) as running:
        original = run(tmp_path / "episode")
    with observe.Observation(True, .002, original) as replaying:
        replay = replay_episode(tmp_path / "episode")
    assert replay["result_equal"] and replaying.teacher_error == 0
    assert running.capture["teacher"].part == replaying.capture["teacher"].part
    assert running.capture["teacher"].last_time == replaying.capture["teacher"].last_time
    for key in ("physics",):
        np.testing.assert_array_equal(running.capture["state"][key], replaying.capture["state"][key])
    for key in ("reference_q", "target", "velocity"):
        np.testing.assert_array_equal(running.capture["state"]["adapter"][key],
                                      replaying.capture["state"]["adapter"][key])


@pytest.mark.parametrize("reason", ["contact_limit", "model_penetration", "ik_failure", "nonfinite_state"])
def test_execution_protection_never_starts_or_continues_branch(tmp_path, monkeypatch, display, reason):
    fail_at_tick(monkeypatch, original_reason=reason)
    with observe.Observation(True, .05, diagnostics=tmp_path / "hard_diagnostics") as observer:
        original = run(tmp_path / "hard")
    assert observer.capture is None
    assert observer.continue_failure(original, tmp_path / "skip")["reason"] == "execution_protection"
    fail_at_tick(monkeypatch, tail_reason=reason)
    with observe.Observation(True, .05, diagnostics=tmp_path / "semantic_diagnostics") as observer:
        original = run(tmp_path / "semantic")
    branch = observer.continue_failure(original, tmp_path / "branch")
    assert branch["reason"] == reason and branch["elapsed_seconds"] == .001
    assert original["failure_reason"] == "food_dropped"


def test_phase_cancellation_preserves_reference_in_branch(tmp_path, monkeypatch, display):
    fail_at_tick(monkeypatch, phase_change=True)
    original_stop = RobotAdapter.stop
    preserved = []

    def stop(adapter, *args, **kwargs):
        before_q, before_tcp = adapter.reference.q.copy(), adapter.target.wxyz_xyz.copy()
        original_stop(adapter, *args, **kwargs)
        if kwargs.get("hold_reference"):
            np.testing.assert_array_equal(adapter.reference.q, before_q)
            np.testing.assert_array_equal(adapter.target.wxyz_xyz, before_tcp)
            preserved.append(float(adapter.data.time))

    monkeypatch.setattr(RobotAdapter, "stop", stop)
    with observe.Observation(True, .002) as observer:
        original = run(tmp_path / "episode")
    observer.continue_failure(original, tmp_path / "branch")
    rows = [json.loads(row) for row in (tmp_path / "branch/actions.jsonl").read_text().splitlines()]
    assert rows == [dict(tick=54, kind="stop", hold_reference=True)]
    import gzip
    with gzip.open(tmp_path / "branch/physics.jsonl.gz", "rt") as stream:
        first, second = [json.loads(row) for row in stream]
    assert first["phase"] == "WAIT_READY"
    assert preserved[-1] == pytest.approx(.054)


def test_window_closed_during_tail_ends_observation(tmp_path, monkeypatch, display):
    fail_at_tick(monkeypatch)

    def update(viewer, **kwargs):
        if viewer.data.time > .054:
            viewer.report.update(status="closed", reason="window_closed")

    monkeypatch.setattr(Display, "update", update)
    with observe.Observation(True, 2., diagnostics=tmp_path / "diagnostic") as observer:
        original = run(tmp_path / "episode")
    report = observer.continue_failure(original, tmp_path / "branch")
    assert report["reason"] == "window_closed" and report["elapsed_seconds"] == .007
    assert original["failure_reason"] == "food_dropped"


def test_success_in_branch_ends_early_without_rewriting_original(tmp_path, monkeypatch, display):
    fail_at_tick(monkeypatch)
    original_step = FeedingTask.step_physics

    def step(task, **kwargs):
        snapshot = original_step(task, **kwargs)
        if not kwargs.get("_settling", False) and task.tick == 54:
            task.logic.success = task.terminated = True
            task.logic.emit("success", task.data.time)
            task.adapter.stop("success")
        return snapshot

    monkeypatch.setattr(FeedingTask, "step_physics", step)
    with observe.Observation(True, 2.) as observer:
        original = run(tmp_path / "episode")
    report = observer.continue_failure(original, tmp_path / "branch")
    assert report["reason"] == "branch_success" and report["elapsed_seconds"] == .001
    assert report["success"] and not original["success"]
    assert original["failure_reason"] == "food_dropped"


@pytest.mark.parametrize("status,reason", [("closed", "window_closed"), ("unavailable", "desktop_unavailable")])
def test_missing_or_closed_window_skips_tail_but_keeps_failure(tmp_path, monkeypatch, display, status, reason):
    monkeypatch.setattr(Display, "status", status)
    monkeypatch.setattr(Display, "reason", reason)
    fail_at_tick(monkeypatch)
    with observe.Observation(True, .02) as observer:
        original = run(tmp_path / "episode")
    report = observer.continue_failure(original, tmp_path / "branch")
    assert report["reason"] == reason and report["elapsed_seconds"] == 0
    assert original["failure_reason"] == "food_dropped" and not (tmp_path / "branch").exists()


@pytest.mark.parametrize("result,reason", [
    [dict(success=True, truncated=False), "original_success"],
    [dict(success=False, truncated=True), "original_time_limit"],
])
def test_success_and_timeout_do_not_append_observation(tmp_path, result, reason):
    observer = observe.Observation(True, 2.)
    result.update(failure_reason=None, time_s=1.)
    assert observer.continue_failure(result, tmp_path / "branch")["reason"] == reason
    assert not (tmp_path / "branch").exists()


@pytest.mark.parametrize("duration", [0., .051])
def test_diagnostics_preserve_original_and_log_branch(tmp_path, monkeypatch, display, duration):
    import gzip
    fail_at_tick(monkeypatch)
    baseline = run(tmp_path / "baseline")
    with observe.Observation(True, duration, diagnostics=tmp_path / "diagnostic") as observer:
        original = run(tmp_path / "episode")
    report = observer.continue_failure(original, tmp_path / "branch")
    _, first = load_episode(tmp_path / "baseline")
    _, second = load_episode(tmp_path / "episode")
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert baseline["events"] == original["events"]
    assert (tmp_path / "baseline/commands.json").read_bytes() == (tmp_path / "episode/commands.json").read_bytes()
    assert replay_episode(tmp_path / "episode")["max_physics_reference_error"] == 0
    with gzip.open(tmp_path / "diagnostic/diagnostics.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 53 and rows[-1]["failure"] == "food_dropped"
    assert rows[0]["actual_tcp_acceleration_world"] is None
    assert rows[1]["derivative_valid"]
    assert all(key in rows[-1] for key in ("food_local_velocity", "food_min_corner_plate_height_m",
               "reference_joint_limit_margin", "actual_tcp_acceleration_world", "reference_tcp_acceleration_world"))
    commands = [json.loads(line) for line in (tmp_path / "diagnostic/diagnostic_commands.jsonl").read_text().splitlines()]
    fault = next(row for row in commands if row.get("fault"))
    assert fault["before"]["command"] is not None and fault["after"]["command"] is None
    assert fault["after"]["fault"] == "food_dropped"
    if duration:
        assert report["elapsed_seconds"] == pytest.approx(.051)
        with gzip.open(tmp_path / "branch/diagnostics.jsonl.gz", "rt") as stream:
            branch = [json.loads(line) for line in stream]
        assert len(branch) == 51 and branch[0]["tick"] == 54
        assert branch[0]["segment"] == "observation_branch"
        assert branch[-1]["time"] - branch[0]["time"] == pytest.approx(.05)


def test_diagnostics_zero_time_protection_boundary(tmp_path, monkeypatch, display):
    original_update = RobotAdapter.update

    def update(adapter, dt):
        if adapter.data.time >= .052 - 1e-12:
            adapter.stop("ik_failure", fault=True)
        else:
            original_update(adapter, dt)

    monkeypatch.setattr(RobotAdapter, "update", update)
    with observe.Observation(True, .05, diagnostics=tmp_path / "diagnostic") as observer:
        original = run(tmp_path / "episode")
    assert original["failure_reason"] == "ik_failure" and observer.capture is None
    assert observer.continue_failure(original, tmp_path / "branch")["reason"] == "execution_protection"
    import gzip
    with gzip.open(tmp_path / "diagnostic/diagnostics.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[-1]["tick"] == rows[-2]["tick"] == 52
    assert rows[-1]["derivative_dt_s"] == 0 and not rows[-1]["derivative_valid"]
    assert rows[-1]["actual_tcp_acceleration_world"] is None
    assert rows[-1]["food_local_velocity"] is None


def test_diagnostics_unavailable_window_keeps_original(tmp_path, monkeypatch, display):
    fail_at_tick(monkeypatch)
    monkeypatch.setattr(Display, "status", "unavailable")
    monkeypatch.setattr(Display, "reason", "desktop_unavailable")
    with observe.Observation(True, .05, diagnostics=tmp_path / "diagnostic") as observer:
        original = run(tmp_path / "episode")
    report = observer.continue_failure(original, tmp_path / "branch")
    assert report["reason"] == "desktop_unavailable" and report["elapsed_seconds"] == 0
    assert original["failure_reason"] == "food_dropped"
    assert (tmp_path / "diagnostic/diagnostics.jsonl.gz").is_file()
    assert not (tmp_path / "branch").exists()
