"""Inspect M4 failures without changing frozen execution or acceptance code."""

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


TASK_FAILURES = {"food_dropped", "food_lost_after_delivery",
                 "withdrawal_before_release", "food_missing"}


@contextmanager
def recorded_asset_paths(source_root, workspace, manifest):
    """Keep recorded mesh paths in MJB signatures when using archived code."""
    if source_root == workspace or manifest is None:
        yield
        return
    from feedingrobot.sim import model
    original = model._merge_asset

    def merge(root, path, *args):
        world = original(root, path, *args)
        for mesh in root.findall("asset/mesh"):
            archived = Path(mesh.get("file"))
            if archived.is_relative_to(source_root):
                relative = archived.relative_to(source_root)
                recorded = workspace / relative
                expected = manifest["input_hashes"].get(str(relative))
                if not recorded.is_file() or hashlib.sha256(recorded.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Recorded mesh is missing or changed: {relative}")
                mesh.set("file", str(recorded))
        return world

    with patch.object(model, "_merge_asset", merge):
        yield


def seconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("Seconds must be finite and nonnegative")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path)
    source.add_argument("--case")
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--post-failure-seconds", type=seconds, default=0.)
    args = parser.parse_args(argv)
    if bool(args.case is not None) != bool(args.cases_file is not None):
        parser.error("--case requires --cases-file; --episode does not use --cases-file")
    if args.post_failure_seconds and not args.viewer:
        parser.error("--post-failure-seconds requires --viewer")
    return args


class Diagnostics:
    """Read-only boundary and command records, separate from episode arrays."""

    def __init__(self, directory, segment):
        directory.mkdir(parents=True, exist_ok=True)
        self.physics = gzip.open(directory / "diagnostics.jsonl.gz", "wt")
        self.commands = (directory / "diagnostic_commands.jsonl").open("w")
        self.segment = segment
        self.previous = None

    def write(self, stream, row):
        from feedingrobot.data.episodes import json_value
        stream.write(json.dumps(row, default=json_value, allow_nan=False) + "\n")

    def adapter_state(self, task):
        import numpy as np
        adapter = task.adapter
        state = dict(reference_q=adapter.reference.q[task.index.qpos].copy(),
                    reference_tcp=adapter.target.wxyz_xyz.copy(),
                    shaped_twist_base=adapter.velocity.copy(),
                    command=None if adapter.command is None else adapter.command[0].copy(),
                    valid_until=None if adapter.command is None else adapter.command[1],
                    status=adapter.status, fault=adapter.fault)
        return {key: None if isinstance(value, (np.ndarray, float)) and not np.isfinite(value).all()
                else value for key, value in state.items()}

    def command(self, task, kind, before, **detail):
        now = float(task.data.time)
        self.write(self.commands, dict(segment=self.segment, tick=task.tick, time=now if math.isfinite(now) else None,
                                      kind=kind, before=before, after=self.adapter_state(task), **detail))

    def boundary(self, task, teacher, before):
        import mujoco
        import numpy as np
        from feedingrobot.sim.events import evidence, geom_corners

        now = float(task.data.time)
        valid = task.failure_reason != "nonfinite_state" and np.isfinite(now)
        row = dict(segment=self.segment, tick=task.tick, time=now if np.isfinite(now) else None,
                   phase=task.logic.phase, failure=task.failure_reason, valid=bool(valid),
                   adapter_before=before, adapter_after=self.adapter_state(task),
                   part=teacher.part if teacher else None,
                   carry_part=teacher.carry_part if teacher else None,
                   lift_complete=teacher.pickup_lift_complete if teacher else None,
                   target_position=teacher.target_position.copy() if teacher else None,
                   target_rotation=teacher.target_rotation.copy() if teacher else None,
                   next_action_tick=(task.tick // round(.05 / task.dt) + 1) * round(.05 / task.dt))
        # Invalid states have no geometry or numerical derivatives to serialize.
        if not valid:
            row["adapter_before"] = row["adapter_after"] = None
            self.write(self.physics, row)
            return
        snapshot, e = task.snapshot(), evidence(task)
        rotation = snapshot["tcp_rotation"]
        local = rotation.T @ (e["food_position"] - e["tcp_position"])
        shaped = task.adapter.velocity.copy()
        base = task.data.site_xmat[task.index.base].reshape(3, 3)
        reference_velocity = np.r_[base @ shaped[:3], base @ shaped[3:]]
        elapsed = None if self.previous is None else now - self.previous["time"]
        derivative_valid = elapsed is not None and elapsed > 1e-12
        previous = self.previous
        ranges = np.array([task.model.jnt_range[mujoco.mj_name2id(
            task.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in task.robot_config["joints"]])
        actual_q = snapshot["q"]
        reference_q = task.adapter.reference.q[task.index.qpos].copy()
        food_geom = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_GEOM, "food_box")
        plate = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
        corners = geom_corners(task.model, task.data, food_geom)
        row.update(actual_tcp_position=snapshot["tcp_position"], actual_tcp_rotation=rotation,
                   actual_tcp_twist_world=snapshot["tcp_twist_world"],
                   reference_tcp_twist_world=reference_velocity,
                   derivative_dt_s=elapsed, derivative_valid=derivative_valid,
                   actual_tcp_acceleration_world=(snapshot["tcp_twist_world"] - previous["velocity"]) / elapsed
                       if derivative_valid else None,
                   reference_tcp_acceleration_world=(reference_velocity - previous["reference_velocity"]) / elapsed
                       if derivative_valid else None,
                   food_position=e["food_position"], food_local=local,
                   food_local_velocity=(local - previous["local"]) / elapsed if derivative_valid else None,
                   pitch_rad=float(np.arcsin(np.clip(-rotation[2, 0], -1., 1.))),
                   food_min_corner_z_m=float(corners[:, 2].min()),
                   food_min_corner_plate_height_m=float(corners[:, 2].min() - task.data.site_xpos[plate, 2]),
                   actual_q=actual_q, reference_q=reference_q,
                   joint_limit_margin=np.minimum(actual_q-ranges[:, 0], ranges[:, 1]-actual_q),
                   reference_joint_limit_margin=np.minimum(reference_q-ranges[:, 0], ranges[:, 1]-reference_q),
                   supported=e["supported"], on_plate=e["on_plate"], off_plate=e["off_plate"],
                   food_ground_contact=e["food_ground_contact"], mouth_supported=e["mouth_supported"],
                   at_wait=e["at_wait"], penetration=e["penetration"],
                   event_unsupported_s=task.logic.timers["unsupported"],
                   contacts=copy.deepcopy(task.contacts))
        self.write(self.physics, row)
        self.previous = dict(time=now, velocity=snapshot["tcp_twist_world"].copy(),
                             reference_velocity=reference_velocity.copy(), local=local.copy())

    def close(self):
        self.physics.close()
        self.commands.close()


class Observation:
    """Hooks live only around the original run/replay in this diagnostic process."""

    def __init__(self, viewer, duration, manifest=None, diagnostics=None):
        self.viewer, self.duration, self.manifest = viewer, duration, manifest
        self.task = self.teacher = self.display = self.capture = None
        self.teacher_error = 0.
        self.disabled_reason = None
        self.diagnostics_directory = diagnostics
        self.diagnostics = None

    def __enter__(self):
        import numpy as np
        from feedingrobot.control.adapter import RobotAdapter
        from feedingrobot.experts import Teacher
        from feedingrobot.sim.observer_viewer import ObserverViewer
        from feedingrobot.sim.task import FeedingTask

        original_init, original_act = FeedingTask.__init__, Teacher.act
        original_command, original_stop = RobotAdapter.set_twist, RobotAdapter.stop
        original_step, original_terminate = FeedingTask.step_physics, FeedingTask._terminate

        def initialize(task, *args, **kwargs):
            original_init(task, *args, **kwargs)
            self.task = task
            if self.manifest is not None:
                self.teacher = Teacher(task.robot_config, self.manifest["teacher_config"])
                from feedingrobot.experts.geometry import teacher_geometry
                self.teacher.reset(self.manifest["teacher_parameters"], geometry=teacher_geometry(task))

        def act(teacher, observation):
            self.teacher = teacher
            return original_act(teacher, observation)

        def command(adapter, twist, now, valid_until):
            if self.manifest is not None and self.task is not None and adapter is self.task.adapter:
                proposed = self.teacher.act(self.task.provider.observe()["policy_obs"])
                error = float(np.max(np.abs(proposed - twist)))
                self.teacher_error = max(self.teacher_error, error)
                if not np.isfinite(error) or error > 1e-12:
                    raise AssertionError("Recorded commands do not reproduce teacher history")
            before = self.diagnostics.adapter_state(self.task) if self.diagnostics and adapter is self.task.adapter else None
            result = original_command(adapter, twist, now, valid_until)
            if before is not None:
                self.diagnostics.command(self.task, "twist", before, command_time=now, valid_until=valid_until)
            return result

        def stop(adapter, reason="stopped", fault=False, *, hold_reference=False):
            before = self.diagnostics.adapter_state(self.task) if self.diagnostics and self.task is not None and adapter is self.task.adapter else None
            result = original_stop(adapter, reason, fault, hold_reference=hold_reference)
            if before is not None:
                self.diagnostics.command(self.task, "stop", before, reason=reason, fault=fault,
                                         hold_reference=hold_reference)
            return result

        def terminate(task, reason):
            if (task is self.task and self.duration and reason in TASK_FAILURES
                    and not task.terminated and not task.adapter.fault and self.capture is None):
                self.capture = dict(state=task.get_state(), teacher=copy.deepcopy(self.teacher), reason=reason)
            return original_terminate(task, reason)

        def step(task, **kwargs):
            active = task is self.task and not kwargs.get("_settling", False)
            if active and self.viewer and self.display is None:
                self.display = ObserverViewer(task.model, task.data)
                self.display.start()
            before = self.diagnostics.adapter_state(task) if active and self.diagnostics else None
            state = original_step(task, **kwargs)
            if before is not None:
                self.diagnostics.boundary(task, self.teacher, before)
            if active and self.display is not None and (task.tick % round(.02 / task.dt) == 0 or task.terminated):
                self.display.update(phase=task.logic.phase,
                                    result=task.failure_reason or ("success" if task.logic.success else "running"))
                if self.display.report["status"] in ("closed", "unavailable") and self.disabled_reason is None:
                    self.disabled_reason = self.display.report.get("reason", self.display.report["status"])
                    print(f"Viewer disabled; original episode continues headless: {self.disabled_reason}", flush=True)
            return state

        self.hooks = ExitStack()
        if self.diagnostics_directory is not None:
            self.diagnostics = Diagnostics(self.diagnostics_directory, "original")
            self.hooks.callback(self.diagnostics.close)
        for owner, name, replacement in ((FeedingTask, "__init__", initialize), (Teacher, "act", act),
                                          (RobotAdapter, "set_twist", command),
                                          (RobotAdapter, "stop", stop),
                                          (FeedingTask, "_terminate", terminate), (FeedingTask, "step_physics", step)):
            self.hooks.enter_context(patch.object(owner, name, replacement))
        return self

    def __exit__(self, kind, value, traceback):
        self.hooks.close()
        if kind is not None and self.display is not None:
            self.display.close()

    def continue_failure(self, original, directory):
        import numpy as np
        from feedingrobot.data.episodes import json_value, write_json
        from feedingrobot.envs import FeedingGymEnv
        from feedingrobot.sim.events import evidence

        report = dict(observation_branch=True, requested_seconds=self.duration, elapsed_seconds=0.,
                      original_failure=original["failure_reason"], original_time_s=original["time_s"])
        if not self.duration:
            return dict(report, reason="disabled")
        if original["success"] or original["truncated"]:
            return dict(report, reason="original_success" if original["success"] else "original_time_limit")
        if original["failure_reason"] not in TASK_FAILURES or self.capture is None:
            return dict(report, reason="execution_protection")
        self.display.update()
        if self.display.report["status"] != "running":
            return dict(report, reason=self.display.report.get("reason", "viewer_unavailable"))

        directory.mkdir()
        # Trusted-local diagnostic snapshot; never consumed as a training episode.
        (directory / "pre_stop.pkl").write_bytes(pickle.dumps(self.capture, protocol=5))
        env = FeedingGymEnv(original["robot_id"], timestep=original["dt"],
                           max_episode_s=original["max_episode_s"])
        diagnostics = None
        try:
            task = env.task
            task.model.opt.iterations = original["solver_iterations"]
            task.model.opt.tolerance = original["solver_tolerance"]
            env.reset(seed=original["seed"], options={"scenario": original["scenario"]})
            state = copy.deepcopy(self.capture["state"])
            state["logic"]["failure_reason"] = None
            task.set_state(state)
            teacher = self.capture["teacher"]
            diagnostics = Diagnostics(directory, "observation_branch") if self.diagnostics_directory is not None else None
            if diagnostics:
                original_stop = task.adapter.stop

                def diagnostic_stop(reason="stopped", fault=False, *, hold_reference=False):
                    before = diagnostics.adapter_state(task)
                    result = original_stop(reason, fault, hold_reference=hold_reference)
                    diagnostics.command(task, "stop", before, reason=reason, fault=fault,
                                        hold_reference=hold_reference)
                    return result

                task.adapter.stop = diagnostic_stop
            terminate = task._terminate

            def observe_termination(reason):
                if reason in TASK_FAILURES and not task.adapter.fault and not task.terminated:
                    task.logic.failure_reason = None
                else:
                    terminate(reason)

            task._terminate = observe_termination
            self.display.model, self.display.data = task.model, task.data
            action_ticks, obs_ticks = round(.05 / task.dt), round(.02 / task.dt)
            start_tick, start_time = task.tick, float(task.data.time)
            start_wall = time.monotonic()
            event_start = len(task.logic.events)
            reason = "duration_reached"
            print(f"Original failure: {original['failure_reason']} at {start_time:.6f}s; "
                  f"teacher observation branch for {self.duration:g}s", flush=True)

            def line(stream, row):
                stream.write(json.dumps(row, default=json_value, allow_nan=False) + "\n")

            with gzip.open(directory / "physics.jsonl.gz", "wt") as physics, \
                    (directory / "actions.jsonl").open("w") as actions, \
                    (directory / "observations.jsonl").open("w") as observations:
                line(observations, dict(tick=task.tick, time=start_time, observation=env.observe_policy()))
                # Integer ticks avoid accumulated floating-point duration errors.
                while (task.tick - start_tick) * task.dt + 1e-12 < self.duration:
                    phase, before_events = task.logic.phase, len(task.logic.events)
                    if task.tick % action_ticks == 0:
                        command = teacher.act(task.provider.observe()["policy_obs"])
                        until = (task.tick + action_ticks) * task.dt
                        before = diagnostics.adapter_state(task) if diagnostics else None
                        task.adapter.set_twist(command, task.data.time, until)
                        if diagnostics:
                            diagnostics.command(task, "twist", before, command_time=float(task.data.time), valid_until=until)
                        line(actions, dict(tick=task.tick, time=float(task.data.time), kind="twist",
                                           command=command, valid_until=until, part=teacher.part,
                                           carry_part=teacher.carry_part, target_position=teacher.target_position,
                                           target_rotation=teacher.target_rotation))
                    before = diagnostics.adapter_state(task) if diagnostics else None
                    snapshot = task.step_physics()
                    if phase != task.logic.phase and not task.terminated:
                        task.adapter.stop(hold_reference=True)
                        line(actions, dict(tick=task.tick, kind="stop", hold_reference=True))
                    if diagnostics:
                        diagnostics.boundary(task, teacher, before)
                    valid = task.failure_reason != "nonfinite_state"
                    row = dict(tick=task.tick, time=float(task.data.time) if np.isfinite(task.data.time) else None,
                               phase=task.logic.phase, valid=valid, failure=task.failure_reason,
                               events=task.logic.events[before_events:])
                    if valid:
                        row.update(snapshot=snapshot, evidence=evidence(task), contacts=task.contacts,
                                   timers=task.logic.timers, reference_q=task.adapter.reference.q,
                                   reference_tcp=task.adapter.target.wxyz_xyz)
                    line(physics, row)
                    final = task.terminated or (task.tick - start_tick) * task.dt + 1e-12 >= self.duration
                    if task.tick % obs_ticks == 0 or final:
                        line(observations, dict(tick=task.tick, time=row["time"], valid=valid,
                                                observation=env.observe_policy() if valid else None))
                        elapsed = (task.tick - start_tick) * task.dt
                        time.sleep(max(0., start_wall + elapsed - time.monotonic()))
                        self.display.update(phase=task.logic.phase, result=task.failure_reason or "observation branch")
                        if self.display.report["status"] != "running":
                            reason = self.display.report.get("reason", "viewer_unavailable")
                            break
                    if task.terminated:
                        reason = "branch_success" if task.logic.success else task.failure_reason
                        break
            report.update(reason=reason, elapsed_seconds=(task.tick - start_tick) * task.dt,
                          wall_seconds=time.monotonic() - start_wall,
                          start_tick=start_tick, end_tick=task.tick, success=task.logic.success,
                          phase=task.logic.phase, events=task.logic.events[event_start:])
            write_json(directory / "summary.json", report)
            return report
        finally:
            if diagnostics:
                diagnostics.close()
            env.close()


def main(argv=None):
    args = parse_args(argv)
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    from feedingrobot.data.episodes import load_episode, write_json
    from feedingrobot.data.replay import replay_episode
    from feedingrobot.data.rollout import run_episode
    from feedingrobot.sim.model import ROOT, load_json

    if ROOT.resolve() != source_root:
        raise ValueError("Selected source root was not loaded")
    inspection_root = Path(__file__).resolve().parents[1] / "outputs/inspection/m4"
    directory = inspection_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = load_episode(args.episode)[0] if args.episode else None
    observer = Observation(args.viewer, args.post_failure_seconds, manifest,
                           directory if args.diagnostics else None)
    with recorded_asset_paths(source_root, Path(__file__).resolve().parents[1], manifest):
        try:
            with observer:
                if args.episode:
                    replay = replay_episode(args.episode)
                    original = manifest
                else:
                    case = json.loads(args.cases_file.read_text())[args.case]
                    original = run_episode("panda", case["seed"], load_json("configs/collect.json"),
                                           directory / "episode", scenario=case["scenario"],
                                           teacher_parameters=case["teacher_parameters"], max_episode_s=60.,
                                           split="inspection", group_id=f"inspection:{args.case}")
                    replay = None
            branch = observer.continue_failure(original, directory / "branch")
            visualization = (observer.display.close() if observer.display else dict(status="disabled"))
            summary = dict(original=dict(time_s=original["time_s"], failure_reason=original["failure_reason"],
                                         success=original["success"], truncated=original["truncated"],
                                         events=original["events"]), branch=branch, visualization=visualization,
                           replay=replay, replay_teacher_max_error=observer.teacher_error,
                           source_root=str(source_root), input_hashes=original["input_hashes"],
                           observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           diagnostics_requested=args.diagnostics,
                           source_episode=str(args.episode.resolve()) if args.episode else str(directory / "episode"))
            write_json(directory / "summary.json", summary)
            print(f"Inspection evidence: {directory}", flush=True)
            print(json.dumps({k: v for k, v in branch.items() if k != "events"}, ensure_ascii=False), flush=True)
        finally:
            if observer.display:
                observer.display.close()


if __name__ == "__main__":
    main()
