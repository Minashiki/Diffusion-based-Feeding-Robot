"""Replay actual command intervals through physics, without running the teacher."""

import json
import pickle
from pathlib import Path

import numpy as np

from feedingrobot.data.episodes import input_hashes, load_episode, physics_row
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.sim.events import PHASES


def replay_episode(directory, *, viewer=False):
    directory = Path(directory)
    manifest, arrays = load_episode(directory)
    current_hashes = input_hashes()
    # Replay bug fixes may change this reader; all recorded physics inputs stay frozen.
    replay_source = "src/feedingrobot/data/replay.py"
    if {k: v for k, v in manifest["input_hashes"].items() if k != replay_source} != {
            k: v for k, v in current_hashes.items() if k != replay_source}:
        raise ValueError("Replay requires the recorded source/configuration hashes")
    env = FeedingGymEnv(manifest["robot_id"], timestep=manifest["dt"],
                       max_episode_s=manifest["max_episode_s"])
    env.task.model.opt.iterations = manifest["solver_iterations"]
    env.task.model.opt.tolerance = manifest["solver_tolerance"]
    env.reset(seed=manifest["seed"], options={"scenario": manifest["scenario"]})
    # Only load snapshots generated locally by this project, never untrusted pickle files.
    env.task.set_state(pickle.loads((directory / "initial_state.pkl").read_bytes()))
    task = env.task
    commands = json.loads((directory / "commands.json").read_text())
    cursor, obs_cursor, max_obs, max_reference = 0, 0, 0., 0.
    display = None
    def compare_observation():
        nonlocal obs_cursor, max_obs
        if obs_cursor < len(arrays["observation_ticks"]) and task.tick == arrays["observation_ticks"][obs_cursor]:
            actual = env.observe_policy() if arrays["observation_valid"][obs_cursor] else arrays["observations"][obs_cursor-1]
            max_obs = max(max_obs, float(np.max(np.abs(actual - arrays["observations"][obs_cursor]))))
            if PHASES.index(task.logic.phase) != arrays["observation_phases"][obs_cursor]:
                raise AssertionError("Replay phase mismatch")
            obs_cursor += 1
    try:
        if viewer:
            from feedingrobot.sim.observer_viewer import ObserverViewer
            display = ObserverViewer(task.model,task.data)
            if display.start()["status"] != "running":
                raise RuntimeError("Replay viewer is unavailable; physical viewer check remains unverified")
        compare_observation()
        for index in range(manifest["physics_rows"]):
            while cursor < len(commands) and commands[cursor]["tick"] == task.tick:
                command = commands[cursor]
                if command["kind"] == "stop":
                    task.adapter.stop(hold_reference=command.get("hold_reference", False))
                else:
                    task.adapter.set_twist(command["twist"], task.data.time, command["valid_until"])
                cursor += 1
            state = task.step_physics()
            # Phase cancellations happened after this physical boundary and before sampling.
            while (cursor < len(commands) and commands[cursor]["tick"] == task.tick
                   and commands[cursor]["kind"] == "stop"
                   and index + 1 < manifest["physics_rows"]):
                task.adapter.stop(hold_reference=commands[cursor].get("hold_reference", False))
                cursor += 1
            row = physics_row(task, state, manifest["physics_fields"])
            np.testing.assert_allclose(row, arrays["physics"][index], rtol=0, atol=1e-10, equal_nan=True)
            finite = np.isfinite(row)
            if np.any(finite):
                max_reference = max(max_reference, float(np.max(np.abs(row[finite] - arrays["physics"][index][finite]))))
            # An adapter fault can terminate before advancing time. The final
            # observation belongs to that fault row, not the preceding equal tick.
            if (index + 1 == manifest["physics_rows"]
                    or arrays["physics"][index + 1, 0] != row[0]
                    or (obs_cursor + 1 < len(arrays["observation_ticks"])
                        and arrays["observation_ticks"][obs_cursor + 1] == task.tick)):
                compare_observation()
            if viewer and task.tick % round(.02 / task.dt) == 0:
                display.update(phase=task.logic.phase, result=task.failure_reason or ("success" if task.logic.success else "running"))
        while cursor < len(commands) and commands[cursor]["tick"] == task.tick:
            if commands[cursor]["kind"] != "stop":
                raise AssertionError("Unexpected command after final physics row")
            task.adapter.stop(hold_reference=commands[cursor].get("hold_reference", False))
            cursor += 1
        if manifest["truncated"]:
            task.logic.emit("time_limit", task.data.time)
        assert cursor == len(commands), "Unconsumed replay command"
        assert obs_cursor == manifest["observation_rows"], "Unconsumed observation"
        assert task.logic.events == manifest["events"], "Replay event mismatch"
        assert task.logic.success == manifest["success"] and task.failure_reason == manifest["failure_reason"]
        assert max_obs <= 1e-7 and max_reference <= 1e-10, (max_obs, max_reference)
        visualization = display.close() if display else dict(status="disabled")
        if viewer and (not visualization.get("displayed_frames") or visualization.get("reason") != "episode_finished"):
            raise AssertionError("Full replay viewer did not display and complete the episode")
        return dict(status="passed", episode=str(directory), max_observation_error=max_obs,
                    visualization=visualization,
                    max_physics_reference_error=max_reference, events_equal=True, result_equal=True,
                    replay_source_hash=current_hashes[replay_source],
                    recorded_replay_source_hash=manifest["input_hashes"][replay_source])
    finally:
        if display:
            display.close()
        env.close()
