"""Integer-tick teacher execution using the existing, sole physics owner."""

import copy
import pickle
import time
from pathlib import Path

import numpy as np

from feedingrobot.data.episodes import EpisodeWriter, annotate, input_hashes
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.sim.events import PHASES, evidence


def run_episode(robot, seed, config, directory, *, scenario=None, teacher_parameters=None,
                timestep=.001, iterations=None, viewer=False, split="calibration", group_id=None,
                max_episode_s=None):
    scenario = dict(config["scene"], **(scenario or {}))
    env = FeedingGymEnv(robot, timestep=timestep, max_episode_s=max_episode_s)
    task = env.task
    if iterations is not None:
        task.model.opt.iterations = iterations
    env.reset(seed=seed, options={"scenario": scenario or {}})
    teacher = Teacher(task.robot_config, config)
    teacher.reset(teacher_parameters or {})
    action_ticks, obs_ticks = round(.05 / task.dt), round(.02 / task.dt)
    if not np.isclose(action_ticks * task.dt, .05) or not np.isclose(obs_ticks * task.dt, .02):
        env.close()
        raise ValueError("Physics dt must divide both observation and action periods")
    writer = EpisodeWriter(directory, env.schema, task.dt, env.max_episode_s)
    Path(directory, "initial_state.pkl").write_bytes(pickle.dumps(task.get_state(), protocol=5))
    start = time.monotonic()
    evidence_events = []
    last_observation = env.observe_policy()
    writer.record_observation(0, PHASES.index(task.logic.phase), last_observation)
    display = None
    try:
        if viewer:
            from feedingrobot.sim.observer_viewer import ObserverViewer
            display = ObserverViewer(task.model, task.data)
            status = display.start()
            if status["status"] == "unavailable":
                print(f"Viewer unavailable; continuing headless: {status['reason']}", flush=True)
        while not task.terminated and task.tick * task.dt < env.max_episode_s - 1e-12:
            phase, event_start = task.logic.phase, len(task.logic.events)
            if task.tick % action_ticks == 0:
                command = teacher.act(task.provider.observe()["policy_obs"])
                until = (task.tick + action_ticks) * task.dt
                task.adapter.set_twist(command, task.data.time, until)
                writer.command(task.tick, command, until)
                writer.action(task.tick, PHASES.index(phase), teacher.proposal, command,
                              env.observe_policy(), task.tick + action_ticks)
            state = task.step_physics()
            if phase != task.logic.phase and not task.terminated:
                task.adapter.stop(hold_reference=True)
                writer.command(task.tick, hold_reference=True)
                writer.interrupt(task.tick)
            writer.record_physics(task, state)
            if task.tick % obs_ticks == 0 or task.terminated:
                valid = task.failure_reason != "nonfinite_state"
                if valid:
                    last_observation = env.observe_policy()
                writer.record_observation(task.tick, PHASES.index(task.logic.phase), last_observation, valid=valid)
                if display is not None:
                    previous = display.report.copy()
                    display.update()
                    if display.report != previous and display.report["status"] in ("closed", "unavailable"):
                        print(f"Viewer disabled; continuing headless: {display.report.get('reason')}", flush=True)
            for event in task.logic.events[event_start:]:
                if event["name"] in ("pickup", "delivery", "success"):
                    e = evidence(task)
                    evidence_events.append(dict(event=copy.deepcopy(event), **{key: e[key] for key in
                                           ("supported", "off_plate", "on_plate", "mouth_supported", "released",
                                            "tool_inside", "tool_mouth_contact", "food_position", "tcp_position")}))
        truncated = not task.terminated
        if truncated:
            task.logic.emit("time_limit", task.data.time)
            task.adapter.stop()
            writer.command(task.tick)
        writer.interrupt(task.tick)
        events = copy.deepcopy(task.logic.events)
        visualization = (dict(requested=True, **display.close()) if display is not None
                         else dict(requested=False, status="disabled"))
        result = dict(robot_id=robot, seed=seed, split=split, group_id=group_id or f"{robot}:{split}:{seed}",
                      scenario=scenario or {}, teacher_parameters=teacher_parameters or {}, teacher_config=config,
                      signature=task.state_signature(), input_hashes=input_hashes(), time_s=float(task.data.time),
                      max_episode_s=env.max_episode_s,
                      wall_s=time.monotonic() - start, success=task.logic.success, truncated=truncated,
                      visualization=visualization,
                      terminated=task.terminated,
                      failure_reason=task.failure_reason, phase=task.logic.phase, events=events,
                      evidence=evidence_events, segments=annotate(events),
                      contact_peak_n=task.monitor.peak_n, contact_impulse_ns=task.monitor.impulse_ns,
                      contact_over_limit_s=task.monitor.over_limit_s,
                      wrist_peak_n=float(np.max(writer.physics[:writer.physics_rows, -3])) if writer.physics_rows else 0.,
                      solver_iterations=int(task.model.opt.iterations),
                      accepted_normal=bool(task.logic.success),
                      accepted_recovery=any(s["recovery_valid"] for s in annotate(events)),
                      rejection_reason=None if task.logic.success else task.failure_reason or "time_limit")
        return writer.finish(result)
    finally:
        if display is not None:
            display.close()
        env.close()
