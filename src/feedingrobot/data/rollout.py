"""Integer-tick teacher execution using the existing, sole physics owner."""

import copy
import pickle
import time
from pathlib import Path

import numpy as np

from feedingrobot.data.episodes import EpisodeWriter, annotate, input_hashes, recovery_action_mask
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.sim.events import PHASES, evidence


def run_episode(robot, seed, config, directory, *, scenario=None, teacher_parameters=None,
                timestep=.001, iterations=None, solver_tolerance=None, viewer=False, split="calibration", group_id=None,
                max_episode_s=None, initial_state=None):
    scenario = dict(config["scene"], **(scenario or {}))
    env = FeedingGymEnv(robot, timestep=timestep, max_episode_s=max_episode_s)
    task = env.task
    if iterations is not None:
        task.model.opt.iterations = iterations
    if solver_tolerance is not None:
        task.model.opt.tolerance = solver_tolerance
    env.reset(seed=seed, options={"scenario": scenario or {}})
    if initial_state is not None:
        from feedingrobot.scripts.validate_m1 import restore_numerical_state
        restore_numerical_state(task, initial_state)
        env.last_observation = env.observe_policy()
    teacher = Teacher(task.robot_config, config)
    geometry = teacher_geometry(task)
    teacher.reset(teacher_parameters or {}, geometry=geometry)
    frozen_hashes = input_hashes()
    action_ticks, obs_ticks = round(.05 / task.dt), round(.02 / task.dt)
    if not np.isclose(action_ticks * task.dt, .05) or not np.isclose(obs_ticks * task.dt, .02):
        env.close()
        raise ValueError("Physics dt must divide both observation and action periods")
    writer = EpisodeWriter(directory, env.schema, task.dt, env.max_episode_s)
    Path(directory, "initial_state.pkl").write_bytes(pickle.dumps(task.get_state(), protocol=5))
    start = time.monotonic()
    evidence_events = []
    action_starts, tcp_samples, event_positions = [], [], []
    last_stage = None
    def comparison_position():
        tcp = task.data.site_xpos[task.index.tcp].copy()
        frame = ("world" if not teacher.pickup_lift_complete else "receiver"
                 if teacher.stage.startswith("release_") else "mouth"
                 if teacher.stage in ("entry", "retract", "recover", "wait_level") else "world")
        if frame == "world":
            return frame, tcp
        site = task.model.site("mouth_receiver" if frame == "receiver" else "mouth_entry").id
        return frame, (tcp-task.data.site_xpos[site]) @ task.data.site_xmat[site].reshape(3, 3)
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
                if teacher.stage != last_stage:
                    action_starts.append(dict(stage=teacher.stage, time=float(task.data.time)))
                    last_stage = teacher.stage
                if teacher.stop_requested:
                    task.adapter.stop(hold_reference=True)
                    writer.command(task.tick, hold_reference=True)
                until = (task.tick + action_ticks) * task.dt
                task.adapter.set_twist(command, task.data.time, until)
                writer.command(task.tick, command, until)
                writer.action(task.tick, PHASES.index(phase), teacher.proposal, command,
                              env.observe_policy(), task.tick + action_ticks)
            state = task.step_physics()
            if phase != task.logic.phase and not task.terminated:
                task.adapter.stop(hold_reference=True)
                writer.command(task.tick, hold_reference=True, after_physics=True)
                writer.interrupt(task.tick)
            writer.record_physics(task, state)
            if task.tick % obs_ticks == 0:
                frame, position = comparison_position()
                tcp_samples.append(dict(time=float(task.data.time), stage=teacher.stage, frame=frame,
                                        position=task.data.site_xpos[task.index.tcp].copy(), comparison_position=position))
            if task.tick % obs_ticks == 0 or task.terminated:
                valid = task.failure_reason != "nonfinite_state"
                if valid:
                    last_observation = env.observe_policy()
                writer.record_observation(task.tick, PHASES.index(task.logic.phase), last_observation, valid=valid)
                if display is not None:
                    previous = display.report.copy()
                    display.update(phase=task.logic.phase, result=task.failure_reason or ("success" if task.logic.success else "running"))
                    if display.report != previous and display.report["status"] in ("closed", "unavailable"):
                        print(f"Viewer disabled; continuing headless: {display.report.get('reason')}", flush=True)
            for event in task.logic.events[event_start:]:
                event_positions.append(comparison_position()[1])
                if event["name"] in ("pickup", "delivery", "success"):
                    e = evidence(task)
                    evidence_events.append(dict(event=copy.deepcopy(event), **{key: e[key] for key in
                                           ("supported", "off_bowl", "on_bowl", "mouth_supported", "released",
                                            "tool_inside", "tool_mouth_contact", "bean_position", "tcp_position")}))
        truncated = not task.terminated
        if truncated:
            task.logic.emit("time_limit", task.data.time)
            event_positions.append(comparison_position()[1])
            task.adapter.stop()
            writer.command(task.tick)
        writer.interrupt(task.tick)
        events = copy.deepcopy(task.logic.events)
        visualization = (dict(requested=True, **display.close()) if display is not None
                         else dict(requested=False, status="disabled"))
        if input_hashes() != frozen_hashes:
            raise ValueError("Runtime inputs changed during the episode; recording cannot be accepted")
        final_frame, final_position = comparison_position()
        comparison = dict(action_starts=action_starts, tcp_samples=tcp_samples,
                          tcp_comparison_frame=final_frame, tcp_comparison_position=final_position,
                          event_tcp_comparison_positions=event_positions)
        segments = annotate(events)
        recovery_rows = int(np.count_nonzero(recovery_action_mask(segments,
            [a['phase'] for a in writer.actions], [a['tick'] for a in writer.actions],
            [a['end_tick'] for a in writer.actions], [a['valid'] for a in writer.actions], task.dt)))
        result = dict(recovery_action_rows=recovery_rows, comparison=comparison, teacher_stage=teacher.stage,
                      pickup_settled=teacher.pickup_lift_complete, robot_id=robot, seed=seed, split=split, group_id=group_id or f"{robot}:{split}:{seed}",
                      scenario=scenario or {}, teacher_parameters=teacher_parameters or {}, teacher_config=config,
                      teacher_geometry_sha256=geometry["sha256"],
                      signature=task.state_signature(), input_hashes=frozen_hashes, time_s=float(task.data.time),
                      max_episode_s=env.max_episode_s,
                      wall_s=time.monotonic() - start, success=task.logic.success, truncated=truncated,
                      visualization=visualization,
                      terminated=task.terminated,
                      failure_reason=task.failure_reason, phase=task.logic.phase, events=events,
                      evidence=evidence_events, segments=segments,
                      contact_peak_n=task.monitor.peak_n, contact_impulse_ns=task.monitor.impulse_ns,
                      contact_group_peaks_n=copy.deepcopy(task.monitor.pair_peaks),
                      contact_group_impulses_ns=copy.deepcopy(task.monitor.pair_impulses),
                      contact_over_limit_s=task.monitor.over_limit_s,
                      wrist_peak_n=float(np.max(writer.physics[:writer.physics_rows, -3])) if writer.physics_rows else 0.,
                      solver_iterations=int(task.model.opt.iterations),
                      solver_tolerance=float(task.model.opt.tolerance),
                      accepted_normal=bool(task.logic.success),
                      accepted_recovery=bool(recovery_rows),
                      rejection_reason=None if task.logic.success else task.failure_reason or "time_limit")
        return writer.finish(result)
    finally:
        if display is not None:
            display.close()
        env.close()
