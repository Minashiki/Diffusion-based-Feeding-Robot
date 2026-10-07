"""Independent copy of the original evaluator for the 1/4-step cadence comparison.

Physics, guards, phase cancellation, features, and chunk semantics remain owned
by the original modules. Changes: replan period, tick-paired noise, extra logs.
The sequential-noise 4-step path is regression-checked against run_policy.
"""

import pickle
from pathlib import Path
import time

import numpy as np
import torch

from feedingrobot.envs import FeedingGymEnv
from feedingrobot.policies.audit import read_json
from feedingrobot.policies.data import features, field_slices
from feedingrobot.policies.dit import sample_actions
from feedingrobot.policies.evaluation import ActionChunk
from feedingrobot.policies.runtime import precision


def noise_seed_at_tick(base_seed, tick):
    # A common physical tick receives identical initial noise in both arms.
    return base_seed * 1000003 + tick // 50


def run_prefix_policy(model, config, normalization, episode, device, *, execute_steps=4, paired_noise=True, viewer=False, max_ticks=None):
    if execute_steps not in (1, 4):
        raise ValueError("This comparison supports only 1 or 4 executed steps")
    episode = Path(episode)
    metadata = read_json(episode / 'manifest.json')
    env = FeedingGymEnv('panda')
    display = None
    traces, inference_times, observation_ticks, observation_values = [], [], [], []
    plans = []
    chunk, last_plan = None, -200
    wrist_peak = 0.
    try:
        env.reset(seed=metadata['seed'], options={'scenario': metadata['scenario']})
        # M4 snapshots are trusted local artifacts already hash-checked by parent_check.
        env.task.set_state(pickle.loads((episode / 'initial_state.pkl').read_bytes()))
        task = env.task
        if metadata['signature'] != task.state_signature():
            raise ValueError('Closed-loop environment differs from the frozen dataset')
        if viewer:
            from feedingrobot.sim.observer_viewer import ObserverViewer
            display = ObserverViewer(task.model, task.data)
            display.start()
        fields = field_slices(env.schema)
        generator = torch.Generator(device=device).manual_seed(config['training']['seed'])
        start = time.perf_counter()
        limit = round(env.max_episode_s / task.dt) if max_ticks is None else min(max_ticks, round(env.max_episode_s / task.dt))
        while not task.terminated and task.tick < limit:
            phase = task.logic.phase
            if task.tick % 20 == 0 or task.tick % 50 == 0:
                observation_ticks.append(task.tick)
                observation_values.append(env.observe_policy())
                if len(observation_ticks) > 32:
                    observation_ticks.pop(0)
                    observation_values.pop(0)
            if task.tick % 50 == 0 and phase != 'SELECT':
                if chunk is None or task.tick - last_plan >= 50 * execute_steps:
                    x = features(np.array(observation_ticks), np.array(observation_values),
                                 np.ones(len(observation_ticks), bool), task.tick, fields, normalization)
                    batch = {k: torch.as_tensor(v, device=device)[None] for k, v in x.items()}
                    if paired_noise:
                        generator.manual_seed(noise_seed_at_tick(config['training']['seed'], task.tick))
                    plans.append(dict(tick=task.tick, phase=phase,
                                      noise_seed=noise_seed_at_tick(config['training']['seed'], task.tick) if paired_noise else None,
                                      condition={k: np.asarray(v).tolist() for k, v in x.items()}))
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    before = time.perf_counter()
                    with precision(device):
                        actions = sample_actions(model, config, batch, normalization, generator)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    inference_times.append(time.perf_counter() - before)
                    chunk = ActionChunk(task.tick, phase, actions[0].float().cpu().numpy())
                    last_plan = task.tick
                action = chunk.current(task.tick, phase)
                if action is None or not np.isfinite(action).all():
                    task._terminate('invalid_command')
                    break
                from feedingrobot.control.adapter import clip_norm
                command = np.r_[clip_norm(action[:3], task.robot_config['linear_speed_limit']),
                                clip_norm(action[3:], task.robot_config['angular_speed_limit'])]
                task.adapter.set_twist(command, task.data.time, (task.tick + 50) * task.dt)
                traces.append(dict(tick=task.tick, phase=phase, queue_age_ticks=task.tick - chunk.issued_tick,
                                   predicted=action.tolist(), command=command.tolist(), observation=observation_values[-1].tolist()))
            task.step_physics()
            wrist_peak = max(wrist_peak, task.substep_wrist_peak_n)
            if task.logic.phase != phase and not task.terminated:
                task.adapter.stop(hold_reference=True)
                chunk = None
            if traces and task.tick % 50 == 0:
                state = task.snapshot()
                traces[-1].update(shaped=task.adapter.velocity.tolist(), measured=state['tcp_twist_world'].tolist())
            if display is not None and task.tick % 20 == 0:
                display.update(phase=task.logic.phase, result=task.failure_reason or 'running')
        if not task.terminated:
            task.adapter.stop(hold_reference=True)
        events = task.logic.events
        entered = {'SELECT'} | {e['phase'] for e in events if e['name'] == 'phase'}
        recovered = any(e['name'] == 'phase' and e.get('previous') == 'RECOVER'
                        and e['phase'] == 'WAIT_READY' for e in events)
        recover_case = bool(metadata['scenario'].get('recover', False))
        success = bool(task.logic.success and (recovered or not recover_case))
        names = {e['name'] for e in events}
        steps = np.array([t['command'] for t in traces])
        differences = np.diff(steps, axis=0) if len(steps) > 1 else np.zeros((1, 6))
        return dict(seed=metadata['seed'], source_episode=str(episode), recover=recover_case, success=success,
                    physical_success=bool(task.logic.success), recovery_completed=recovered,
                    failure_reason=task.failure_reason or (None if success else 'time_limit' if task.tick >= round(env.max_episode_s/task.dt)
                                                          else 'diagnostic_prefix' if max_ticks is not None else 'missing_recovery'),
                    entered=sorted(entered), pickup='pickup' in names, delivery='delivery' in names,
                    events=events, simulated_s=task.tick * task.dt, wall_s=time.perf_counter()-start,
                    execute_steps=execute_steps, replan_ticks=50 * execute_steps, paired_noise=paired_noise, plans=plans,
                    failure_contacts=[dict(group1=c['group1'], group2=c['group2'], force_n=c['force_n'], distance=c['distance'])
                                      for c in sorted(task.contacts + task.applied_contacts, key=lambda c: c['force_n'], reverse=True)[:10]],
                    contact_peak_n=task.monitor.peak_n, wrist_peak_n=wrist_peak,
                    contact_impulse_ns=task.monitor.impulse_ns,
                    linear_command_jump_max=float(np.linalg.norm(differences[:, :3], axis=1).max()),
                    angular_command_jump_max=float(np.linalg.norm(differences[:, 3:], axis=1).max()),
                    inference_s=inference_times, trace=traces,
                    visualization=display.close() if display is not None else dict(status='disabled'))
    finally:
        if display is not None:
            display.close()
        env.close()
