"""UR5e seed-0 substep evidence and same-state local contact experiments."""

import json
import pickle

import mujoco
import numpy as np

from feedingrobot.scripts.validate_m1 import SETTINGS, input_hashes, restore_numerical_state, write_json
from feedingrobot.scripts.validate_m1c import move_pose, sweep_path
from feedingrobot.sim.model import ROOT, load_json
from feedingrobot.sim.task import FeedingTask


OUTPUT = ROOT / 'outputs/calibration/beans_native/m1d/ur5e_seed0_diagnosis'
SOURCE = ROOT / 'outputs/beans_native/v1/m1/ur5e/m1d/baseline/sweep_seed_0_states.pkl'


def task_from(state, setting):
    task = FeedingTask('ur5e', timestep=setting[0])
    task.scene_config['head_fixed'] = True
    task.model.opt.iterations, task.model.opt.tolerance = setting[1:]
    restore_numerical_state(task, state)
    return task


def sample(task, phase):
    state = task.snapshot()
    contacts = {}
    for name in ('applied_contacts', 'contacts'):
        contacts[name] = [dict(row,
            geom1_name=task.model.geom(row['geom1']).name,
            geom2_name=task.model.geom(row['geom2']).name)
            for row in getattr(task, name) if row['bean1_id'] or row['bean2_id']]
    return dict(time_s=state['time'], applied_state_time_s=state['time'] - task.dt,
                phase=phase, **{key: state[key] for key in
                ('bean_positions', 'bean_quaternions', 'bean_linear_velocities_world',
                 'bean_angular_velocities_world', 'tcp_position', 'tcp_rotation')},
                wrist_peak_n=task.substep_wrist_peak_n, **contacts)


def focused_force(row, source='contacts'):
    return sum(c['force_n'] for c in row[source]
               if 'bean_011' in (c['bean1_id'], c['bean2_id']))


def compare(left, right):
    # Compare only exact shared physical times; never interpolate contact forces.
    right = {round(r['time_s'], 9): r for r in right}
    rows = []
    for a in left:
        b = right.get(round(a['time_s'], 9))
        if b is None:
            continue
        rows.append(dict(time_s=a['time_s'], phase=a['phase'],
            position_error_m=float(np.linalg.norm(np.array(a['bean_positions'])[11] - b['bean_positions'][11])),
            velocity_error_m_s=float(np.linalg.norm(np.array(a['bean_linear_velocities_world'])[11] - b['bean_linear_velocities_world'][11])),
            tcp_error_m=float(np.linalg.norm(np.array(a['tcp_position']) - b['tcp_position'])),
            force_error_n=abs(focused_force(a) - focused_force(b))))
    return dict(common_samples=len(rows),
        first_position_crossings={str(t): next((r for r in rows if r['position_error_m'] > t), None)
                                  for t in (1e-8, 1e-6, 1e-5, 1e-4, 1e-3)},
        first_force_crossings={str(t): next((r for r in rows if r['force_error_n'] > t), None)
                               for t in (1e-4, .01, .1)},
        max_position_error_m=max(r['position_error_m'] for r in rows),
        max_tcp_error_m=max(r['tcp_error_m'] for r in rows))


def save_trace(path, rows):
    from feedingrobot.scripts.validate_m1b import serializable
    with path.open('w') as file:
        for row in rows:
            file.write(json.dumps(serializable(row), allow_nan=False) + '\n')


def summarize(rows):
    peak = max(rows, key=focused_force)
    return dict(samples=len(rows), bean011_peak=peak,
                wrist_peak_n=max(r['wrist_peak_n'] for r in rows))


def record(state, name):
    task = task_from(state, SETTINGS[name])
    rows, tape, checkpoints = [], [], {}
    original = task.step_physics

    class Observer:
        phase = 'control'

    observer = Observer()

    def step():
        before = float(task.data.time)
        if name == 'baseline' and round(before, 9) in (10.4, 12.4):
            checkpoints[str(round(before, 1))] = task.get_state()
        result = original()
        tape.append(dict(time_s=before, ctrl=task.data.ctrl.copy(), phase=observer.phase))
        rows.append(sample(task, observer.phase))
        assert not task.terminated and task.adapter.fault is None, task.failure_reason
        return result

    task.step_physics = step
    for phase, position, rotation, speed in sweep_path(task, load_json('configs/acceptance.json')):
        observer.phase = phase
        move_pose(task, observer, position, rotation, load_json('configs/acceptance.json'), speed=speed)
        print(name, phase, task.data.time, flush=True)
        if phase == 'sweep':
            break
    save_trace(OUTPUT / f'{name}_substeps.jsonl', rows)
    if name == 'baseline':
        with (OUTPUT / 'checkpoints_and_drivers.pkl').open('wb') as file:
            pickle.dump(dict(checkpoints=checkpoints, tape=tape), file)
    return rows, tape, checkpoints


def local_replay(state, tape, name, single, duration=1.):
    task = task_from(state, SETTINGS[name])
    initial_matches = np.array_equal(task.get_state()['physics'], state['physics'])
    if single:
        # Remove interference only at the experiment's initial boundary. Preserve
        # bean_011, robot, spoon, bowl, mass, pose, velocity and contact parameters.
        geoms = np.delete(task.index.bean_collision_geoms, 11)
        task.model.geom_contype[geoms] = 0
        task.model.geom_conaffinity[geoms] = 0
        mujoco.mj_forward(task.model, task.data)
    start = float(task.data.time)
    drivers = [r for r in tape if r['time_s'] >= start - 1e-9]
    cursor = 0

    def update(dt):
        nonlocal cursor
        while cursor + 1 < len(drivers) and drivers[cursor + 1]['time_s'] <= task.data.time + 1e-9:
            cursor += 1
        task.data.ctrl[:] = drivers[cursor]['ctrl']

    task.adapter.update = update
    rows = []
    end = min(start + duration, tape[-1]['time_s'] + .001)
    for _ in range(round((end - start) / task.dt)):
        task.step_physics()
        row = sample(task, drivers[cursor]['phase'])
        if single:
            assert all(c['bean1_id'] in (None, 'bean_011') and c['bean2_id'] in (None, 'bean_011')
                       for source in ('contacts', 'applied_contacts') for c in row[source])
        rows.append(row)
        assert not task.terminated, task.failure_reason
    label = f'replay_{start:.1f}_{name}_{"single" if single else "all"}'
    save_trace(OUTPUT / f'{label}.jsonl', rows)
    return rows, dict(initial_integration_matches=initial_matches, **summarize(rows))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes = input_hashes()
    with SOURCE.open('rb') as file:
        state = pickle.load(file)['episode_reset']
    report = dict(source=str(SOURCE), input_hashes=hashes, settings=SETTINGS,
                  status='diagnostic_only', parameter_status='candidate',
                  replay_protocol='Same full state; baseline actuator commands held over each 1 ms interval; existing step_physics owns integration',
                  trajectories={}, comparisons={}, local_replays={})
    baseline, tape, checkpoints = record(state, 'baseline')
    report['trajectories']['baseline'] = summarize(baseline)
    for name in ('dt05ms', 'iterations200'):
        rows, _, _ = record(state, name)
        report['trajectories'][name] = summarize(rows)
        report['comparisons'][name] = compare(baseline, rows)
        write_json(OUTPUT / 'report.json', report)
    for clock, checkpoint in checkpoints.items():
        results = {}
        for single in (False, True):
            reference = None
            for name in SETTINGS:
                rows, detail = local_replay(checkpoint, tape, name, single,
                                           duration=3. if clock == '10.4' else 1.)
                if reference is None:
                    reference = rows
                    if not single:
                        original = [r for r in baseline if float(clock) < r['time_s'] + 1e-9 <= rows[-1]['time_s'] + 1e-9]
                        detail['baseline_reproduction'] = compare(original, rows)
                else:
                    detail['comparison'] = compare(reference, rows)
                results[f'{name}_{"single" if single else "all"}'] = detail
                print('replay', clock, name, single, detail['wrist_peak_n'], flush=True)
        report['local_replays'][clock] = results
        write_json(OUTPUT / 'report.json', report)
    report['inputs_unchanged'] = hashes == input_hashes()
    write_json(OUTPUT / 'report.json', report)


if __name__ == '__main__':
    main()
