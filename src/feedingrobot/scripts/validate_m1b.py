"""M1-B native contact/reset acceptance; M1-C/D remain unverified."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import mujoco
import numpy as np

from feedingrobot.sim.beans import bean_diagnostics
from feedingrobot.sim.contacts import read_contacts
from feedingrobot.sim.model import ROOT, asset_files, load_json
from feedingrobot.sim.task import FeedingTask
from feedingrobot.scripts.validate_m1a import beans_check, integration_state, validate as validate_m1a


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def parameters_match(rows, expected):
    friction = np.array(expected['friction'])[[0, 0, 1, 2, 2]]
    active = expected['condim'] - 1
    return bool(rows) and all(r['condim'] == expected['condim']
                             and np.allclose(r['friction'][:active], friction[:active])
                             and np.allclose(r['solref'], expected['solref'])
                             and np.allclose(r['solimp'], expected['solimp'])
                             and np.isclose(r['includemargin'],
                                            sum(bool(r.get(k)) for k in ('bean1_id', 'bean2_id'))
                                            * (expected.get('margin', 0) - expected.get('gap', 0)))
                             for r in rows)


def replay_initial(task, initial, preset, seed):
    task.reset(preset='empty')
    task.seed = seed
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_resetData(task.model, task.data)
    mujoco.mj_setState(task.model, task.data, np.array(initial), kind)
    mujoco.mj_forward(task.model, task.data)
    mujoco.mj_setState(task.model, task.data, np.array(initial), kind)
    task.tick = round(task.data.time / task.dt)
    task.contacts = read_contacts(task.model, task.data, task.index)
    task.applied_contacts = []
    task.physics_timing = dict(mj_step_s=0., forward_s=0., steps=0)
    task.reset_diagnostics = dict(preset=preset, seed=seed, status='passed')


def contact_case(task, kind, initial_state=None):
    if initial_state is None:
        task.reset(preset='empty')
    else:
        replay_initial(task, initial_state, 'beans_on_spoon' if kind == 'spoon' else kind, 0)
    m, d, idx = task.model, task.data, task.index
    origin = np.array(task.scene_config['bowl_frame_position_m'])
    if kind == 'spoon':
        if initial_state is None:
            task.reset(preset='beans_on_spoon')
        else:
            task._settle_beans('beans_on_spoon')
        # Contact evidence and full reset stability are separate mandatory cases.
        reset_result = dict(task.reset_diagnostics)
        task.spoon_reset_result = reset_result
        rows = [r for r in task.contacts if (r.get('bean1_id') == 'bean_000' or r.get('bean2_id') == 'bean_000')
                and (r['geom1'] in idx.scoop_geoms or r['geom2'] in idx.scoop_geoms)]
        parameters = parameters_match(rows, task.scene_config['beans'])
        supported = bool(bean_diagnostics(task)['spoon_supported'][0])
        return dict(status='passed' if rows and supported and parameters else 'failed',
                    reset_status=task.reset_diagnostics['status'], contacts=rows,
                    real_required_contact=bool(rows), support_verified=supported, effective_parameters_match=parameters,
                    final=bean_diagnostics(task))
    # Isolated fixture placement occurs during reset setup, before episode advancement.
    count = 2 if kind == 'double' else 1
    if initial_state is None:
        for i in range(count):
            d.qpos[idx.bean_qpos[i]] = np.r_[origin + [0, 0, .0075 + i * .010], [1, 0, 0, 0]]
            d.qvel[idx.bean_dofs[i]] = 0
        mujoco.mj_forward(m, d)
    task.contacts = read_contacts(m, d, idx)
    task.applied_contacts = []
    initial = integration_state(m, d)
    evidence = []
    maximum = 0.
    for _ in range(round(1. / task.dt)):
        task.step_physics(_settling=True)
        diag = bean_diagnostics(task)
        maximum = max(maximum, diag['max_penetration_m'])
        for row in task.applied_contacts + task.contacts:
            pair = {row['group1'], row['group2']}
            if pair == {'bowl', 'food'} or (kind == 'double' and pair == {'food'}):
                if not any((e['geom1'], e['geom2']) == (row['geom1'], row['geom2']) for e in evidence):
                    evidence.append(dict(time_s=task.data.time, **row))
        if task.terminated:
            break
    required = any({r['group1'], r['group2']} == {'bowl', 'food'} for r in evidence)
    if kind == 'double':
        required &= any(r.get('bean1_id') == 'bean_000' and r.get('bean2_id') == 'bean_001'
                        or r.get('bean2_id') == 'bean_000' and r.get('bean1_id') == 'bean_001' for r in evidence)
    expected = task.scene_config['beans']
    parameters = parameters_match(evidence, expected)
    passed = required and parameters and diag['bowl_supported'][:count].all() and diag['in_bowl'][:count].all() and not task.terminated
    passed &= maximum <= task.bean_acceptance['penetration_limit_m'] and not any(w.number for w in d.warning)
    return dict(status='passed' if passed else 'failed', initial_state=initial,
                final_state=integration_state(m, d), contacts=evidence, effective_parameters_match=parameters,
                real_required_contact=bool(required), support_verified=bool(diag['bowl_supported'][:count].all()),
                max_penetration_m=maximum, final=diag, warning_counts=[w.number for w in d.warning])


def viewer_child(robot, output):
    from feedingrobot.sim.model import load_model
    from feedingrobot.sim.viewer import passive_viewer
    m, idx, cfg, _ = load_model(robot)
    d = mujoco.MjData(m)
    d.qpos[idx.qpos] = cfg['reset_q']
    mujoco.mj_forward(m, d)
    mujoco.mj_setState(m, d, np.array(json.loads((Path(output) / 'viewer_state.json').read_text())), mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)
    with passive_viewer(m, d) as viewer:
        for visual, collision in [(1, 0), (0, 1), (1, 1)]:
            viewer.opt.geomgroup[1] = visual
            viewer.opt.geomgroup[3] = collision
            viewer.sync()
            time.sleep(.1)
            assert viewer.is_running()
    print('VIEWER_OK')


def visual_check(task, output):
    state_file = output / 'viewer_state.json'
    state_file.write_text(json.dumps(integration_state(task.model, task.data)))
    result = subprocess.run([sys.executable, '-m', 'feedingrobot.scripts.validate_m1b', '--robot', task.robot_id,
                             '--output', str(output), '--viewer-child'],
                            cwd=ROOT, text=True, capture_output=True, timeout=30)
    result_render = subprocess.run([sys.executable, '-m', 'feedingrobot.scripts.validate_m1b', '--robot', task.robot_id,
                                   '--render-child', str(output)], cwd=ROOT, text=True,
                                  capture_output=True, timeout=30, env={**os.environ, 'MUJOCO_GL': 'egl'})
    images = [str((output / f'{mode}.png').relative_to(ROOT)) if output.is_relative_to(ROOT)
              else str(output / f'{mode}.png') for mode in ('visual', 'collision', 'overlay')]
    aligned = beans_check(task.model, task.data, task.index, task.scene_config)
    passed = result.returncode == 0 and 'VIEWER_OK' in result.stdout and result_render.returncode == 0
    return dict(status='passed' if passed else 'failed', viewer_open_sync_close= result.returncode == 0,
                viewer_stderr=result.stderr, render_stderr=result_render.stderr, images=images,
                alignment=aligned)


def render_child(robot, output):
    from PIL import Image
    from feedingrobot.sim.model import load_model
    output = Path(output)
    m, idx, _, _ = load_model(robot)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, np.array(json.loads((output / 'viewer_state.json').read_text())), mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.45, -.18, .01]
    camera.distance, camera.azimuth, camera.elevation = .25, 140., -55.
    option = mujoco.MjvOption()
    with mujoco.Renderer(m, 480, 640) as renderer:
        for mode, visual, collision in [('visual', 1, 0), ('collision', 0, 1), ('overlay', 1, 1)]:
            option.geomgroup[1] = visual
            option.geomgroup[3] = collision
            renderer.update_scene(d, camera=camera, scene_option=option)
            closeup = Image.fromarray(renderer.render())
            renderer.update_scene(d, camera='overview', scene_option=option)
            overview = Image.fromarray(renderer.render())
            tool_camera = mujoco.MjvCamera()
            tool_camera.lookat[:] = d.site_xpos[idx.tcp]
            tool_camera.distance, tool_camera.azimuth, tool_camera.elevation = .16, 140., -35.
            renderer.update_scene(d, camera=tool_camera, scene_option=option)
            tool = Image.fromarray(renderer.render())
            combined = Image.new('RGB', (1920, 480))
            combined.paste(closeup, (0, 0))
            combined.paste(overview, (640, 0))
            combined.paste(tool, (1280, 0))
            combined.save(output / f'{mode}.png')


def numerical_check(robot, baseline):
    variants = {}
    for name, dt, iterations, tolerance in [('dt05ms', .0005, 100, 1e-8),
                                          ('iterations200', .001, 200, 1e-10)]:
        task = FeedingTask(robot, timestep=dt)
        task.model.opt.iterations = iterations
        task.model.opt.tolerance = tolerance
        cases = {}
        for fixture in ('single', 'spoon'):
            reference = baseline['cases']['beans_on_spoon_reset']['metrics'] if fixture == 'spoon' else baseline['cases'][fixture]
            initial = reference['initial_physics_state' if fixture == 'spoon' else 'initial_state']
            case = contact_case(task, fixture, initial_state=initial)
            actual_initial = task.spoon_reset_result['initial_physics_state'] if fixture == 'spoon' else case['initial_state']
            case['initial_state_matches'] = bool(np.array_equal(initial, actual_initial))
            if fixture == 'spoon':
                case['reset_metrics'] = task.spoon_reset_result
                if task.spoon_reset_result['status'] != 'passed':
                    case['status'] = 'failed'
            if not case['initial_state_matches']:
                case['status'] = 'failed'
            cases[fixture] = case
        for seed in baseline['acceptance']['seeds']:
            case_name = f'beans_in_bowl_seed_{seed}'
            reference = baseline['cases'][case_name]['metrics']
            replay_initial(task, reference['initial_physics_state'], 'beans_in_bowl', seed)
            task._settle_beans('beans_in_bowl')
            metrics = task.reset_diagnostics
            matches = bool(np.array_equal(reference['initial_physics_state'], metrics['initial_physics_state']))
            base_positions = np.array([r['bean_positions'] for r in reference['trajectory']])
            positions = np.array([r['bean_positions'] for r in metrics['trajectory']])
            common = min(len(positions), len(base_positions))
            cases[case_name] = dict(status=metrics['status'] if matches else 'failed', metrics=metrics,
                                   initial_state_matches=matches,
                                   position_divergence_m=np.max(np.linalg.norm(positions[:common] - base_positions[:common], axis=2), axis=0),
                                   settle_time_difference_s=metrics['bean_settle_s'] - reference['bean_settle_s'])
            print(robot, name, case_name, cases[case_name]['status'], flush=True)
        variants[name] = dict(status='passed' if all(c['status'] == 'passed' for c in cases.values()) else 'failed',
                              solver=dict(timestep=dt, iterations=iterations, tolerance=tolerance), cases=cases)
    return dict(status='passed' if all(v['status'] == 'passed' for v in variants.values()) else 'failed',
                scope='M1-B contact/settling only; M1-C/D motion, carry/drop and force/impulse convergence remain unverified',
                trace_interval_s=.01, variants=variants)


def validate(robot, output, *, numerical=False):
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_json('configs/acceptance.json')['beans_native']
    report = dict(schema_version=1, model_version='single_bean_native_v1', stage='M1-B', robot_id=robot,
                  status='failed', m1_status='incomplete', mujoco_version=mujoco.__version__,
                  acceptance=cfg, stages={'M1-A': 'failed', 'M1-B': 'failed', 'M1-C': 'not_verified', 'M1-D': 'not_verified'}, cases={})
    inputs = asset_files(robot) + list((ROOT / 'src/feedingrobot').rglob('*.py'))
    inputs += list((ROOT / 'tests').glob('test_beans_native*.py'))
    inputs += [ROOT / 'tests/test_contracts.py', ROOT / 'tests/test_guard_physics.py', ROOT / 'tests/test_tableware.py',
               ROOT / 'third_party_manifest.json', ROOT / 'requirements.lock.txt']
    report['input_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    prerequisite = validate_m1a(robot)
    report['stages']['M1-A'] = prerequisite['status']
    report['cases']['m1a_prerequisite'] = dict(status=prerequisite['status'], cases=prerequisite['cases'])
    started = time.monotonic()
    task = FeedingTask(robot)
    report['load_and_initial_reset_wall_s'] = time.monotonic() - started
    report['model_load_wall_s'] = task.model_load_wall_s
    report['solver'] = prerequisite.get('solver')
    report['bean_index'] = prerequisite.get('bean_index')
    report['contact_parameters'] = task.scene_config['beans']
    report['excluded_cases'] = {'double': 'not_applicable: single-bean feeding prototype'}
    report['reset_layout'] = 'fixed position/quaternion; seeds test repetition, not layout coverage'
    for name in ('single', 'spoon'):
        try:
            report['cases'][name] = contact_case(task, name)
        except Exception:
            report['cases'][name] = dict(status='failed', error=traceback.format_exc())
        print(robot, name, report['cases'][name]['status'], flush=True)
    report['cases']['beans_on_spoon_reset'] = dict(status=getattr(task, 'spoon_reset_result', {}).get('status', 'failed'),
                                                   metrics=getattr(task, 'spoon_reset_result', {}))
    task.reset(preset='empty')
    report['cases']['empty_reset'] = dict(status='passed' if not task.terminated and task.data.time == 0
                                        and task.adapter.command is None and task.adapter.fault is None else 'failed',
                                        metrics=task.reset_diagnostics)
    for seed in cfg['seeds']:
        name = f'beans_in_bowl_seed_{seed}'
        try:
            task.reset(seed=seed)
            report['cases'][name] = dict(status=task.reset_diagnostics['status'], metrics=task.reset_diagnostics,
                                        final_state=integration_state(task.model, task.data))
        except Exception:
            report['cases'][name] = dict(status='failed', error=traceback.format_exc())
        print(robot, name, report['cases'][name]['status'], flush=True)
        (output / 'm1b_report.json').write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + '\n')
    try:
        report['cases']['viewer'] = visual_check(task, output)
    except Exception:
        report['cases']['viewer'] = dict(status='failed', error=traceback.format_exc())
    if numerical:
        try:
            report['cases']['numerical_check'] = numerical_check(robot, report)
        except Exception:
            report['cases']['numerical_check'] = dict(status='failed', error=traceback.format_exc())
    if all(c['status'] == 'passed' for c in report['cases'].values()):
        report['status'] = report['stages']['M1-B'] = 'passed'
    report['performance_scope'] = 'headless reset including diagnostics; full IK/control loop RTF deferred to M1-C/D'
    (output / 'm1b_report.json').write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=['panda', 'ur5e'], default='panda')
    parser.add_argument('--output')
    parser.add_argument('--numerical-check', action='store_true', help='Replay M1-B fixtures at 0.5 ms and 200 iterations')
    parser.add_argument('--viewer-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--render-child', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.viewer_child:
        viewer_child(args.robot, args.output)
        return
    if args.render_child:
        render_child(args.robot, args.render_child)
        return
    output = ROOT / (args.output or f'outputs/single_bean/v1/m1/{args.robot}')
    try:
        report = validate(args.robot, output, numerical=args.numerical_check)
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        report = dict(stage='M1-B', status='failed', m1_status='incomplete', error=traceback.format_exc())
        (output / 'm1b_report.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
