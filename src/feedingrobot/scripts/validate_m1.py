"""Complete native Beans M1 acceptance and joint Panda/UR5e freeze."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback

import mujoco
import numpy as np

from feedingrobot.sim.model import ROOT, asset_files, load_json
from feedingrobot.sim.task import FeedingTask
from feedingrobot.scripts import validate_m1a as a, validate_m1b as b, validate_m1c as c

SETTINGS = {'baseline': (.001, 100, 1e-8), 'dt05ms': (.0005, 100, 1e-8),
            'iterations200': (.001, 200, 1e-10)}
CONTROL_CASES = ('hold', 'tracking', 'wrench', 'faults', 'reset', 'head', 'carry',
                 'tilt', 'acceleration', 'reachability', 'guards',
                 'sweep_seed_0', 'sweep_seed_1', 'sweep_seed_2')
MOTION_CASES = tuple(n for n in CONTROL_CASES if n not in ('reset', 'guards'))
REQUIRED = CONTROL_CASES + ('assembly', 'contacts', 'convergence', 'viewer', 'performance', 'manifest')


def write_json(path, value):
    path.write_text(json.dumps(b.serializable(value), indent=2, allow_nan=False) + '\n')


def input_hashes():
    # Both robot assets/configs belong to a single shared version. Outputs and
    # the freeze manifest are deliberately excluded to avoid recursive hashes.
    paths = set(asset_files())
    for folder, pattern in [('src/feedingrobot', '*.py'), ('tests', '*.py'), ('docs', '*')]:
        paths.update(p for p in (ROOT / folder).rglob(pattern) if p.is_file())
    paths.update(ROOT / name for name in ('README.md', 'SimModelPlann.md', 'assets/task/tableware/README.md',
                 'requirements.txt', 'requirements.lock.txt', 'pyproject.toml', 'third_party_manifest.json'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def restore_numerical_state(task, state):
    """Acceptance-only replay; public set_state retains strict compatibility."""
    if state is None:
        raise ValueError('Missing complete episode reset state')
    opt = task.model.opt
    saved = (opt.timestep, opt.iterations, opt.tolerance)
    try:
        opt.timestep, opt.iterations, opt.tolerance = SETTINGS['baseline']
        # The serialized compiled model verifies structure, assets and all other
        # numeric/control parameters. Only these three solver settings may vary.
        if task.state_signature() != state['signature']:
            raise ValueError('Numerical replay has incompatible non-numerical inputs')
        task.set_state(state)
    finally:
        opt.timestep, opt.iterations, opt.tolerance = saved
    task.tick = round(task.data.time / task.dt)
    task.physics_timing = dict(mj_step_s=0., forward_s=0., steps=0)


def compare_motion(base, other, left, right, cfg):
    if base['status'] != 'passed' or other['status'] != 'passed':
        raise AssertionError('Each numerical setting must pass the original case')
    def samples(rows):
        times = np.array([row['time_s'] for row in rows])
        if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise AssertionError('Missing, nonfinite or duplicate trajectory samples')
        grid = np.rint(times / .01).astype(int)
        if not np.allclose(times, grid * .01, atol=1e-8) or np.any(np.diff(grid) != 1) or grid[0] != 1:
            raise AssertionError('Missing common-grid trajectory samples')
        for row in rows:
            for key in ('tcp_position', 'tcp_rotation', 'bean_positions', 'compensated_wrench'):
                if not np.isfinite(row[key]).all():
                    raise AssertionError(f'Nonfinite trajectory {key}')
        return rows
    left, right = samples(left), samples(right)
    for result in (base, other):
        for key in ('final_tcp_position', 'final_tcp_rotation', 'max_wrist_force_n'):
            assert np.isfinite(result[key]).all(), f'Nonfinite terminal metric {key}'
    count = min(len(left), len(right))
    positions = [np.linalg.norm(np.array(x['tcp_position']) - y['tcp_position']) for x,y in zip(left[:count],right[:count])]
    rotations = [c.angle_error(np.array(x['tcp_rotation']), np.array(y['tcp_rotation'])) for x,y in zip(left[:count],right[:count])]
    positions.append(np.linalg.norm(np.array(base['final_tcp_position']) - other['final_tcp_position']))
    rotations.append(c.angle_error(np.array(base['final_tcp_rotation']), np.array(other['final_tcp_rotation'])))
    metrics = dict(common_samples=count, max_tcp_position_error_m=float(max(positions)),
                   max_tcp_rotation_error_rad=float(max(rotations)),
                   duration_difference_s=right[-1]['time_s']-left[-1]['time_s'],
                   bean_position_divergence_m=np.max([np.linalg.norm(np.array(x['bean_positions'])-y['bean_positions'],axis=1)
                       for x,y in zip(left[:count],right[:count])], axis=0))
    assert metrics['max_tcp_position_error_m'] <= cfg['convergence_position_m'], metrics
    assert metrics['max_tcp_rotation_error_rad'] <= cfg['convergence_rotation_rad'], metrics
    def difference(x, y, absolute, relative, label):
        assert np.isfinite([x,y]).all(), f'Nonfinite {label}'
        delta, limit = abs(x-y), max(absolute, abs(x)*relative)
        assert delta <= limit, f'{label}: difference={delta}, tolerance={limit}, baseline={x}, variant={y}'
        return dict(difference=delta, tolerance=limit)
    metrics['wrist_peak'] = difference(base['max_wrist_force_n'], other['max_wrist_force_n'],
        cfg['convergence_force_absolute_n'], cfg['convergence_force_relative'], 'wrist_peak_n')
    for field, absolute, relative in [('semantic_pair_peaks_n', 'convergence_force_absolute_n', 'convergence_force_relative'),
                                     ('semantic_pair_impulses_ns', 'convergence_impulse_absolute_ns', 'convergence_impulse_relative')]:
        metrics[field] = {key: difference(base[field].get(key,0.),other[field].get(key,0.),cfg[absolute],cfg[relative], f'{field}:{key}')
                         for key in base[field].keys() | other[field].keys()}
    bm, om = base['metrics'], other['metrics']
    if 'picked_ids' in bm:
        metrics.update(baseline_picked_ids=bm['picked_ids'], picked_ids=om['picked_ids'],
                       baseline_phases=[r['phase'] for r in bm['achieved']], phases=[r['phase'] for r in om['achieved']])
    if 'dropped' in bm:
        assert bm['bean_id'] == om['bean_id'] == 'bean_000' and bm['dropped'] and om['dropped']
        metrics['drop_time_difference_s'] = om['time_s'] - bm['time_s']
    return metrics


def convergence(robot, output, baseline):
    cfg = load_json('configs/acceptance.json')
    variants = {}
    for name, setting in list(SETTINGS.items())[1:]:
        folder = output / name
        report = c.validate(robot, folder, MOTION_CASES, numerical=setting, replay=output/'baseline')
        comparisons = {}
        for case in MOTION_CASES:
            left = right = matches = None
            try:
                left = json.loads((output/'baseline'/f'{case}_trajectory.json').read_text())
                right = json.loads((folder/f'{case}_trajectory.json').read_text())
                with (output/'baseline'/f'{case}_states.pkl').open('rb') as file:
                    reference = pickle.load(file)['episode_reset']
                with (folder/f'{case}_states.pkl').open('rb') as file:
                    restored = pickle.load(file)['episode_reset']
                assert reference is not None and restored is not None, 'Missing complete reset snapshot'
                matches = dict(integration=np.array_equal(reference['physics'], restored['physics']),
                    boundary=all(np.array_equal(reference['boundary'][key], restored['boundary'][key])
                                 for key in ('qacc', 'sensordata')),
                    adapter=all(np.array_equal(reference['adapter'][key], restored['adapter'][key])
                                for key in ('reference_q', 'target', 'velocity', 'last_ik_velocity')),
                    monitor=reference['monitor']==restored['monitor'],
                    driver_clock=reference['task']['scenario_state']==restored['task']['scenario_state'])
                assert all(matches.values()), f'Initial replay mismatch: {matches}'
                comparisons[case] = dict(status='passed', initial_state_matches=matches, metrics=compare_motion(
                    baseline['cases'][case], report['cases'][case], left, right, cfg))
            except Exception:
                comparisons[case] = dict(status='failed', error=traceback.format_exc(), initial_state_matches=matches,
                    baseline_statistics={key: baseline['cases'][case].get(key) for key in
                        ('metrics', 'max_wrist_force_n', 'semantic_pair_peaks_n', 'semantic_pair_impulses_ns',
                         'final_tcp_position', 'final_tcp_rotation')},
                    variant_statistics={key: report['cases'][case].get(key) for key in
                        ('metrics', 'max_wrist_force_n', 'semantic_pair_peaks_n', 'semantic_pair_impulses_ns',
                         'final_tcp_position', 'final_tcp_rotation')})
            bm = baseline['cases'][case].get('metrics', {})
            om = report['cases'][case].get('metrics', {})
            if bm.get('dropped') and om.get('dropped'):
                comparisons[case]['drop_time_difference_s'] = om['time_s'] - bm['time_s']
            if 'picked_ids' in bm and 'picked_ids' in om:
                comparisons[case].update(baseline_picked_ids=bm['picked_ids'], picked_ids=om['picked_ids'],
                    baseline_phases=[row['phase'] for row in bm['achieved']],
                    phases=[row['phase'] for row in om['achieved']])
            if left and right:
                count = min(len(left), len(right))
                if np.allclose([row['time_s'] for row in left[:count]], [row['time_s'] for row in right[:count]], atol=1e-8):
                    divergence = np.array([np.linalg.norm(np.array(x['bean_positions'])-y['bean_positions'],axis=1)
                                          for x,y in zip(left[:count], right[:count])])
                    if np.isfinite(divergence).all():
                        comparisons[case]['bean_position_divergence_m'] = divergence.max(axis=0)
        variants[name] = dict(status='passed' if all(v['status']=='passed' for v in comparisons.values()) else 'failed',
                             solver=setting, comparisons=comparisons, report=str(folder/'m1c_report.json'))
    return dict(status='passed' if all(v['status']=='passed' for v in variants.values()) else 'failed', variants=variants)


def viewer(robot, output):
    task = FeedingTask(robot)
    results = {}
    for preset in ('bowl_reset', 'pickup_supported'):
        folder = output/'viewer'/preset
        folder.mkdir(parents=True, exist_ok=True)
        if preset == 'bowl_reset':
            task.scene_config['head_fixed'] = True
            task.reset(seed=0)
            state = task.get_state()
        else:
            with (output/'baseline'/'sweep_seed_0_pickup.pkl').open('rb') as file:
                state = pickle.load(file)
            task.set_state(state)
        with (folder/'snapshot.pkl').open('wb') as file:
            pickle.dump(state,file)
        results[preset] = b.visual_check(task,folder)
    return dict(status='passed' if all(v['status']=='passed' for v in results.values()) else 'failed', views=results)


def source_manifest():
    manifest = load_json('third_party_manifest.json')
    rows = list(manifest['derived_files'])
    for entry in manifest['sources']:
        rows.extend(entry.get('files', []))
    mismatches = [row['path'] for row in rows if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']]
    assert not mismatches, f'Stale source manifest hashes: {mismatches}'
    return dict(status='passed', checked_files=len(rows), parameter_status=load_json('configs/scene.json')['beans']['contact_parameter_status'])


def validate(robot, output, selected=None, expected_hashes=None):
    output.mkdir(parents=True, exist_ok=True)
    chosen = set(REQUIRED if selected is None else selected)
    if chosen - set(REQUIRED):
        raise ValueError(f'Unknown cases: {chosen-set(REQUIRED)}')
    start_hashes = input_hashes()
    report = dict(schema_version=1, snapshot_schema_version=3, model_version='single_bean_native_v1', stage='M1-D',
                  scope='Single-bean feeding prototype M1; fixed layout; M3/M4 not verified',
                  robot_id=robot, status='incomplete', m1_status='incomplete',
                  stages={stage:'not_verified' for stage in ('M1-A','M1-B','M1-C','M1-D','M3','M4')},
                  input_hashes=start_hashes, numerical_settings=SETTINGS,
                  cases={name:dict(status='not_verified') for name in REQUIRED})
    path = output/'m1d_report.json'
    def run(name, check):
        started=time.monotonic()
        try:
            report['cases'][name]=check()
        except Exception:
            report['cases'][name]=dict(status='failed',error=traceback.format_exc())
        report['cases'][name]['wall_s']=time.monotonic()-started
        write_json(path,report)
        print(robot,name,report['cases'][name]['status'],flush=True)
    if 'assembly' in chosen:
        run('assembly',lambda: a.validate(robot))
        report['stages']['M1-A']=report['cases']['assembly']['status']
    if 'contacts' in chosen:
        run('contacts',lambda: b.validate(robot,output/'contacts',numerical=True))
        report['stages']['M1-B']=report['cases']['contacts']['status']
    controls = chosen & set(CONTROL_CASES)
    baseline = None
    baseline_cases = set(controls)
    if 'convergence' in chosen:
        baseline_cases.update(MOTION_CASES)
    if 'viewer' in chosen:
        baseline_cases.add('sweep_seed_0')
    if 'performance' in chosen:
        baseline_cases.update(CONTROL_CASES)
    if baseline_cases:
        baseline = c.validate(robot,output/'baseline',sorted(baseline_cases))
        for name in controls:
            report['cases'][name]=copy.deepcopy(baseline['cases'][name])
            for field in ('state_file', 'trajectory_file'):
                if field in report['cases'][name]:
                    report['cases'][name][field]=str(output/'baseline'/report['cases'][name][field])
        if controls == set(CONTROL_CASES):
            report['stages']['M1-C']=baseline['stages']['M1-C']
        elif any(report['cases'][n]['status']=='failed' for n in controls):
            report['stages']['M1-C']='failed'
        write_json(path,report)
    if 'convergence' in chosen:
        run('convergence',lambda: convergence(robot,output,baseline))
    if 'viewer' in chosen:
        run('viewer',lambda: viewer(robot,output))
    if 'performance' in chosen:
        def performance():
            assert baseline and all(baseline['cases'][n]['status']=='passed' for n in CONTROL_CASES)
            rows={n:{**baseline['cases'][n]['performance'], 'load_wall_s':baseline['cases'][n]['load_wall_s'],
                     'model_load_wall_s':baseline['cases'][n]['model_load_wall_s']} for n in CONTROL_CASES}
            return dict(status='passed', full_control_rtf_gate=False, cases=rows)
        run('performance',performance)
    if 'manifest' in chosen:
        run('manifest',source_manifest)
    end_hashes=input_hashes()
    report['hashes_unchanged']=start_hashes==end_hashes and (expected_hashes is None or start_hashes==expected_hashes)
    report['final_input_hashes']=end_hashes
    if not report['hashes_unchanged']:
        report['cases']['manifest']=dict(status='failed',error='Input hash drift or different robot input version')
    statuses=[v['status'] for v in report['cases'].values()]
    report['status']='passed' if all(v=='passed' for v in statuses) else ('failed' if 'failed' in statuses else 'incomplete')
    report['stages']['M1-D']=report['status']
    report['m1_status']=report['status']
    write_json(path,report)
    return report


def publish_freeze(reports, folders, destination, expected_hashes):
    assert set(reports)=={'panda','ur5e'}, 'Freeze requires both robots'
    assert input_hashes()==expected_hashes, 'Input hash drift before publication'
    beans = load_json('configs/scene.json')['beans']
    assert beans['contact_parameter_status']=='frozen', 'Parameters are still candidate'
    assert beans['count']==1, 'Freeze requires exactly one bean'
    for robot, report in reports.items():
        assert report['model_version']=='single_bean_native_v1' and report['snapshot_schema_version']==3
        assert report['status']=='passed' and report['hashes_unchanged']
        assert report['input_hashes']==expected_hashes==report['final_input_hashes']
        assert all(report['cases'].get(n,{}).get('status')=='passed' for n in REQUIRED)
        assert all(report['stages'][n]=='passed' for n in ('M1-A','M1-B','M1-C','M1-D'))
        assert all(report['stages'].get(n)=='not_verified' for n in ('M3','M4'))
    evidence={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in folders.values() for p in sorted(folder.rglob('*')) if p.is_file()}
    versions={name:importlib.metadata.version(name) for name in ('mujoco','numpy','mink','qpsolvers','daqp')}
    manifest=dict(schema_version=1, model_version='single_bean_native_v1',
                  scope='Single-bean M1, fixed layout; no randomized-layout or full feeding release',status='frozen',input_sha256=expected_hashes,
                  actual_parameters={r:dict(solver=reports[r]['cases']['assembly']['solver'],
                      beans=reports[r]['cases']['contacts']['contact_parameters'],
                      acceptance=load_json('configs/acceptance.json'),
                      controls={n:reports[r]['cases'][n].get('actual_control_parameters', {}) for n in CONTROL_CASES}) for r in reports},
                  environment=dict(python=sys.version,platform=platform.platform(),versions=versions),
                  reports={r:str(folders[r]/'m1d_report.json') for r in reports},evidence_sha256=evidence,
                  stages={'M1-A':'passed','M1-B':'passed','M1-C':'passed','M1-D':'passed','M3':'not_verified','M4':'not_verified'})
    assert input_hashes()==expected_hashes, 'Input hash drift during receipt generation'
    write_json(destination,manifest)
    return manifest



def parameter_status(status):
    """Prepare a candidate/frozen input version before its next formal run."""
    scene = load_json('configs/scene.json')
    scene['beans']['contact_parameter_status'] = status
    write_json(ROOT/'configs/scene.json', scene)
    manifest = load_json('third_party_manifest.json')
    for row in manifest['derived_files']:
        row['sha256'] = hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()
    write_json(ROOT/'third_party_manifest.json', manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',choices=['panda','ur5e','all'],default='panda')
    parser.add_argument('--cases',nargs='+',choices=REQUIRED)
    parser.add_argument('--output')
    args=parser.parse_args()
    robots=('panda','ur5e') if args.robot=='all' else (args.robot,)
    output=ROOT/(args.output or 'outputs/single_bean/v1/m1')
    folders={r:(output/r/'m1d' if args.robot=='all' else output if args.output else output/r/'m1d') for r in robots}
    expected=input_hashes()
    final_run = (args.robot=='all' and args.cases is None
                 and load_json('configs/scene.json')['beans']['contact_parameter_status']=='frozen')
    try:
        reports={r:validate(r,folders[r],args.cases,expected) for r in robots}
        passed=all(v['status']=='passed' for v in reports.values())
        if final_run and passed:
            publish_freeze(reports,folders,output/'freeze_manifest.json',expected)
    except BaseException:
        if final_run:
            parameter_status('candidate')
        raise
    if not passed:
        if final_run:
            parameter_status('candidate')
        raise SystemExit(1)


if __name__=='__main__':
    main()
