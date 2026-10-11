"""Train-only initial-position extension; the frozen teacher and physics stay intact."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import shutil

for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import mink
import mujoco
import numpy as np

from feedingrobot.control.adapter import clip_norm
from feedingrobot.data.episodes import input_hashes, json_value, load_episode, write_json
from feedingrobot.data.recipes import recipe
from feedingrobot.data.replay import replay_episode
from feedingrobot.data.rollout import run_episode
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.policies.audit import parent_check, read_json, sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.scripts.collect import dataset_statistics
from feedingrobot.sim.events import geom_corners
from feedingrobot.sim.model import ROOT

HERE = Path(__file__).resolve().parent
CONFIG = read_json(HERE / 'config.json')
CELLS = list(itertools.product(*(range(n) for n in CONFIG['grid'])))
DATASET = ROOT / 'datasets/single_bean/start_pose_v2/panda'
OUTPUT = ROOT / 'outputs/single_bean/start_pose_v2/collection_001'


def tool_hashes():
    return {p.name: sha256(p) for p in sorted(HERE.iterdir())
            if p.suffix in ('.py', '.json', '.md')}


def cell_bounds(cell):
    bounds = np.asarray(CONFIG['bounds_m'])
    width = (bounds[:, 1] - bounds[:, 0]) / CONFIG['grid']
    low = bounds[:, 0] + np.asarray(cell) * width
    return low, low + width


def cell_for(position):
    bounds = np.asarray(CONFIG['bounds_m'])
    position = np.asarray(position)
    if not np.isfinite(position).all() or np.any(position < bounds[:, 0]) or np.any(position >= bounds[:, 1]):
        return None
    return tuple(((position - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) * CONFIG['grid']).astype(int))


def collision_pairs(task):
    m, idx = task.model, task.index
    pairs = set(tuple(map(int, p)) for p in task.adapter.limits[2].geom_id_pairs)
    obstacles = [g for g in range(m.ngeom)
                 if idx.group(m, g) in ('table', 'bowl', 'mouth', 'floor', 'food')]
    # The rigid tool intentionally overlaps its mounting link; exclude that assembly only.
    mount = m.body(task.robot_config['mount_body']).id
    other_arm = [g for g in idx.arm_geoms if m.geom_bodyid[g] != mount]
    pairs.update((a, b) for a in idx.spoon_geoms for b in obstacles + other_arm)
    pairs.update((a, int(b)) for a in idx.arm_geoms for b in idx.bean_collision_geoms)
    return np.asarray(sorted(pairs), dtype=int)


def clearance(task, data, pairs):
    # Bounding spheres are a conservative broadphase; planes always reach the narrowphase.
    a, b = pairs.T
    lower = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1)
    lower -= task.model.geom_rbound[a] + task.model.geom_rbound[b]
    planes = ((task.model.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE)
              | (task.model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE))
    minimum = .02
    for x, y in pairs[(lower < minimum) | planes]:
        minimum = min(minimum, float(mujoco.mj_geomDistance(task.model, data, int(x), int(y), .02, None)))
    return minimum


def solve(task, configuration, position, rotation, pairs):
    frame = mink.FrameTask(task.robot_config['tcp_site'], 'site', position_cost=1.,
                           orientation_cost=1., lm_damping=1e-4)
    minimum = clearance(task, configuration.data, pairs)
    for _ in range(1200):
        if minimum < CONFIG['clearance_m']:
            raise ValueError('path_collision')
        actual = configuration.get_transform_frame_to_world(task.robot_config['tcp_site'], 'site')
        error_p = np.asarray(position) - actual.translation()
        error_r = (mink.SO3.from_matrix(rotation) @ actual.rotation().inverse()).log()
        if np.linalg.norm(error_p) < .0002 and np.linalg.norm(error_r) < .002:
            return minimum
        incremental = mink.SE3.from_rotation_and_translation(
            mink.SO3.exp(clip_norm(error_r, .005)) @ actual.rotation(),
            actual.translation() + clip_norm(error_p, .0005))
        frame.set_target(incremental)
        try:
            velocity = mink.solve_ik(configuration, [frame], .01, solver='daqp',
                                     limits=task.adapter.limits, constraints=[task.adapter.freeze],
                                     damping=1e-5, safety_break=True)
            configuration.integrate_inplace(velocity, .01)
        except (mink.exceptions.MinkError, ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError('ik_unreachable') from exc
        minimum = min(minimum, clearance(task, configuration.data, pairs))
    raise ValueError('ik_unreachable')


def check_pose(task, target, rotation, cell, pairs):
    idx, data = task.index, task.data
    position = data.site_xpos[idx.tcp].copy()
    measured_rotation = data.site_xmat[idx.tcp].reshape(3, 3).copy()
    q = data.qpos[idx.qpos].copy()
    ranges = task.model.jnt_range[idx.joints]
    margin = float(np.min(np.minimum(q - ranges[:, 0], ranges[:, 1] - q)))
    distance = clearance(task, data, pairs)
    angle = float(np.linalg.norm(mink.SO3.from_matrix(rotation @ measured_rotation.T).log()))
    if cell_for(position) != tuple(cell):
        raise ValueError('settled_outside_cell')
    if margin < CONFIG['joint_margin_rad']:
        raise ValueError('joint_margin')
    if distance < CONFIG['clearance_m']:
        raise ValueError('initial_collision')
    if np.linalg.norm(position - target) > CONFIG['position_tolerance_m'] or angle > CONFIG['rotation_tolerance_rad']:
        raise ValueError('settled_pose_error')
    table = task.model.geom('table').id
    corners = geom_corners(task.model, data, table)
    tool = np.concatenate([geom_corners(task.model, data, g) for g in idx.spoon_geoms])
    if (np.any(tool[:, :2] < corners[:, :2].min(0) + CONFIG['clearance_m'])
            or np.any(tool[:, :2] > corners[:, :2].max(0) - CONFIG['clearance_m'])
            or tool[:, 2].min() < corners[:, 2].max() + CONFIG['clearance_m']):
        raise ValueError('tool_outside_table_top')
    return dict(target_position_m=target, actual_position_m=position, actual_rotation=measured_rotation,
                q=q, joint_margin_rad=margin, initial_clearance_m=distance,
                clearance_distance_cap_m=.02, rotation_error_rad=angle, cell=list(cell))


def prepare(seed, scenario, parameters, teacher_config, target, cell):
    env = FeedingGymEnv('panda')
    try:
        env.reset(seed=seed, options={'scenario': scenario})
        task = env.task
        # V1's configured TCP orientation is identity; gravity settling is checked separately.
        rotation = np.eye(3)
        pairs = collision_pairs(task)
        configuration = mink.Configuration(task.model)
        configuration.update(task.data.qpos.copy())
        solve(task, configuration, target, rotation, pairs)
        original_q = task.robot_config['reset_q']
        try:
            task.robot_config['reset_q'] = configuration.q[task.index.qpos].tolist()
            env.reset(seed=seed, options={'scenario': scenario})
        except (RuntimeError, ValueError) as exc:
            raise ValueError('settling_failed') from exc
        finally:
            task.robot_config['reset_q'] = original_q
        if task.terminated or task.reset_diagnostics['status'] != 'passed':
            raise ValueError('settling_failed')
        pose = check_pose(task, target, rotation, cell, pairs)
        teacher = Teacher(task.robot_config, teacher_config)
        teacher.reset(parameters, geometry=teacher_geometry(task))
        configuration.update(task.data.qpos.copy())
        pose['approach_path_clearance_m'] = solve(task, configuration, *teacher.path[0][1:3], pairs)
        if task.tick != 0 or task.data.time != 0 or task.logic.events:
            raise ValueError('nonfresh_initial_state')
        return task.get_state(), pose
    finally:
        env.close()


def rejection(result, replay):
    if result.get('truncated') or result.get('time_s', 61) > 60. or result.get('failure_reason') == 'time_limit':
        return 'time_limit'
    events = {e['name'] for e in result.get('events', [])}
    if (not result.get('success') or not result.get('accepted_normal') or result.get('failure_reason')
            or not result.get('terminated') or result.get('scenario', {}).get('recover', False)
            or result.get('max_episode_s') != 60. or not {'pickup', 'delivery', 'success'} <= events):
        return result.get('failure_reason') or 'task_failed'
    if (replay.get('status') != 'passed' or replay.get('max_physics_reference_error', 1) > 1e-10
            or replay.get('max_observation_error', 1) > 1e-7
            or not replay.get('events_equal') or not replay.get('result_equal')):
        return 'replay_failed'
    return None


def worker_setup():
    # All numerical pools inherit the four-CPU process-tree affinity and one-thread settings.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def trial(cell_index, attempt, output, teacher_config):
    cell = CELLS[cell_index]
    index = CONFIG['recipe_index_base'] + cell_index * CONFIG['max_candidates_per_cell'] + attempt
    seed, scenario, parameters, base_group = recipe(teacher_config, 'train', index, recover=False)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261007]))
    low, high = cell_bounds(cell)
    # Leave space for the few-mm natural settling displacement.
    target = rng.uniform(low + .005, high - .005)
    name = f'normal_{seed}'
    directory = Path(output) / 'pending' / name
    row = dict(seed=seed, recipe_index=index, cell_index=cell_index, cell=list(cell), attempt=attempt,
               target_position_m=target, base_group_id=base_group, name=name)
    try:
        state, pose = prepare(seed, scenario, parameters, teacher_config, target, cell)
        row['initial_pose'] = pose
        result = run_episode('panda', seed, teacher_config, directory, scenario=scenario,
                             teacher_parameters=parameters, split='train',
                             group_id=f'start_pose_v2:{seed}:{base_group}', initial_state=state, viewer=False)
        result['initial_pose'] = pose
        result['base_group_id'] = base_group
        result['dataset_version'] = CONFIG['version']
        write_json(directory / 'manifest.json', result)
        try:
            replay = replay_episode(directory)
        except Exception as exc:
            replay = dict(status='failed', error=str(exc))
        row.update(replay=replay, time_s=result['time_s'], success=result['success'],
                   failure_reason=result['failure_reason'], contact_peak_n=result['contact_peak_n'],
                   wrist_peak_n=result['wrist_peak_n'])
        reason = rejection(result, replay)
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        row['error_type'] = type(exc).__name__
    row.update(status='accepted' if reason is None else 'rejected', rejection_reason=reason)
    return json.loads(json.dumps(row, default=json_value))


def historical_ids(teacher_config):
    seeds, heldout, interrupted = set(), set(), []
    def record_seeds(value):
        if isinstance(value, dict):
            if isinstance(value.get('seed'), int):
                seeds.add(value['seed'])
            for item in value.values():
                record_seeds(item)
        elif isinstance(value, list):
            for item in value:
                record_seeds(item)
    for base in (ROOT / 'datasets/single_bean', ROOT / 'outputs/single_bean/v1'):
        for path in base.rglob('manifest.json'):
            if path.stat().st_size == 0 and 'interrupted' in path.parts:
                interrupted.append(str(path.relative_to(ROOT)))
                continue
            m = read_json(path)
            if 'seed' in m:
                seeds.add(m['seed'])
            if m.get('split') in ('validation', 'test'):
                heldout.add(m.get('base_group_id', m.get('group_id')))
        for path in base.rglob('report.json'):
            record_seeds(read_json(path))
    for c in range(len(CELLS)):
        for attempt in range(CONFIG['max_candidates_per_cell']):
            index = CONFIG['recipe_index_base'] + c * CONFIG['max_candidates_per_cell'] + attempt
            seed, _, _, group = recipe(teacher_config, 'train', index, recover=False)
            if seed in seeds or group in heldout:
                raise ValueError(f'Historical seed/group conflict: {seed}')
    return interrupted


def save_report(output, report):
    counts = Counter(row['cell_index'] for row in report['accepted'])
    report['cell_counts'] = [counts[i] for i in range(len(CELLS))]
    report['accepted_count'] = len(report['accepted'])
    report['rejection_counts'] = dict(Counter(row['rejection_reason'] for row in report['rejected']))
    write_json(output / 'report.json', report)


def record(row, dataset, output, report):
    name = row['name']
    pending = output / 'pending' / name
    destination = dataset / 'train' / name if row['status'] == 'accepted' else output / 'rejected' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if pending.exists():
        pending.rename(destination)
    else:
        destination.mkdir()
    row['path'] = str(destination.relative_to(dataset if row['status'] == 'accepted' else output))
    if row['status'] == 'accepted':
        row['replay']['episode'] = str(destination)
        report['accepted'].append(row)
    else:
        write_json(destination / 'rejection.json', row)
        report['rejected'].append(row)
    report['active'] = [job for job in report.get('active', []) if job['name'] != name]
    save_report(output, report)
    print(json.dumps(dict(cell=row['cell'], status=row['status'], accepted=len(report['accepted']),
                          seed=row['seed'], reason=row['rejection_reason'])), flush=True)


def verify_binding(report):
    if report['tools_sha256'] != tool_hashes() or report['input_hashes'] != input_hashes():
        raise ValueError('Collection inputs/tools changed')
    if report['parent'] != parent_check(read_json(ROOT / 'configs/minadit_v5.json')):
        raise ValueError('V1 parent/data binding changed')
    for name, digest in report['tools_sha256'].items():
        if sha256(Path(report['output']) / 'tool_snapshot' / name) != digest:
            raise ValueError('Collection tool snapshot changed')


def recover_interrupted(dataset, output, report):
    """Quarantine uncommitted work; never reuse an interrupted attempt's seed."""
    for row in list(report.get('active', [])):
        orphan = dataset / 'train' / row['name']
        pending = output / 'pending' / row['name']
        if orphan.exists():
            orphan.rename(pending)
        row.update(status='rejected', rejection_reason='interrupted')
        record(row, dataset, output, report)
    if any((output / 'pending').iterdir()):
        raise ValueError('Unjournaled pending work requires inspection')


def audit(dataset, output, report, *, complete=False):
    verify_binding(report)
    expected = {r['path'] for r in report['accepted']}
    actual = {str(p.parent.relative_to(dataset)) for p in dataset.glob('*/*/manifest.json')}
    if actual != expected:
        raise ValueError('Accepted dataset coverage differs from the success list')
    if dataset.is_relative_to(output) or output.is_relative_to(dataset):
        raise ValueError('Training and rejection roots must be separate')
    seeds = set()
    for row in report['accepted']:
        m, arrays = load_episode(dataset / row['path'])
        if (m['seed'] in seeds or m['seed'] != row['seed'] or m['split'] != 'train'
                or m.get('dataset_version') != CONFIG['version']
                or m['initial_pose'] != row['initial_pose'] or m['input_hashes'] != report['input_hashes']
                or rejection(m, row['replay']) is not None):
            raise ValueError('Unqualified, duplicate or unbound training episode')
        seeds.add(m['seed'])
        pose = m['initial_pose']
        if (cell_for(pose['actual_position_m']) != tuple(row['cell'])
                or pose['joint_margin_rad'] < CONFIG['joint_margin_rad']
                or min(pose['initial_clearance_m'], pose['approach_path_clearance_m']) < CONFIG['clearance_m']):
            raise ValueError('Initial-position safety/coverage evidence differs')
        for name in ('physics', 'observations', 'action_observations', 'actions', 'proposals'):
            if not np.isfinite(arrays[name]).all():
                raise ValueError('Nonfinite accepted data')
    counts = Counter(r['cell_index'] for r in report['accepted'])
    if complete and any(counts[i] != CONFIG['per_cell'] for i in range(len(CELLS))):
        raise ValueError('Requires exactly five accepted trajectories in every cell')
    if report['accepted'] and (dataset / 'normalization.json').exists():
        windows = ActionWindows(dataset, 'train', horizon=16)
        if len(windows.episodes) != len(expected):
            raise ValueError('Training reader episode coverage differs')
        for episode in range(len(windows.episodes)):
            first = next(i for i, w in enumerate(windows.windows) if w[0] == episode)
            item = windows[first]
            if not item['action_mask'].any():
                raise ValueError('Missing valid training labels')
        return len(windows)
    return 0


def finish(dataset, output, report):
    audit(dataset, output, report, complete=True)
    statistics = dataset_statistics(dataset)
    for episode in statistics['episodes']:
        episode['replay'] = next(r['replay'] for r in report['accepted'] if r['path'] == episode['path'])
    statistics.update(status='ready_train_extension', replay_status='passed', version=CONFIG['version'])
    write_json(dataset / 'statistics.json', statistics)
    report['training_windows'] = audit(dataset, output, report, complete=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions = np.array([r['initial_pose']['actual_position_m'] for r in report['accepted']])
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*positions.T, c=positions[:, 2], s=12)
    ax.set(xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)', title='200 accepted initial TCP positions')
    ax = fig.add_subplot(122)
    ax.bar(range(len(CELLS)), report['cell_counts'])
    ax.set(xlabel='Position cell', ylabel='Accepted trajectories', ylim=(0, 6), title='40 cells, 5 trajectories each')
    fig.tight_layout()
    fig.savefig(output / 'initial_positions.png', dpi=160)
    plt.close(fig)
    report['dataset_sha256'] = {str(p.relative_to(dataset)): sha256(p)
                                for p in sorted(dataset.rglob('*')) if p.is_file()}
    report.update(status='ready_train_extension', source_unchanged=True, model_training_run=False)
    save_report(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'audit'))
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-episodes', type=int, help='Stop safely after this many newly accepted episodes')
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if dataset.is_relative_to(output) or output.is_relative_to(dataset):
        parser.error('Dataset and evidence/rejection directories must be separate')
    affinity = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, affinity)
    os.nice(10)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    if args.mode == 'audit':
        report = read_json(output / 'report.json')
        count = audit(dataset, output, report, complete=True)
        actual = {str(p.relative_to(dataset)): sha256(p) for p in dataset.rglob('*') if p.is_file()}
        if actual != report.get('dataset_sha256') or report['status'] != 'ready_train_extension':
            raise ValueError('Final dataset binding/status changed')
        print(json.dumps(dict(status='passed', episodes=len(report['accepted']), training_windows=count)), flush=True)
        return
    teacher_config = read_json(ROOT / 'configs/collect.json')
    if args.resume:
        report = read_json(output / 'report.json')
        if report['dataset'] != str(dataset):
            raise ValueError('Resume dataset differs')
        verify_binding(report)
        recover_interrupted(dataset, output, report)
        audit(dataset, output, report)
    else:
        interrupted = historical_ids(teacher_config)
        parent = parent_check(read_json(ROOT / 'configs/minadit_v5.json'))
        output.mkdir(parents=True, exist_ok=False)
        dataset.mkdir(parents=True, exist_ok=False)
        (dataset / 'train').mkdir()
        (output / 'pending').mkdir()
        (output / 'rejected').mkdir()
        (output / 'tool_snapshot').mkdir()
        for name in tool_hashes():
            shutil.copyfile(HERE / name, output / 'tool_snapshot' / name)
        report = dict(version=CONFIG['version'], config=CONFIG, dataset=str(dataset), output=str(output), status='incomplete',
                      accepted=[], rejected=[], active=[], parent=parent, input_hashes=input_hashes(),
                      tools_sha256=tool_hashes(), teacher_config=teacher_config, cpu_affinity=affinity,
                      workers=2, model_training_run=False, historical_empty_interrupted_manifests=interrupted)
        save_report(output, report)
    start_count = len(report['accepted'])
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_setup) as pool:
            while len(report['accepted']) < len(CELLS) * CONFIG['per_cell']:
                counts = Counter(r['cell_index'] for r in report['accepted'])
                attempts = Counter(r['cell_index'] for r in report['accepted'] + report['rejected'])
                available = [i for i in range(len(CELLS)) if counts[i] < CONFIG['per_cell']
                             and attempts[i] < CONFIG['max_candidates_per_cell']]
                # Breadth first: establish coverage of every cell before filling extra slots.
                available.sort(key=lambda i: (counts[i], i))
                if not available:
                    raise ValueError('Candidate budget exhausted before complete coverage')
                slots = 2
                if args.max_new_episodes is not None:
                    slots = min(slots, args.max_new_episodes - (len(report['accepted']) - start_count))
                if slots <= 0:
                    break
                report['active'] = []
                for i in available[:slots]:
                    index = CONFIG['recipe_index_base'] + i * CONFIG['max_candidates_per_cell'] + attempts[i]
                    seed, _, _, _ = recipe(teacher_config, 'train', index, recover=False)
                    report['active'].append(dict(cell_index=i, cell=list(CELLS[i]), attempt=attempts[i],
                                                 recipe_index=index, seed=seed, name=f'normal_{seed}'))
                save_report(output, report)
                futures = [pool.submit(trial, i, attempts[i], output, teacher_config) for i in available[:slots]]
                for future in futures:
                    record(future.result(), dataset, output, report)
                if report['tools_sha256'] != tool_hashes() or report['input_hashes'] != input_hashes():
                    raise ValueError('Inputs changed during collection')
        if len(report['accepted']) == len(CELLS) * CONFIG['per_cell']:
            finish(dataset, output, report)
    except BaseException as exc:
        report.update(status='incomplete', error=str(exc) or type(exc).__name__)
        save_report(output, report)
        raise


if __name__ == '__main__':
    main()
