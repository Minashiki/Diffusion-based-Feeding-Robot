"""New-run reuse of frozen M4 and audited v4 teacher windows."""

from pathlib import Path
import sys

from torch.utils.data import Dataset

from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.legacy import RUNTIME, verify_runtime
from feedingrobot.sim.model import ROOT


def start_pose_check(config):
    """Bind the completed normal extension without rewriting its collection evidence."""
    if not config.get('train_extension_report'):
        return None
    if config.get('correction_report') or config['model'].get('name') not in ('MinaDiTV4', 'MinaDiTV5'):
        raise ValueError('Start-pose mixing requires V4 or V5 without correction data')
    path = ROOT / config['train_extension_report']
    report = read_json(path)
    directory = Path(report['dataset'])
    audit = read_json(path.parent / 'audit.json')
    if (report['status'] != 'ready_train_extension' or not report['source_unchanged']
            or audit['status'] != 'passed' or report['accepted_count'] != 200
            or audit['episodes'] != 200 or report['cell_counts'] != [5] * 40
            or directory.is_relative_to(path.parent) or path.parent.is_relative_to(directory)):
        raise ValueError('Requires the complete, separately stored 200-episode extension')
    actual = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}
    if actual != report['dataset_sha256']:
        raise ValueError('Start-pose dataset file coverage or SHA256 changed')
    expected = {row['path'] for row in report['accepted']}
    if (len(expected) != 200 or expected != {str(p.parent.relative_to(directory))
                                           for p in directory.glob('*/*/manifest.json')}):
        raise ValueError('Start-pose accepted episode coverage differs')
    for row in report['accepted']:
        m = read_json(directory / row['path'] / 'manifest.json')
        replay, pose = row['replay'], row['initial_pose']
        if (Path(row['path']).parts[:1] != ('train',) or m['split'] != 'train'
                or m['scenario'].get('recover') or not m['success'] or not m['accepted_normal']
                or m['status'] != 'complete' or m['seed'] != row['seed']
                or m['initial_pose'] != pose or m['input_hashes'] != report['input_hashes']
                or row['status'] != 'accepted' or row['rejection_reason'] is not None
                or not 0 < row['time_s'] <= 60 or replay['status'] != 'passed'
                or not 0 <= replay['max_physics_reference_error'] <= 1e-10
                or not 0 <= replay['max_observation_error'] <= 1e-7
                or not replay['events_equal'] or not replay['result_equal']
                or pose['joint_margin_rad'] < .02
                or min(pose['initial_clearance_m'], pose['approach_path_clearance_m']) < .005):
            raise ValueError('Unqualified start-pose training episode')
    return dict(report=str(path.resolve()), report_sha256=sha256(path),
                audit_sha256=sha256(path.parent / 'audit.json'), dataset=str(directory),
                dataset_sha256=actual)


def correction_check(config):
    if not config.get('correction_report'):
        return None
    path, audit_path = (ROOT / config[name] for name in ('correction_report', 'correction_audit'))
    report, audit = read_json(path), read_json(audit_path)
    if (report.get('status') != 'ready_experimental' or not report.get('source_unchanged')
            or audit.get('status') != 'audit_passed' or audit.get('binding') != report['binding']
            or audit.get('collection_sha256') != sha256(path) or audit.get('episodes') != 44
            or not audit.get('model_exact') or not audit.get('ema_exact')
            or audit.get('optimizer_updates') != 0 or audit.get('optimizer_created') is not False):
        raise ValueError('Requires the unchanged, passed v4 collection and legacy zero-update audit')
    legacy = report['binding']['legacy']
    verify_runtime(legacy['source_hashes'])
    groups = [(RUNTIME / 'tools/m5', legacy['v1_tools']),
              (RUNTIME / 'tools/m5/correction_v2', legacy['v2_tools']),
              (RUNTIME / 'tools/m5/correction_v3', legacy['v3_tools']),
              (RUNTIME / 'tools/m5/correction_v4', report['binding']['v4_tools'])]
    for directory, expected in groups:
        actual = {p.name: sha256(p) for p in directory.glob('*.py') if not p.name.startswith('test_')}
        if actual != expected:
            raise ValueError('Historical collection tool snapshot differs')
    for name in ('__init__.py', 'corpus.py'):
        if sha256(ROOT / 'tools/m5/correction_v4' / name) != report['binding']['v4_tools'][name]:
            raise ValueError('The active v4 reader differs from its collection evidence')
    directory = Path(report['dataset'])
    actual = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}
    if actual != report['dataset_sha256']:
        raise ValueError('Correction dataset file coverage or SHA256 changed')
    evidence = {str(p.relative_to(path.parent)): sha256(p) for p in path.parent.rglob('*')
                if p.is_file() and p != path and '__pycache__' not in p.parts}
    if evidence != report['evidence_sha256']:
        raise ValueError('Correction collection evidence changed')
    return dict(report=str(path.resolve()), report_sha256=sha256(path), audit_sha256=sha256(audit_path),
                dataset=str(directory), dataset_sha256=actual, legacy_source_hashes=legacy['source_hashes'])


class TrainingWindows(Dataset):
    def __init__(self, config, correction=None, normalization=None, extension=None):
        self.base = ActionWindows(ROOT / config['dataset'], 'train', config['model']['horizon'], normalization)
        self.extra = None
        self.split, self.schema, self.fields = self.base.split, self.base.schema, self.base.fields
        self.normalization = normalization
        self.by_phase = self.base.by_phase
        self.extension = None
        if extension:
            if correction:
                raise ValueError('Start-pose mixing excludes correction data')
            self.extension = ActionWindows(extension['dataset'], 'train', config['model']['horizon'], normalization)
            # Check every base split against the extension, including scenario groups.
            base_manifests = [read_json(p) for p in self.base.directory.glob('*/*/manifest.json')]
            for key in ('seed', 'group_id', 'base_group_id'):
                seen = {m.get(key, m['group_id']) for m in base_manifests
                        if key != 'base_group_id' or m['split'] != 'train'}
                for _, m in self.extension.episodes:
                    value = m.get(key, m['group_id'])
                    if value in seen:
                        raise ValueError(f'Cross-source duplicate or split leakage: {key}')
                    if key != 'base_group_id':
                        seen.add(value)
            self.by_phase = {phase: dict(episodes) for phase, episodes in self.base.by_phase.items()}
            for phase, episodes in self.extension.by_phase.items():
                target = self.by_phase.setdefault(phase, {})
                for episode, indices in episodes.items():
                    target[len(self.base.episodes) + episode] = [len(self.base) + i for i in indices]
        if correction:
            sys.path.insert(0, str(ROOT / 'tools/m5'))
            from correction_v4.corpus import PickupWindows
            self.extra = PickupWindows(correction['dataset'], 'train', config['model']['horizon'], normalization)
            self.indices = sorted(self.extra.tags)
            self.lookup = {index: len(self.base)+i for i, index in enumerate(self.indices)}

    def set_normalization(self, normalization):
        timing_changed = ((self.normalization or {}).get('observation_timing')
                          != normalization.get('observation_timing'))
        self.normalization = self.base.normalization = normalization
        if timing_changed:
            self.base.cache.clear()
        if self.extension:
            self.extension.normalization = normalization
            if timing_changed:
                self.extension.cache.clear()
        if self.extra:
            self.extra.normalization = normalization

    def __len__(self):
        return len(self.base) + (len(self.extension) if self.extension else 0) + (len(self.indices) if self.extra else 0)

    def __getitem__(self, index):
        if self.extension and index >= len(self.base):
            return self.extension[index-len(self.base)]
        return (self.base[index] if index < len(self.base)
                else self.extra[self.indices[index-len(self.base)]])

    def sample_indices(self, rng, count):
        if not self.extra:
            return ActionWindows.sample_indices(self, rng, count)
        if count < 4 or count % 4:
            raise ValueError('Mixed v4 batch size must be divisible by four')
        indices = self.base.sample_indices(rng, count*3//4)
        names = tuple(self.extra.pools)
        for _ in range(count//4):
            pool = self.extra.pools[names[int(rng.integers(3))]]
            episodes = sorted(pool)
            stages = pool[episodes[int(rng.integers(len(episodes)))]]
            names_in_stage = sorted(stages)
            windows = stages[names_in_stage[int(rng.integers(len(names_in_stage)))]]
            indices.append(self.lookup[windows[int(rng.integers(len(windows)))]])
        return [indices[i] for i in rng.permutation(count)]

    def summary(self):
        result = dict(base_windows=len(self.base), correction_windows=len(self.indices) if self.extra else 0,
                    correction=self.extra.summary() if self.extra else None)
        if self.extension:
            result.update(extension_windows=len(self.extension), base_episodes=len(self.base.episodes),
                          extension_episodes=len(self.extension.episodes))
        return result
