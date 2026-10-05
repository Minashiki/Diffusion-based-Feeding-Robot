"""Read-only M4 ancestry and M5 checkpoint provenance."""

import hashlib
import json
from pathlib import Path

from feedingrobot.data.episodes import input_hashes
from feedingrobot.sim.model import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def parent_check(config, *, root=ROOT):
    """Verify archived evidence; deliberately never calls a replay or M4 publisher."""
    root = Path(root)
    parent, dataset = root / config['parent'], root / config['dataset']
    frozen = read_json(parent / 'freeze_manifest.json')
    audit = read_json(parent / 'acceptance_audit.json')
    if frozen.get('evidence_sha256', {}).get(str((dataset / 'statistics.json').relative_to(root))) != sha256(dataset / 'statistics.json'):
        raise ValueError('M4 statistics are not bound to the parent freeze')
    stats = read_json(dataset / 'statistics.json')
    if (config['robot_id'] != 'panda' or frozen['status'] != 'frozen'
            or frozen['acceptance_robots'] != ['panda']
            or frozen['model_version'] != 'single_bean_native_v1'
            or frozen['observation_schema_version'] != 3
            or any(audit[k] != 'passed' for k in ('status', 'teacher_gate', 'dataset_gate'))
            or stats['status'] != 'passed' or stats['replay_status'] != 'passed'
            or stats['robot_id'] != 'panda' or frozen['dataset'] != config['dataset']
            or audit['input_sha256'] != frozen['input_sha256']
            or stats['input_hashes'] != frozen['input_sha256']
            or audit['evidence_sha256'] != frozen['evidence_sha256']):
        raise ValueError('M5 requires the frozen Panda M4 audit and dataset')
    from feedingrobot.scripts.collect import statistics_digest
    if stats['validation_sha256'] != statistics_digest(stats):
        raise ValueError('M4 statistics binding changed')
    changed = []
    for relative, expected in frozen['input_sha256'].items():
        if sha256(parent / 'frozen_inputs' / relative) != expected:
            raise ValueError(f'M4 archived input changed: {relative}')
        if sha256(root / relative) != expected:
            if relative not in ('README.md', 'SimModelPlann.md'):
                raise ValueError(f'M4 runtime input changed: {relative}')
            changed.append(relative)
    additions = []
    if root == ROOT:
        additions = sorted(set(input_hashes()) - set(frozen['input_sha256']))
        for relative in additions:
            if not (relative.startswith('src/feedingrobot/policies/')
                    or relative in ('configs/dp_dit.json', 'docs/m5_training.md', 'tests/test_m5.py')
                    or relative in tuple(f'src/feedingrobot/scripts/{name}.py' for name in ('train_dp', 'eval_dp', 'validate_m5'))):
                raise ValueError(f'Unexpected runtime addition outside M5: {relative}')
    files = {str(p.relative_to(dataset)) for p in dataset.rglob('*') if p.is_file()
             and p.name not in ('statistics.json', 'replay_progress.json')}
    if files != set(stats['evidence_sha256']):
        raise ValueError('M4 dataset file coverage changed')
    for relative, expected in stats['evidence_sha256'].items():
        p = dataset / relative
        repo_relative = str(p.relative_to(root))
        if frozen['evidence_sha256'].get(repo_relative) != expected or sha256(p) != expected:
            raise ValueError(f'M4 dataset evidence changed: {relative}')
    for name in ('statistics.json',):
        if frozen['evidence_sha256'].get(str((dataset / name).relative_to(root))) != sha256(dataset / name):
            raise ValueError('M4 statistics are not bound to the parent freeze')
    episodes = {str(p.parent.relative_to(dataset)) for p in dataset.glob('*/*/manifest.json')}
    if (episodes != {e['path'] for e in stats['episodes']} or len(episodes) != stats['attempts']
            or any(e['replay']['status'] != 'passed' for e in stats['episodes'])
            or any(stats[key].get(split, 0) < count for key in ('counts', 'recovery_counts')
                   for split, count in frozen['teacher_config']['quotas'].items())):
        raise ValueError('M4 quotas or replay coverage changed')
    return dict(status='passed', parent_sha256=sha256(parent / 'freeze_manifest.json'),
                audit_sha256=sha256(parent / 'acceptance_audit.json'),
                statistics_sha256=sha256(dataset / 'statistics.json'),
                dataset_binding=stats['validation_sha256'], changed_documents=changed, m5_additions=additions,
                old_acceptance_reruns=0, old_replay_reruns=0)


def checkpoint_check(checkpoint, config, parent):
    if (checkpoint['schema_version'] != 1 or checkpoint['config'] != config
            or checkpoint['parent']['parent_sha256'] != parent['parent_sha256']
            or checkpoint['parent']['dataset_binding'] != parent['dataset_binding']
            or checkpoint['source_hashes'] != input_hashes()):
        raise ValueError('Checkpoint configuration, data or source differs from this run')
