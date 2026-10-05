"""Single-bean parent verification and immutable M4 input archives."""

import hashlib
import json
import shutil

from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.sim.model import ROOT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_m3_check(*, evidence=False):
    path = ROOT / 'outputs/single_bean/v1/m3/freeze_manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['status'] != 'frozen' or manifest['model_version'] != 'single_bean_native_v1':
        raise ValueError('A frozen single-bean M3 parent is required')
    fixed = {p: h for p, h in manifest['input_sha256'].items()
             if p.startswith(('assets/', 'configs/robots/'))
             or p in ('configs/scene.json', 'configs/acceptance.json', 'configs/task.json', 'configs/acceptance_m3.json')}
    for relative, digest in fixed.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'M3 frozen physical input differs: {relative}')
    if evidence:
        from feedingrobot.scripts.validate_m3 import parent_freeze_check
        parent_freeze_check()
        for relative, digest in manifest['evidence_sha256'].items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f'M3 frozen evidence differs: {relative}')
    return dict(status='passed', manifest=str(path.relative_to(ROOT)), sha256=sha256(path),
                physical_inputs=len(fixed), evidence_files=len(manifest['evidence_sha256']))


def freeze_inputs(directory, config, seeds):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'freeze_manifest.json'
    hashes = input_hashes()
    parent = parent_m3_check(evidence=True)
    if path.exists():
        frozen = json.loads(path.read_text())
        if (frozen['input_sha256'] != hashes or frozen['teacher_config'] != config
                or frozen['parent_m3'] != parent or frozen['seeds'] != seeds):
            raise ValueError('M4 freeze belongs to different inputs; use a new directory')
    else:
        frozen = dict(schema_version=1, status='candidate_frozen', model_version='single_bean_native_v1',
                      teacher_version=config['teacher_version'], observation_schema_version=3,
                      teacher_config=config, parent_m3=parent, input_sha256=hashes, seeds=seeds)
        archive = directory / 'frozen_inputs'
        for relative, digest in hashes.items():
            source, destination = ROOT / relative, archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(destination) != digest:
                raise ValueError(f'Input archive mismatch: {relative}')
        write_json(path, frozen)
    for relative, digest in hashes.items():
        if sha256(directory / 'frozen_inputs' / relative) != digest:
            raise ValueError(f'Frozen M4 input mismatch: {relative}')
    return frozen


def publish_acceptance(directory, config, dataset):
    from feedingrobot.experts.gate import matching_teachers_passed
    reports = {robot: json.loads((directory / robot / 'report.json').read_text()) for robot in ('panda', 'ur5e')}
    if (not matching_teachers_passed(reports['panda'], reports['ur5e'])
            or any(any(case['status'] != 'passed' for case in report['cases'].values()) for report in reports.values())):
        return None
    frozen = json.loads((directory / 'freeze_manifest.json').read_text())
    freeze_inputs(directory, config, frozen['seeds'])
    statistics = json.loads((dataset / 'statistics.json').read_text())
    if (statistics['status'] != 'passed' or statistics['replay_status'] != 'passed'
            or any(statistics[key].get(split, 0) < quota for key in ('counts', 'recovery_counts')
                   for split, quota in config['quotas'].items())):
        raise ValueError('M4 requires a complete, replayed dataset')
    excluded = {directory / 'freeze_manifest.json', directory / 'acceptance_audit.json'}
    evidence = {str(p.relative_to(ROOT)): sha256(p) for folder in (directory, dataset)
                for p in sorted(folder.rglob('*')) if p.is_file() and p not in excluded
                and not p.is_relative_to(directory / 'frozen_inputs')}
    if input_hashes() != frozen['input_sha256']:
        raise ValueError('M4 inputs changed during final audit')
    frozen.update(status='frozen', evidence_sha256=evidence,
                  reports={robot: str((directory / robot / 'report.json').relative_to(ROOT)) for robot in reports},
                  dataset=str(dataset.relative_to(ROOT)))
    write_json(directory / 'freeze_manifest.json', frozen)
    audit = dict(status='passed', teacher_gate='passed', dataset_gate='passed',
                 parent_m3=parent_m3_check(evidence=True), model_version=frozen['model_version'],
                 observation_schema_version=3, input_files=len(frozen['input_sha256']),
                 evidence_files=len(evidence), input_sha256=frozen['input_sha256'], evidence_sha256=evidence,
                 counts=statistics['counts'], recovery_counts=statistics['recovery_counts'],
                 attempts=statistics['attempts'])
    write_json(directory / 'acceptance_audit.json', audit)
    return audit
