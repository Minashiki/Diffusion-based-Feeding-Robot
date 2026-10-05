"""Carry the pinned revision-3 Panda evidence across the single-robot scope change."""

import ast
import copy
import difflib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from feedingrobot.data.episodes import input_hashes, load_episode, write_json
from feedingrobot.experts.freeze import freeze_inputs, parent_m3_check, sha256
from feedingrobot.experts.gate import panda_teacher_passed
from feedingrobot.sim.model import ROOT

SOURCE = 'outputs/single_bean/v1/m4/revision_3'
SOURCE_SHA256 = {
    'freeze_manifest.json': '06af3c68d2c0e44bdd919cda5da6a0b5eb925269e2cbd8daeb17d408d83a5735',
    'panda/report.json': 'db1a11bef058c5c26f4fec6961e5dc09700be3640b60ac73e6b0b4c091ad8ed5',
    'range_calibration.json': 'b7c69d02cade76f4999fe82e720dd853729125282693d8ceab3a80850c7a29a0',
    'shared_pytest.json': '2b8af55456218de1c8dc499275cb2db4cc39de3dd5caee4355061a00fffb7af7',
    'tolerance_authorization.json': '12595ed747385ac695bae05e35d934f66788543f0c2ee10915429f89ea192c3e',
}
# Only these scope/publishing changes are authorized for this one-time handoff.
CHANGED_FUNCTIONS = {
    'src/feedingrobot/experts/gate.py': {'matching_teachers_passed', 'panda_teacher_passed'},
    'src/feedingrobot/experts/freeze.py': {'freeze_inputs', 'publish_acceptance'},
    'src/feedingrobot/scripts/collect.py': {'check_gate', 'dataset_statistics', 'statistics_digest', 'verified_statistics', 'main'},
    'src/feedingrobot/scripts/validate_m4.py': {'update_gate', 'main'},
}
ALLOWED_CHANGES = set(CHANGED_FUNCTIONS) | {
    'src/feedingrobot/experts/reuse.py', 'tests/test_m4_rebuild.py', 'tests/test_m4_reuse.py',
    'README.md', 'SimModelPlann.md', 'docs/m4_interfaces.md',
    'docs/m4_execution_plan.md', 'docs/m4_rebuild_status.md',
}


def input_changes(archive, original, current):
    changes = {}
    for relative in sorted(set(original) | set(current)):
        if original.get(relative) == current.get(relative):
            continue
        if relative not in ALLOWED_CHANGES:
            raise ValueError(f'Unapproved acceptance input change: {relative}')
        before = (archive / relative).read_text() if relative in original else ''
        after = (ROOT / relative).read_text() if relative in current else ''
        if relative in CHANGED_FUNCTIONS:
            def protected(text):
                return {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                        and node.name not in CHANGED_FUNCTIONS[relative]}
            if protected(before) != protected(after):
                raise ValueError(f'Runtime function changed during scope handoff: {relative}')
        changes[relative] = dict(old_sha256=original.get(relative), new_sha256=current.get(relative),
            diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile='revision_3/' + relative, tofile='revision_4/' + relative)))
    return changes


def verify_reuse_audit(report):
    origin = report['evidence_origin']
    path = ROOT / origin['audit']
    if sha256(path) != origin['audit_sha256']:
        raise ValueError('Teacher reuse audit changed')
    audit = json.loads(path.read_text())
    if (audit['status'] != 'passed' or audit['input_sha256'] != input_hashes()
            or audit['source_report_sha256'] != SOURCE_SHA256['panda/report.json']
            or report['input_hashes'] != audit['input_sha256']
            or origin['input_sha256'] != audit['source_input_sha256']):
        raise ValueError('Teacher reuse audit belongs to different evidence/inputs')
    original = json.loads((ROOT / audit['source_report']).read_text())
    for key in ('teacher_config', 'parent_m3', 'baseline', 'recovery_baseline', 'convergence_cases'):
        if report[key] != original[key]:
            raise ValueError(f'Reused teacher report differs: {key}')
    for name in original['cases']:
        if name != 'dataset' and report['cases'][name] != original['cases'][name]:
            raise ValueError(f'Reused physical result differs: {name}')
    for relative, digest in audit['evidence_sha256'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'Reused evidence changed: {relative}')
    return audit


def reuse_teacher_report(source_path, output, config):
    source = ROOT / SOURCE
    if source_path.resolve() != (source / 'panda/report.json').resolve():
        raise ValueError('Only the authorized revision-3 Panda report can be reused')
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Reuse must preserve revision-3 evidence in a separate output directory')
    for relative, digest in SOURCE_SHA256.items():
        if sha256(source / relative) != digest:
            raise ValueError(f'Original acceptance evidence changed: {relative}')
    original = json.loads(source_path.read_text())
    frozen = json.loads((source / 'freeze_manifest.json').read_text())
    parent = parent_m3_check(evidence=True)
    if (not panda_teacher_passed(original) or original['teacher_config'] != config
            or config.get('teacher_status') != 'frozen'
            or original['input_hashes'] != frozen['input_sha256']
            or original['parent_m3'] != frozen['parent_m3'] or parent != original['parent_m3']):
        raise ValueError('Original Panda teacher/parent acceptance is incomplete or inconsistent')
    archive = source / 'frozen_inputs'
    for relative, digest in frozen['input_sha256'].items():
        if sha256(archive / relative) != digest:
            raise ValueError(f'Original frozen input changed: {relative}')
    current = input_hashes()
    changes = input_changes(archive, frozen['input_sha256'], current)
    # This is an integrity check of saved recordings, not a physical rerun.
    manifests = sorted((source / 'panda').glob('**/manifest.json'))
    expected = {str(Path(r['episode']).resolve()) for r in original['cases']['replay']['episodes']}
    expected |= {str(Path(r['episode']).resolve()) for r in original['cases']['prior_revision']['replays']}
    actual = {str(p.parent.resolve()) for p in manifests}
    if (len(original['cases']['replay']['episodes']) != 163 or expected != actual
            or any(r['status'] != 'passed' for r in original['cases']['replay']['episodes']
                   + original['cases']['prior_revision']['replays'])):
        raise ValueError('Original replay coverage is incomplete')
    for path in manifests:
        manifest, _ = load_episode(path.parent)
        if manifest['input_hashes'] != frozen['input_sha256'] or manifest['teacher_config'] != config:
            raise ValueError(f'Original episode inputs differ: {path}')
    ranges = json.loads((source / 'range_calibration.json').read_text())
    if ranges['status'] != 'passed' or len(ranges['trials']) != 45 or not all(t['success'] for t in ranges['trials']):
        raise ValueError('Original diversity acceptance is incomplete')
    range_directory = ROOT / ranges['evidence_directory']
    for name, digest in ranges['episode_manifest_sha256'].items():
        if sha256(range_directory / name / 'manifest.json') != digest:
            raise ValueError(f'Diversity evidence changed: {name}')
        load_episode(range_directory / name)
    shared = json.loads((source / 'shared_pytest.json').read_text())
    if (shared['status'] != 'passed' or shared['tests'] != 413
            or shared['input_sha256'] != frozen['input_sha256']
            or sha256(ROOT / shared['log']) != shared['log_sha256']
            or sha256(ROOT / shared['xml']) != shared['xml_sha256']):
        raise ValueError('Original 413-test evidence differs')
    delta = json.loads((output.parent / 'delta_tests.json').read_text())
    suites = ET.parse(ROOT / delta['xml']).getroot().iter('testsuite')
    if (delta['status'] != 'passed' or delta['input_sha256'] != current
            or sha256(ROOT / delta['log']) != delta['log_sha256']
            or sha256(ROOT / delta['xml']) != delta['xml_sha256']
            or any(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites)
            or delta['tests'] < 1):
        raise ValueError('Current scope/gate tests must pass before teacher release')
    evidence = {}
    folders = (source / 'panda', archive, range_directory)
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                evidence[str(path.relative_to(ROOT))] = sha256(path)
    for path in [source / p for p in SOURCE_SHA256] + [ROOT / shared['log'], ROOT / shared['xml'],
            output.parent / 'delta_tests.json', ROOT / delta['log'], ROOT / delta['xml']]:
        evidence[str(path.relative_to(ROOT))] = sha256(path)
    for check in original['cases']['regressions']['checks'] + original['cases']['viewer']['checks']:
        path = Path(check['log'])
        if not path.is_file():
            raise ValueError(f'Original acceptance log missing: {path}')
        evidence[str(path.relative_to(ROOT))] = sha256(path)
    if input_hashes() != current:
        raise ValueError('Inputs changed during evidence handoff')
    freeze_inputs(output.parent, config, {'panda': frozen['seeds']['panda']})
    audit = dict(status='passed', acceptance_robots=['panda'], source_report=str(source_path.relative_to(ROOT)),
                 source_report_sha256=sha256(source_path), source_input_sha256=frozen['input_sha256'],
                 input_sha256=current, input_changes=changes, parent_m3=parent,
                 evidence_sha256=evidence, original_tests=413, delta_tests=delta['tests'],
                 physical_reruns=0, replay_reruns=0)
    audit_path = output.parent / 'teacher_release_audit.json'
    write_json(audit_path, audit)
    report = copy.deepcopy(original)
    report.update(status='incomplete', teacher_gate='passed', acceptance_robots=['panda'], input_hashes=current,
                  scope='Panda-only M4; revision-3 physical results reused with audited input differences',
                  evidence_origin=dict(report=str(source_path.relative_to(ROOT)), report_sha256=sha256(source_path),
                      input_sha256=frozen['input_sha256'], audit=str(audit_path.relative_to(ROOT)),
                      audit_sha256=sha256(audit_path)))
    report['cases']['dataset'] = dict(status='not_verified')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'report.json', report)
    return report
