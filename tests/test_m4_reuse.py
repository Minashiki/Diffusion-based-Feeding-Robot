"""Single-Panda handoff integrity and reuse of completed dataset replays."""

import json
import sys

import pytest

from feedingrobot.data.episodes import write_json
from feedingrobot.experts import freeze, reuse
from feedingrobot.experts.gate import PRECOLLECTION_CASES, panda_teacher_passed
from feedingrobot.scripts import collect, validate_m4


def report():
    return dict(robot_id='panda', teacher_gate='passed', parent_m3={'status': 'passed'},
                baseline=dict(attempts=100, successes=100),
                recovery_baseline=dict(attempts=10, successes=10),
                cases={name: {'status': 'passed'} for name in PRECOLLECTION_CASES})


def test_panda_gate_needs_no_ur5e_report():
    r = report()
    assert panda_teacher_passed(r)
    validate_m4.update_gate(r)
    assert r['teacher_gate'] == 'passed' and r['acceptance_robots'] == ['panda']
    r['robot_id'] = 'ur5e'
    assert not panda_teacher_passed(r)


@pytest.mark.parametrize('normal,recovery', [(94, 10), (100, 9)])
def test_panda_gate_preserves_success_thresholds(normal, recovery):
    r = report()
    r['baseline']['successes'] = normal
    r['recovery_baseline']['successes'] = recovery
    assert not panda_teacher_passed(r)


def test_scope_change_records_both_hashes_and_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, 'ROOT', tmp_path)
    archive = tmp_path / 'archive'
    archive.mkdir()
    (archive / 'README.md').write_text('dual robot\n')
    (tmp_path / 'README.md').write_text('Panda only\n')
    changes = reuse.input_changes(archive, {'README.md': 'old'}, {'README.md': 'new'})
    assert changes['README.md']['old_sha256'] == 'old'
    assert changes['README.md']['new_sha256'] == 'new'
    assert '-dual robot' in changes['README.md']['diff']


@pytest.mark.parametrize('relative', ['configs/collect.json', 'configs/acceptance_m4.json',
                                      'src/feedingrobot/experts/teacher.py', 'src/feedingrobot/data/rollout.py'])
def test_scope_handoff_rejects_physics_teacher_and_sampling_changes(tmp_path, relative):
    with pytest.raises(ValueError, match='Unapproved'):
        reuse.input_changes(tmp_path, {relative: 'old'}, {relative: 'new'})


def test_scope_handoff_preserves_collection_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, 'ROOT', tmp_path)
    relative = 'src/feedingrobot/scripts/collect.py'
    archive = tmp_path / 'archive'
    before, after = archive / relative, tmp_path / relative
    before.parent.mkdir(parents=True)
    after.parent.mkdir(parents=True)
    before.write_text('def collection_trial():\n    return 1\n')
    after.write_text('def collection_trial():\n    return 2\n')
    with pytest.raises(ValueError, match='Runtime function'):
        reuse.input_changes(archive, {relative: 'old'}, {relative: 'new'})


def test_reuse_rejects_modified_original_report(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, 'ROOT', tmp_path)
    source = tmp_path / reuse.SOURCE
    source.mkdir(parents=True)
    (source / 'freeze_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='Original acceptance evidence changed'):
        reuse.reuse_teacher_report(source / 'panda/report.json', tmp_path / 'new/panda', {})


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    config = {'quotas': {'train': 1}}
    monkeypatch.setattr(collect, 'input_hashes', lambda: {'teacher': 'fixed'})
    episode = tmp_path / 'train/normal_1'
    episode.mkdir(parents=True)
    (episode / 'manifest.json').write_text('{}')
    (episode / 'actions.npy').write_bytes(b'bound recording')
    (tmp_path / 'normalization.json').write_text('{"source_split":"train"}')
    summary = dict(status='passed', replay_status='passed', counts={'train': 1}, recovery_counts={'train': 1},
                   attempts=1, input_hashes={'teacher': 'fixed'}, teacher_config=config, robot_id='panda',
                   episodes=[dict(path='train/normal_1', replay={'status': 'passed'})],
                   evidence_sha256={str(p.relative_to(tmp_path)): freeze.sha256(p)
                                    for p in tmp_path.rglob('*') if p.is_file()})
    summary['validation_sha256'] = collect.statistics_digest(summary)
    write_json(tmp_path / 'statistics.json', summary)
    return tmp_path, config, summary


def test_completed_replay_is_verified_without_physics(dataset, monkeypatch):
    directory, config, summary = dataset
    monkeypatch.setattr(collect, 'replay_many', lambda *a, **k: pytest.fail('Repeated physical replay'))
    assert collect.verified_statistics(directory, config=config, robot='panda') == summary
    assert collect.dataset_statistics(directory, config=config, robot='panda', replay=True) == summary


@pytest.mark.parametrize('change', ['recording', 'normalization', 'summary', 'extra_file', 'quotas', 'inputs'])
def test_dataset_replay_binding_rejects_changes(dataset, change, monkeypatch):
    directory, config, summary = dataset
    if change == 'recording':
        (directory / 'train/normal_1/actions.npy').write_bytes(b'changed')
    elif change == 'normalization':
        (directory / 'normalization.json').write_text('{"source_split":"test"}')
    elif change == 'summary':
        summary['counts']['train'] = 10
        write_json(directory / 'statistics.json', summary)
    elif change == 'extra_file':
        (directory / 'train/normal_1/unbound.npy').write_bytes(b'extra')
    elif change == 'quotas':
        summary['counts']['train'] = 0
        summary['validation_sha256'] = collect.statistics_digest(summary)
        write_json(directory / 'statistics.json', summary)
    else:
        monkeypatch.setattr(collect, 'input_hashes', lambda: {'teacher': 'changed'})
    with pytest.raises(ValueError):
        collect.verified_statistics(directory, config=config, robot='panda')


def test_reuse_cli_does_not_launch_acceptance(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(validate_m4, 'ROOT', tmp_path)
    write_json(tmp_path / 'config.json', {})
    monkeypatch.setattr(reuse, 'reuse_teacher_report', lambda *args: calls.append(args))
    monkeypatch.setattr(validate_m4, 'run_episode', lambda *a, **k: pytest.fail('Repeated acceptance'))
    monkeypatch.setattr(sys, 'argv', ['validate_m4', '--config', 'config.json', '--reuse-report', 'old/report.json',
                                    '--output', 'new/panda'])
    assert validate_m4.main() == 0
    assert len(calls) == 1


def test_final_publish_uses_only_panda_and_bound_statistics(tmp_path, monkeypatch):
    directory, data = tmp_path / 'acceptance', tmp_path / 'dataset'
    (directory / 'panda').mkdir(parents=True)
    data.mkdir()
    config = {'quotas': {'train': 1}}
    r = report()
    r.update(input_hashes={'teacher': 'fixed'}, teacher_config=config)
    r['cases']['dataset'] = {'status': 'passed'}
    write_json(directory / 'panda/report.json', r)
    write_json(directory / 'freeze_manifest.json', dict(input_sha256=r['input_hashes'], seeds={}, model_version='test'))
    monkeypatch.setattr(freeze, 'ROOT', tmp_path)
    monkeypatch.setattr(freeze, 'input_hashes', lambda: r['input_hashes'])
    monkeypatch.setattr(freeze, 'freeze_inputs', lambda *args: None)
    monkeypatch.setattr(freeze, 'parent_m3_check', lambda **kwargs: {'status': 'passed'})
    monkeypatch.setattr(collect, 'verified_statistics', lambda *args, **kwargs:
                        dict(counts={'train': 1}, recovery_counts={'train': 1}, attempts=2))
    monkeypatch.setattr(collect, 'replay_many', lambda *a, **k: pytest.fail('Repeated replay'))
    audit = freeze.publish_acceptance(directory, config, data)
    assert audit['status'] == 'passed' and audit['acceptance_robots'] == ['panda']
    final = json.loads((directory / 'freeze_manifest.json').read_text())
    assert final['status'] == 'frozen' and set(final['reports']) == {'panda'}


@pytest.fixture
def reuse_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, 'ROOT', tmp_path)
    monkeypatch.setattr(reuse, 'input_hashes', lambda: {'current': 'hash'})
    original = report()
    original.update(teacher_config={'teacher': 'v2'}, input_hashes={'old': 'hash'}, convergence_cases=[])
    source = tmp_path / 'old_report.json'
    write_json(source, original)
    source_hash = freeze.sha256(source)
    monkeypatch.setattr(reuse, 'SOURCE_SHA256', {'panda/report.json': source_hash})
    audit = dict(status='passed', input_sha256={'current': 'hash'}, source_input_sha256={'old': 'hash'},
                 source_report_sha256=source_hash, source_report='old_report.json',
                 evidence_sha256={'old_report.json': source_hash})
    audit_path = tmp_path / 'audit.json'
    write_json(audit_path, audit)
    r = json.loads(json.dumps(original))
    r['input_hashes'] = {'current': 'hash'}
    r['evidence_origin'] = dict(audit='audit.json', audit_sha256=freeze.sha256(audit_path),
                              input_sha256={'old': 'hash'})
    return tmp_path, r


def test_reuse_audit_keeps_original_and_current_inputs_separate(reuse_audit):
    _, r = reuse_audit
    result = reuse.verify_reuse_audit(r)
    assert result['source_input_sha256'] != result['input_sha256']


@pytest.mark.parametrize('change', ['audit', 'teacher', 'result', 'source', 'current_inputs'])
def test_reuse_audit_rejects_modified_evidence_or_report(reuse_audit, change, monkeypatch):
    directory, r = reuse_audit
    if change == 'audit':
        (directory / 'audit.json').write_text('{}')
    elif change == 'teacher':
        r['teacher_config']['teacher'] = 'changed'
    elif change == 'result':
        r['cases']['replay']['status'] = 'failed'
    elif change == 'source':
        original = json.loads((directory / 'old_report.json').read_text())
        original['unbound_change'] = True
        write_json(directory / 'old_report.json', original)
    else:
        monkeypatch.setattr(reuse, 'input_hashes', lambda: {'current': 'changed'})
    with pytest.raises(ValueError):
        reuse.verify_reuse_audit(r)
