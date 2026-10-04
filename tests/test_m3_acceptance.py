"""M3 release must be complete, paired, and tied to its physical M1 parent."""

import copy

import pytest

from feedingrobot.scripts import validate_m3 as m3


def reports(hashes):
    return {r:dict(status='passed', hashes_unchanged=True, input_hashes=hashes,
                   final_input_hashes=hashes, stages={'M3':'passed','M4':'not_verified'},
                   cases={n:dict(status='passed') for n in m3.CASES})
            for r in ('panda','ur5e')}


@pytest.mark.parametrize('damage', ['missing_robot', 'missing_case', 'failed_case', 'input_drift', 'm4_release'])
def test_freeze_rejects_incomplete_or_changed_evidence(tmp_path, monkeypatch, damage):
    hashes = {'input.py':'frozen'}
    data = reports(hashes)
    monkeypatch.setattr(m3, 'input_hashes', lambda: copy.deepcopy(hashes))
    monkeypatch.setattr(m3, 'parent_freeze_check', lambda: {'status':'passed'})
    if damage == 'missing_robot':
        del data['ur5e']
    elif damage == 'missing_case':
        del data['panda']['cases']['full_dynamic']
    elif damage == 'failed_case':
        data['ur5e']['cases']['viewer']['status'] = 'failed'
    elif damage == 'input_drift':
        data['panda']['final_input_hashes'] = {'input.py':'changed'}
    else:
        data['panda']['stages']['M4'] = 'passed'
    with pytest.raises((AssertionError, KeyError)):
        m3.publish_freeze(data, tmp_path, hashes)
    assert not (tmp_path/'freeze_manifest.json').exists()


def test_matrix_contains_natural_pickup_and_both_full_flows():
    assert len(m3.PHYSICAL_CASES) == 21
    assert set(m3.FULL_CASES) == {'full_static','full_dynamic'}
    assert {'pickup_lift','bowl','bowl_return','m1_regression','manifest','viewer'} <= set(m3.CASES)
