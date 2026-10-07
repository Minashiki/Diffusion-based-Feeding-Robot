"""Reject incomplete, duplicate, and nonfinite training evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('check_trained',Path(__file__).with_name('check_trained.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def records(tmp_path,rows):
    path=tmp_path/'metrics.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    return path


def test_continuous_finite_training_records(tmp_path):
    rows=[dict(step=s,loss=.01,gradient_norm=.2) for s in range(50001,50004)]
    rows[-1]['validation_loss']=.009
    report=module.check_records(records(tmp_path,rows),50003)
    assert report['rows']==3 and report['continuous'] and report['finite']
    assert report['validation']==[rows[-1]]


@pytest.mark.parametrize('steps',[[50001,50003],[50001,50002,50002],[50002,50003]])
def test_missing_or_duplicate_records_rejected(tmp_path,steps):
    rows=[dict(step=s,loss=.01,gradient_norm=.2) for s in steps]
    with pytest.raises(ValueError,match='continuous'):
        module.check_records(records(tmp_path,rows),50003)


@pytest.mark.parametrize('field',['loss','gradient_norm','validation_loss'])
def test_nonfinite_record_rejected(tmp_path,field):
    row=dict(step=50001,loss=.01,gradient_norm=.2);row[field]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        module.check_records(records(tmp_path,[row]),50001)
