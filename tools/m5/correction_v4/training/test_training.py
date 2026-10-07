import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

DIRECTORY=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v4_training_common',DIRECTORY/'common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_cuda_six_threads_without_mutating_ancestor_config(monkeypatch):
    config={'training':{'workers':2,'cpu_budget':6,'torch_threads':4}}
    calls=[]
    monkeypatch.setattr(c,'setup',lambda runtime,device:calls.append((runtime,device)))
    c.compute(config,6)
    assert calls==[({'training':{'workers':0,'cpu_budget':6,'torch_threads':6}},'cuda')]
    assert config['training']=={'workers':2,'cpu_budget':6,'torch_threads':4}
    with pytest.raises(ValueError):c.compute(config,8)


def test_resume_rejects_wrong_corpus_and_nonfinite_weights():
    checkpoint=dict(schema_version=4,fork_kind=c.FORK,binding={'data':'v4'},config={},normalization={},
        step=100001,model={'w':torch.tensor(1.)},ema={'w':torch.tensor(1.)})
    c.check_fork(checkpoint,{'data':'v4'},{},{})
    with pytest.raises(ValueError):c.check_fork(checkpoint,{'data':'v3'},{},{})
    checkpoint['ema']['w']=torch.tensor(float('nan'))
    with pytest.raises(FloatingPointError):c.check_fork(checkpoint,{'data':'v4'},{},{})


def test_training_subdirectory_does_not_change_collection_tool_hashes():
    from correction_v4.run import tools_hashes
    assert 'common.py' not in tools_hashes() and 'train.py' not in tools_hashes()
    assert set(c.entry_hashes())=={'common.py','train.py'}


def test_preflight_creates_no_model_optimizer_or_cuda_context(monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'common',c)
    spec=importlib.util.spec_from_file_location('v4_training_entry',DIRECTORY/'train.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    fail=lambda *a,**k:pytest.fail('Preflight must not construct training state')
    monkeypatch.setattr(entry,'ActionDiT',fail);monkeypatch.setattr(entry,'compute',fail)
    monkeypatch.setattr(torch.optim,'AdamW',fail)
    config={'training':{}}
    binding={'collection':{'legacy':{'source_hashes':{}}}}
    monkeypatch.setattr(entry,'evidence',lambda a:(dict(config=config,normalization={}),None,{},dict(dataset='v4'),None,binding))
    monkeypatch.setattr(entry,'MixedV4',lambda *a:SimpleNamespace(extra=SimpleNamespace(summary=lambda:{'episodes':44})))
    monkeypatch.setattr(entry,'check_unchanged',lambda b:None)
    monkeypatch.setattr(entry,'sampler_settings',lambda *a:{})
    args=SimpleNamespace(cpu_threads=6,resume=None,mode='preflight',output=str(tmp_path/'preflight'),
        base_checkpoint='base',parent_checkpoint='parent',data_report='data',audit_report='audit')
    result=entry.run(args)
    assert result['status']=='preflight_passed' and result['optimizer_created'] is False
    assert result['optimizer_updates']==0 and result['model_executed'] is False
    assert result['data']=={'episodes':44}
