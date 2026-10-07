"""Correction provenance, invalid labels, and deterministic mixed batches."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))
import correction_data as data
import train_correction as train


def example():
    metadata=dict(success=True,accepted_normal=True,qualified_correction=True,split='train',
        m5_extension=True,scenario=dict(recover=False),perturbation=dict(pulse_start_tick=50,release_tick=150))
    arrays=dict(action_mask=np.array([True,False,False,True]),teacher_owned=np.array([True,False,False,True]),
        action_ticks=np.array([0,50,100,150]))
    for name in ('physics','observations','action_observations','actions','proposals'):
        arrays[name]=np.zeros((4,6))
    return metadata,arrays


def test_labels_exclude_pulse_settling_and_nonfinite():
    metadata,arrays=example()
    assert data.label_check(metadata,arrays)['valid_labels']==2
    arrays['action_mask'][1]=True
    with pytest.raises(ValueError,match='Perturbation'): data.label_check(metadata,arrays)
    arrays['action_mask'][1]=False;arrays['observations'][0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'): data.label_check(metadata,arrays)


@pytest.mark.parametrize('field,value',[('success',False),('qualified_correction',False),('split','validation')])
def test_incomplete_or_validation_episode_rejected(field,value):
    metadata,arrays=example();metadata[field]=value
    with pytest.raises(ValueError,match='complete'): data.label_check(metadata,arrays)


def test_pulse_mask_checked_even_if_ownership_tampered():
    metadata,arrays=example();arrays['action_mask'][1]=True;arrays['teacher_owned'][1]=True
    with pytest.raises(ValueError,match='Pulse'): data.label_check(metadata,arrays)


class FakeWindows:
    def __init__(self,kind): self.kind=kind
    def sample_indices(self,rng,count): return rng.integers(0,100,count).tolist()
    def __getitem__(self,index): return dict(kind=torch.tensor(self.kind),index=torch.tensor(index))


def test_mixed_batch_exact_fraction_and_paired_resume_rng():
    mixed=object.__new__(data.MixedCorrection)
    mixed.base=FakeWindows(0);mixed.extra=FakeWindows(1);mixed.pools={0:[3,4],1:[5,6]}
    rng=np.random.default_rng(12);state=rng.bit_generator.state
    a=mixed.batch(rng,64);rng.bit_generator.state=state;b=mixed.batch(rng,64)
    assert int(a['kind'].sum())==16
    assert torch.equal(a['kind'],b['kind']) and torch.equal(a['index'],b['index'])
    assert set(a['index'][a['kind']==1].tolist())<={3,4,5,6}
    with pytest.raises(ValueError,match='divisible'): mixed.batch(rng,6)


def test_warm_start_uses_ema_not_training_weights():
    model=torch.nn.Linear(2,1)
    base=dict(ema={k:torch.ones_like(v) for k,v in model.state_dict().items()},
        model={k:torch.zeros_like(v) for k,v in model.state_dict().items()})
    train.initialize(model,base)
    assert all(torch.equal(v,base['ema'][k]) for k,v in model.state_dict().items())
    optimizer=torch.optim.AdamW(model.parameters())
    assert not optimizer.state


def test_resume_rejects_formal_wrong_data_and_wrong_directory(tmp_path):
    binding=dict(data_report_sha256='a',tool_hashes={'train_correction.py':'b'})
    checkpoint=dict(schema_version=2,diagnostic=True,fork_kind=train.FORK,binding=binding,step=50001)
    train.fork_check(checkpoint,binding,tmp_path,tmp_path/'last.pt')
    with pytest.raises(ValueError,match='ancestry'): train.fork_check(checkpoint,dict(binding,data_report_sha256='x'),tmp_path,tmp_path/'last.pt')
    with pytest.raises(ValueError,match='directory'): train.fork_check(checkpoint,binding,tmp_path/'other',tmp_path/'last.pt')
    with pytest.raises(ValueError,match='ancestry'): train.fork_check(dict(checkpoint,diagnostic=False),binding,tmp_path,tmp_path/'last.pt')
    with pytest.raises(ValueError,match='ancestry'): train.fork_check(checkpoint,dict(binding,tool_hashes={}),tmp_path,tmp_path/'last.pt')


def test_correction_pools_only_valid_post_release_acquire(monkeypatch):
    class Windows:
        windows=[(0,0,1,1),(0,1,2,1),(0,2,3,1),(0,3,4,2)]
        episodes=[(Path('episode'),dict(perturbation=dict(release_tick=100),corrected_tick=150))]
        def __init__(self,*args): pass
        def arrays(self,e): return dict(action_ticks=np.array([50,100,150,200]))
    monkeypatch.setattr(data,'ActionWindows',Windows)
    mixed=data.MixedCorrection(dict(dataset='ignored',model=dict(horizon=16)),{},dict(dataset='ignored'))
    assert mixed.pools=={0:[1,2]}


@pytest.fixture
def audited_corpus(tmp_path,monkeypatch):
    import json
    from feedingrobot.policies.audit import sha256
    root=tmp_path/'extra';original=tmp_path/'original';original.mkdir();root.mkdir()
    for p in (original/'normalization.json',root/'normalization.json'): p.write_text('{}')
    checkpoint_path=tmp_path/'base.pt';checkpoint_path.write_bytes(b'base')
    checkpoint=dict(config=dict(dataset='original'),parent=dict(dataset_binding='parent'),normalization={})
    accepted=[]
    for i in range(4):
        directory=root/'train'/f'episode_{i}';directory.mkdir(parents=True)
        (directory/'manifest.json').write_text('{}')
        accepted.append(dict(path=f'train/episode_{i}',replay=dict(status='passed')))
    report=dict(mode='correction_dataset',status='ready_experimental',parent=checkpoint['parent'],normalization={},
        checkpoint=str(checkpoint_path),checkpoint_sha256=sha256(checkpoint_path),dataset=str(root),accepted=accepted,
        dataset_sha256={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()})
    monkeypatch.setattr(data,'ROOT',tmp_path);monkeypatch.setattr(data,'check_checkpoint',lambda x:None)
    monkeypatch.setattr(data,'verify_evidence',lambda path,digest:report)
    def load(path):
        m,a=example();m.update(seed=int(path.name.split('_')[-1])+9000,group_id=path.name)
        return m,a
    monkeypatch.setattr(data,'load_episode',load)
    return checkpoint,checkpoint_path,tmp_path/'report.json',report,root,original


def test_dataset_file_and_episode_coverage_are_exact(audited_corpus):
    checkpoint,path,report_path,report,root,original=audited_corpus
    assert data.audit_data(checkpoint,path,report_path)==report
    (root/'unexpected.txt').write_text('new')
    with pytest.raises(ValueError,match='file coverage'): data.audit_data(checkpoint,path,report_path)
    (root/'unexpected.txt').unlink();report['accepted'].pop()
    with pytest.raises(ValueError,match='episode coverage'): data.audit_data(checkpoint,path,report_path)


def test_dataset_changed_file_and_normalization_rejected(audited_corpus):
    checkpoint,path,report_path,report,root,original=audited_corpus
    (root/'train/episode_0/manifest.json').write_text('changed')
    with pytest.raises(ValueError,match='dataset changed'): data.audit_data(checkpoint,path,report_path)
    (root/'train/episode_0/manifest.json').write_text('{}')
    (original/'normalization.json').write_text('changed')
    with pytest.raises(ValueError,match='normalization changed'): data.audit_data(checkpoint,path,report_path)


def test_original_split_seed_leakage_rejected(audited_corpus):
    import json
    checkpoint,path,report_path,report,root,original=audited_corpus
    directory=original/'validation'/'episode';directory.mkdir(parents=True)
    (directory/'manifest.json').write_text(json.dumps(dict(seed=9000,group_id='validation_group')))
    with pytest.raises(ValueError,match='split leakage'): data.audit_data(checkpoint,path,report_path)


def test_unreplayed_and_wrong_ancestry_rejected(audited_corpus):
    checkpoint,path,report_path,report,root,original=audited_corpus
    report['accepted'][0]['replay']['status']='failed'
    with pytest.raises(ValueError,match='Unverified'): data.audit_data(checkpoint,path,report_path)
    report['normalization']={'changed':True}
    with pytest.raises(ValueError,match='ancestry'): data.audit_data(checkpoint,path,report_path)


def test_tiny_model_checkpoint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    """Three toy updates test optimizer/RNG restore, never the 80M checkpoint."""
    import copy
    from types import SimpleNamespace
    from feedingrobot.policies.audit import read_json,checkpoint_check
    from feedingrobot.policies.dit import ActionDiT
    from feedingrobot.sim.model import ROOT
    config=copy.deepcopy(read_json(ROOT/'configs/dp_dit.json'))
    config['model'].update(hidden_size=16,heads=2,depth=1,mlp_hidden=32,horizon=2)
    config['training']['batch_size']=4
    model=ActionDiT(config)
    path=tmp_path/'base.pt';torch.save(dict(config=config,ema=model.state_dict(),normalization={}),path)
    data_report=tmp_path/'data.json';data_report.write_text('{}')
    monkeypatch.setattr(train,'check_checkpoint',lambda x:dict(dataset_binding='toy'))
    monkeypatch.setattr(train,'audit_data',lambda *args: {})
    monkeypatch.setattr(train,'input_hashes',lambda:{'original':'fixed'})
    class Mixed:
        def __init__(self,*args): pass
        def summary(self): return dict(correction_fraction=.25)
        def batch(self,rng,count):
            return dict(states=torch.zeros(count,2,122),history=torch.zeros(count,10,28),
                phase=torch.ones(count,dtype=torch.long),interaction=torch.zeros(count,4),
                state_mask=torch.ones(count,2,dtype=torch.bool),history_mask=torch.ones(count,10,dtype=torch.bool),
                actions=torch.tensor(rng.normal(size=(count,2,6)),dtype=torch.float32),
                action_mask=torch.ones(count,2,dtype=torch.bool))
    monkeypatch.setattr(train,'MixedCorrection',Mixed)
    monkeypatch.setattr(train,'ActionWindows',lambda *args:None)
    def args(output,updates,resume=None,mode='train'):
        return SimpleNamespace(checkpoint=str(path),data_report=str(data_report),output=str(output),
            updates=updates,resume=resume,mode=mode,device='cpu')
    audit=train.run(args(tmp_path/'audit',None,mode='audit'))
    assert audit['optimizer_updates']==0 and audit['ema_exact'] and not audit['optimizer_created']
    train.run(args(tmp_path/'continuous',50003))
    train.run(args(tmp_path/'split',50002))
    train.run(args(tmp_path/'split',50003,str(tmp_path/'split/last.pt')))
    a=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    b=torch.load(tmp_path/'split/last.pt',weights_only=False)
    assert a['step']==b['step']==50003
    for key in ('model','ema'):
        for name in a[key]: torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
    assert a['rng']['sampler']==b['rng']['sampler']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    with pytest.raises(ValueError): checkpoint_check(a,config,dict(dataset_binding='toy'))
