"""Toy-only update/resume tests and physical diagnostic replay; no production training."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).parent))

import common
import evaluate
import train
import numpy as np
import pytest
import torch

from feedingrobot.data.episodes import load_episode
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.sim.model import ROOT


def test_collection_bindings_remain_unchanged():
    from correction_v3.run import v3_hashes,unchanged
    report=read_json(ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_dataset_001/report.json')
    assert v3_hashes()==report['binding']['v3_tools']
    unchanged(report['binding'])


@pytest.mark.parametrize('threads',[0,5,9,16])
def test_cpu_budget_outside_six_to_eight_rejected(threads):
    with pytest.raises(ValueError,match='6..8'):common.compute({},threads)


def test_compute_uses_cuda_and_preserves_original_config(monkeypatch):
    original=dict(training=dict(cpu_budget=6,torch_threads=4,workers=2))
    captured={}
    def setup(config,device): captured.update(config=config,device=device);return None,{}
    monkeypatch.setattr(common,'setup',setup)
    common.compute(original,8)
    assert captured['device']=='cuda'
    assert captured['config']['training']==dict(cpu_budget=8,torch_threads=8,workers=0)
    assert original['training']['torch_threads']==4


@pytest.mark.parametrize('field,value',[('status','failed'),('mode','correction_warmstart_audit_v2'),
    ('step',50000),('optimizer_updates',1),('optimizer_created',True),('ema_exact',False),('data_report_sha256','wrong')])
def test_unmatched_zero_update_audit_blocks_training(monkeypatch,field,value):
    binding=dict(parent_checkpoint_sha256='fixed')
    monkeypatch.setattr(common,'ancestry',lambda args:({'config':{'dataset':'unused'},'normalization':{}},{},{},None,[],None,binding))
    monkeypatch.setattr(common,'audit_corpus',lambda *args:dict(calibration_report='cal',calibration_report_sha256='fixed'))
    monkeypatch.setattr(common,'calibration_check',lambda *args:{})
    monkeypatch.setattr(common,'sha256',lambda *args:'fixed')
    audit=dict(mode='correction_warmstart_audit_v3',status='audit_passed',binding=binding,step=100000,
        optimizer_created=False,optimizer_updates=0,model_exact=True,ema_exact=True,data_report_sha256='fixed')
    audit[field]=value
    monkeypatch.setattr(common,'verify_evidence',lambda *args:audit)
    args=SimpleNamespace(base_checkpoint='base',parent_checkpoint='parent',v1_report='v1',data_report='v3',audit_report='audit',cpu_threads=6)
    with pytest.raises(ValueError,match='zero-update'):common.evidence(args)


@pytest.fixture
def toy(tmp_path,monkeypatch):
    config=deepcopy(read_json(ROOT/'configs/dp_dit.json'))
    config['model'].update(hidden_size=16,heads=2,depth=1,mlp_hidden=32,horizon=2)
    config['training'].update(batch_size=4,validation_every=1000,checkpoint_every=2)
    torch.manual_seed(7);model=ActionDiT(config)
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    parent=dict(ema=ema,model={k:torch.zeros_like(v) for k,v in ema.items()},optimizer={'must_not_load':True},rng={'must_not_load':True})
    binding=dict(collection=dict(source_hashes={'frozen':'yes'}),entry_tools=common.entry_hashes())
    monkeypatch.setattr(train,'evidence',lambda args:(dict(config=config,normalization={}),parent,{}, {},{},binding))
    monkeypatch.setattr(train,'check_unchanged',lambda binding:None)
    monkeypatch.setattr(train,'compute',lambda *args:(torch.device('cpu'),dict(device='toy_cpu_only')))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
    monkeypatch.setattr(train,'ActionWindows',lambda *args:None)
    captured=[];real_load=ActionDiT.load_state_dict
    def capture(self,weights,*args,**kwargs):
        captured.append({k:v.clone() for k,v in weights.items()});return real_load(self,weights,*args,**kwargs)
    monkeypatch.setattr(ActionDiT,'load_state_dict',capture)
    class Mixed:
        def __init__(self,*args):pass
        def summary(self):return dict(correction_fraction=.25)
        def batch(self,rng,count):
            return dict(states=torch.zeros(count,2,122),history=torch.zeros(count,10,28),
                phase=torch.ones(count,dtype=torch.long),interaction=torch.zeros(count,4),
                state_mask=torch.ones(count,2,dtype=torch.bool),history_mask=torch.ones(count,10,dtype=torch.bool),
                actions=torch.tensor(rng.normal(size=(count,2,6)),dtype=torch.float32),
                action_mask=torch.ones(count,2,dtype=torch.bool))
    monkeypatch.setattr(train,'MixedV3',Mixed)
    def args(name,updates=None,resume=None,mode='train'):
        return SimpleNamespace(base_checkpoint='formal',parent_checkpoint='parent',v1_report='v1',
            data_report='v3',audit_report='audit',output=str(tmp_path/name),cpu_threads=6,
            updates=updates,resume=resume,mode=mode)
    return args,config,binding,ema,captured


def test_preflight_has_no_model_optimizer_or_backward(toy,monkeypatch):
    args,_,_,_,_=toy
    def forbidden(*args,**kwargs):raise AssertionError('Preflight must not train or build a model')
    monkeypatch.setattr(train,'ActionDiT',forbidden);monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    monkeypatch.setattr(torch.Tensor,'backward',forbidden);monkeypatch.setattr(train,'compute',forbidden)
    result=train.run(args('preflight',mode='preflight'))
    assert result['step']==100000 and result['optimizer_updates']==0 and not result['model_executed']


def test_toy_warmstart_and_exact_resume(toy):
    args,config,binding,ema,captured=toy
    train.run(args('continuous',100003))
    assert all(torch.equal(captured[0][k],v) for k,v in ema.items())
    train.run(args('split',100002))
    path=Path(args('split').output)/'last.pt'
    train.run(args('split',100003,str(path)))
    a=torch.load(Path(args('continuous').output)/'last.pt',weights_only=False)
    b=torch.load(path,weights_only=False)
    for key in ('model','ema'):
        for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert a['rng']['sampler']==b['rng']['sampler']
    assert a['step']==b['step']==100003 and b['optimizer_updates']==3
    common.check_fork(b,binding,config,{})
    with pytest.raises(ValueError,match='mismatch'):common.check_fork(dict(b,fork_kind='m5_rotation_correction_v1'),binding,config,{})
    with pytest.raises(ValueError,match='mismatch'):common.check_fork(b,dict(binding,entry_tools={}),config,{})
    corrupted=deepcopy(b);next(iter(corrupted['ema'].values())).fill_(float('nan'))
    with pytest.raises(FloatingPointError):common.check_fork(corrupted,binding,config,{})


def test_cuda_unavailable_never_creates_training_output(toy,monkeypatch):
    args,_,_,_,_=toy
    def unavailable(*args):raise RuntimeError('CUDA unavailable; no CPU fallback')
    monkeypatch.setattr(train,'compute',unavailable)
    with pytest.raises(RuntimeError,match='CUDA'):train.run(args('cuda_missing',100001))
    assert not Path(args('cuda_missing').output).exists()


def test_targets_and_resume_directory_guard(toy):
    args,_,_,_,_=toy
    with pytest.raises(ValueError,match='cumulative'):train.run(args('bad',100000))
    with pytest.raises(ValueError,match='directory'):train.run(args('bad',100003,'other/last.pt'))


def test_discontinuous_or_nonfinite_metrics_rejected(tmp_path):
    path=tmp_path/'metrics.jsonl'
    for rows in ([dict(step=100002,loss=0.,gradient_norm=1.)],
                 [dict(step=100001,loss=float('nan'),gradient_norm=1.)]):
        path.write_text('\n'.join(json.dumps(r) for r in rows))
        with pytest.raises(ValueError,match='continuous'):train.check_records(path,100001)


def test_alignment_requires_original_height_and_continuous_stability(monkeypatch):
    r=dict(angle_deg=.3,lateral_m=.0004,linear_speed=.001,angular_speed=.02)
    monkeypatch.setattr(evaluate,'residual',lambda *args:dict(r))
    obs=dict(tcp_position=[0,0,.05])
    _,_,stable,done=evaluate.alignment_measure(None,obs,.05,100,None)
    assert stable==100 and not done
    assert evaluate.alignment_measure(None,obs,.05,300,stable)[3]
    r['angular_speed']=.1
    assert evaluate.alignment_measure(None,obs,.05,300,stable)[2] is None
    r['angular_speed']=.02;obs['tcp_position'][2]+=.00071
    assert not evaluate.alignment_measure(None,obs,.05,500,100)[3]


def test_small_and_full_cases_are_validation_and_independent_calibration():
    config=read_json(ROOT/'configs/dp_dit.json')
    root=ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_calibration_001'
    calibration=dict(read_json(root/'report.json'),calibration_directory=str(root))
    small=evaluate.select_cases(config,calibration,False);full=evaluate.select_cases(config,calibration,True)
    assert len(small)==8 and len(full)==38
    assert {read_json(p/'manifest.json')['seed'] for p,c in small if c}=={780001,780003,780005,780007}
    assert all(read_json(p/'manifest.json')['split']=='validation' for p,c in full if not c)


def test_real_low_speed_suffix_is_model_owned_and_replays(tmp_path,monkeypatch):
    config=read_json(ROOT/'configs/dp_dit.json')
    source=ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_calibration_001/episodes/780003_calibration'
    m=read_json(source/'manifest.json')
    normalization=read_json(source.parents[1]/'report.json')['normalization']
    conditions=[]
    def sample(model,config,batch,norm,generator):
        conditions.append(batch);return torch.zeros(1,16,6)
    monkeypatch.setattr(evaluate,'sample_actions',sample)
    def forbidden(*args):raise AssertionError('Teacher must not execute after handover')
    monkeypatch.setattr(evaluate.Teacher,'act',forbidden)
    output=tmp_path/'suffix'
    result=evaluate.rollout(None,config,normalization,source,output,torch.device('cpu'),
        correction=True,viewer=False,max_ticks=m['handover']['release_tick']+300)
    assert result['initial_tick']==m['handover']['release_tick']
    assert conditions[0]['state_mask'].all() and conditions[0]['history_mask'].all()
    assert result['external_braking']['prefix_commands']>0
    assert not result['external_braking']['autonomous_high_speed_braking_proven']
    assert not result['model_alignment']['success']
    _,arrays=load_episode(output)
    assert not arrays['action_mask'].any() and set(arrays['action_owner'])=={'model'}
    assert replay_episode(output)['status']=='passed'


def test_viewer_failure_is_explicit(tmp_path,monkeypatch):
    from feedingrobot.sim import observer_viewer
    class Unavailable:
        def __init__(self,*args):pass
        def start(self):return dict(status='unavailable')
        def close(self):return {}
    monkeypatch.setattr(observer_viewer,'ObserverViewer',Unavailable)
    config=read_json(ROOT/'configs/dp_dit.json')
    source=ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_calibration_001/episodes/780001_calibration'
    with pytest.raises(RuntimeError,match='viewer unavailable'):
        evaluate.rollout(None,config,{},source,tmp_path/'visible',torch.device('cpu'),correction=True)


def test_real_height_violation_brakes_without_teacher_and_replays(tmp_path,monkeypatch):
    import mink
    config=read_json(ROOT/'configs/dp_dit.json')
    source=ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_calibration_001/episodes/780003_calibration'
    m=read_json(source/'manifest.json');normalization=read_json(source.parents[1]/'report.json')['normalization']
    rotation=mink.SO3(np.asarray(read_json(ROOT/'configs/robots/panda.json')['base_quaternion'])).as_matrix()
    action=np.r_[rotation.T@np.array([0.,0.,-.02]),np.zeros(3)]
    monkeypatch.setattr(evaluate,'sample_actions',lambda *args:torch.tensor(action,dtype=torch.float32).repeat(1,16,1))
    output=tmp_path/'unsafe'
    result=evaluate.rollout(None,config,normalization,source,output,torch.device('cpu'),
        correction=True,viewer=False,max_ticks=m['handover']['release_tick']+1000)
    assert result['external_braking']['interventions'][0]['reason']=='alignment_height_drift'
    assert result['external_braking']['commands']>0 and not result['model_alignment']['success']
    assert not result['success'] and result['model_alignment']['maximum_height_drift_m']>=.0007
    assert replay_episode(output)['status']=='passed'
