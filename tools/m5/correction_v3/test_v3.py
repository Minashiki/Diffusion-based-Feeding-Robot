"""Regression gates for retained momentum, safety stops and frozen ancestry."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from correction_v2.test_v2 import fake_teacher,obs,label_example,physical_config
from correction_v3 import controller as c,corpus,rollout,run
from feedingrobot.data.episodes import load_episode,write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.sim.model import ROOT


@pytest.mark.parametrize('speed,reference,expected',[(.05,.05,False),(.005,.0045,True),(.004,.0,True),
    (.002,.0,False),(.0,.0,False),(.004,.005,False)])
def test_moving_release_requires_measured_nonzero_low_speed_and_reference(speed,reference,expected):
    assert c.moving_release(obs(vz=-speed),np.array([0.,0.,-reference,0.,0.,0.]))==expected


def test_geometry_qualification_remains_separate_from_safety(monkeypatch):
    teacher=fake_teacher();progress=c.Progress();progress.above=True
    task=SimpleNamespace(adapter=SimpleNamespace(velocity=np.zeros(6)),robot_config={'linear_acceleration_limit':.5})
    monkeypatch.setattr(c,'clearance',lambda task:.026)
    outside=obs(y=-.002014,vz=-.05)
    assert not progress.candidate('p1_moving',teacher,outside)
    assert c.safety_gate(task,outside)['stop_required']
    inside=obs(y=-.0015,vz=-.05)
    assert progress.candidate('p1_moving',teacher,inside)
    assert c.safety_gate(task,inside)['stop_required']
    outside['stage']='TRANSPORT'
    assert not c.safety_gate(task,outside)['stop_required']


def test_high_speed_geometry_screen_reserves_physical_braking(monkeypatch):
    task=SimpleNamespace(adapter=SimpleNamespace(velocity=np.array([0,0,-.05,0,0,0])))
    original=dict(minimum_m=.03,reserve_m=.005,net_m=.025,required_m=.002,passed=True)
    monkeypatch.setattr(c,'v2_swept_clearance',lambda *args:deepcopy(original))
    gate=c.swept_clearance(task,fake_teacher(),obs(vz=-.05))
    assert gate['reserve_m']==pytest.approx(.025)
    assert gate['net_m']==pytest.approx(.005) and gate['passed']
    assert original['reserve_m']==.005


def v3_labels():
    m,a=label_example();m.update(teacher_version=c.VERSION,safety_checks=[],alignment_max_drift_m=.0003)
    m['handover']['release_velocity']=np.zeros(6).tolist()
    return m,a


def test_braking_and_saved_prefixes_cannot_become_training_labels():
    m,a=v3_labels();assert corpus.check_labels(m,a)['valid_labels']==2
    a['action_mask'][2]=True
    with pytest.raises(ValueError,match='Non-teacher'): corpus.check_labels(m,a)
    m,a=v3_labels();m['prefix_source']='diagnostic'
    with pytest.raises(ValueError,match='actual DP prefix'): corpus.check_labels(m,a)
    m,a=v3_labels();m['safety_checks']=[dict(stop_required=True)]
    with pytest.raises(ValueError,match='Safety intervention'): corpus.check_labels(m,a)


@pytest.mark.parametrize('speed,reference',[(.05,0.),(.004,.005)])
def test_corpus_rejects_unverified_moving_release(speed,reference):
    m,a=v3_labels();m['category']='p1_moving'
    m['handover']['release_residual'].update(downward_speed=speed,linear_speed=speed)
    m['handover']['release_velocity'][2]=-reference
    with pytest.raises(ValueError,match='category gates|moving reference'): corpus.check_labels(m,a)


def test_calibration_requires_perturbation_before_high_speed_braking():
    case=dict(category='p1_moving',alignment_max_drift_m=.0003,handover=dict(pulse_end_tick=100,
        trigger_tick=200,release_tick=500,trigger_residual=dict(downward_speed=.05),
        release_residual=dict(downward_speed=.004,linear_speed=.004),release_velocity=np.zeros(6)))
    assert run.calibrated_release(case)
    case['handover']['trigger_residual']['downward_speed']=.0028
    assert not run.calibrated_release(case)
    case['handover']['trigger_residual']['downward_speed']=.05
    case['handover']['pulse_end_tick']=300
    assert not run.calibrated_release(case)


def test_seed_isolation_reads_historical_accepted_data(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    path=tmp_path/'outputs/single_bean/v1/m5/dit/correction_v2_dataset_001/data/train/correction_780101'
    path.mkdir(parents=True);write_json(path/'manifest.json',dict(seed=780101))
    with pytest.raises(ValueError,match='overlaps'):run.seed_check([780101],[],dict(accepted=[],rejected=[]))


@pytest.mark.parametrize('bf16',(False,True))
def test_collection_refuses_cpu_or_precision_fallback(monkeypatch,tmp_path,bf16):
    monkeypatch.setattr(run,'ancestry',lambda args:({'config':{}},None,{},None,[],None,{}))
    monkeypatch.setattr(run,'calibration_check',lambda *args:None)
    monkeypatch.setattr(run,'seed_check',lambda *args:None)
    if bf16:
        monkeypatch.setattr(run,'setup',lambda *args:(torch.device('cuda'),{}))
        monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
    else:
        def unavailable(*args):raise RuntimeError('CUDA unavailable')
        monkeypatch.setattr(run,'setup',unavailable)
    args=SimpleNamespace(calibration_report='unused',output=str(tmp_path/'data'))
    with pytest.raises(RuntimeError):run.collect(args)
    assert not Path(args.output).exists()


def test_saved_real_momentum_brakes_without_reset_and_replays(tmp_path,physical_config,monkeypatch):
    source=ROOT/'outputs/single_bean/v1/m5/dit/correction_v2_dataset_001/attempts/770005_candidate'
    m=read_json(source/'manifest.json');directory=tmp_path/'saved';captured={};cancellations=[]
    real_env=rollout.FeedingGymEnv
    def make_env(*args,**kwargs):
        captured['env']=real_env(*args,**kwargs);return captured['env']
    def cancel(chunk):
        task=captured['env'].task;cancellations.append(task.adapter.velocity.copy());return None
    monkeypatch.setattr(rollout,'FeedingGymEnv',make_env);monkeypatch.setattr(rollout,'cancel_chunk',cancel)
    case=rollout.episode(770005,'p1_moving',m['teacher_config'],m['scenario'],directory,
        source='diagnostic',resume_from=source,max_s=6.)
    h=case['handover'];assert abs(cancellations[0][2])>.049
    assert case['qualified_correction'] and not case['abort_reason'] and h['release_tick']-h['trigger_tick']==300
    assert .002<h['release_residual']['downward_speed']<=.005
    assert .006<h['brake_max_drop_m']<.008 and case['alignment_max_drift_m']<.0007
    _,arrays=load_episode(directory)
    assert np.any(arrays['action_owner']=='external_brake')
    assert not arrays['action_mask'][arrays['action_owner']=='external_brake'].any()
    assert replay_episode(directory)['status']=='passed'


@pytest.mark.parametrize('seed',(770007,770010))
def test_unqualified_real_prefix_stops_before_collision(tmp_path,physical_config,seed):
    source=ROOT/f'outputs/single_bean/v1/m5/dit/correction_v2_dataset_001/attempts/{seed}_candidate'
    m=read_json(source/'manifest.json');directory=tmp_path/str(seed)
    case=rollout.episode(seed,m['category'],m['teacher_config'],m['scenario'],directory,
        source='diagnostic',prefix_from=source,max_s=12.)
    assert case['abort_reason']=='unsafe_unqualified_prefix' and not case['failure_reason']
    assert case['handover'].get('release_tick') is None and not case['qualified_correction']
    assert case['simulated_s']<m['simulated_s'] and case['contact_peak_n']<1.
    assert case['handover']['safety_stop_tick']>case['handover']['safety_tick']
    assert replay_episode(directory)['status']=='passed'


def test_v3_zero_update_audit_loads_only_100k_ema(tmp_path,monkeypatch):
    config=deepcopy(read_json(ROOT/'configs/dp_dit.json'))
    config['model'].update(hidden_size=16,heads=2,depth=1,mlp_hidden=32,horizon=2)
    model=ActionDiT(config);ema={k:torch.ones_like(v)*.01 for k,v in model.state_dict().items()}
    trained=dict(ema=ema,model={k:torch.zeros_like(v) for k,v in ema.items()})
    binding=dict(parent_checkpoint_sha256='parent',normalization={})
    monkeypatch.setattr(run,'ancestry',lambda args:(dict(config=config,normalization={}),trained,{}, {},[],{},binding))
    calibration=tmp_path/'calibration.json';calibration.write_text('{}')
    report=dict(calibration_report=str(calibration),calibration_report_sha256=sha256(calibration),dataset='ignored')
    monkeypatch.setattr(run,'audit_corpus',lambda *args:report)
    monkeypatch.setattr(run,'calibration_check',lambda *args:{})
    item=dict(states=torch.zeros(2,122),history=torch.zeros(10,28),phase=torch.tensor(1),interaction=torch.zeros(4),
        state_mask=torch.ones(2,dtype=torch.bool),history_mask=torch.ones(10,dtype=torch.bool),
        actions=torch.zeros(2,6),action_mask=torch.ones(2,dtype=torch.bool))
    class Mixed:
        pools=dict(alignment={0:[0]},transition={0:[0]},aligned={1:[0]})
        extra=[item]
        def __init__(self,*args): pass
        def batch(self,rng,count):
            from torch.utils.data import default_collate
            assert count==64
            return default_collate([item]*count)
        def summary(self): return dict(correction_fraction=.25)
    monkeypatch.setattr(run,'MixedV3',Mixed)
    def forbidden(*args,**kwargs): raise AssertionError('Optimizer/backward must not run')
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    monkeypatch.setattr(torch.Tensor,'backward',forbidden)
    monkeypatch.setattr(run,'start_output',lambda *args:(tmp_path,dict(optimizer_created=False,optimizer_updates=0)))
    monkeypatch.setattr(run,'finish',lambda output,provenance,binding,**kwargs:dict(provenance,**kwargs))
    (tmp_path/'data.json').write_text('{}')
    args=SimpleNamespace(data_report=str(tmp_path/'data.json'))
    result=run.audit(args)
    assert result['step']==100000 and result['model_exact'] and result['ema_exact']
    assert result['optimizer_updates']==0 and not result['optimizer_created'] and result['batch_size']==64



def test_unpassed_diagnosis_blocks_new_calibration(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ancestry',lambda args:({},None,{},None,[],None,{'v3_tools':{}}))
    monkeypatch.setattr(run,'v2_calibration_check',lambda *args:None)
    def refuse(*args):raise ValueError('Unpassed intervention diagnosis')
    monkeypatch.setattr(run,'diagnosis_check',refuse)
    args=SimpleNamespace(previous_report='old',diagnosis_report='failed',output=str(tmp_path/'calibration'))
    with pytest.raises(ValueError,match='Unpassed'):run.calibrate(args)
    assert not Path(args.output).exists()
