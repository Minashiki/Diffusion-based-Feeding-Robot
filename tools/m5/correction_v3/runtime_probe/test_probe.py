"""Fixed-action physical tests; never update production model weights."""

from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import rollout as probe
import evaluate as legacy
from run import compare_baseline,compatible_runtime
import numpy as np
import pytest
import torch

from feedingrobot.data.episodes import load_episode
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.sim.model import ROOT


def test_viewer_ready_waits_for_status_message_and_rejects_unavailable(tmp_path):
    class Display:
        def __init__(self,status): self.report={'status':'starting'};self.status=status
        def start(self): return self.report.copy()
        def _messages(self): self.report.update(status=self.status)
    probe.start_viewer(Display('running'),tmp_path)
    with pytest.raises(RuntimeError,match='unavailable'):
        probe.start_viewer(Display('unavailable'),tmp_path)
    assert read_json(tmp_path/'viewer_error.json')['status']=='unavailable'


def test_runtime_allows_eight_cpu_threads_without_accepting_gpu_changes():
    old=dict(device='cuda',gpu='same',torch='same',cpu_affinity=list(range(6)),torch_threads=6,interop_threads=1)
    eight=dict(old,cpu_affinity=list(range(8)),torch_threads=8)
    assert compatible_runtime(eight,old)
    assert not compatible_runtime(dict(eight,device='cpu'),old)
    assert not compatible_runtime(dict(eight,gpu='different'),old)


def test_rotated_base_gate_preserves_world_horizontal_and_angular():
    rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    world=np.array([.01,.02,-.03]);command=np.r_[rotation.T@world,[.1,.2,.3]]
    applied,changed=probe.hold_gate(command,rotation,True)
    assert changed
    np.testing.assert_allclose(rotation@applied[:3],[.01,.02,0.],atol=1e-15)
    np.testing.assert_array_equal(applied[3:],command[3:])
    for active,value in ((False,command),(True,np.r_[rotation.T@[.01,.02,.03],[.1,.2,.3]])):
        result,modified=probe.hold_gate(value,rotation,active)
        assert not modified
        np.testing.assert_array_equal(result,value)


def test_local_leading_sampler_matches_original():
    from diagnose_sampling import spacing_override
    from feedingrobot.policies.dit import sample_actions
    config=read_json(ROOT/'configs/dp_dit.json')
    config['model'].update(depth=1,hidden_size=32,heads=4,mlp_hidden=64)
    torch.manual_seed(11);model=ActionDiT(config).eval()
    state_dim=model.encoder.state[0].in_features
    batch=dict(states=torch.randn(1,2,state_dim),history=torch.randn(1,10,28),phase=torch.ones(1,dtype=torch.long),
        interaction=torch.zeros(1,4),state_mask=torch.ones(1,2,dtype=torch.bool),history_mask=torch.ones(1,10,dtype=torch.bool))
    norm=dict(action_mean=[0.]*6,action_std=[1.]*6)
    with spacing_override('leading'):
        expected=sample_actions(model,config,batch,norm,torch.Generator().manual_seed(3))
    actual=probe.sample_actions(model,config,batch,norm,torch.Generator().manual_seed(3))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.fixture
def physical():
    source=ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_calibration_001/episodes/780003_calibration'
    metadata=read_json(source/'manifest.json')
    return source,metadata,read_json(ROOT/'configs/dp_dit.json'),read_json(source.parents[1]/'report.json')['normalization']


@pytest.mark.parametrize('variant,expected_indices,expected_plans',[
    ('baseline200',[0,1,2,3,0,1],2),('replan50',[0,0,0,0,0,0],6)])
def test_physical_execution_window(tmp_path,monkeypatch,physical,variant,expected_indices,expected_plans):
    source,m,config,norm=physical;plans=[]
    def sample(*args):
        plans.append(1);actions=torch.zeros(1,16,6)
        actions[0,:,1]=torch.arange(16)*.0001
        return actions
    monkeypatch.setattr(probe,'sample_actions',sample)
    monkeypatch.setattr(probe.Teacher,'act',lambda *args:pytest.fail('Teacher must not execute'))
    directory=tmp_path/'case'
    result=probe.rollout(None,config,norm,source,directory,torch.device('cpu'),correction=True,
        viewer=False,variant=variant,max_ticks=m['handover']['release_tick']+300)
    trace=read_json(directory/'trace.json')
    assert [t['chunk_action_index'] for t in trace]==expected_indices
    assert len(plans)==expected_plans
    _,a=load_episode(directory)
    assert not a['action_mask'].any() and set(a['action_owner'])=={'model'}
    assert not result['training_export']
    assert replay_episode(directory)['status']=='passed'


def test_baseline_physical_equivalence_and_mismatch(tmp_path,monkeypatch,physical):
    source,m,config,norm=physical
    sample=lambda *args:torch.zeros(1,16,6)
    monkeypatch.setattr(probe,'sample_actions',sample);monkeypatch.setattr(legacy,'sample_actions',sample)
    kw=dict(correction=True,viewer=False,max_ticks=m['handover']['release_tick']+300)
    legacy.rollout(None,config,norm,source,tmp_path/'old',torch.device('cpu'),**kw)
    probe.rollout(None,config,norm,source,tmp_path/'new',torch.device('cpu'),**kw)
    assert compare_baseline(tmp_path/'new',tmp_path/'old')['status']=='passed'
    actions=np.load(tmp_path/'new/actions.npy');actions[0,0]+=1.
    np.save(tmp_path/'new/actions.npy',actions)
    with pytest.raises(ValueError,match='hash mismatch'):compare_baseline(tmp_path/'new',tmp_path/'old')


def test_gate_records_raw_prediction_and_remains_assistance(tmp_path,monkeypatch,physical):
    source,m,config,norm=physical
    action=torch.tensor([0.,0.,-.02,0.,0.,0.]).repeat(1,16,1)
    monkeypatch.setattr(probe,'sample_actions',lambda *args:action)
    result=probe.rollout(None,config,norm,source,tmp_path/'gate',torch.device('cpu'),correction=True,
        variant='holdgate200',viewer=False,max_ticks=m['handover']['release_tick']+300)
    _,a=load_episode(tmp_path/'gate')
    assert result['hold_gate']['applications']>0 and not result['model_alignment']['success']
    assert set(a['action_owner'])=={'model_hold_gate'} and not a['action_mask'].any()
    assert np.all(a['proposals'][:,2]<0.) and np.all(a['actions'][:,2]==0.)
    assert replay_episode(tmp_path/'gate')['status']=='passed'


def test_gate_releases_at_original_continuous_stability(tmp_path,monkeypatch,physical):
    source,m,config,norm=physical;release=m['handover']['release_tick']
    monkeypatch.setattr(probe,'sample_actions',lambda *args:torch.tensor([0.,0.,-.02,0.,0.,0.]).repeat(1,16,1))
    def good(reference,obs,height,tick,stable):
        stable=release if stable is None else stable
        return dict(angle_deg=.1,lateral_m=.0001,linear_speed=.001,angular_speed=.001),0.,stable,tick-stable>=200
    monkeypatch.setattr(probe,'alignment_measure',good)
    result=probe.rollout(None,config,norm,source,tmp_path/'release',torch.device('cpu'),correction=True,
        variant='holdgate200',viewer=False,max_ticks=release+300)
    trace=read_json(tmp_path/'release/trace.json')
    assert all(t['gate_modified'] for t in trace[:4])
    assert all(not t['gate_active'] and t['command'][2]<0. for t in trace[4:])
    assert result['hold_gate']['assisted_alignment_pass'] and not result['model_alignment']['success']
    assert replay_episode(tmp_path/'release')['status']=='passed'


def test_gate_preserves_physical_height_violation(tmp_path,monkeypatch,physical):
    source,m,config,norm=physical
    monkeypatch.setattr(probe,'sample_actions',lambda *args:torch.tensor([0.,0.,.02,0.,0.,0.]).repeat(1,16,1))
    result=probe.rollout(None,config,norm,source,tmp_path/'unsafe',torch.device('cpu'),correction=True,
        variant='holdgate200',viewer=False,max_ticks=m['handover']['release_tick']+1000)
    assert result['failure_reason']=='alignment_height_drift'
    assert result['external_braking']['interventions'][0]['reason']=='alignment_height_drift'
    assert result['model_alignment']['maximum_height_drift_m']>=.0007
    assert not result['model_alignment']['success']
    assert replay_episode(tmp_path/'unsafe')['status']=='passed'
