"""Real-control pulse transitions, expansion gates and original teacher prefix."""

from pathlib import Path
import sys

import mink
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from teacher_rotation_calibration import may_expand,pulse_command,run_case
from feedingrobot.data.rollout import run_episode
from feedingrobot.data.episodes import load_episode
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT


class FakeTeacher:
    part=0
    path=[('above',np.zeros(3),np.eye(3),.03)]

    def reached(self,target,rotation,position,actual):
        return np.linalg.norm(position-target)<.0007 and np.linalg.norm((mink.SO3.from_matrix(rotation)@mink.SO3.from_matrix(actual).inverse()).log())<.01


def observed(angle,speed=0.):
    return dict(tcp_position=np.zeros(3),tcp_rotation=mink.SO3.exp(np.array([0.,-angle,0.])).as_matrix(),tcp_twist_world=np.array([0.,0.,0.,0.,speed,0.]))


def test_pulse_settle_and_release_use_real_observation_without_mutation():
    state=dict(mode='waiting',requested_deg=5.,angular_acceleration_limit=2.)
    obs=observed(0.);original=obs['tcp_rotation'].copy()
    command=pulse_command(state,FakeTeacher(),obs,np.zeros(6),np.eye(3),1000)
    np.testing.assert_allclose(command,[0,0,0,0,-.05,0])
    np.testing.assert_array_equal(obs['tcp_rotation'],original)
    assert state['mode']=='pulse'
    np.testing.assert_array_equal(pulse_command(state,FakeTeacher(),observed(np.deg2rad(5.3),-.1),command,np.eye(3),1550),np.zeros(6))
    assert state['mode']=='settling'
    assert pulse_command(state,FakeTeacher(),observed(np.deg2rad(5.4)),np.zeros(6),np.eye(3),1800) is None
    assert state['mode']=='released' and state['residual_slow_match']
    assert state['achieved_deg']==pytest.approx(5.4)


def test_pulse_never_starts_before_the_above_waypoint_is_reached():
    state=dict(mode='waiting',requested_deg=5.,angular_acceleration_limit=2.)
    assert pulse_command(state,FakeTeacher(),observed(.2),np.zeros(6),np.eye(3),1000) is None
    assert state['mode']=='waiting'


def test_settling_does_not_release_a_fast_state():
    state=dict(mode='settling',requested_deg=5.,angular_acceleration_limit=2.,pulse_end_tick=1000)
    assert pulse_command(state,FakeTeacher(),observed(.1,.1),np.zeros(6),np.eye(3),1400) is not None
    assert state['mode']=='settling'


def test_expansion_requires_baseline_success_real_correction_and_pickup():
    baseline=dict(success=True)
    pulse=dict(qualified_correction=True,pickup=True,failure_reason=None)
    assert may_expand(baseline,pulse)
    pulse['pickup']=False
    assert not may_expand(baseline,pulse)
    pulse['pickup']=True;pulse['qualified_correction']=False
    assert not may_expand(baseline,pulse)


def test_unperturbed_real_physics_prefix_matches_original_teacher(tmp_path):
    setup(read_json(ROOT/'configs/dp_dit.json'),'cpu')
    manifest=read_json(sorted((ROOT/'datasets/single_bean/v1/m4/panda/train').glob('normal_*/manifest.json'))[0])
    config=manifest['teacher_config'];scenario=dict(config['scene'],recover=False)
    result=run_case(730001,5.,False,config,scenario,tmp_path,max_ticks=300)
    original=run_episode('panda',730001,config,tmp_path/'original',scenario=scenario,max_episode_s=.3,split='calibration')
    actions=np.load(tmp_path/'original/actions.npy')
    observations=np.load(tmp_path/'original/action_observations.npy')
    for i,row in enumerate(result['trace']):
        np.testing.assert_allclose(row['command'],actions[i],rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(row['observation'],observations[i])
    assert result['simulated_s']==pytest.approx(.3)
    assert not result['model_policy_executed'] and not result['training_export']


def test_recorded_pulse_excludes_labels_and_replays_real_physics(tmp_path):
    setup(read_json(ROOT/'configs/dp_dit.json'),'cpu')
    manifest=read_json(sorted((ROOT/'datasets/single_bean/v1/m4/panda/train').glob('normal_*/manifest.json'))[0])
    config=manifest['teacher_config'];scenario=dict(config['scene'],recover=False)
    directory=tmp_path/'episode'
    row=run_case(740001,3.,True,config,scenario,tmp_path,max_ticks=4500,episode_directory=directory)
    metadata,arrays=load_episode(directory)
    assert row['perturbation']['mode']=='released'
    assert arrays['teacher_owned'].any() and (~arrays['teacher_owned']).any()
    assert not np.any(arrays['action_mask'] & ~arrays['teacher_owned'])
    start=row['perturbation']['pulse_start_tick'];end=row['perturbation']['release_tick']
    pulse=(arrays['action_ticks']>=start)&(arrays['action_ticks']<end)
    assert pulse.any() and not arrays['action_mask'][pulse].any()
    assert metadata['truncated'] and not metadata['accepted_normal']
    assert replay_episode(directory)['status']=='passed'
