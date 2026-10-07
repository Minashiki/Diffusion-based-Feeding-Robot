"""Perturbation ownership and stopping gates at a safe descent point."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mink
import numpy as np

spec=importlib.util.spec_from_file_location('calibrate_descent',Path(__file__).with_name('calibrate_descent.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def observation(y=0.,angle=0.,linear=0.,angular=0.,z=.069):
    return dict(tcp_position=np.array([0.,y,z]),tcp_rotation=mink.SO3.exp(np.array([0.,-angle,0.])).as_matrix(),
        tcp_twist_world=np.array([0.,linear,0.,0.,angular,0.]))


def state():
    return dict(mode='waiting',requested_deg=2.5,requested_lateral_m=.0015,trigger_z=.07)


def teacher(part=1):
    return SimpleNamespace(part=part,path=[None,('pre_entry',np.array([0.,0.,.025]),np.eye(3),.03)])


def test_only_safe_pre_entry_progress_can_trigger():
    for part,z in ((0,.069),(2,.069),(1,.08)):
        s=state();assert module.control(s,teacher(part),observation(z=z),np.zeros(6),np.eye(3),1000) is None
        assert s['mode']=='waiting'


def test_brake_pulse_stop_and_release_preserve_observation():
    s=state();obs=observation();original=obs['tcp_position'].copy()
    np.testing.assert_array_equal(module.control(s,teacher(),obs,np.zeros(6),np.eye(3),1000),np.zeros(6))
    assert s['mode']=='braking'
    command=module.control(s,teacher(),obs,np.zeros(6),np.eye(3),1250)
    np.testing.assert_allclose(command,[0,-.0025,0,0,-.05,0]);assert s['mode']=='pulse'
    np.testing.assert_array_equal(obs['tcp_position'],original)
    command=module.control(s,teacher(),observation(y=-.001,angle=np.deg2rad(2.6)),np.zeros(6),np.eye(3),2200)
    np.testing.assert_allclose(command,[0,-.0025,0,0,0,0])
    command=module.control(s,teacher(),observation(y=-.0016,angle=np.deg2rad(2.9)),np.zeros(6),np.eye(3),2250)
    np.testing.assert_array_equal(command,np.zeros(6));assert s['mode']=='settling'
    assert module.control(s,teacher(),observation(y=-.0016,angle=np.deg2rad(3.0)),np.zeros(6),np.eye(3),2500) is None
    assert s['mode']=='released' and s['release_part']==1


def test_fast_motion_cannot_release_teacher_ownership():
    s=state();s.update(mode='settling',rotation_target=np.eye(3).tolist(),origin_y=0.,pulse_end_tick=1000)
    command=module.control(s,teacher(),observation(angular=.1),np.zeros(6),np.eye(3),1400)
    np.testing.assert_array_equal(command,np.zeros(6));assert s['mode']=='settling'


def test_unperturbed_prefix_matches_original_teacher(tmp_path,monkeypatch):
    from feedingrobot.envs import FeedingGymEnv
    from feedingrobot.data.rollout import run_episode
    from feedingrobot.policies.audit import read_json
    from feedingrobot.policies.runtime import setup
    from feedingrobot.sim.model import ROOT
    setup(read_json(ROOT/'configs/dp_dit.json'),'cpu')
    manifest=read_json(sorted((ROOT/'datasets/single_bean/v1/m4/panda/train').glob('normal_*/manifest.json'))[0])
    config=manifest['teacher_config'];scenario=dict(config['scene'],recover=False)
    monkeypatch.setattr(module,'FeedingGymEnv',lambda robot:FeedingGymEnv(robot,max_episode_s=.3))
    row=module.run_case(config,scenario,False,tmp_path)
    original=tmp_path/'original'
    run_episode('panda',module.SEED,config,original,scenario=scenario,max_episode_s=.3,split='calibration')
    actions=np.load(original/'actions.npy');observations=np.load(original/'action_observations.npy')
    for i,record in enumerate(row['trace']):
        np.testing.assert_allclose(record['command'],actions[i],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(record['observation'],observations[i])
    assert not row['training_export'] and not row['model_policy_executed']
