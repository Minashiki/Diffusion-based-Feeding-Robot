"""Inference-only config, paired noise and candidate-gate regressions."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))
import step_sensitivity as probe
from acquire_diagnosis import sample_from_noise
from diagnose_sampling import spacing_override
from test_diagnose_sampling import SmallModel
from feedingrobot.policies.audit import read_json
from feedingrobot.sim.model import ROOT


def test_inference_count_does_not_mutate_training_config():
    config = read_json(ROOT/'configs/dp_dit.json')
    before = deepcopy(config)
    revised = probe.inference_config(config,50)
    assert revised['diffusion']['inference_steps']==50
    assert config==before
    with spacing_override('leading'):
        assert [probe.settings(config,n)['timesteps'][0] for n in probe.COUNTS]==[90,95,98]


def summaries():
    metric = dict(finite=True,angular=dict(vector_rmse=1.,clipped_fraction=0.),linear=dict(vector_rmse=1.,clipped_fraction=0.))
    result = {str(n):{str(s):dict(metrics=deepcopy(metric)) for s in probe.SEEDS} for n in probe.COUNTS}
    for seed in probe.SEEDS:
        result['20'][str(seed)]['metrics']['angular']['vector_rmse']=.8
    return result


def test_gate_accepts_consistent_error_improvement():
    assert probe.select_candidate(summaries())==20


@pytest.mark.parametrize('field,value',[('angular_rmse',.91),('linear_rmse',1.06),('clipping',.001),('finite',False)])
def test_gate_rejects_one_bad_seed(field,value):
    data=summaries()
    metric=data['20']['2']['metrics']
    if field=='finite': metric['finite']=value
    elif field=='clipping': metric['angular']['clipped_fraction']=value
    elif field=='angular_rmse': metric['angular']['vector_rmse']=value
    else: metric['linear']['vector_rmse']=value
    assert probe.select_candidate(data) is None


def test_initial_noise_is_reused_and_not_modified_by_sampling():
    cases=[dict(tick=50),dict(tick=250)]
    noise=probe.noise_for(cases,1,16,torch.device('cpu'))
    before=noise.clone()
    model=SmallModel()
    batch=dict(states=torch.zeros(2,2,122))
    config=read_json(ROOT/'configs/dp_dit.json')
    norm=dict(action_mean=[0.]*6,action_std=[1.]*6)
    with spacing_override('leading'):
        for count in probe.COUNTS:
            sample_from_noise(model,probe.inference_config(config,count),batch,norm,noise)
            torch.testing.assert_close(noise,before,rtol=0,atol=0)
            torch.testing.assert_close(noise,probe.noise_for(cases,1,16,torch.device('cpu')),rtol=0,atol=0)


def test_known_noise_probe_recovers_teacher_and_ignores_padding(tmp_path):
    config=read_json(ROOT/'configs/dp_dit.json')
    _,scheduler=probe.dit.schedulers(config)

    class Oracle:
        horizon=16

        def encoder(self,batch):
            return batch['actions']

        def denoiser(self,noisy,t,clean):
            alpha=scheduler.alphas_cumprod[t][:,None,None]
            epsilon=(noisy-alpha.sqrt()*clean)/(1-alpha).sqrt()
            epsilon[:,4:]+=100
            return epsilon

    items=[dict(actions=torch.ones(16,6)*.01,states=torch.zeros(2,122))]
    truth=np.full((1,16,6),.01)
    mask=np.zeros((1,16),dtype=bool)
    mask[:,:4]=True
    norm=dict(action_mean=[0.]*6,action_std=[1.]*6)
    result=probe.epsilon_probe(Oracle(),config,norm,torch.device('cpu'),items,[dict(tick=50)],truth,mask,(.05,.5),tmp_path)
    for row in result['by_timestep'].values():
        assert row['normalized_epsilon_component_rmse']<1e-5
        assert row['reconstruction']['angular']['vector_rmse']<1e-5
