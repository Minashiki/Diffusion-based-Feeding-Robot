"""SO3 residual, coverage descriptor and exact nearest-condition tests."""

from pathlib import Path
import sys

import mink
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))
from rotation_feedback import correction_projection,nearest_conditions,plan_item,residual_slow,rotation_descriptor


def test_rotation_descriptor_uses_so3_error_and_observed_angular_speed():
    fields=dict(tcp_rotation=slice(0,9),tcp_twist_world=slice(9,15))
    target=mink.SO3.exp(np.array([0.,.1,0.])).as_matrix()
    observation=np.r_[np.eye(3).reshape(-1),[0,0,0,0,.02,0]]
    row=rotation_descriptor(observation,target,fields)
    assert row['target_error_deg']==pytest.approx(np.rad2deg(.1))
    assert row['measured_angular_speed']==pytest.approx(.02)
    assert residual_slow(row)
    assert correction_projection(np.array([0.,-.003,0.]),np.array([0.,.3,0.]))==pytest.approx(-.01)


def test_residual_slow_is_not_just_a_pose_error_threshold():
    assert not residual_slow(dict(target_error_deg=5.,measured_angular_speed=.2))
    assert not residual_slow(dict(target_error_deg=1.,measured_angular_speed=.001))
    assert not residual_slow(dict(target_error_deg=60.,measured_angular_speed=.001))


def test_nearest_search_includes_later_blocks():
    matrix=np.zeros((600,2),np.float32)
    matrix[550]=[2.,4.]
    result=nearest_conditions(matrix,[np.array([2.,4.]),np.array([1.,2.])])
    assert result[0]['index']==550 and result[0]['normalized_condition_rms_distance']==0.
    assert result[1]['normalized_condition_rms_distance']==pytest.approx(np.sqrt(2.5))


def test_saved_live_plan_retains_causal_condition_and_dtypes():
    condition=dict(states=[[1.,2.]],history=[[3.]],phase=1,interaction=[1.,0.],state_mask=[True],history_mask=[True])
    batch=plan_item(dict(condition=condition,tick=2050))
    assert 'tick' not in batch
    assert batch['phase'].dtype==torch.long
    assert batch['state_mask'].dtype==torch.bool
    assert batch['states'].dtype==torch.float32
    assert condition['states']==[[1.,2.]]
