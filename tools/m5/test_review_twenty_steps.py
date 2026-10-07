"""Protected review statistics and exploratory rollout rejection tests."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
from review_twenty_steps import protected_review,supports_exploratory_rollout


def test_speed_projection_uses_original_limits_and_ignores_padding():
    truth=np.array([[[.05,0,0,0,.5,0],[0,0,0,0,0,0]]])
    predictions=truth*1.04
    predictions[:,1]=1e6
    mask=np.array([[True,False]])
    result=protected_review(predictions,truth,mask,(.05,.5))
    assert result['raw']['linear']['clipped_fraction']==1
    assert result['speed_projected']['linear']['vector_rmse']<1e-12
    assert result['speed_projected']['angular']['vector_rmse']<1e-12
    assert result['overshoot']['linear']['maximum']<.053


def review():
    row=dict(raw=dict(finite=True),speed_projected=dict(linear=dict(vector_rmse=1.),angular=dict(vector_rmse=1.)))
    result={count:{str(seed):deepcopy(row) for seed in (0,1,2)} for count in ('10','20')}
    for row in result['20'].values():
        for component in ('linear','angular'): row['speed_projected'][component]['vector_rmse']=.8
    return result


def test_review_requires_projected_improvement_for_every_seed():
    result=review()
    assert supports_exploratory_rollout(result)
    result['20']['2']['speed_projected']['angular']['vector_rmse']=1.1
    assert not supports_exploratory_rollout(result)


def test_nonfinite_candidate_never_supports_rollout():
    result=review()
    result['20']['1']['raw']['finite']=False
    assert not supports_exploratory_rollout(result)
