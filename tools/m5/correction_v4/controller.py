"""Real direction perturbations and physics-boundary stable qualification."""

import itertools
import numpy as np

from correction_v3.controller import (AlignmentTeacher as PreviousTeacher,Progress,
    alignment_residual,moving_release,stopped,residual,swept_clearance,clearance,
    braking_reserve,BRAKE_HORIZON_S,REQUIRED_CLEARANCE,TEACHER_OWNERS)

VERSION='descent_alignment_v4_003'
CATEGORIES=('p1_stopped','p1_moving','p2_stopped','p2_moving')
CELLS=tuple(dict(category=c,quadrant=list(q),velocity=v)
    for c,q,v in itertools.product(CATEGORIES,itertools.product((-1,1),repeat=2),('toward','away')))
CALIBRATION_SEEDS=tuple(range(800015,800047))


def direction_velocity(cell):
    sign=-1 if cell['velocity']=='toward' else 1
    return sign*np.asarray(cell['quadrant'],dtype=float)/np.sqrt(2)*.001


def perturbation_feedback(cell,teacher,obs):
    target=teacher.path[2][1][:2]+np.asarray(cell['quadrant'])*.0015/np.sqrt(2)
    return np.clip(2*teacher.parameters['position_gain']*(target-np.asarray(obs['tcp_position'])[:2]),-.0018,.0018)


def release_direction(cell,teacher,obs):
    error=np.asarray(obs['tcp_position'])[:2]-teacher.path[2][1][:2]
    speed=float(error@np.asarray(obs['tcp_twist_world'])[:2]/max(np.linalg.norm(error),1e-12))
    sign=-1 if cell['velocity']=='toward' else 1
    return bool(np.all(error*np.asarray(cell['quadrant'])>0) and .0005<=sign*speed<=.0015)


class AlignmentTeacher(PreviousTeacher):
    def __init__(self,*args):
        super().__init__(*args)
        self.last_tick=None
        self.records=[]

    def observe(self,obs,tick):
        if self.completed_tick is not None: return
        if self.last_tick is not None and tick!=self.last_tick+1:
            raise ValueError('Missing physical stability boundary')
        r=residual(self.teacher,obs)
        good=(r['angle_deg']<np.rad2deg(.01) and r['lateral_m']<.0007
            and r['linear_speed']<.002 and r['angular_speed']<.05)
        reset=self.stable is not None and not good
        self.stable=(tick if self.stable is None else self.stable) if good else None
        error=np.asarray(obs['tcp_position'])[:2]-self.teacher.path[2][1][:2]
        self.records.append([tick,*error,r['angle_deg'],r['linear_speed'],r['angular_speed'],
            good,-1 if self.stable is None else self.stable,reset])
        self.last_tick=tick

    def act(self,obs,tick):
        if self.completed_tick is None and self.last_tick!=tick:
            raise ValueError('Teacher requires the current physical stability boundary')
        return super().act(obs,tick)
