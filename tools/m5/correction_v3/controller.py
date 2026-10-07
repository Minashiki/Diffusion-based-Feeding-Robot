"""Brake real momentum before alignment; stop unsafe, unqualified prefixes."""

import numpy as np

from correction_v2.controller import (AlignmentTeacher as V2AlignmentTeacher,ATTEMPTS,CATEGORIES,QUOTAS,
    TEACHER_OWNERS,Progress as V2Progress,cancel_chunk,clearance,residual,stopped,
    swept_clearance as v2_swept_clearance)


VERSION='descent_alignment_v3_001'
MOVING_MIN=.002
MOVING_MAX=.005
REFERENCE_MAX=.0045
# Recorded 50mm/s prefixes require about 400ms to stop physically; include
# one 50ms decision interval. This envelope still requires paired calibration.
BRAKE_HORIZON_S=.45
REQUIRED_CLEARANCE=.002


class Progress(V2Progress):
    def corridor(self,category,teacher,obs):
        if obs['stage']!='ACQUIRE' or not self.above: return False
        z=obs['tcp_position'][2]
        if category.startswith('p1'):
            upper=teacher.geometry['bowl_position'][2]+(.070 if category.endswith('stopped') else .062)
            return bool(teacher.path[1][1][2]+.003<z<=upper)
        entry=teacher.path[2][1][2]
        return bool(self.pre_entry and entry+.003<z<=entry+.008)

    def candidate(self,category,teacher,obs,calibration=False):
        return bool(self.corridor(category,teacher,obs) and (calibration or category_residual(teacher,obs)))


def category_residual(teacher,obs):
    r=residual(teacher,obs)
    return bool(alignment_residual(teacher,obs) and r['downward_speed']>MOVING_MIN)


def alignment_residual(teacher,obs):
    r=residual(teacher,obs)
    return bool(2.<=r['angle_deg']<=4. and .001<=r['lateral_m']<=.002 and r['angular_speed']<.05)


def moving_release(obs,velocity):
    twist=np.asarray(obs['tcp_twist_world'])
    return bool(MOVING_MIN<-twist[2]<=MOVING_MAX and np.linalg.norm(twist[:3])<=MOVING_MAX
        and np.linalg.norm(twist[3:])<.05 and np.linalg.norm(velocity[:3])<=REFERENCE_MAX
        and np.linalg.norm(velocity[3:])<.05)


def braking_reserve(task,obs):
    speed=max(np.linalg.norm(obs['tcp_twist_world'][:3]),np.linalg.norm(task.adapter.velocity[:3]))
    return float(speed*BRAKE_HORIZON_S+speed**2/(2*task.robot_config['linear_acceleration_limit']))


def swept_clearance(task,teacher,obs):
    gate=v2_swept_clearance(task,teacher,obs)
    speed=max(np.linalg.norm(obs['tcp_twist_world'][:3]),np.linalg.norm(task.adapter.velocity[:3]))
    gate['reserve_m']+=float(speed*(BRAKE_HORIZON_S-.05))
    gate.update(net_m=float(gate['minimum_m']-gate['reserve_m']),brake_horizon_s=BRAKE_HORIZON_S)
    gate['passed']=bool(gate['net_m']>gate['required_m'])
    return gate


def safety_gate(task,obs):
    distance=clearance(task);reserve=braking_reserve(task,obs)
    return dict(minimum_m=distance,reserve_m=reserve,required_m=REQUIRED_CLEARANCE,
        stop_required=bool(obs['stage']=='ACQUIRE' and distance<=reserve+REQUIRED_CLEARANCE))


class AlignmentTeacher(V2AlignmentTeacher):
    def qualified(self):
        r=self.release_residual
        return bool(super().qualified() and (self.category.endswith('stopped')
            or (r['downward_speed']<=MOVING_MAX and r['linear_speed']<=MOVING_MAX)))
