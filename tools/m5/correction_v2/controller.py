"""Versioned height-holding expert and observation-based handover gates."""

import copy

import mink
import mujoco
import numpy as np

from feedingrobot.control.adapter import clip_norm
from feedingrobot.experts.teacher import rotation_error


VERSION='descent_alignment_v2_001'
CATEGORIES=('p1_stopped','p1_moving','p2_stopped','p2_moving','aligned')
QUOTAS=dict(zip(CATEGORIES,(2,2,2,2,4)))
ATTEMPTS=dict(zip(CATEGORIES,(4,4,4,4,8)))
TEACHER_OWNERS=('alignment_teacher','path_teacher')


def residual(teacher,obs):
    target=teacher.path[2]
    return dict(angle_deg=float(np.rad2deg(np.linalg.norm(rotation_error(target[2],obs['tcp_rotation'])))),
        lateral_m=float(np.linalg.norm(np.asarray(obs['tcp_position'])[:2]-target[1][:2])),
        linear_speed=float(np.linalg.norm(obs['tcp_twist_world'][:3])),
        angular_speed=float(np.linalg.norm(obs['tcp_twist_world'][3:])),
        downward_speed=float(-obs['tcp_twist_world'][2]))


def stopped(obs,velocity):
    return (np.linalg.norm(obs['tcp_twist_world'][:3])<.002 and np.linalg.norm(obs['tcp_twist_world'][3:])<.02
        and np.linalg.norm(velocity[:3])<.002 and np.linalg.norm(velocity[3:])<.02)


class Progress:
    """Only observed waypoint corridors; never elapsed time or nearest paths."""
    def __init__(self):
        self.above=False
        self.pre_entry=False

    def update(self,teacher,obs):
        p=np.asarray(obs['tcp_position'])
        if np.linalg.norm(p[:2]-teacher.path[0][1][:2])<.005 and abs(p[2]-teacher.path[0][1][2])<.007:
            self.above=True
        if self.above and np.linalg.norm(p[:2]-teacher.path[1][1][:2])<.003 and abs(p[2]-teacher.path[1][1][2])<.001:
            self.pre_entry=True

    def candidate(self,category,teacher,obs,calibration=False):
        if obs['stage']!='ACQUIRE' or not self.above: return False
        p=np.asarray(obs['tcp_position']);entry=teacher.path[2][1]
        if category.startswith('p1'):
            # Stop at the previously calibrated trigger; moving handover near 62mm.
            z=teacher.geometry['bowl_position'][2]+(.070 if category.endswith('stopped') else .062)
            corridor=p[2]<=z and p[2]>teacher.path[1][1][2]+.003
        else:
            corridor=self.pre_entry and entry[2]+.003<p[2]<=entry[2]+.008
        r=residual(teacher,obs)
        return bool(corridor and (calibration or (2.<=r['angle_deg']<=4. and .001<=r['lateral_m']<=.002
            and r['angular_speed']<.05 and r['downward_speed']>.002)))


def clearance(task):
    return min(float(mujoco.mj_geomDistance(task.model,task.data,a,b,.2,None))
        for a in task.index.spoon_geoms for b in task.index.bowl_geoms)


def swept_clearance(task,teacher,obs):
    """Screen the whole spoon, with interpolation and measured stopping reserve.

    This geometric screen is not proof of actuator stopping distance. Paired
    physical calibration remains mandatory, including moving handovers.
    """
    data=copy.copy(task.data)
    p=np.asarray(obs['tcp_position']);r=np.asarray(obs['tcp_rotation'])
    target=teacher.path[2];delta=target[1]-p;delta[2]=0.
    angular=rotation_error(target[2],r)
    spoon=list(task.index.spoon_geoms)
    positions=(task.data.geom_xpos[spoon]-p)@r
    rotations=task.data.geom_xmat[spoon].reshape(-1,3,3).copy()
    radius=float(np.max(np.linalg.norm(teacher.geometry['tool_points'],axis=1)))
    minimum=.2
    for fraction in np.linspace(0.,1.,9):
        current=mink.SO3.exp(fraction*angular).as_matrix()@r
        data.geom_xpos[spoon]=positions@current.T+p+fraction*delta
        data.geom_xmat[spoon]=(current@r.T@rotations).reshape(-1,9)
        minimum=min(minimum,min(float(mujoco.mj_geomDistance(task.model,data,a,b,.2,None))
            for a in spoon for b in task.index.bowl_geoms))
    cfg=task.robot_config
    linear=max(np.linalg.norm(obs['tcp_twist_world'][:3]),np.linalg.norm(task.adapter.velocity[:3]))
    speed=max(np.linalg.norm(obs['tcp_twist_world'][3:]),np.linalg.norm(task.adapter.velocity[3:]))
    reserve=linear*.05+linear**2/(2*cfg['linear_acceleration_limit'])
    reserve+=radius*(speed*.05+speed**2/(2*cfg['angular_acceleration_limit']))
    reserve+=(np.linalg.norm(delta)+radius*np.linalg.norm(angular))/16
    return dict(minimum_m=minimum,reserve_m=float(reserve),net_m=float(minimum-reserve),
        required_m=.002,passed=bool(minimum-reserve>.002))


def cancel_chunk(chunk):
    # Queue cancellation must not call adapter.stop/reset or overwrite velocity.
    return None


class AlignmentTeacher:
    def __init__(self,teacher,obs,tick,category,progress):
        if not progress.above or (category.startswith('p2') and not progress.pre_entry):
            raise ValueError('Unobserved acquisition progress')
        self.teacher=teacher
        self.category=category
        self.tick=tick
        self.height=float(obs['tcp_position'][2])
        self.stable=None
        self.completed_tick=None
        self.path_part=2 if category.startswith('p2') else 1
        teacher.part=self.path_part
        teacher.stage=teacher.path[self.path_part][0]
        self.release_residual=residual(teacher,obs)

    def act(self,obs,tick):
        teacher=self.teacher
        if self.completed_tick is not None:
            return teacher.act(obs),'path_teacher'
        p=np.asarray(obs['tcp_position']);r=np.asarray(obs['tcp_rotation'])
        target=teacher.path[2]
        error=rotation_error(target[2],r)
        gap=np.linalg.norm(p[:2]-target[1][:2])
        height_gap=abs(p[2]-self.height)
        if height_gap>=.0007:
            raise ValueError('alignment_height_drift')
        if (np.linalg.norm(error)<.01 and gap<.0007 and np.linalg.norm(obs['tcp_twist_world'][3:])<.05
                and np.linalg.norm(obs['tcp_twist_world'][:3])<.002):
            self.stable=tick if self.stable is None else self.stable
            if tick-self.stable>=200:
                self.completed_tick=tick
                return teacher.act(obs),'path_teacher'
        else: self.stable=None
        target_position=np.r_[target[1][:2],self.height]
        gain=teacher.parameters['position_gain']*(2. if self.path_part==1 else 1.)
        linear=teacher.base_rotation.T@(gain*(target_position-p))
        angular=teacher.base_rotation.T@(teacher.parameters['orientation_gain']*error)
        teacher.proposal=np.r_[linear,angular]
        teacher.target_position,teacher.target_rotation=target_position,target[2]
        teacher.stop_requested=False
        speed=teacher.parameters['transport_speed_m_s'] if self.path_part==1 else teacher.parameters['linear_speed_m_s']
        return np.r_[clip_norm(linear,min(speed,teacher.robot_config['linear_speed_limit'])),
            clip_norm(angular,teacher.robot_config['angular_speed_limit'])],'alignment_teacher'

    def qualified(self):
        r=self.release_residual
        return (self.completed_tick is not None and 2.<=r['angle_deg']<=4. and .001<=r['lateral_m']<=.002
            and r['angular_speed']<.05 and (r['linear_speed']<.002 if self.category.endswith('stopped')
                else r['downward_speed']>.002))
