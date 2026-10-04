"""Minimal state-driven M3 proof driver; not a qualified M4 teacher."""

import mink
import mujoco
import numpy as np

from feedingrobot.scripts.validate_m1c import sweep_path
from feedingrobot.sim.events import evidence
from feedingrobot.sim.model import load_json


class FeedingDriver:
    def __init__(self, task):
        self.path = sweep_path(task, load_json('configs/acceptance.json'))
        self.part = 0
        self.stage = 'above'
        self.transfer_start = None
        self.lifted = False
        self.first_entry_contact = None
        self.sweep_start = self.sweep_end = None

    def action(self, task):
        e = evidence(task)
        tcp = e['tcp_position']
        rotation = task.data.site_xmat[task.index.tcp].reshape(3, 3)
        phase = task.logic.phase
        speed = .025
        if self.part < len(self.path):
            self.stage, target, target_r, speed = self.path[self.part]
            error = (mink.SO3.from_matrix(target_r) @ mink.SO3.from_matrix(rotation).inverse()).log()
            if np.linalg.norm(target-tcp) < .0007 and np.linalg.norm(error) < .01:
                task.adapter.stop(hold_reference=True)
                self.part += 1
                return self.action(task)
        elif not task.logic.acquired:
            self.stage = 'pickup_hold'
            return np.zeros(6)
        else:
            target_r = e['mouth_rotation']
            if not self.lifted:
                self.stage = 'transport_raise'
                target = np.r_[self.path[-1][1][:2], e['mouth_position'][2]]
                if np.linalg.norm(target-tcp) < .002:
                    self.lifted = True
            elif phase in ('TRANSPORT', 'WAIT_READY', 'RECOVER', 'RETRACT'):
                self.stage = phase.lower()
                target = e['wait_position']
                if phase == 'RETRACT':
                    target_r = e['mouth_rotation'] @ mink.SO3.exp([-1.2, 0., 0.]).as_matrix()
            else:
                self.stage = 'entry' if phase == 'APPROACH' else 'release'
                target = e['mouth_position'] + e['mouth_rotation'] @ [.004, 0, -.006]
                if phase == 'TRANSFER' and (self.transfer_start is not None or np.linalg.norm(target-tcp) < .001):
                    if self.transfer_start is None:
                        self.transfer_start = float(task.data.time)
                    target = e['mouth_position'] + e['mouth_rotation'] @ [.014, -.008, -.004]
                    target_r = e['mouth_rotation'] @ mink.SO3.exp([-1.2, 0., 0.]).as_matrix()
            speed = .05 if phase == 'TRANSPORT' else .025
        linear = 3 * (target-tcp)
        mouth_angular = np.zeros(3)
        if self.lifted and phase != 'ACQUIRE':
            velocity = np.zeros(6)
            mouth = task.model.site('mouth_entry').id
            mujoco.mj_objectVelocity(task.model, task.data, mujoco.mjtObj.mjOBJ_SITE, mouth, velocity, 0)
            linear += velocity[3:] + np.cross(velocity[:3], target-e["mouth_position"])
            mouth_angular = velocity[:3]
        scale = min(1., speed / max(np.linalg.norm(linear), 1e-12))
        linear *= scale
        angular = 3 * (mink.SO3.from_matrix(target_r) @ mink.SO3.from_matrix(rotation).inverse()).log() + mouth_angular
        if self.stage.startswith('wall_'):
            angular *= scale
        base = task.data.site_xmat[task.index.base].reshape(3, 3)
        limits = np.r_[[task.robot_config['linear_speed_limit']]*3,
                       [task.robot_config['angular_speed_limit']]*3]
        return np.clip(np.r_[base.T @ linear, base.T @ angular] / limits, -1, 1)

    def record(self, task, before=None, radius=None):
        if self.stage == 'sweep':
            position = task.data.geom_xpos[task.index.bean_collision_geoms[0]].copy()
            if self.sweep_start is None:
                self.sweep_start = position
            self.sweep_end = position
        if self.first_entry_contact is not None:
            return
        bean = int(task.index.bean_collision_geoms[0])
        for row in task.applied_contacts:
            if row['force_n'] <= .0001:
                continue
            other = row['geom1'] if row['geom2'] == bean else row['geom2'] if row['geom1'] == bean else None
            if other not in task.index.scoop_geoms:
                continue
            centre = task.data.geom_xpos[bean] if before is None else before
            if radius is None:
                radius = np.linalg.norm(task.data.geom_xmat[bean].reshape(3,3)[2] * task.model.geom_size[bean])
            self.first_entry_contact = dict(time_s=float(task.data.time-task.dt),
                height_relative_radius=float((row['position'][2]-centre[2])/radius),
                force_on_bean_world_n=(row['force_on_geom2_world'] * (1 if row['geom2']==bean else -1)).tolist(),
                position_m=row['position'].tolist(), spoon_geom=task.model.geom(int(other)).name)
            break
