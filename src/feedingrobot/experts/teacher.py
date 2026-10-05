"""Single-bean teacher: immutable geometry and current/past policy observations."""

import copy

import mink
import numpy as np

from feedingrobot.control.adapter import clip_norm
from feedingrobot.experts.bean_path import pickup_path


def rotation_error(target, actual):
    return mink.SO3.from_matrix(target @ actual.T).log()


class Teacher:
    def __init__(self, robot_config, config):
        self.robot_config = copy.deepcopy(robot_config)
        self.config = copy.deepcopy(config)
        if config.get("schema_version") != 2:
            raise ValueError("Single-bean teacher requires configuration schema 2")
        if config["teacher"]["pickup_hold_s"] < 1.:
            raise ValueError("Single-bean seating requires at least one second")
        self.base_rotation = mink.SO3(np.asarray(robot_config["base_quaternion"])).as_matrix()
        self.geometry = None

    def reset(self, parameters, *, geometry):
        self.parameters = dict(self.config["teacher"], **copy.deepcopy(parameters))
        self.geometry = copy.deepcopy(geometry)
        for value in self.geometry.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self.path = pickup_path(self.geometry, self.parameters)
        self.part = self.release_part = 0
        self.stage = 'above'
        self.phase = None
        self.pickup_hold_start = self.roll_start = self.transfer_start = None
        self.pickup_lift_complete = False
        self.wait_level_complete = False
        self.release_rotation = np.zeros(3)
        self.retreat_offset = None
        self.last_roll_target = self.last_roll_time = None
        self.last_time = self.last_mouth = self.last_mouth_rotation = self.last_bean_local = None
        self.bean_velocity_local = np.zeros(3)
        self.mouth_velocity = self.mouth_angular_velocity = np.zeros(3)
        self.proposal = np.zeros(6)
        self.target_position = self.target_rotation = None
        self.stop_requested = False

    def acquisition_waypoints(self, policy_obs=None):
        return [(name, position.copy(), rotation.copy()) for name, position, rotation, _ in self.path]

    def reached(self, target, target_r, tcp, rotation, *, seating=False):
        return (np.linalg.norm(target-tcp) < (.0002 if seating else .0007)
                and np.linalg.norm(rotation_error(target_r, rotation)) < (.002 if seating else .01))

    def act(self, policy_obs):
        if self.geometry is None:
            raise ValueError("Teacher.reset requires measured geometry before act")
        p, g = self.parameters, self.geometry
        tcp, rotation = np.asarray(policy_obs['tcp_position']), np.asarray(policy_obs['tcp_rotation'])
        mouth = tcp + np.asarray(policy_obs['mouth_relative_world'])
        mouth_r = np.asarray(policy_obs['mouth_rotation'])
        phase, now = policy_obs['stage'], float(policy_obs['time'])
        bean_local = rotation.T @ np.asarray(policy_obs['bean_relative_world'])[0]
        if self.last_time is not None and now > self.last_time:
            dt = now-self.last_time
            self.bean_velocity_local = (bean_local-self.last_bean_local)/dt
            self.mouth_velocity = clip_norm((mouth-self.last_mouth)/dt, .025)
            self.mouth_angular_velocity = clip_norm(rotation_error(mouth_r, self.last_mouth_rotation)/dt, .5)
        self.last_time, self.last_mouth, self.last_mouth_rotation = now, mouth.copy(), mouth_r.copy()
        self.last_bean_local = bean_local.copy()
        self.phase, self.stop_requested = phase, False
        speed = p['linear_speed_m_s']
        while self.part < len(self.path):
            self.stage, target, target_r, speed = self.path[self.part]
            if not self.reached(target, target_r, tcp, rotation,
                                seating=self.stage in ('wall_seat_tip', 'wall_seat_level')):
                break
            self.part += 1
            self.stop_requested = True
        if self.part == len(self.path) and not self.pickup_lift_complete:
            self.stage = 'pickup_hold'
            if self.pickup_hold_start is None:
                self.pickup_hold_start = now
            self.pickup_lift_complete = bool(phase == 'TRANSPORT' and bool(policy_obs['interaction'][0])
                                        and now-self.pickup_hold_start >= p['pickup_hold_s']
                                        and now+1e-9 >= p['earliest_transport_start_s']
                                        and np.linalg.norm(self.bean_velocity_local) < g['pickup_linear_speed_m_s'])
            if not self.pickup_lift_complete:
                self.target_position, self.target_rotation = self.path[-1][1:3]
                self.proposal = np.zeros(6)
                return self.proposal.copy()
        lifted = self.pickup_lift_complete
        if lifted:
            wait = mouth-mouth_r[:, 0]*g['task_config']['wait_offset_m']
            target_r = mouth_r
            speed = p['transport_speed_m_s'] if phase == 'TRANSPORT' else p['linear_speed_m_s']
            if phase in ('TRANSPORT', 'WAIT_READY', 'RECOVER') or (phase == 'RETRACT' and self.release_part == 3):
                self.stage, target = phase.lower(), wait
                if phase == 'TRANSPORT' and np.linalg.norm(wait-tcp) > .015:
                    target_r = mouth_r @ mink.SO3.exp(np.array([0., p['carry_pitch_rad'], 0.])).as_matrix()
                if phase == 'RECOVER':
                    self.wait_level_complete = False
                if phase == 'RETRACT':
                    target_r = mouth_r @ mink.SO3.exp(self.release_rotation).as_matrix()
                    target = mouth+mouth_r @ self.retreat_offset
                    receiver = tcp+np.asarray(policy_obs['receiver_relative_world'])
                    normal = np.asarray(policy_obs['receiver_rotation'])[:, 2]
                    lowest = np.min((g['scoop_points'] @ target_r.T) @ normal)
                    clearance = g['task_config']['clearance_margin_m']
                    target += ((clearance-lowest-normal @ (target-receiver))/(normal @ mouth_r[:, 2]))*mouth_r[:, 2]
                elif phase in ('TRANSPORT', 'WAIT_READY') and self.reached(wait, mouth_r, tcp, rotation, seating=True):
                    self.wait_level_complete = True
            elif phase == 'APPROACH' and not self.wait_level_complete:
                self.stage, target = 'wait_level', wait
                if self.reached(wait, mouth_r, tcp, rotation, seating=True):
                    self.wait_level_complete = True
            else:
                self.stage = 'entry' if phase == 'APPROACH' else 'release'
                target = mouth+mouth_r @ np.array([p['insert_depth_m'], 0., p['insert_height_m']])
                if phase in ('TRANSFER', 'RETRACT') and (self.transfer_start is not None or np.linalg.norm(target-tcp) < .001):
                    if self.transfer_start is None:
                        self.transfer_start = now
                    self.stage = ('release_lower', 'release_roll', 'release_clear')[self.release_part]
                    fraction = 0. if self.roll_start is None else min(1., p['release_rate_s_inv']*(now-self.roll_start))
                    self.release_rotation = fraction*np.asarray(p['release_rotation_rad'])
                    target_r = mouth_r @ mink.SO3.exp(self.release_rotation).as_matrix()
                    target = mouth+mouth_r @ np.asarray(p['release_offset_m'])
                    receiver = tcp+np.asarray(policy_obs['receiver_relative_world'])
                    normal = np.asarray(policy_obs['receiver_rotation'])[:, 2]
                    lowest = np.min((g['scoop_points'] @ target_r.T) @ normal)
                    height = (p['release_clearance_m']-lowest-normal @ (target-receiver))/(normal @ mouth_r[:, 2])
                    target += height*mouth_r[:, 2]
                    if self.release_part == 2:
                        target += mouth_r @ np.asarray(p['release_avoidance_m'])
                    reached = self.reached(target, target_r, tcp, rotation)
                    if self.release_part == 0 and reached:
                        self.roll_start, self.release_part = now, 1
                    elif (self.release_part == 1 and fraction == 1. and reached and phase == 'RETRACT'
                          and now-self.roll_start >= 1./p['release_rate_s_inv']+p['release_settle_s']):
                        self.release_part = 2
                    elif self.release_part == 2 and reached:
                        self.retreat_offset = mouth_r.T @ (target-mouth)
                        extent = np.max((g['tool_points'] @ mink.SO3.exp(self.release_rotation).as_matrix().T)[:, 0])
                        self.retreat_offset[0] = -extent-g['task_config']['clearance_margin_m']
                        self.release_part = 3
        if not lifted:
            if self.stage == 'sweep':
                speed *= .7
            elif self.stage == 'wall_15':
                speed *= .8
            elif self.stage in ('above', 'pre_entry'):
                speed = p['transport_speed_m_s']
        wait_multiplier = 4. if self.stage in ('wait_ready', 'wait_level', 'recover') else 2.
        gain = 1. if self.stage == 'clearance' and np.linalg.norm(target-tcp) < .005 else p['position_gain']*(wait_multiplier if lifted else 2. if self.stage in ('above', 'pre_entry') else 1.)
        linear = gain*(target-tcp)
        mouth_linear = (self.mouth_velocity + np.cross(self.mouth_angular_velocity, target-mouth)
                        if lifted else np.zeros(3))
        roll_angular = np.zeros(3)
        if self.stage == 'release_roll':
            if self.last_roll_time is not None and now > self.last_roll_time:
                mouth_linear = (target-self.last_roll_target)/(now-self.last_roll_time)
            self.last_roll_target, self.last_roll_time = target.copy(), now
            if fraction < 1.:
                roll_angular = mouth_r @ np.asarray(p['release_rotation_rad'])*p['release_rate_s_inv']
        else:
            self.last_roll_target = self.last_roll_time = None
        if phase == 'TRANSPORT' and lifted:
            linear += mouth_linear
        scale = min(1., speed/max(np.linalg.norm(linear), 1e-12))
        linear *= scale
        if phase != 'TRANSPORT' and lifted:
            linear += mouth_linear
        orientation_gain = p['orientation_gain']*(wait_multiplier if lifted else 1.)
        angular = orientation_gain*rotation_error(target_r, rotation)
        if lifted:
            angular += self.mouth_angular_velocity+roll_angular
        if self.stage.startswith('wall_'):
            angular *= scale
        if self.stage in ('wall_seat_tip', 'wall_seat_level'):
            angular = clip_norm(angular, .3)
        self.proposal = np.r_[self.base_rotation.T @ (gain*(target-tcp)+mouth_linear),
                             self.base_rotation.T @ (orientation_gain*rotation_error(target_r, rotation)
                                                    + (self.mouth_angular_velocity if lifted else 0.)+roll_angular)]
        self.target_position, self.target_rotation = target.copy(), target_r.copy()
        return np.r_[clip_norm(self.base_rotation.T @ linear, np.nextafter(self.robot_config['linear_speed_limit'], 0.)),
                     clip_norm(self.base_rotation.T @ angular, self.robot_config['angular_speed_limit'])]
