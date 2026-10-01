"""State-feedback P0 teacher. This module never owns or writes physics state."""

import copy

import mink
import numpy as np

from feedingrobot.control.adapter import clip_norm


class Teacher:
    def __init__(self, robot_config, config):
        self.robot_config = copy.deepcopy(robot_config)
        self.config = copy.deepcopy(config)
        self.base_rotation = mink.SO3(np.asarray(robot_config["base_quaternion"])).as_matrix()
        self.reset({})

    def reset(self, parameters):
        self.parameters = dict(self.config["teacher"], **copy.deepcopy(parameters))
        self.phase = None
        self.part = 0
        self.food_start = None
        self.approach_position = None
        self.capture_position = None
        self.carry_part = 0
        self.carry_position = None
        self.lift_ready_since = None
        self.pickup_lift_complete = False
        self.last_mouth = None
        self.last_time = None
        self.mouth_velocity = np.zeros(3)
        self.proposal = np.zeros(6)
        self.target_position = None
        self.target_rotation = None

    def act(self, policy_obs):
        p = self.parameters
        tcp = np.asarray(policy_obs["tcp_position"])
        rotation = np.asarray(policy_obs["tcp_rotation"])
        food = tcp + policy_obs["food_relative_world"]
        mouth = tcp + policy_obs["mouth_relative_world"]
        mouth_r = np.asarray(policy_obs["mouth_rotation"])
        phase, now = policy_obs["stage"], policy_obs["time"]
        if self.last_time is not None and now > self.last_time:
            estimate = (mouth - self.last_mouth) / (now - self.last_time)
            self.mouth_velocity = .5 * self.mouth_velocity + .5 * clip_norm(estimate, .025)
        self.last_mouth, self.last_time = mouth.copy(), now
        if phase != self.phase:
            if phase not in ("SELECT", "ACQUIRE", "TRANSPORT"):
                self.part = 0
            self.phase = phase
        if self.food_start is None:
            self.food_start = food.copy()
            self.approach_position = tcp.copy()
        yaw = mink.SO3.exp(np.array([0., 0., p["scoop_yaw_rad"]])).as_matrix()
        pitch = p["scoop_pitch_rad"]
        tilted = yaw @ mink.SO3.exp(np.array([0., pitch, 0.])).as_matrix()
        target_r = yaw
        target = tcp.copy()
        feedforward = np.zeros(3)
        acquiring = phase in ("SELECT", "ACQUIRE") or (phase == "TRANSPORT" and not self.pickup_lift_complete)
        if acquiring:
            direction = yaw[:, 0]
            entry = self.food_start - direction * p["scoop_start_offset_m"]
            if (phase == "TRANSPORT" and self.capture_position is None
                    and (self.part != 5 or policy_obs["interaction"][0])):
                self.capture_position = tcp.copy()
                self.part = 6
            if self.part == 0:
                target = self.approach_position.copy()
            elif self.part == 1:
                target = self.approach_position.copy()
                target[2] = p["approach_height_m"]
            elif self.part == 2:
                target = entry.copy()
                target[2] = p["approach_height_m"]
            elif self.part == 3:
                target = entry.copy()
                target[2], target_r = p["approach_height_m"], tilted
            elif self.part == 4:
                target = entry.copy()
                target[2], target_r = p["scoop_height_m"], tilted
            elif self.part == 5:
                target = self.food_start + direction * p["scoop_travel_m"]
                target[2], target_r = p["scoop_height_m"], tilted
                target[2] += max(0., np.dot(target[:2] - self.food_start[:2], direction[:2])) * np.tan(pitch)
                if (policy_obs["interaction"][0]
                        and abs((rotation.T @ policy_obs["food_relative_world"])[0]) < p["scoop_capture_m"]):
                    self.capture_position = tcp.copy()
                    self.part = 6
            if self.part == 6:
                if phase == "TRANSPORT" and tcp[2] > p["tilt_start_height_m"]:
                    self.carry_part = 1
                target = self.capture_position.copy()
                target[2], target_r = p["lift_height_m"], tilted
                if self.carry_part:
                    target_r = yaw @ mink.SO3.exp(np.array([0., p["early_carry_pitch_rad"], 0.])).as_matrix()
            error_r = mink.SO3.from_matrix(target_r @ rotation.T).log()
            if np.linalg.norm(target - tcp) < p["waypoint_tolerance_m"] and np.linalg.norm(error_r) < .02:
                if self.part < 5:
                    self.part += 1
                elif self.part == 6 and phase == "TRANSPORT" and policy_obs["interaction"][0]:
                    if self.lift_ready_since is None:
                        self.lift_ready_since = now
                    self.pickup_lift_complete = now - self.lift_ready_since >= p["pickup_hold_s"]
            if self.part == 6 and (np.linalg.norm(target - tcp) >= p["waypoint_tolerance_m"]
                                  or np.linalg.norm(error_r) >= .02 or not policy_obs["interaction"][0]):
                self.lift_ready_since = None
        elif phase == "TRANSPORT":
            if self.carry_position is None:
                self.carry_position = mouth - mouth_r[:, 0] * p["wait_offset_m"]
                self.carry_position[2] = p["transport_height_m"]
                self.carry_part = 2
            food_local = rotation.T @ policy_obs["food_relative_world"]
            if self.carry_part == 2:
                target = self.carry_position.copy()
                target_r = yaw @ mink.SO3.exp(np.array([0., p["transport_pitch_rad"], 0.])).as_matrix()
                error_r = mink.SO3.from_matrix(target_r @ rotation.T).log()
                if (np.linalg.norm(target - tcp) < p["waypoint_tolerance_m"]
                        and np.linalg.norm(error_r) < .02 and food_local[0] <= p["capture_center_m"]
                        and policy_obs["interaction"][0]):
                    self.carry_part = 3
            else:
                target = mouth - mouth_r[:, 0] * p["wait_offset_m"]
                target_r = mouth_r
                feedforward = self.mouth_velocity
        elif phase in ("WAIT_READY", "RECOVER", "RETRACT"):
            target = mouth - mouth_r[:, 0] * p["wait_offset_m"]
            target_r, feedforward = mouth_r, self.mouth_velocity
        elif phase in ("APPROACH", "TRANSFER"):
            target = mouth + mouth_r @ np.array([p["insert_depth_m"], 0., p["insert_height_m"]])
            target_r, feedforward = mouth_r, self.mouth_velocity
            if phase == "TRANSFER" and np.linalg.norm(tcp - target) < .002:
                self.part = 1
            if self.part:
                target_r = mouth_r @ mink.SO3.exp(np.array([0., p["release_pitch_rad"], 0.])).as_matrix()
                if np.linalg.norm(mink.SO3.from_matrix(target_r @ rotation.T).log()) < .03:
                    self.part = 2
            if self.part == 2:
                target = mouth + mouth_r @ np.array([p["release_retreat_m"], 0., p["insert_height_m"]])
        error_r = mink.SO3.from_matrix(target_r @ rotation.T).log()
        world = np.r_[p["position_gain"] * (target - tcp) + feedforward,
                      p["orientation_gain"] * error_r]
        self.proposal = np.r_[self.base_rotation.T @ world[:3], self.base_rotation.T @ world[3:]]
        speed = p["transport_speed_m_s"] if phase == "TRANSPORT" and not acquiring else p["linear_speed_m_s"]
        command = np.r_[clip_norm(self.proposal[:3], min(speed, self.robot_config["linear_speed_limit"])),
                        clip_norm(self.proposal[3:], min(p["angular_speed_rad_s"], self.robot_config["angular_speed_limit"]))]
        force = np.linalg.norm(policy_obs["compensated_wrench"][:3])
        command *= min(1., max(0., (8. - force) / 2.))
        self.target_position, self.target_rotation = target.copy(), target_r.copy()
        return command
