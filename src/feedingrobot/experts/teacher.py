"""Geometry-based P0 teacher; dynamic inputs are only current/past policy_obs."""

import copy

import mink
import numpy as np

from feedingrobot.control.adapter import clip_norm


class Teacher:
    def __init__(self, robot_config, config):
        self.robot_config = copy.deepcopy(robot_config)
        self.config = copy.deepcopy(config)
        if config.get("schema_version") != 2:
            raise ValueError("New-tableware teacher requires configuration schema 2")
        self.base_rotation = mink.SO3(np.asarray(robot_config["base_quaternion"])).as_matrix()
        self.geometry = None

    def reset(self, parameters, *, geometry):
        self.parameters = dict(self.config["teacher"], **copy.deepcopy(parameters))
        self.geometry = copy.deepcopy(geometry)
        for value in self.geometry.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self.phase = None
        self.part = self.carry_part = 0
        self.food_start = self.approach_position = self.capture_position = None
        self.carry_position = None
        self.lift_ready_since = None
        self.pickup_lift_complete = False
        self.capture_roll_complete = False
        self.capture_food = self.capture_food_local = self.capture_rotation = None
        self.last_mouth = self.last_food_local = self.last_time = None
        self.mouth_velocity = np.zeros(3)
        self.food_velocity_local = np.zeros(3)
        self.proposal = np.zeros(6)
        self.target_position = self.target_rotation = None

    def acquisition_waypoints(self, policy_obs):
        p, g = self.parameters, self.geometry
        plate_r, plate_p = g["plate_rotation"], g["plate_position"]
        yaw = mink.SO3.exp(np.array([0., 0., p["scoop_yaw_rad"]])).as_matrix()
        direction = yaw[:, 0]
        pitch = p["scoop_pitch_rad"]
        if self.part >= 3:
            advance = np.dot(plate_r.T @ (policy_obs["tcp_position"]-self.food_start), direction)
            fraction = np.clip((advance+p["arc_start_offset_m"])/p["arc_length_m"], 0., 1.)
            pitch += fraction*(p["scoop_exit_pitch_rad"]-pitch)
        rotation = plate_r @ yaw @ mink.SO3.exp(np.array([0., pitch, 0.])).as_matrix()
        local = (g["scoop_points"] @ rotation.T) @ plate_r
        food = plate_r.T @ (self.food_start - plate_p)
        entry = food - direction * p["scoop_start_offset_m"]
        height = -local[:, 2].min() + p["plate_gap_m"]
        if self.part == 3:
            actual = (g["scoop_points"] @ np.asarray(policy_obs["tcp_rotation"]).T) @ plate_r
            height = max(height, -actual[:, 2].min() + p["plate_gap_m"])
        entry[2] = height
        end = food + direction * p["scoop_travel_m"]
        end[2] = height
        above = entry.copy()
        above[2] = p["approach_clearance_m"]
        lift = plate_r.T @ ((self.capture_position if self.capture_position is not None else plate_p + plate_r @ end) - plate_p)
        lift[2] = p["lift_clearance_m"]
        return rotation, [plate_p + plate_r @ point for point in (above, entry, end, lift)]

    def act(self, policy_obs):
        if self.geometry is None:
            raise ValueError("Teacher.reset requires measured geometry before act")
        p, g = self.parameters, self.geometry
        tcp, rotation = np.asarray(policy_obs["tcp_position"]), np.asarray(policy_obs["tcp_rotation"])
        food = tcp + policy_obs["food_relative_world"]
        mouth = tcp + policy_obs["mouth_relative_world"]
        mouth_r = np.asarray(policy_obs["mouth_rotation"])
        phase, now = policy_obs["stage"], policy_obs["time"]
        supported = policy_obs["interaction"][0]
        food_local = rotation.T @ policy_obs["food_relative_world"]
        bounds = g["scoop_points"][:, :2]
        interior = (np.all(food_local[:2] >= bounds.min(0) + p["capture_margin_m"])
                    and np.all(food_local[:2] <= bounds.max(0) - p["capture_margin_m"]))
        if self.last_time is not None and now > self.last_time:
            dt = now - self.last_time
            self.mouth_velocity = .5 * self.mouth_velocity + .5 * clip_norm((mouth-self.last_mouth)/dt, .025)
            self.food_velocity_local = .5*self.food_velocity_local + .5*(food_local-self.last_food_local)/dt
        self.last_mouth, self.last_food_local, self.last_time = mouth.copy(), food_local.copy(), now
        if phase != self.phase:
            if phase not in ("SELECT", "ACQUIRE", "TRANSPORT"):
                self.part = 0
            self.phase = phase
        if self.food_start is None:
            self.food_start, self.approach_position = food.copy(), tcp.copy()
        scoop_r, waypoints = self.acquisition_waypoints(policy_obs)
        target, target_r, feedforward = tcp.copy(), scoop_r, np.zeros(3)
        acquiring = phase in ("SELECT", "ACQUIRE") or (phase == "TRANSPORT" and not self.pickup_lift_complete)
        if acquiring:
            if self.part == 0:
                target = self.approach_position.copy()
            elif self.part == 1:
                target = waypoints[0]
            elif self.part == 2:
                target = waypoints[1]
            elif self.part == 3:
                target = waypoints[2]
                if supported and interior:
                    self.capture_position, self.part = tcp.copy(), 4
                    self.capture_food, self.capture_food_local, self.capture_rotation = food.copy(), food_local.copy(), rotation.copy()
            if phase == "TRANSPORT" and self.part < 4 and supported:
                self.capture_position, self.part = tcp.copy(), 4
                self.capture_food, self.capture_food_local, self.capture_rotation = food.copy(), food_local.copy(), rotation.copy()
            if self.part == 4:
                level_r = (g["plate_rotation"] @ mink.SO3.exp(np.array([0.,0.,p["scoop_yaw_rad"]])).as_matrix()
                           @ mink.SO3.exp(np.array([0.,p["scoop_exit_pitch_rad"],0.])).as_matrix())
                if not self.capture_roll_complete:
                    turn = mink.SO3.from_matrix(level_r @ self.capture_rotation.T).log()
                    angle = np.linalg.norm(turn)
                    remaining = np.linalg.norm(mink.SO3.from_matrix(level_r @ rotation.T).log())
                    fraction = 1. if angle < 1e-6 else np.clip(1.-remaining/angle,0.,1.)
                    next_fraction = 1. if angle < 1e-6 else min(1.,fraction+p["angular_speed_rad_s"]*.1/angle)
                    target_r = mink.SO3.exp(turn*next_fraction).as_matrix() @ self.capture_rotation
                    normal = g["plate_rotation"][:,2]
                    level_tcp = self.capture_food-level_r @ self.capture_food_local
                    rise = max(0.,p["approach_clearance_m"]-np.dot(level_tcp-g["plate_position"],normal))
                    # Lift the contact pivot while turning, instead of forcing
                    # the curved cup down onto the plate as it becomes level.
                    pivot = self.capture_food+normal*rise*next_fraction
                    target = pivot-target_r @ self.capture_food_local
                    centered = (np.all(food_local[:2] >= bounds.min(0)+p["carry_margin_m"])
                                and np.all(food_local[:2] <= bounds.max(0)-p["carry_margin_m"]))
                    if (supported and centered and np.linalg.norm(mink.SO3.from_matrix(level_r @ rotation.T).log())
                            < p["orientation_tolerance_rad"]):
                        self.capture_roll_complete = True
                        self.capture_position = tcp.copy()
                else:
                    target = self.acquisition_waypoints(policy_obs)[1][3]
                    target_r = level_r
                if self.capture_roll_complete and phase == "TRANSPORT" and np.dot(tcp-g["plate_position"], g["plate_rotation"][:, 2]) >= p["tilt_clearance_m"]:
                    self.carry_part = 1
                if self.carry_part:
                    target_r = g["plate_rotation"] @ mink.SO3.exp(np.array([0., p["carry_pitch_rad"], p["scoop_yaw_rad"]])).as_matrix()
            reached = (np.linalg.norm(target-tcp) < p["waypoint_tolerance_m"]
                       and np.linalg.norm(mink.SO3.from_matrix(target_r @ rotation.T).log()) < p["orientation_tolerance_rad"])
            if reached and self.part < 3:
                self.part += 1
            elif reached and self.part == 4 and self.capture_roll_complete and phase == "TRANSPORT" and supported:
                if self.lift_ready_since is None:
                    self.lift_ready_since = now
                self.pickup_lift_complete = now-self.lift_ready_since >= p["pickup_hold_s"]
            else:
                self.lift_ready_since = None
        elif phase == "TRANSPORT":
            if self.carry_position is None:
                self.carry_position = tcp.copy()
                self.carry_position[2] = max(tcp[2], mouth[2])
                self.carry_part = 2
            if self.carry_part == 2:
                target, target_r = self.carry_position.copy(), mouth_r
                if (np.linalg.norm(target-tcp) < p["waypoint_tolerance_m"]
                        and np.linalg.norm(mink.SO3.from_matrix(target_r @ rotation.T).log()) < p["orientation_tolerance_rad"]
                        and supported and interior and np.linalg.norm(self.food_velocity_local[:2]) <= p["max_slide_speed_m_s"]):
                    self.carry_part = 3
            else:
                target = mouth-mouth_r[:, 0]*g["task_config"]["wait_offset_m"]
                target_r, feedforward = mouth_r, self.mouth_velocity
        elif phase in ("WAIT_READY", "RECOVER", "RETRACT"):
            target = mouth-mouth_r[:, 0]*g["task_config"]["wait_offset_m"]
            target_r, feedforward = mouth_r, self.mouth_velocity
        elif phase in ("APPROACH", "TRANSFER"):
            target = mouth+mouth_r @ np.array([p["insert_depth_m"], 0., p["insert_height_m"]])
            target_r, feedforward = mouth_r, self.mouth_velocity
            if phase == "TRANSFER" and np.linalg.norm(tcp-target) < p["waypoint_tolerance_m"]:
                self.part = 1
            if self.part:
                target_r = mouth_r @ mink.SO3.exp(np.array([0., p["release_pitch_rad"], 0.])).as_matrix()
            # Stay in the receiver until the environment confirms physical delivery.
            # RETRACT is owned exclusively by TaskEvents, never by the teacher.
        error_r = mink.SO3.from_matrix(target_r @ rotation.T).log()
        world = np.r_[p["position_gain"]*(target-tcp)+feedforward, p["orientation_gain"]*error_r]
        self.proposal = np.r_[self.base_rotation.T @ world[:3], self.base_rotation.T @ world[3:]]
        speed = p["transport_speed_m_s"] if phase == "TRANSPORT" and not acquiring else p["linear_speed_m_s"]
        if acquiring and self.part == 3:
            sweep = abs(p["scoop_exit_pitch_rad"]-p["scoop_pitch_rad"])
            if sweep:
                speed = min(speed, .8*p["angular_speed_rad_s"]*p["arc_length_m"]/sweep)
        if acquiring and self.part == 4 and not self.capture_roll_complete:
            speed = min(speed, p["roll_speed_m_s"])
        command = np.r_[clip_norm(self.proposal[:3], min(speed, self.robot_config["linear_speed_limit"])),
                        clip_norm(self.proposal[3:], min(p["angular_speed_rad_s"], self.robot_config["angular_speed_limit"]))]
        force = np.linalg.norm(policy_obs["compensated_wrench"][:3])
        command *= min(1., max(0., (8.-force)/2.))
        self.target_position, self.target_rotation = target.copy(), target_r.copy()
        return command
