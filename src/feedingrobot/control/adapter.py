"""A small, time-stamped TCP reference adapter; never advances physics."""

from __future__ import annotations

import itertools

import mink
import mujoco
import numpy as np


def clip_norm(vector, limit):
    return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-15))


class RobotAdapter:
    def __init__(self, model, data, index, config):
        self.model, self.data, self.index, self.config = model, data, index, config
        self.frame = mink.FrameTask(config["tcp_site"], "site", position_cost=1., orientation_cost=1., lm_damping=1e-4)
        self.frozen_dofs = np.setdiff1d(np.arange(model.nv), index.dofs)
        self.freeze = mink.DofFreezingTask(model, self.frozen_dofs.tolist())
        # Self collision and robot/environment avoidance. Task contact by the spoon is allowed.
        by_body = [[g for g in index.arm_geoms if model.geom_bodyid[g] == b] for b in sorted(index.arm_bodies)]
        pairs = [(a, b) for a, b in itertools.combinations(by_body, 2) if a and b]
        obstacles = [g for g in range(model.ngeom) if index.group(model, g) in {"table", "plate", "mouth", "floor"}]
        pairs.append((index.arm_geoms, obstacles))
        self.limits = [mink.ConfigurationLimit(model),
                       mink.VelocityLimit(model, {n: config["joint_velocity_limit"] for n in config["joints"]}),
                       mink.CollisionAvoidanceLimit(model, pairs, minimum_distance_from_collisions=0.005,
                                                    collision_detection_distance=0.02)]
        self.reset()

    def reset(self):
        # Scenario reset may change the jaw range. Frozen non-arm DOFs must
        # use those current bounds, rather than the constructor's cached range.
        self.limits[0] = mink.ConfigurationLimit(self.model)
        self.reference = mink.Configuration(self.model)
        self.reference.update(self.data.qpos.copy())
        self.target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.data.site_xmat[self.index.tcp].reshape(3, 3)),
            self.data.site_xpos[self.index.tcp].copy())
        self.frame.set_target(self.target)
        self.command = None
        self.last_command_time = -np.inf
        self.velocity = np.zeros(6)
        self.status = "idle"
        self.fault = None
        self.last_ik_velocity = np.zeros(self.model.nv)
        self.ik_failures = 0
        self.error_detail = None
        self._hold()

    def _hold(self):
        ids = self.index.actuators
        q = self.data.qpos[self.index.qpos]
        self.data.ctrl[ids] = np.clip(q, self.model.actuator_ctrlrange[ids, 0], self.model.actuator_ctrlrange[ids, 1])
        self.reference.update(self.data.qpos.copy())
        self.target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.data.site_xmat[self.index.tcp].reshape(3, 3)), self.data.site_xpos[self.index.tcp].copy())
        self.frame.set_target(self.target)

    def stop(self, reason="stopped", fault=False, *, hold_reference=False):
        self.command = None
        if fault or not hold_reference:
            self.velocity[:] = 0
        self.last_ik_velocity[:] = 0
        self.status = reason
        if fault:
            self.fault = reason
        # Phase cancellation preserves the PD load offset and lets the existing
        # acceleration limits bring the reference to rest without a velocity jump.
        if np.isfinite(self.data.qpos).all() and (fault or not hold_reference):
            self._hold()

    def set_twist(self, twist, command_time, valid_until):
        value = np.asarray(twist, dtype=float)
        now = float(self.data.time)
        if value.shape != (6,) or not np.isfinite(value).all() or not np.isfinite([command_time, valid_until]).all():
            self.stop("invalid_command", fault=True)
            raise ValueError("Expected a finite six-dimensional twist and finite timestamps")
        if self.fault:
            raise RuntimeError(f"Reset required after {self.fault}")
        if command_time < self.last_command_time or command_time > now + 1e-9:
            raise ValueError("Commands must be ordered and cannot be future-dated")
        self.last_command_time = float(command_time)
        if valid_until <= now or valid_until <= command_time:
            self.stop("expired")
            return
        self.command = (value.copy(), float(valid_until))
        self.status = "active"

    def update(self, dt):
        if self.command is None:
            if self.status != "stopped" or np.linalg.norm(self.velocity) <= 1e-12:
                self.velocity[:] = 0
                return
        elif self.data.time >= self.command[1] - 1e-12:
            self.stop("expired")
            return
        cfg, idx = self.config, self.index
        actual_p = self.data.site_xpos[idx.tcp]
        actual_r = mink.SO3.from_matrix(self.data.site_xmat[idx.tcp].reshape(3, 3))
        pos_error = np.linalg.norm(self.target.translation() - actual_p)
        rot_error = np.linalg.norm((self.target.rotation() @ actual_r.inverse()).log())
        if pos_error >= cfg["reference_position_limit"] - 1e-9 or rot_error >= cfg["reference_rotation_limit"] - 1e-9:
            self.stop("blocked", fault=True)
            return
        requested = self.command[0] if self.command is not None else np.zeros(6)
        desired = np.r_[clip_norm(requested[:3], cfg["linear_speed_limit"]),
                        clip_norm(requested[3:], cfg["angular_speed_limit"])]
        self.velocity[:3] += clip_norm(desired[:3] - self.velocity[:3], cfg["linear_acceleration_limit"] * dt)
        self.velocity[3:] += clip_norm(desired[3:] - self.velocity[3:], cfg["angular_acceleration_limit"] * dt)
        base_r = self.data.site_xmat[idx.base].reshape(3, 3)
        p = self.target.translation() + base_r @ self.velocity[:3] * dt
        p_base = base_r.T @ (p - self.data.site_xpos[idx.base])
        if np.any(p_base < cfg["workspace_min"]) or np.any(p_base > cfg["workspace_max"]):
            self.stop("workspace_limit", fault=True)
            return
        r = mink.SO3.exp(base_r @ self.velocity[3:] * dt) @ self.target.rotation()
        # Clamp the proposed target too, so one update cannot cross the deviation bound.
        p = actual_p + clip_norm(p - actual_p, cfg["reference_position_limit"])
        delta_r = clip_norm((r @ actual_r.inverse()).log(), cfg["reference_rotation_limit"])
        self.target = mink.SE3.from_rotation_and_translation(mink.SO3.exp(delta_r) @ actual_r, p)
        self.frame.set_target(self.target)
        # Synchronize scenario coordinates without modifying the arm reference.
        q = self.data.qpos.copy()
        q[idx.qpos] = self.reference.q[idx.qpos]
        self.reference.update(q)
        try:
            velocity = mink.solve_ik(self.reference, [self.frame], dt, solver="daqp", limits=self.limits,
                                     constraints=[self.freeze], damping=1e-5, safety_break=True)
            if not np.isfinite(velocity).all():
                raise ValueError("Non-finite IK result")
            if np.max(np.abs(velocity[self.frozen_dofs])) > 1e-7:
                raise ValueError("IK moved a frozen degree of freedom")
            self.last_ik_velocity = velocity.copy()
            self.reference.integrate_inplace(velocity, dt)
            ids = idx.actuators
            self.data.ctrl[ids] = np.clip(self.reference.q[idx.qpos], self.model.actuator_ctrlrange[ids, 0],
                                        self.model.actuator_ctrlrange[ids, 1])
        except (mink.exceptions.MinkError, ValueError, np.linalg.LinAlgError) as exc:
            self.ik_failures += 1
            self.stop("ik_failure", fault=True)
            self.error_detail = str(exc)
