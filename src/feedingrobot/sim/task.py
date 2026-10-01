"""Single physics owner, with opt-in M3 phase and event evaluation."""

from __future__ import annotations

import copy
import hashlib
import json

import mink
import mujoco
import numpy as np

from feedingrobot.control.adapter import RobotAdapter
from feedingrobot.sim.contacts import ContactMonitor, read_contacts
from feedingrobot.sim.model import load_model, load_json, named_id
from feedingrobot.sim.events import TaskEvents, evidence
from feedingrobot.sim.sensors import wrist_state


class FeedingTask:
    def __init__(self, robot_id="panda", timestep=None, *, task_mode=False):
        self.model, self.index, self.robot_config, self.scene_config = load_model(robot_id, timestep)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.robot_id = robot_id
        self.task_mode = task_mode
        self.task_config = load_json("configs/task.json")
        self.logic = None
        self.monitor = ContactMonitor(self.scene_config["contact_force_limit_n"])
        self.adapter = None
        self.default_food_mass = float(self.model.body_mass[self.index.food_body])
        self.default_food_inertia = self.model.body_inertia[self.index.food_body].copy()
        self.food_geom = named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "food_box")
        self.default_food_friction = self.model.geom_friction[self.food_geom].copy()
        self.default_jaw_range = self.model.jnt_range[self.index.head_joints[-1]].copy()
        self.head_body = named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.default_head_origin = self.model.body_pos[self.head_body].copy()
        self.reset()
        self.adapter = RobotAdapter(self.model, self.data, self.index, self.robot_config)
        self.provider = StateProvider(self)

    def reset(self, seed=0, preset="food_on_plate", *, scenario=None):
        if preset not in {"food_on_plate", "food_on_spoon", "empty"}:
            raise ValueError(f"Unknown reset preset: {preset}")
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        from feedingrobot.sim.scenarios import validate_scenario
        scenario = validate_scenario(scenario)
        mass = scenario.get("food_mass_kg", self.default_food_mass)
        self.model.body_mass[self.index.food_body] = mass
        self.model.body_inertia[self.index.food_body] = self.default_food_inertia * mass / self.default_food_mass
        self.model.geom_friction[self.food_geom] = self.default_food_friction
        self.model.geom_friction[self.food_geom, 0] = scenario.get("food_friction", self.default_food_friction[0])
        self.model.jnt_range[self.index.head_joints[-1]] = self.default_jaw_range
        self.model.body_pos[self.head_body] = scenario.get("head_origin_m", self.default_head_origin)
        if scenario.get("recover", False):
            self.model.jnt_range[self.index.head_joints[-1], 0] = -.65
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.tick = 0
        self.terminated = False
        self.failure_reason = None
        self.external_wrench = None
        self.monitor.reset()
        self.contacts = []
        self.substep_contact_peak_n = self.substep_wrist_peak_n = 0.
        self.scenario_state = dict(seed=self.seed, head_phase=scenario.get("head_phase_rad", 0.),
                                   future_events=[], parameters=scenario, closure_start=None)
        self.logic = TaskEvents(self.task_config) if self.task_mode else None
        self.data.qpos[self.index.qpos] = self.robot_config["reset_q"]
        self.data.ctrl[self.index.actuators] = self.robot_config["reset_q"]
        # Configuration placement is permitted only during reset.
        address = self.index.food_qpos
        self.data.qpos[address:address + 7] = [2, 2, .1, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        # Settle the robot under gravity before placing food on its actual spoon.
        for _ in range(round(self.scene_config["settle_seconds"] / self.dt)):
            self.step_physics(_settling=True)
            if self.terminated:
                raise RuntimeError(f"Reset settling failed: {self.failure_reason}")
        if preset != "empty":
            site = self.index.tcp if preset == "food_on_spoon" else named_id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
            rotation = self.data.site_xmat[site].reshape(3, 3)
            offset = (rng.uniform(-.001, .001, 2) if preset == "food_on_spoon"
                      else scenario.get("food_offset_m", rng.uniform(-.01, .01, 2)))
            xy = np.r_[offset, 0.]
            geoms = self.index.scoop_geoms if preset == "food_on_spoon" else [named_id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "collision_plate_fast_base_disk")]
            half = self.model.geom_size[self.food_geom]
            heights = []
            for x in (-half[0], 0., half[0]):
                for y in (-half[1], 0., half[1]):
                    point = self.data.site_xpos[site] + rotation @ (xy + [x, y, .05])
                    distances = []
                    for geom in geoms:
                        if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                            hit = mujoco.mj_rayMesh(self.model, self.data, geom, point, -rotation[:, 2])
                        else:
                            hit = mujoco.mju_rayGeom(self.data.geom_xpos[geom], self.data.geom_xmat[geom],
                                self.model.geom_size[geom], point, -rotation[:, 2], self.model.geom_type[geom])
                        if hit >= 0:
                            distances.append(hit)
                    if not distances:
                        raise ValueError("Food footprint is outside the collision support surface")
                    heights.append(.05 - min(distances))
            xy[2] = max(heights) + half[2] + self.scene_config["food_spawn_gap_m"]
            position = self.data.site_xpos[site] + rotation @ xy
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rotation.ravel())
        else:
            position, quat = np.array([2., 2., .1]), np.array([1., 0, 0, 0])
        self.data.qpos[address:address + 7] = np.r_[position, quat]
        food_joint = self.model.body_jntadr[self.index.food_body]
        food_dof = self.model.jnt_dofadr[food_joint]
        self.data.qvel[food_dof:food_dof + 6] = 0
        self.data.time = 0.
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[:] = 0
        self.tick = 0
        self.monitor.reset()
        self.contacts = []
        self.scenario_state["closure_start"] = None
        self.scenario_state["future_events"] = []
        self.substep_contact_peak_n = self.substep_wrist_peak_n = 0.
        mujoco.mj_forward(self.model, self.data)
        if self.adapter:
            self.adapter.reset()
        return self.snapshot()

    def state_signature(self):
        buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, None, buffer)
        digest = hashlib.sha256(buffer.tobytes())
        digest.update(json.dumps([self.robot_config, self.scene_config, self.task_config], sort_keys=True).encode())
        return dict(schema_version=2, event_rules_version=2, robot_id=self.robot_id, task_mode=self.task_mode,
                    model_config_hash=digest.hexdigest(), mujoco_version=mujoco.__version__)

    def get_state(self):
        """Complete replay state; unlike snapshot(), this includes execution memory."""
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        physics = np.empty(mujoco.mj_stateSize(self.model, kind))
        mujoco.mj_getState(self.model, self.data, physics, kind)
        adapter = {key: copy.deepcopy(getattr(self.adapter, key)) for key in
                   ("command", "last_command_time", "velocity", "status", "fault", "last_ik_velocity", "ik_failures")}
        adapter.update(reference_q=self.adapter.reference.q.copy(), target=self.adapter.target.wxyz_xyz.copy(),
                       error_detail=getattr(self.adapter, "error_detail", None))
        return dict(signature=self.state_signature(), physics=physics, adapter=adapter,
                    boundary=dict(qacc=self.data.qacc.copy(), sensordata=self.data.sensordata.copy()),
                    warnings=[(w.lastinfo, w.number) for w in self.data.warning],
                    task=copy.deepcopy({key: getattr(self, key) for key in
                                       ("seed", "tick", "terminated", "failure_reason", "external_wrench",
                                        "contacts", "scenario_state", "substep_contact_peak_n", "substep_wrist_peak_n")}),
                    monitor=copy.deepcopy(vars(self.monitor)),
                    logic=copy.deepcopy(vars(self.logic)) if self.logic else None)

    def set_state(self, state):
        if state["signature"] != self.state_signature():
            raise ValueError("Incompatible robot/model/configuration or snapshot schema")
        state = copy.deepcopy(state)
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state["physics"], kind)
        mujoco.mj_forward(self.model, self.data)
        # Forward computes derived geometry/sensors but may touch integration memory.
        mujoco.mj_setState(self.model, self.data, state["physics"], kind)
        self.data.qacc[:] = state["boundary"]["qacc"]
        self.data.sensordata[:] = state["boundary"]["sensordata"]
        for warning, (lastinfo, number) in zip(self.data.warning, state["warnings"]):
            warning.lastinfo, warning.number = lastinfo, number
        for key, value in state["task"].items():
            setattr(self, key, value)
        self.monitor.__dict__.update(state["monitor"])
        if self.logic:
            self.logic.__dict__.update(state["logic"])
        adapter = state["adapter"]
        self.adapter.reference.update(adapter.pop("reference_q"))
        self.adapter.target = mink.SE3(adapter.pop("target"))
        self.adapter.frame.set_target(self.adapter.target)
        self.adapter.__dict__.update(adapter)

    def set_external_wrench(self, force, torque, point):
        values = np.asarray([force, torque, point], dtype=float)
        if values.shape != (3, 3) or not np.isfinite(values).all():
            raise ValueError("External force, torque and point must be finite 3-vectors")
        self.external_wrench = values.copy()

    def clear_external_wrench(self):
        self.external_wrench = None

    def _write_drivers(self):
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[:] = 0
        if self.external_wrench is not None:
            force, torque, point = self.external_wrench
            body = self.index.tool_body
            self.data.xfrc_applied[body, :3] = force
            self.data.xfrc_applied[body, 3:] = torque + np.cross(point - self.data.xipos[body], force)
        cfg = self.scene_config["head"]
        scenario = self.scenario_state["parameters"]
        w = 2 * np.pi * scenario.get("head_freq_hz", cfg["freq_hz"])
        t = self.tick * self.dt
        amp = scenario.get("head_amp_m", cfg["amp_m"])
        amplitude = np.array([amp, amp, 0., cfg["yaw_amp_rad"], cfg["jaw_amp_rad"]])
        if self.scene_config["head_fixed"]:
            amplitude[:] = 0
        phase = self.scenario_state["head_phase"]
        desired = amplitude * np.sin(w * t + phase)
        desired[:3] += scenario.get("head_offset_m", [0., 0., 0.])
        desired[4] += cfg["jaw_center_rad"]
        velocity = amplitude * w * np.cos(w * t + phase)
        if scenario.get("recover", False) and self.logic and self.logic.phase == "APPROACH":
            if self.scenario_state["closure_start"] is None:
                self.scenario_state["closure_start"] = t
                self.scenario_state["future_events"].append(dict(name="jaw_closure", time=t))
        start = self.scenario_state["closure_start"]
        if start is not None and t - start < 1.5:
            desired[4], velocity[4] = -.6, 0.
        joints = self.index.head_joints
        dofs = self.model.jnt_dofadr[joints]
        position = self.data.qpos[self.model.jnt_qposadr[joints]]
        command = (np.array(cfg["kp"]) * (desired - position)
                   + np.array(cfg["kd"]) * (velocity - self.data.qvel[dofs]) + self.data.qfrc_bias[dofs])
        ids = self.index.head_actuators
        self.data.ctrl[ids] = np.clip(command, self.model.actuator_ctrlrange[ids, 0], self.model.actuator_ctrlrange[ids, 1])

    def _terminate(self, reason):
        self.terminated = True
        self.failure_reason = reason
        if self.task_mode and reason == "nonfinite_state" and self.logic.failure_reason is None:
            self.logic.failure_reason = reason
            self.logic.emit("failure", self.data.time, reason=reason)
        if self.adapter:
            self.adapter.stop(reason, fault=True)

    def step_physics(self, *, _settling=False):
        if self.terminated:
            raise RuntimeError(f"Episode terminated: {self.failure_reason}; reset required")
        self.substep_contact_peak_n = self.substep_wrist_peak_n = 0.
        if not all(np.isfinite(a).all() for a in [self.data.qpos, self.data.qvel, self.data.ctrl]):
            self._terminate("nonfinite_state")
            return {"terminated": True, "failure_reason": self.failure_reason}
        if self.adapter and not _settling:
            self.adapter.update(self.dt)
            if self.task_mode and self.adapter.fault:
                self._terminate(self.adapter.fault)
                self.logic.update(evidence(self), 0., self.data.time, failure=self.failure_reason)
                return self.snapshot()
        self._write_drivers()
        mujoco.mj_step(self.model, self.data)
        self.tick += 1
        # These are the loads actually used by the integrator. Sampling only
        # after mj_forward could miss a short contact that has already separated.
        applied_contacts = read_contacts(self.model, self.data, self.index)
        overloaded = self.monitor.update(applied_contacts, self.dt, float(self.data.time))
        self.substep_contact_peak_n = self.monitor.last_peak_n
        applied_wrist = wrist_state(self.model, self.data, self.index)
        mujoco.mj_forward(self.model, self.data)
        if (not all(np.isfinite(a).all() for a in [self.data.qpos, self.data.qvel, self.data.qacc])
                or any(self.data.warning[w].number for w in [mujoco.mjtWarning.mjWARN_BADQPOS,
                                                            mujoco.mjtWarning.mjWARN_BADQVEL,
                                                            mujoco.mjtWarning.mjWARN_BADQACC])):
            self._terminate("nonfinite_state")
            return {"terminated": True, "failure_reason": self.failure_reason}
        self.contacts = read_contacts(self.model, self.data, self.index)
        # Also guard the current boundary state, without counting impulse twice.
        overloaded = self.monitor.update(self.contacts, 0., float(self.data.time)) or overloaded
        self.substep_contact_peak_n = max(self.substep_contact_peak_n, self.monitor.last_peak_n)
        sensors = wrist_state(self.model, self.data, self.index)
        wrist_peak = max(np.linalg.norm(sensors["compensated_wrench"][:3]),
                         np.linalg.norm(applied_wrist["compensated_wrench"][:3]))
        self.substep_wrist_peak_n = float(wrist_peak)
        if overloaded or wrist_peak > self.scene_config["wrist_force_limit_n"]:
            self._terminate("contact_limit")
        elif np.max(np.abs(self.data.qvel[self.index.dofs])) > self.scene_config["joint_speed_fault_rad_s"]:
            self._terminate("joint_speed_limit")
        if self.task_mode and not _settling:
            deep = any(r["distance"] < -self.task_config["penetration_limit_m"] for r in applied_contacts)
            self.logic.update(evidence(self), self.dt, self.data.time, failure=self.failure_reason, penetration=deep)
            if self.logic.failure_reason:
                self._terminate(self.logic.failure_reason)
            elif self.logic.success:
                self.terminated = True
                self.adapter.stop("success")
        return self.snapshot(sensors)

    def snapshot(self, sensors=None):
        idx = self.index
        if sensors is None:
            sensors = wrist_state(self.model, self.data, idx)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, idx.tcp, velocity, 0)
        return dict(robot_id=self.robot_id, time=float(self.data.time), tick=self.tick,
                    q=self.data.qpos[idx.qpos].copy(), dq=self.data.qvel[idx.dofs].copy(),
                    tcp_position=self.data.site_xpos[idx.tcp].copy(),
                    tcp_rotation=self.data.site_xmat[idx.tcp].reshape(3, 3).copy(),
                    tcp_twist_world=np.r_[velocity[3:], velocity[:3]],
                    food_position=self.data.xpos[idx.food_body].copy(),
                    execution_status=self.adapter.status if self.adapter else "resetting",
                    terminated=self.terminated, failure_reason=self.failure_reason,
                    contact_peak_n=self.monitor.peak_n, contact_impulse_ns=self.monitor.impulse_ns,
                    contact_over_limit_s=self.monitor.over_limit_s,
                    actuator_force=self.data.actuator_force[idx.actuators].copy(),
                    **sensors)


class StateProvider:
    def __init__(self, task):
        self.task = task

    def observe(self):
        task = self.task
        state = task.snapshot()
        mouth = named_id(task.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_entry")
        tcp = state["tcp_position"]
        policy_keys = ["robot_id", "time", "q", "dq", "tcp_position", "tcp_rotation", "tcp_twist_world",
                       "raw_wrench_sensor", "wrench_world_at_tcp", "compensated_wrench", "execution_status"]
        obs = {k: state[k] for k in policy_keys}
        obs.update(food_relative_world=state["food_position"] - tcp,
                   mouth_relative_world=task.data.site_xpos[mouth].copy() - tcp,
                   frame_age_s=0., stage="m1_diagnostic",
                   current_tool_contact=any("spoon" in [r["group1"], r["group2"]] for r in task.contacts))
        if task.task_mode:
            e = evidence(task)
            obs.update(stage=task.logic.phase, mouth_rotation=e["mouth_rotation"], mouth_aperture_m=e["aperture_m"],
                       interaction=np.array([e[k] for k in ("supported", "mouth_supported", "tool_mouth_contact", "ready")],
                                            dtype=np.float32))
        return dict(policy_obs=obs,
                    oracle_info=dict(contacts=copy.deepcopy(task.contacts), terminated=task.terminated,
                                     failure_reason=task.failure_reason, contact_peak_n=task.monitor.peak_n,
                                     contact_impulse_ns=task.monitor.impulse_ns,
                                     contact_over_limit_s=task.monitor.over_limit_s,
                                     **(dict(phase=task.logic.phase, timers=copy.deepcopy(task.logic.timers),
                                             events=copy.deepcopy(task.logic.events), success=task.logic.success)
                                        if task.task_mode else {})),
                    scenario_state=copy.deepcopy(task.scenario_state))
