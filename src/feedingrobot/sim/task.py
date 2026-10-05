"""Single physics owner, with opt-in M3 phase and event evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import time

import mink
import mujoco
import numpy as np

from feedingrobot.control.adapter import RobotAdapter
from feedingrobot.sim.contacts import ContactMonitor, read_contacts
from feedingrobot.sim.model import load_model, load_json, named_id
from feedingrobot.sim.events import TaskEvents, evidence
from feedingrobot.sim.sensors import wrist_state
from feedingrobot.sim.beans import bean_state, bean_diagnostics, place_beans, spawn_clearance


class FeedingTask:
    def __init__(self, robot_id="panda", timestep=None, *, task_mode=False):
        started = time.perf_counter()
        self.model, self.index, self.robot_config, self.scene_config = load_model(robot_id, timestep)
        self.model_load_wall_s = time.perf_counter() - started
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.robot_id = robot_id
        self.task_mode = task_mode
        self.task_config = load_json("configs/task.json")
        self.bean_acceptance = load_json("configs/acceptance.json")["beans_native"]
        self.logic = None
        self.monitor = ContactMonitor(self.scene_config["contact_force_limit_n"])
        self.adapter = None
        self.default_jaw_range = self.model.jnt_range[self.index.head_joints[-1]].copy()
        self.head_body = named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.default_head_origin = self.model.body_pos[self.head_body].copy()
        self.reset()
        self.adapter = RobotAdapter(self.model, self.data, self.index, self.robot_config)
        if self.terminated:
            self.adapter.stop(self.failure_reason, fault=True)
        self.provider = StateProvider(self)

    def reset(self, seed=0, preset="beans_in_bowl", *, scenario=None):
        if preset not in {"beans_in_bowl", "beans_on_spoon", "empty"}:
            raise ValueError(f"Unknown reset preset: {preset}")
        from feedingrobot.sim.scenarios import validate_scenario
        scenario = validate_scenario(scenario)
        if (set(scenario) & {"food_mass_kg", "food_friction", "food_offset_m"}
                or ("recover" in scenario and not self.task_mode)):
            raise ValueError("Single-food scenario parameters are not supported by native Beans")
        self.seed = int(seed)
        self.model.jnt_range[self.index.head_joints[-1]] = self.default_jaw_range
        self.model.body_pos[self.head_body] = scenario.get("head_origin_m", self.default_head_origin)
        if scenario.get("recover", False):
            self.model.jnt_range[self.index.head_joints[-1], 0] = -.65
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.tick = 0
        self.physics_timing = dict(mj_step_s=0., forward_s=0., steps=0)
        self.terminated = False
        self.failure_reason = None
        self.external_wrench = None
        self.monitor.reset()
        self.contacts, self.applied_contacts = [], []
        self.substep_contact_peak_n = self.substep_wrist_peak_n = 0.
        self.scenario_state = dict(seed=self.seed, head_phase=scenario.get("head_phase_rad", 0.),
                                  future_events=[], parameters=scenario, closure_start=None)
        self.logic = None
        self.data.qpos[self.index.qpos] = self.robot_config["reset_q"]
        self.data.ctrl[self.index.actuators] = self.robot_config["reset_q"]
        place_beans(self, "empty", self.seed)
        mujoco.mj_forward(self.model, self.data)
        if self.adapter:
            self.adapter.reset()
        started = time.monotonic()
        for _ in range(round(self.scene_config["settle_seconds"] / self.dt)):
            self.step_physics(_settling=True)
            if self.terminated:
                raise RuntimeError(f"Robot reset settling failed: {self.failure_reason}")
        robot_wall = time.monotonic() - started
        robot_timing = dict(self.physics_timing)
        self.physics_timing = dict(mj_step_s=0., forward_s=0., steps=0)
        place_beans(self, preset, self.seed)
        mujoco.mj_forward(self.model, self.data)
        self.contacts = read_contacts(self.model, self.data, self.index)
        self.applied_contacts = []
        clearance = spawn_clearance(self, preset)
        cfg = self.bean_acceptance
        self.reset_diagnostics = dict(preset=preset, seed=self.seed, status="passed",
                                      spawn_clearance_m=clearance, robot_settle_wall_s=robot_wall,
                                      robot_performance=robot_timing)
        if clearance < cfg["spawn_gap_m"] - 1e-9:
            raise ValueError(f"Bean spawn clearance {clearance} is below required gap")
        if preset != "empty":
            self._settle_beans(preset)
        self.data.time = 0.
        self.tick = 0
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[:] = 0
        self.monitor.reset()
        self.contacts = read_contacts(self.model, self.data, self.index)
        self.applied_contacts = []
        self.substep_contact_peak_n = self.substep_wrist_peak_n = 0.
        if self.adapter:
            self.adapter.reset()
            if self.terminated:
                self.adapter.stop(self.failure_reason, fault=True)
        if self.task_mode:
            self.logic = TaskEvents(self.task_config)
        return self.snapshot()

    def _settle_beans(self, preset):
        cfg = self.bean_acceptance
        window = best = maximum = low_window = best_low = 0.
        peak = None
        started = time.monotonic()
        initial = self.data.qpos.copy()
        state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
        initial_physics = np.empty(mujoco.mj_stateSize(self.model, state_kind))
        mujoco.mj_getState(self.model, self.data, initial_physics, state_kind)
        trace = [dict(time_s=0., **bean_state(self.model, self.data, self.index))]
        violations = []
        trace_stride = round(.01 / self.dt)
        contact_evidence = {}
        violating_ids = set()
        for step in range(round(cfg["max_settle_s"] / self.dt)):
            self.step_physics(_settling=True)
            diag = bean_diagnostics(self)
            for row in self.applied_contacts + self.contacts:
                if row.get("bean1_id") or row.get("bean2_id"):
                    key = (row["geom1"], row["geom2"])
                    if key not in contact_evidence or row["force_n"] > contact_evidence[key]["force_n"]:
                        contact_evidence[key] = dict(time_s=(step + 1) * self.dt, **row)
            if diag["max_penetration_m"] > maximum:
                maximum = diag["max_penetration_m"]
                peak = dict(diag["penetration_peak"],
                            time_s=(step + (0 if diag["penetration_peak"]["source"] == "applied_contact" else 1)) * self.dt)
            violating_ids.update(diag["penetration_ids"])
            supported = diag["bowl_supported"].copy()
            if preset == "beans_on_spoon":
                supported[0] = diag["spoon_supported"][0]
            inside = diag["in_bowl"].copy()
            if preset == "beans_on_spoon":
                inside[0] = True
            low = (diag["linear_speed_m_s"] < cfg["linear_speed_m_s"]) & (diag["angular_speed_rad_s"] < cfg["angular_speed_rad_s"])
            low_window = low_window + self.dt if low.all() else 0.
            best_low = max(best_low, low_window)
            window = window + self.dt if low.all() and supported.all() and inside.all() else 0.
            best = max(best, window)
            if diag["max_penetration_m"] > cfg["penetration_limit_m"] and not any(v["reason"] == "penetration" for v in violations):
                violations.append(dict(time_s=(step + 1) * self.dt, reason="penetration", bean_ids=diag["penetration_ids"]))
            violating_ids.update(np.array(self.index.bean_ids)[~inside].tolist())
            if not inside.all() and not any(v["reason"] == "outside_bowl" for v in violations):
                violations.append(dict(time_s=(step + 1) * self.dt, reason="outside_bowl", bean_ids=np.array(self.index.bean_ids)[~inside].tolist(),
                                       boundary_violations=[v for v in diag["boundary_violations"] if v["bean_id"] in np.array(self.index.bean_ids)[~inside]]))
            if (step + 1) % trace_stride == 0:
                trace.append(dict(time_s=(step + 1) * self.dt, **bean_state(self.model, self.data, self.index)))
            if self.terminated or any(w.number for w in self.data.warning) or time.monotonic() - started > cfg["wall_budget_s"]:
                break
            if window + 1e-12 >= cfg["stable_window_s"]:
                break
        passed = (window + 1e-12 >= cfg["stable_window_s"] and not violations and not self.terminated
                  and not any(w.number for w in self.data.warning) and time.monotonic() - started <= cfg["wall_budget_s"])
        self.physics_timing["pure_physics_rtf"] = ((step + 1) * self.dt / self.physics_timing["mj_step_s"])
        self.physics_timing["headless_reset_rtf"] = ((step + 1) * self.dt / (time.monotonic() - started))
        self.physics_timing["ik_wall_s"] = 0.
        final_physics = np.empty_like(initial_physics)
        mujoco.mj_getState(self.model, self.data, final_physics, state_kind)
        self.reset_diagnostics.update(state_spec="mjSTATE_INTEGRATION", initial_physics_state=initial_physics,
            final_physics_state=final_physics, performance=dict(self.physics_timing), status="passed" if passed else "failed", stable_window_s=best,
            bean_settle_s=(step + 1) * self.dt, bean_settle_wall_s=time.monotonic() - started,
            max_penetration_m=maximum, penetration_peak=peak, low_speed_window_s=best_low,
            violations=violations, initial_qpos=initial, failure_ids=sorted(violating_ids | set(np.array(self.index.bean_ids)[~(low & supported & inside)].tolist())),
            final=diag, trajectory=trace, contact_evidence=list(contact_evidence.values()), warning_counts=[w.number for w in self.data.warning])
        if not passed:
            self.terminated = True
            self.failure_reason = self.failure_reason or "bean_reset_settling_failed"

    def state_signature(self):
        buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, None, buffer)
        digest = hashlib.sha256(buffer.tobytes())
        digest.update(json.dumps([self.robot_config, self.scene_config, self.task_config, self.bean_acceptance], sort_keys=True).encode())
        return dict(schema_version=4 if self.task_mode else 3, event_rules_version=3 if self.task_mode else 2, robot_id=self.robot_id, task_mode=self.task_mode,
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
                                        "contacts", "applied_contacts", "reset_diagnostics", "physics_timing", "scenario_state", "substep_contact_peak_n", "substep_wrist_peak_n")}),
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
            mouth = self.model.site('mouth_entry').id
            wait = self.data.site_xpos[mouth]-self.data.site_xmat[mouth].reshape(3, 3)[:, 0]*self.task_config['wait_offset_m']
            departed = np.linalg.norm(self.data.site_xpos[self.index.tcp]-wait) >= scenario.get('recover_departure_m', 0.)
            if self.scenario_state["closure_start"] is None and departed:
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
        if self.task_mode and self.logic and reason == "nonfinite_state" and self.logic.failure_reason is None:
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
            started = time.perf_counter()
            self.adapter.update(self.dt)
            self.physics_timing['ik_wall_s'] = self.physics_timing.get('ik_wall_s', 0.) + time.perf_counter() - started
            if self.task_mode and self.adapter.fault:
                self._terminate(self.adapter.fault)
                self.logic.update(evidence(self), 0., self.data.time, failure=self.failure_reason)
                return self.snapshot()
        self._write_drivers()
        started = time.perf_counter()
        mujoco.mj_step(self.model, self.data)
        self.physics_timing["mj_step_s"] += time.perf_counter() - started
        self.physics_timing["steps"] += 1
        self.tick += 1
        # These are the loads actually used by the integrator. Sampling only
        # after mj_forward could miss a short contact that has already separated.
        applied_contacts = read_contacts(self.model, self.data, self.index)
        self.applied_contacts = applied_contacts
        overloaded = self.monitor.update(applied_contacts, self.dt, float(self.data.time))
        self.substep_contact_peak_n = self.monitor.last_peak_n
        applied_wrist = wrist_state(self.model, self.data, self.index)
        started = time.perf_counter()
        mujoco.mj_forward(self.model, self.data)
        self.physics_timing["forward_s"] += time.perf_counter() - started
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
                    **bean_state(self.model, self.data, idx),
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
        obs.update(bean_relative_world=state["bean_positions"] - tcp,
                   mouth_relative_world=task.data.site_xpos[mouth].copy() - tcp,
                   frame_age_s=0., stage="m1_diagnostic",
                   current_tool_contact=any("spoon" in [r["group1"], r["group2"]] for r in task.contacts))
        if task.task_mode:
            e = evidence(task)
            receiver = named_id(task.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_receiver")
            obs.update(receiver_relative_world=task.data.site_xpos[receiver].copy()-tcp,
                       receiver_rotation=task.data.site_xmat[receiver].reshape(3, 3).copy(),
                       stage=task.logic.phase, mouth_rotation=e["mouth_rotation"], mouth_aperture_m=e["aperture_m"],
                       interaction=np.array([e[k] for k in ("supported", "mouth_supported", "tool_mouth_contact", "ready")],
                                            dtype=np.float32))
        return dict(policy_obs=obs,
                    oracle_info=dict(contacts=copy.deepcopy(task.contacts),
                                     bean_ids=list(task.index.bean_ids),
                                     bean_masses_kg=task.model.body_mass[task.index.bean_bodies].copy(),
                                     beans=bean_diagnostics(task, spoon_frame=True),
                                     reset=copy.deepcopy(task.reset_diagnostics), terminated=task.terminated,
                                     failure_reason=task.failure_reason, contact_peak_n=task.monitor.peak_n,
                                     contact_impulse_ns=task.monitor.impulse_ns,
                                     contact_over_limit_s=task.monitor.over_limit_s,
                                     **(dict(phase=task.logic.phase, timers=copy.deepcopy(task.logic.timers),
                                             events=copy.deepcopy(task.logic.events), success=task.logic.success)
                                        if task.task_mode else {})),
                    scenario_state=copy.deepcopy(task.scenario_state))
