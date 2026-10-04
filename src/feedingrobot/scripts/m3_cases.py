"""Reset-only directed P0 fixtures; never a full feeding teacher."""

import mink
import mujoco
import numpy as np
import tempfile
from pathlib import Path
import time

from feedingrobot.envs import FeedingGymEnv
from feedingrobot.sim.contacts import read_contacts
from feedingrobot.sim.events import evidence
from feedingrobot.sim.model import named_id, RobotIndex
from feedingrobot.control.adapter import RobotAdapter
from feedingrobot.sim.task import FeedingTask

PHYSICAL_CASES = ("bowl", "carry", "pickup_lift", "bowl_return", "receiver", "receiver_edge",
                  "receiver_outside", "unsupported", "unsupported_recovered", "force",
                  "contact_safe", "contact_force", "penetration", "shallow", "entry", "closed",
                  "recover", "unreleased", "early_withdrawal", "post_delivery_loss", "handle")
MOUTH_CASES = {"receiver", "receiver_edge", "receiver_outside", "unsupported_recovered", "entry",
               "closed", "recover", "unreleased", "early_withdrawal", "post_delivery_loss"}


class DiagnosticTask(FeedingTask):
    def __init__(self, robot_id, scenario, timestep=None, *, iterations=100, refined=False):
        self.scenario = scenario
        self.configured = False
        # Freeze reset IK and compiled diagnostic geometry at the baseline dt;
        # numerical comparisons must not silently move the contact target.
        super().__init__(robot_id, .001, task_mode=True)
        self.scene_config["head_fixed"] = True
        if scenario in {"closed", "recover"}:
            self.default_head_origin = np.array([.67, .12, .35])
        self.configured = True
        # Independent IK sets a diagnostic reset configuration, never runtime qpos.
        super().reset(preset="empty")
        if scenario in {"contact_safe", "contact_force"}:
            rotation = self.data.site_xmat[self.index.tcp].reshape(3, 3)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "contact.xml"
                mujoco.mj_saveLastXML(str(path), self.model)
                spec = mujoco.MjSpec.from_file(str(path))
                table = spec.geom("table")
                table.pos = self.data.site_xpos[self.index.tcp] + rotation @ [.027, 0, .004]
                table.size = [.005, .005, .01]
                # A separately compiled soft diagnostic target, identical for
                # the safe/overload pair; normal scene contact parameters stay intact.
                table.solref = [.25, 1]
                table.priority = 2
                table.solimp = [.05, .995, .001, .5, 2]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, rotation.ravel())
                table.quat = quat
                self.model = spec.compile()
            self.index = RobotIndex(self.model, self.robot_config)
            self.data = mujoco.MjData(self.model)
            self.adapter = RobotAdapter(self.model, self.data, self.index, self.robot_config)
        if scenario in MOUTH_CASES:
            e = evidence(self)
            target, rotation = e["wait_position"], e["mouth_rotation"]
            if scenario in {"receiver", "receiver_edge", "unsupported_recovered", "unreleased",
                            "early_withdrawal", "post_delivery_loss"}:
                target = e["mouth_position"] + rotation @ [.004, 0, -.006]
            configuration = mink.Configuration(self.model)
            configuration.update(self.data.qpos.copy())
            frame = mink.FrameTask(self.robot_config["tcp_site"], "site", position_cost=1.,
                                   orientation_cost=1., lm_damping=1e-4)
            goal = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotation), target)
            frame.set_target(goal)
            for _ in range(1200):
                error = frame.compute_error(configuration)
                if np.linalg.norm(error[:3]) < .0002 and np.linalg.norm(error[3:]) < .002:
                    break
                velocity = mink.solve_ik(configuration, [frame], .01, solver="daqp",
                                         limits=self.adapter.limits, constraints=[self.adapter.freeze],
                                         damping=1e-5, safety_break=True)
                configuration.integrate_inplace(velocity, .01)
            actual = configuration.get_transform_frame_to_world(self.robot_config["tcp_site"], "site")
            assert np.linalg.norm(actual.translation() - target) < .002, (scenario, actual.translation(), target)
            self.robot_config["reset_q"] = configuration.q[self.index.qpos].tolist()
        if scenario == "closed":
            self.default_jaw_range[0] = -.65
            self.scene_config["head"]["jaw_center_rad"] = -.6
        self.dt = .001 if timestep is None else timestep
        self.model.opt.timestep = self.dt
        self.model.opt.iterations = iterations
        if refined:
            self.model.opt.tolerance /= 10

    def reset(self, seed=0, preset="beans_in_bowl"):
        if not self.configured:
            return super().reset(seed, preset)
        s = self.scenario
        loaded = s in {"carry", "entry", "closed", "recover", "unreleased", "early_withdrawal", "contact_safe", "contact_force"}
        super().reset(seed, "beans_on_spoon" if loaded else "empty" if s in {"contact_safe", "contact_force"}
                      else "beans_in_bowl", scenario={"recover": True} if s == "recover" else None)
        a = int(self.index.bean_qpos[0, 0])
        self.braked = False
        if s in {"contact_safe", "contact_force"}:
            self.logic.phase = "ACQUIRE"
        if s in {"receiver", "receiver_edge", "receiver_outside", "unsupported_recovered", "post_delivery_loss"}:
            receiver = named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_receiver")
            rotation = self.data.site_xmat[receiver].reshape(3, 3)
            local = [.012, .0185 if s == "receiver_edge" else 0, .0045]
            if s == "receiver_outside":
                local[0] = -.017
            if s == "unsupported_recovered":
                local[2] = .0085
            self.data.qpos[a:a + 3] = self.data.site_xpos[receiver] + rotation @ local
            mujoco.mju_mat2Quat(self.data.qpos[a + 3:a + 7], rotation.ravel())
            self.logic.acquired, self.logic.phase = True, "TRANSFER"
        elif s in {"unsupported", "penetration", "shallow", "bowl_return"}:
            bowl = named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "bowl_frame")
            local = [0, -.13, .18] if s == "unsupported" else [0, 0, .0055] if s == "bowl_return" else [0, 0, .002] if s == "penetration" else [0, 0, .00395]
            rotation = self.data.site_xmat[bowl].reshape(3, 3)
            self.data.qpos[a:a + 3] = self.data.site_xpos[bowl] + rotation @ local
            mujoco.mju_mat2Quat(self.data.qpos[a + 3:a + 7], rotation.ravel())
            if s in {"unsupported", "bowl_return"}:
                self.logic.acquired, self.logic.phase = True, "TRANSPORT"
        elif s in {"entry", "closed", "recover", "unreleased", "early_withdrawal"}:
            self.logic.acquired, self.logic.phase = True, "TRANSPORT" if s in {"entry", "closed", "recover"} else "APPROACH"
        elif s == "handle":
            rotation = self.data.site_xmat[self.index.tcp].reshape(3, 3)
            heights = []
            for x in (-.006, 0, .006):
                for y in (-.006, 0, .006):
                    point = self.data.site_xpos[self.index.tcp] + rotation @ [-.035 + x, y, .05]
                    hits = [mujoco.mj_rayMesh(self.model, self.data, g, point, -rotation[:, 2])
                            for g in self.index.handle_geoms if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
                    heights.extend(.05 - h for h in hits if h >= 0)
            local = [-.035, 0, max(heights) + .0045]
            self.data.qpos[a:a + 3] = self.data.site_xpos[self.index.tcp] + rotation @ local
            mujoco.mju_mat2Quat(self.data.qpos[a + 3:a + 7], rotation.ravel())
        if s == "carry":
            self.logic.acquired, self.logic.phase = True, "TRANSPORT"
        mujoco.mj_forward(self.model, self.data)
        self.contacts = read_contacts(self.model, self.data, self.index)
        return self.snapshot()

    def _write_drivers(self):
        super()._write_drivers()
        # A recorded, bounded physical pulse after actual delivery; no teleport.
        if self.scenario == "post_delivery_loss" and self.logic and self.logic.delivered:
            delivery = next(e["time"] for e in self.logic.events if e["name"] == "delivery")
            if delivery + .05 <= self.data.time < delivery + .25:
                mouth = named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_entry")
                self.data.xfrc_applied[self.index.bean_bodies[0], :3] = self.data.site_xmat[mouth].reshape(3, 3) @ [-.1, 0, .3]


def diagnostic_action(task):
    s, t = task.scenario, task.data.time
    e = evidence(task)
    base = task.data.site_xmat[task.index.base].reshape(3, 3)
    world = np.zeros(6)
    if s == "carry":
        world[0] = .005
    elif s in {"contact_safe", "contact_force"} and t > .1:
        if s == "contact_safe" and task.monitor.pair_peaks.get("spoon|table", 0) >= .05:
            if not task.braked:
                task.adapter.stop(hold_reference=True)
                task.braked = True
            world[0] = 0
        else:
            world[0] = .003 if s == "contact_safe" else .02
    elif s == "entry" and task.logic.phase in {"APPROACH", "TRANSFER"}:
        target = e["mouth_position"] + e["mouth_rotation"] @ [.004, 0, -.006]
        world[:3] = (target - e["tcp_position"]) * 3
    elif s in {"receiver", "receiver_edge", "unsupported_recovered", "post_delivery_loss"} and task.logic.delivered:
        world[:3] = (e["wait_position"] - e["tcp_position"]) * 3
    elif s == "early_withdrawal" and t > .15:
        world[:3] = (e["wait_position"] - e["tcp_position"]) * 3
    scale = np.array([task.robot_config["linear_speed_limit"]] * 3 + [task.robot_config["angular_speed_limit"]] * 3)
    return np.clip(np.r_[base.T @ world[:3], base.T @ world[3:]] / scale, -1, 1)


def capture_frame(task, output, name):
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    e = evidence(task)
    camera.lookat[:] = e["bean_position"] if name == "pickup" else e["mouth_position"] + e["mouth_rotation"] @ [.018, 0, -.005]
    camera.distance, camera.azimuth, camera.elevation = .12, 45, -45
    paths = []
    with mujoco.Renderer(task.model, height=480, width=640) as renderer:
        for mode in ("visual", "collision"):
            option = mujoco.MjvOption()
            option.geomgroup[1], option.geomgroup[3] = (1, 0) if mode == "visual" else (0, 1)
            renderer.update_scene(task.data, camera=camera, scene_option=option)
            path = output / f"{name}_{mode}.png"
            Image.fromarray(renderer.render()).save(path)
            paths.append(path.name)
    return dict(time=float(task.data.time), phase=task.logic.phase, files=paths)


def physical_case(robot, scenario, timestep=.001, *, seed=0, iterations=100, refined=False,
                  viewer=False, frame_output=None, initial_state=None):
    if scenario not in PHYSICAL_CASES + ("full_static", "full_dynamic"):
        raise ValueError(scenario)
    from feedingrobot.scripts.m3_driver import FeedingDriver
    from feedingrobot.scripts.validate_m1 import restore_numerical_state
    natural = scenario in {"pickup_lift", "full_static", "full_dynamic"}
    env = FeedingGymEnv(robot, render_mode="human" if viewer else None)
    if natural:
        env.task.scene_config["head_fixed"] = scenario != "full_dynamic"
    else:
        env.task = DiagnosticTask(robot, scenario)
        env.config = env.task.task_config
    env.reset(seed=seed)
    if initial_state is not None:
        env.set_state(initial_state)
    initial_state = env.get_state()
    env.task.model.opt.timestep = env.task.dt = timestep
    env.task.model.opt.iterations = iterations
    env.task.model.opt.tolerance = 1e-10 if refined else 1e-8
    if timestep != .001 or iterations != 100 or refined:
        restore_numerical_state(env.task, initial_state['task'])
    env.substeps = round(env.control_dt/timestep)
    driver = FeedingDriver(env.task) if natural else None
    if scenario == "force":
        env.task.set_external_wrench([0, 0, 12], [0, 0, 0], env.task.data.site_xpos[env.task.index.tcp])
    trace = []
    # Capture every physical boundary, including contact geometry and F/T.
    step = env.task.step_physics
    def recorded_step(**kwargs):
        bean = int(env.task.index.bean_collision_geoms[0])
        before = env.task.data.geom_xpos[bean].copy()
        radius = np.linalg.norm(env.task.data.geom_xmat[bean].reshape(3,3)[2] * env.task.model.geom_size[bean])
        state = step(**kwargs)
        e = evidence(env.task)
        if driver:
            driver.record(env.task, before, radius)
        trace.append(dict(time=state["time"], phase=env.task.logic.phase, success=env.task.logic.success,
                          failure_reason=env.task.failure_reason, driver_stage=driver.stage if driver else scenario, events=env.task.logic.events[len_events[0]:].copy(),
                          **{k: e[k] for k in ("supported", "off_bowl", "on_bowl", "mouth_supported", "released",
                                               "tool_inside", "ready", "bowl_clearance_m", "spoon_support_force_n",
                                               "required_height_m", "required_width_m", "aperture_m")},
                          tcp_position=state["tcp_position"].tolist(), bean_position=state["bean_positions"][0].tolist(),
                          bean_quaternion=state["bean_quaternions"][0].tolist(),
                          bean_linear_velocity=state["bean_linear_velocities_world"][0].tolist(),
                          bean_angular_velocity=state["bean_angular_velocities_world"][0].tolist(),
                          tcp_rotation=state["tcp_rotation"].tolist(), q=state["q"].tolist(), dq=state["dq"].tolist(),
                          compensated_wrench=state["compensated_wrench"].tolist(),
                          contact_peak_n=env.task.substep_contact_peak_n,
                          contact_impulse_ns=env.task.monitor.impulse_ns,
                          minimum_contact_distance_m=min((r["distance"] for r in env.task.contacts), default=0.),
                      contacts=[dict(r, force_on_geom2_world=r["force_on_geom2_world"].tolist(),
                                         position=r["position"].tolist()) for r in env.task.contacts]))
        len_events[0] = len(env.task.logic.events)
        trace[-1]["diagnostic_stop"] = False
        if (scenario == "contact_safe" and not env.task.braked and not env.task.terminated
                and env.task.monitor.pair_peaks.get("spoon|table", 0) >= .05):
            env.task.adapter.stop(hold_reference=True)
            env.task.braked = True
            trace[-1]["diagnostic_stop"] = True
        trace[-1]['applied_contacts'] = env.task.applied_contacts
        return state
    len_events = [0]
    env.task.step_physics = recorded_step
    duration = 2.5 if scenario in MOUTH_CASES or scenario in {"contact_safe", "contact_force"} else .7
    if scenario in {"closed", "unreleased"}:
        duration = .5
    if scenario == "entry":
        duration = 1.7
    if scenario == "recover":
        duration = 2.
    if natural:
        duration = 60.
    transfer_state = None
    rewards = dict(pickup=0., delivery=0., success=0., failure=0.)
    frames = {}
    try:
        terminated = False
        while env.task.data.time < duration - 1e-12:
            started = time.monotonic()
            action = driver.action(env.task) if driver else diagnostic_action(env.task)
            if driver and driver.transfer_start is not None and transfer_state is None:
                transfer_state = env.get_state()
            _, _, terminated, truncated, info = env.step(action)
            for k in rewards:
                rewards[k] += info["reward_terms"][k]
            if viewer and not env.viewer.is_running():
                raise RuntimeError("Viewer closed before diagnostic completed")
            if frame_output is not None:
                current = evidence(env.task)
                entry_target = current["mouth_position"] + current["mouth_rotation"] @ [.004, 0, -.006]
                label = ("pickup" if natural and env.task.logic.acquired and not env.task.logic.delivered
                         else "entry" if scenario == "entry" and env.task.logic.phase == "TRANSFER"
                         and np.linalg.norm(current["tcp_position"] - entry_target) < .004
                         else "delivery" if env.task.logic.delivered else None)
                if label and label not in frames:
                    frames[label] = capture_frame(env.task, frame_output, label)
                if env.task.logic.success:
                    frames["retracted"] = capture_frame(env.task, frame_output, "retracted")
            if viewer:
                env.viewer.opt.geomgroup[3] = 0
                time.sleep(max(0, .02 - (time.monotonic() - started)))
            if terminated or truncated or (scenario == "pickup_lift" and env.task.logic.acquired):
                break
        e = evidence(env.task)
        result = dict(scenario=scenario, seed=seed, fixture="natural bowl reset and physical commands" if natural else "reset-only directed P0 diagnostic",
                      head_fixed=env.task.scene_config["head_fixed"], dt=timestep,
                      head_origin_m=env.task.default_head_origin.tolist(),
                      diagnostic_contact_solref=[.25, 1] if scenario in {"contact_safe", "contact_force"} else None,
                      diagnostic_contact_solimp=[.05, .995, .001, .5, 2] if scenario in {"contact_safe", "contact_force"} else None,
                      diagnostic_substep_stop_n=.05 if scenario == "contact_safe" else None,
                      iterations=iterations, solver_tolerance=float(env.task.model.opt.tolerance),
                      reset_q=env.task.robot_config["reset_q"], time=float(env.task.data.time),
                      final_q=env.task.snapshot()["q"].tolist(), adapter_error=env.task.adapter.error_detail,
                      diagnostic_jaw_range_rad=env.task.model.jnt_range[env.task.index.head_joints[-1]].tolist(),
                      success=env.task.logic.success, failure_reason=env.task.failure_reason,
                      phase=env.task.logic.phase, peak_force_n=env.task.monitor.peak_n,
                      impulse_ns=env.task.monitor.impulse_ns,
                      contact_pair_peaks_n=env.task.monitor.pair_peaks.copy(),
                      contact_pair_impulses_ns=env.task.monitor.pair_impulses.copy(),
                      tcp_position=env.task.data.site_xpos[env.task.index.tcp].tolist(),
                      events=env.task.logic.events, reward_terms=rewards,
                      tcp_samples=[dict(time=r["time"], position=r["tcp_position"]) for r in trace
                                   if abs(r["time"] / .02 - round(r["time"] / .02)) < 1e-8],
                      event_tcp_positions=[next(r["tcp_position"] for r in trace
                                                if abs(r["time"] - event["time"]) < 1e-10)
                                           for event in env.task.logic.events],
                      frames=frames, initial_state=initial_state, transfer_state=transfer_state,
                      first_entry_contact=driver.first_entry_contact if driver else None,
                      sweep_displacement_m=(driver.sweep_end-driver.sweep_start).tolist() if driver and driver.sweep_end is not None else None,
                      food_pulse=dict(force_mouth_n=[-.1, 0, .3], duration_s=.2) if scenario == "post_delivery_loss" else None)
        failures = {"unsupported": "food_dropped", "bowl_return": "food_dropped", "receiver_outside": "food_dropped",
                    "force": "contact_limit", "contact_force": "contact_limit", "penetration": "model_penetration",
                    "early_withdrawal": "withdrawal_before_release", "post_delivery_loss": "food_lost_after_delivery",
                    "handle": "food_dropped"}
        successes = {"receiver", "receiver_edge", "unsupported_recovered", "full_static", "full_dynamic"}
        names = [r["name"] for r in result["events"]]
        result["expected_failure"] = failures.get(scenario)
        result["expected_success"] = scenario in successes
        assert result["failure_reason"] == failures.get(scenario), result
        assert result["success"] == (scenario in successes), result
        assert rewards["failure"] == (-50 if scenario in failures else 0), result
        if scenario in {"carry", "pickup_lift"}:
            assert env.task.logic.acquired and e["supported"], result
        if natural:
            contact = driver.first_entry_contact
            assert contact and contact['height_relative_radius'] < 0 and contact['force_on_bean_world_n'][2] > 0, result
            assert driver.sweep_end[2] > driver.sweep_start[2], result
            assert names.count('pickup') == 1 and rewards['pickup'] == 10, result
        if scenario in successes:
            assert names.count("delivery") == names.count("success") == 1 and e["mouth_supported"] and e["released"], result
        if scenario in {"entry", "unreleased", "early_withdrawal"}:
            assert any(r["tool_inside"] and r["supported"] for r in trace), result
            assert "delivery" not in names and rewards["delivery"] == rewards["success"] == 0, result
        if scenario == "closed":
            assert env.task.logic.phase == "WAIT_READY" and not e["ready"] and "delivery" not in names, result
        if scenario == "recover":
            assert any(r.get("phase") == "RECOVER" for r in result["events"]), result
        if scenario == "handle":
            assert any(any({c["group1"], c["group2"]} == {"food", "spoon"} for c in r["contacts"]) for r in trace), result
            assert not any(r["supported"] for r in trace) and "pickup" not in names, result
        if scenario in {"bowl", "shallow"}:
            assert e["on_bowl"] and "pickup" not in names, result
        if scenario == "contact_safe":
            assert result["contact_pair_peaks_n"].get("spoon|table", 0) > 0 and not terminated, result
        if scenario == "force":
            assert env.task.data.time <= timestep + 1e-12 and result["impulse_ns"] == 0, result
        return result, trace
    except AssertionError as exc:
        exc.metrics, exc.trace = result, trace
        raise
    finally:
        env.close()
