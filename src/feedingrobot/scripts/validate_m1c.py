"""Native Beans M1-C control and pickup evidence; M1-D remains unverified."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import sys
import time
import traceback

import mink
import mujoco
import numpy as np

from feedingrobot.control.adapter import RobotAdapter
from feedingrobot.sim.beans import bean_diagnostics, spoon_frame_state
from feedingrobot.sim.model import ROOT, asset_files, load_json
from feedingrobot.sim.task import FeedingTask
from feedingrobot.scripts.validate_m1b import serializable



def angle_error(a, b):
    return float(np.linalg.norm((mink.SO3.from_matrix(a) @ mink.SO3.from_matrix(b).inverse()).log()))


def advance(task, seconds, trace=None):
    state = task.snapshot()
    for _ in range(round(seconds / task.dt)):
        state = task.step_physics()
        if trace is not None:
            trace.append({"time": state["time"], "tcp_x": state["tcp_position"][0],
                          "tcp_y": state["tcp_position"][1], "tcp_z": state["tcp_position"][2],
                          "peak_force_n": state["contact_peak_n"],
                          "impulse_ns": state["contact_impulse_ns"],
                          "max_joint_speed": float(np.max(np.abs(state["dq"]))),
                          "wrist_force_n": float(np.linalg.norm(state["compensated_wrench"][:3])),
                          "status": state["execution_status"]})
        assert not task.terminated, task.failure_reason
        assert task.adapter.fault is None, (task.adapter.fault, getattr(task.adapter, "error_detail", ""))
    return state


def holding(task, cfg, trace):
    task.reset(preset="empty")
    start = task.snapshot()
    final = advance(task, cfg["hold_duration_s"], trace)
    pos = float(np.linalg.norm(final["tcp_position"] - start["tcp_position"]))
    rot = angle_error(final["tcp_rotation"], start["tcp_rotation"])
    assert pos < cfg["hold_position_error_m"], pos
    assert rot < cfg["hold_orientation_error_rad"], rot
    return dict(position_drift_m=pos, rotation_drift_rad=rot,
                gravity_joint_offset_rad=(final["q"] - task.robot_config["reset_q"]).tolist(),
                gravcomp=task.robot_config["gravcomp"])


def tracking(task, cfg, trace):
    task.reset(preset="empty")
    errors, rotation_errors, speeds, saturation = [], [], [], 0
    # Each of the six action axes has its own excitation.
    for axis in range(6):
        twist = np.zeros(6)
        twist[axis] = .01 if axis < 3 else .05
        now = task.data.time
        task.adapter.set_twist(twist, now, now + .45)
        for _ in range(round(.4 / task.dt)):
            state = advance(task, task.dt, trace)
            errors.append(float(np.linalg.norm(state["tcp_position"] - task.adapter.target.translation())))
            rotation_errors.append(angle_error(state["tcp_rotation"], task.adapter.target.rotation().as_matrix()))
            speeds.append(float(np.max(np.abs(state["dq"]))))
            force = state["actuator_force"]
            limits = task.model.actuator_forcerange[task.index.actuators]
            saturation += int(np.any(np.isclose(force, limits[:, 0], atol=1e-5) | np.isclose(force, limits[:, 1], atol=1e-5)))
    task.adapter.stop()
    final = advance(task, cfg["stop_settle_s"], trace)
    assert max(errors) < cfg["tracking_position_error_m"], max(errors)
    assert max(rotation_errors) < cfg["tracking_orientation_error_rad"], max(rotation_errors)
    assert np.linalg.norm(final["tcp_twist_world"][:3]) < cfg["stop_linear_speed_m_s"]
    assert np.linalg.norm(final["tcp_twist_world"][3:]) < cfg["stop_angular_speed_rad_s"]
    return dict(max_tracking_error_m=max(errors), max_rotation_error_rad=max(rotation_errors),
                max_joint_speed_rad_s=max(speeds), actuator_saturation_ticks=saturation,
                final_twist=final["tcp_twist_world"].tolist())


def wrench_check(task, cfg, trace):
    task.reset(preset="empty")
    force_errors, torque_errors, residuals = [], [], []
    for direction in np.r_[np.eye(3), -np.eye(3)]:
        point = task.data.site_xpos[task.index.tcp].copy() + np.array([.03, -.02, .01])
        torque = np.array([.01, -.02, .03])
        task.set_external_wrench(direction, torque, point)
        state = advance(task, .02, trace)
        expected_torque = torque + np.cross(point - state["tcp_position"], direction)
        force_errors.append(float(np.max(np.abs(state["compensated_wrench"][:3] - direction))))
        torque_errors.append(float(np.max(np.abs(state["compensated_wrench"][3:] - expected_torque))))
    task.clear_external_wrench()
    now = task.data.time
    task.adapter.set_twist([.01, -.01, .01, .03, -.02, .01], now, now + .5)
    for _ in range(round(.4 / task.dt)):
        state = advance(task, task.dt, trace)
        residuals.append(state["compensated_wrench"].copy())
    maximum = np.max(np.abs(residuals), axis=0)
    assert max(force_errors) < cfg["wrench_force_error_n"], force_errors
    assert max(torque_errors) < cfg["wrench_torque_error_nm"], torque_errors
    assert np.max(maximum[:3]) < cfg["wrench_force_error_n"], maximum
    assert np.max(maximum[3:]) < cfg["wrench_torque_error_nm"], maximum
    return dict(force_error_n=max(force_errors), torque_error_nm=max(torque_errors), dynamic_residual=maximum.tolist())


def fault_checks(task, cfg, trace):
    task.reset(preset="empty")
    # Unreachable workspace request must clear and latch the reference.
    base_p = task.data.site_xpos[task.index.base]
    base_r = task.data.site_xmat[task.index.base].reshape(3, 3)
    p = base_r.T @ (task.data.site_xpos[task.index.tcp] - base_p)
    task.robot_config["workspace_max"] = (p + [.0001, 1, 1]).tolist()
    now = task.data.time
    task.adapter.set_twist([.05, 0, 0, 0, 0, 0], now, now + 1)
    for _ in range(round(.2 / task.dt)):
        task.step_physics()
        if task.adapter.fault:
            break
    assert task.adapter.fault == "workspace_limit" and task.adapter.command is None
    reference = task.data.ctrl[task.index.actuators].copy()
    for _ in range(round(.02 / task.dt)):
        task.step_physics()
    np.testing.assert_array_equal(reference, task.data.ctrl[task.index.actuators])
    return {"unreachable_cancelled": True}


def update_window(window, eligible, dt):
    return np.where(eligible, window + dt, 0.)


def pickup_eligible(initial, diag, positions, rim_z, radii):
    return (initial & ~diag['in_bowl'] & ~diag['bowl_supported']
            & diag['spoon_supported'] & diag['in_spoon_head']
            & (positions[:, 2] - radii > rim_z))


def control_parameters(task):
    return dict(robot=copy.deepcopy(task.robot_config), head_fixed=task.scene_config['head_fixed'],
                joint_speed_fault_rad_s=task.scene_config['joint_speed_fault_rad_s'],
                actuator_gain=task.model.actuator_gainprm[task.index.actuators].copy(),
                actuator_bias=task.model.actuator_biasprm[task.index.actuators].copy())


class Evidence:
    """Observe the existing physical entry point without adding integration steps."""
    def __init__(self, task):
        self.task = task
        self.original_step = task.step_physics
        region = spoon_frame_state(task)
        self.bounds = (region['spoon_region_min_tcp'], region['spoon_region_max_tcp'])
        self.initial = None
        self.initial_parameters = None
        self.trace, self.contacts = [], {}
        self.max_penetration_m = 0.
        self.penetration_peak = None
        self.phase = 'control'
        self.wall_s = self.diagnostic_s = 0.
        self.max_speed = self.max_acceleration = self.max_wrist_force = 0.
        self.previous_velocity = None
        self.diag = None
        self.expected_fault = None
        self.first_entry_contact = None
        self.phase_motion = {}
        task.step_physics = self.step

    def step(self, *, _settling=False):
        if _settling:
            return self.original_step(_settling=True)
        if self.initial is None:
            self.episode_started = time.perf_counter()
            self.initial = self.task.get_state()
            self.initial_parameters = control_parameters(self.task)
            self.start_timing = dict(self.task.physics_timing)
        started = time.perf_counter()
        entry = self.phase in ('entry', 'sweep')
        if entry:
            geom = self.task.index.bean_collision_geoms[0]
            before = self.task.data.geom_xpos[geom].copy()
            radius = np.linalg.norm(self.task.data.geom_xmat[geom].reshape(3, 3)[2]
                                    * self.task.model.geom_size[geom])
            tcp_position = self.task.data.site_xpos[self.task.index.tcp].copy()
            tcp_rotation = self.task.data.site_xmat[self.task.index.tcp].reshape(3, 3).copy()
        state = self.original_step()
        self.wall_s += time.perf_counter() - started
        started = time.perf_counter()
        assert 'bean_positions' in state, self.task.failure_reason
        self.diag = bean_diagnostics(self.task, spoon_frame=True, spoon_bounds=self.bounds)
        if entry:
            motion = self.phase_motion.setdefault(self.phase, dict(start_position_m=before))
            motion['displacement_m'] = state['bean_positions'][0] - motion['start_position_m']
            for row in self.task.applied_contacts:
                if self.first_entry_contact is not None or row['force_n'] <= .0001:
                    continue
                if geom not in (row['geom1'], row['geom2']):
                    continue
                other = row['geom2'] if row['geom1'] == geom else row['geom1']
                if other not in self.task.index.scoop_geoms:
                    continue
                force = np.array(row['force_on_geom2_world']) * (-1 if row['geom1'] == geom else 1)
                point = np.array(row['position'])
                self.first_entry_contact = dict(time_s=state['time'] - self.task.dt,
                    phase=self.phase, bean_id='bean_000', spoon_geom=self.task.model.geom(other).name,
                    position_world_m=point, position_tcp_m=(point-tcp_position) @ tcp_rotation,
                    height_relative_radius=float((point[2]-before[2])/radius),
                    force_on_bean_world_n=force)
        if self.diag['max_penetration_m'] > self.max_penetration_m:
            self.max_penetration_m = self.diag['max_penetration_m']
            self.penetration_peak = dict(self.diag['penetration_peak'], time_s=state['time'])
        forbidden = [r for r in self.task.contacts + self.task.applied_contacts
                     if 'arm' in (r['group1'], r['group2']) and r['force_n'] > 1e-5]
        for row in self.task.applied_contacts + self.task.contacts:
            if row['bean1_id'] or row['bean2_id']:
                key = (row['geom1'], row['geom2'])
                if key not in self.contacts or row['force_n'] > self.contacts[key]['force_n']:
                    self.contacts[key] = dict(row, time_s=state['time'])
        self.max_wrist_force = max(self.max_wrist_force, self.task.substep_wrist_peak_n)
        velocity = state['tcp_twist_world'][:3]
        self.max_speed = max(self.max_speed, float(np.linalg.norm(velocity)))
        if self.previous_velocity is not None:
            self.max_acceleration = max(self.max_acceleration,
                                        float(np.linalg.norm(velocity - self.previous_velocity) / self.task.dt))
        self.previous_velocity = velocity.copy()
        invalid = (self.task.terminated or self.task.adapter.fault not in (None, self.expected_fault) or forbidden
                   or any(w.number for w in self.task.data.warning)
                   or self.max_penetration_m > self.task.bean_acceptance['penetration_limit_m'])
        if self.task.tick % round(.01 / self.task.dt) == 0 or invalid:
            self.trace.append(dict(time_s=state['time'], phase=self.phase,
                last_command_time_s=self.task.adapter.last_command_time,
                **{k: state[k] for k in ('q', 'dq', 'tcp_position', 'tcp_rotation', 'tcp_twist_world',
                    'bean_positions', 'bean_quaternions', 'bean_linear_velocities_world',
                    'bean_angular_velocities_world', 'compensated_wrench', 'contact_peak_n', 'contact_impulse_ns')},
                **{k: self.diag[k] for k in ('in_bowl', 'bowl_supported', 'spoon_supported', 'in_spoon_head',
                    'bean_positions_tcp', 'bean_linear_velocities_tcp', 'bean_angular_velocities_tcp',
                    'direct_support', 'support_edges')}, contacts=self.task.contacts,
                applied_contacts=self.task.applied_contacts))
        self.diagnostic_s += time.perf_counter() - started
        assert not self.task.terminated, self.task.failure_reason
        assert self.task.adapter.fault in (None, self.expected_fault), (self.task.adapter.fault, self.task.adapter.error_detail)
        assert not any(w.number for w in self.task.data.warning), 'MuJoCo warning'
        assert self.max_penetration_m <= self.task.bean_acceptance['penetration_limit_m'], self.penetration_peak
        assert not forbidden, forbidden
        return state

    def advance(self, seconds):
        for _ in range(round(seconds / self.task.dt)):
            state = self.task.step_physics()
        return state

    def result(self):
        timing = self.task.physics_timing
        physics = timing['mj_step_s'] - getattr(self, 'start_timing', timing)['mj_step_s']
        steps = timing['steps'] - getattr(self, 'start_timing', timing)['steps']
        elapsed = self.wall_s + self.diagnostic_s
        episode_wall = time.perf_counter() - self.episode_started if self.initial is not None and hasattr(self, 'episode_started') else None
        return dict(max_penetration_m=self.max_penetration_m, penetration_peak=self.penetration_peak,
                    contact_evidence=list(self.contacts.values()), warning_counts=[w.number for w in self.task.data.warning],
                    max_tcp_speed_m_s=self.max_speed, max_tcp_acceleration_m_s2=self.max_acceleration,
                    max_semantic_contact_force_n=self.task.monitor.peak_n, max_wrist_force_n=self.max_wrist_force,
                    performance=dict(steps=steps, mj_step_s=physics,
                        forward_s=timing['forward_s'] - getattr(self, 'start_timing', timing)['forward_s'],
                        ik_s=timing.get('ik_wall_s', 0.) - getattr(self, 'start_timing', {}).get('ik_wall_s', 0.),
                        physical_entry_s=self.wall_s, diagnostic_s=self.diagnostic_s,
                        episode_wall_s=episode_wall,
                        pure_physics_rtf=steps*self.task.dt/physics if physics else None,
                        control_diagnostic_rtf=steps*self.task.dt/elapsed if elapsed else None,
                        full_control_loop_rtf=steps*self.task.dt/episode_wall if episode_wall else None))


def move_pose(task, observer, position, rotation, cfg, *, speed=None):
    base = task.data.site_xmat[task.index.base].reshape(3, 3)
    limits = cfg['beans_native']['m1c']
    speed = task.robot_config['linear_speed_limit'] if speed is None else speed
    for _ in range(round(cfg['reachability_timeout_s'] / task.dt)):
        state = task.snapshot()
        delta = np.asarray(position) - state['tcp_position']
        angular = (mink.SO3.from_matrix(rotation) @ mink.SO3.from_matrix(state['tcp_rotation']).inverse()).log()
        if np.linalg.norm(delta) < limits['pose_position_error_m'] and np.linalg.norm(angular) < limits['pose_orientation_error_rad']:
            task.adapter.stop(hold_reference=True)
            return state
        if task.tick % round(.02 / task.dt) == 0:
            linear = delta * 3.
            scale = min(1., speed / max(np.linalg.norm(linear), 1e-12))
            linear *= scale
            if observer.phase.startswith('wall_'):
                angular *= scale
            now = task.data.time
            task.adapter.set_twist(np.r_[base.T @ linear, base.T @ angular * 3.], now, now + .04)
        task.step_physics()
    raise AssertionError(f'Pose timeout: target={position}, actual={task.snapshot()["tcp_position"]}')


def collision_points(task, geoms):
    m, d, idx = task.model, task.data, task.index
    rotation = d.site_xmat[idx.tcp].reshape(3, 3)
    points = []
    for geom in geoms:
        if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[geom]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = m.mesh_vert[start:start+count]
        else:
            vertices = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)]) * m.geom_size[geom]
        world = vertices @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
        points.append((world - d.site_xpos[idx.tcp]) @ rotation)
    return np.concatenate(points)


def scoop_points(task):
    return collision_points(task, task.index.scoop_geoms)


def sweep_path(task, cfg):
    c = cfg['beans_native']['m1c']
    centre = np.array(task.scene_config['bowl_frame_position_m'])
    yaw = mink.SO3.exp([0., 0., np.deg2rad(c['entry_yaw_deg'])])
    rotation = (yaw @ mink.SO3.exp([0., np.deg2rad(c['entry_pitch_deg']), 0.])).as_matrix()
    points = scoop_points(task)
    tip = points[np.argmin((points @ rotation.T)[:, 2])]
    z = -float((rotation @ tip)[2]) + c['contact_clearance_m']
    x, y = c['sweep_start_xy_m']
    path = [('above', centre + [x, y, .14], rotation, .03),
            ('pre_entry', centre + [x, y, z + .025], rotation, .03),
            ('entry', centre + [x, y, z], rotation, c['sweep_speed_m_s'])]
    forward, lateral = np.array([1.,0.,0.]), np.array([0.,1.,0.])
    wall_yaw = mink.SO3.exp([0., 0., np.deg2rad(-10.)])
    all_points = collision_points(task, task.index.spoon_geoms)
    walls = [g for g in task.index.bowl_geoms if task.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    rim = max(task.data.geom_xpos[g,2] + np.abs(task.data.geom_xmat[g].reshape(3,3)[2]) @ task.model.geom_size[g] for g in walls)
    last_forward = float(np.dot([x,y,0.], forward))
    # Push toward the wall, then follow its finite inner faces while returning
    # the head to level. Vertex half-spaces only generate conservative targets;
    # actual contact/penetration and wrist clearance are checked in physics.
    for phase, height, angle, side, stage_yaw, gap in [
            ('sweep', c['contact_clearance_m'], c['entry_pitch_deg'], y, yaw, .003),
            ('wall_align', .0085, 60., .006, wall_yaw, .001),
            ('wall_45', .028, 45., .006, wall_yaw, .001),
            ('wall_30', .045, 30., .006, wall_yaw, .001),
            ('wall_15', .080, 15., .006, wall_yaw, .001),
            ('wall_level', .110, 0., .006, wall_yaw, .001)]:
        r = (stage_yaw @ mink.SO3.exp([0., np.deg2rad(angle), 0.])).as_matrix()
        vertical = height - (r @ tip)[2]
        vertices = all_points @ r.T
        active = vertices[centre[2] + vertical + vertices[:,2] <= rim]
        upper, lower = np.inf, -np.inf
        if len(active):
            for geom in walls:
                normal = task.data.geom_xmat[geom].reshape(3,3)[:,0]
                denominator = np.dot(normal, forward)
                if abs(denominator) < 1e-8:
                    continue
                value = (np.dot(normal, task.data.geom_xpos[geom]-centre)
                         - task.model.geom_size[geom,0] - c['contact_clearance_m']
                         - np.dot(normal, side*lateral+[0.,0.,vertical])
                         - np.max(active @ normal)) / denominator
                if denominator > 0:
                    upper = min(upper, value)
                else:
                    lower = max(lower, value)
            assert lower <= upper, f'No whole-tool wall clearance at {phase}: {lower}, {upper}'
            last_forward = upper - gap
        position = centre + last_forward*forward + side*lateral + [0.,0.,vertical]
        path.append((phase, position, r, c['sweep_speed_m_s']))
    clearance = np.r_[path[-1][1][:2], centre[2]+.18]
    path.append(('clearance', clearance, wall_yaw.as_matrix(), .03))
    # A small tip-up motion seats a carried bean away from the front lip;
    # the final observation remains level and uses the original low-speed gates.
    path.append(('wall_seat_tip', clearance,
                 (wall_yaw @ mink.SO3.exp([0., np.deg2rad(-5.), 0.])).as_matrix(), .005))
    path.append(('wall_seat_level', clearance, wall_yaw.as_matrix(), .005))
    return path


def sweep(task, cfg, observer, seed):
    task.reset(seed=seed, preset='beans_in_bowl')
    initial = bean_diagnostics(task)['bowl_supported'].copy()
    path = sweep_path(task, cfg)
    achieved = []
    for phase, position, rotation, speed in path:
        observer.phase = phase
        state = move_pose(task, observer, position, rotation, cfg, speed=speed)
        achieved.append(dict(phase=phase, position_error_m=float(np.linalg.norm(state['tcp_position']-position)),
            orientation_error_rad=float(np.linalg.norm((mink.SO3.from_matrix(rotation)
                @ mink.SO3.from_matrix(state['tcp_rotation']).inverse()).log()))))
        print(task.robot_id, seed, phase, 'reached', flush=True)
    observer.phase = 'pickup_hold'
    bottom = task.model.geom('collision_bowl_fast_bottom_disk').id
    walls = [g for g in task.index.bowl_geoms if g != bottom]
    rim = max(task.data.geom_xpos[g, 2] + np.abs(task.data.geom_xmat[g].reshape(3, 3)[2]) @ task.model.geom_size[g] for g in walls)
    window, best = np.zeros(len(task.index.bean_ids)), np.zeros(len(task.index.bean_ids))
    pickup_state = None
    c = cfg['beans_native']
    for _ in range(round(c['m1c']['pickup_observation_s'] / task.dt)):
        state = task.step_physics()
        diag = observer.diag
        radii = np.linalg.norm(task.data.geom_xmat[task.index.bean_collision_geoms].reshape(-1, 3, 3)[:, 2, :]
                               * task.model.geom_size[task.index.bean_collision_geoms], axis=1)
        eligible = pickup_eligible(initial, diag, state['bean_positions'], rim, radii)
        eligible &= (diag['linear_speed_m_s'] < c['linear_speed_m_s']) & (diag['angular_speed_rad_s'] < c['angular_speed_rad_s'])
        window = update_window(window, eligible, task.dt)
        best = np.maximum(best, window)
        if pickup_state is None and np.any(window + 1e-12 >= c['m1c']['pickup_window_s']):
            pickup_state = task.get_state()
    observer.pickup_state = pickup_state
    ids = np.array(task.index.bean_ids)[best + 1e-12 >= c['m1c']['pickup_window_s']].tolist()
    assert ids, f'No real pickup: best_windows={best}, supported={observer.diag["spoon_supported"]}'
    contact = observer.first_entry_contact
    assert contact is not None, 'Missing actual front entry contact'
    assert contact['height_relative_radius'] < 0 and contact['force_on_bean_world_n'][2] > 0, contact
    assert observer.phase_motion['sweep']['displacement_m'][2] > 0, observer.phase_motion
    return dict(seed=seed, reset_preset='beans_in_bowl', picked_ids=ids, support_windows_s=best,
                first_entry_contact=contact, phase_motion=observer.phase_motion,
                observation_duration_s=c['m1c']['pickup_observation_s'],
                achieved=achieved,
                path=[dict(phase=p, position=pos, rotation=r, speed_m_s=v) for p, pos, r, v in path])


def carry(task, cfg, observer):
    task.reset(preset='beans_on_spoon')
    observer.advance(.3)
    start = task.snapshot()
    loads = []
    for _ in range(round(cfg['support_hold_s'] / task.dt)):
        state = task.step_physics()
        assert observer.diag['spoon_supported'][0] and observer.diag['in_spoon_head'][0], 'bean_000 lost static support'
        loads.append(state['compensated_wrench'][:3])
    drift = np.linalg.norm(state['bean_positions'][0] - start['bean_positions'][0])
    assert drift < cfg['support_drift_m'], drift
    expected = task.model.body_mass[task.index.bean_bodies[0]] * task.model.opt.gravity
    measured = np.mean(loads[-round(.5/task.dt):], axis=0)
    relative_error = float(np.linalg.norm(measured - expected) / np.linalg.norm(expected))
    assert relative_error < cfg['beans_native']['m1c']['load_relative_error'], relative_error
    assert set(task.index.tool_bodies).isdisjoint(task.index.bean_bodies)
    base = task.data.site_xmat[task.index.base].reshape(3, 3)
    now = task.data.time
    task.adapter.set_twist(np.r_[base.T @ [.008, 0., .003], np.zeros(3)], now, now + 1.1)
    carried_loads = []
    for _ in range(round(cfg['beans_native']['m1c']['carry_duration_s'] / task.dt)):
        state = task.step_physics()
        carried_loads.append(state['compensated_wrench'][:3])
        assert observer.diag['spoon_supported'][0] and observer.diag['in_spoon_head'][0], 'bean_000 lost during gentle carry'
    carry_load = np.mean(carried_loads, axis=0)
    carry_error = float(np.linalg.norm(carry_load - expected) / np.linalg.norm(expected))
    assert carry_error < cfg['beans_native']['m1c']['load_relative_error'], carry_error
    return dict(bean_id='bean_000', reset_preset='beans_on_spoon', counts_as_sweep=False,
                carry_measured_load_n=carry_load, carry_load_relative_error=carry_error,
                static_drift_m=float(drift), measured_load_n=measured, expected_load_n=expected,
                load_relative_error=relative_error, static_duration_s=cfg['support_hold_s'],
                carry_duration_s=cfg['beans_native']['m1c']['carry_duration_s'])


def drop(task, cfg, observer, kind):
    task.reset(preset='beans_on_spoon')
    observer.advance(.3)
    assert observer.diag['spoon_supported'][0]
    state = task.snapshot()
    rotation = state['tcp_rotation']
    base = task.data.site_xmat[task.index.base].reshape(3, 3)
    if kind == 'tilt':
        command, duration = np.r_[np.zeros(3), base.T @ rotation[:, 1] * .4], 4.
    else:
        settings = cfg['acceleration_diagnostic']
        task.robot_config.update({k:v for k,v in settings.items() if k in task.robot_config})
        task.scene_config['joint_speed_fault_rad_s'] = settings['joint_speed_fault_rad_s']
        task.adapter = RobotAdapter(task.model, task.data, task.index, task.robot_config)
        ids, gain = task.index.actuators, settings['servo_gain_scale']
        task.model.actuator_gainprm[ids, 0] *= gain
        task.model.actuator_biasprm[ids, 1] *= gain
        task.model.actuator_biasprm[ids, 2] *= np.sqrt(gain)
        command = np.r_[-base.T @ rotation[:, 1] * settings['linear_speed_limit'], np.zeros(3)]
        duration = .65
    observer.phase = kind
    now = task.data.time
    task.adapter.set_twist(command, now, now + duration + .01)
    window = 0.
    confirmed = None
    scoop = set(task.index.scoop_geoms)
    # A fast pulse amplifies a one-substep confirmation-time difference into
    # millimetres. Retain the first drop event, compare at one fixed pulse time.
    observation = .2 if kind == 'acceleration' else duration
    for _ in range(round(observation / task.dt)):
        state = task.step_physics()
        head_touch = any(((row['bean1_id'] == 'bean_000' and row['geom2'] in scoop)
                          or (row['bean2_id'] == 'bean_000' and row['geom1'] in scoop))
                         and row['force_n'] > task.bean_acceptance['support_min_force_n']
                         for row in task.contacts + task.applied_contacts)
        lost = not head_touch and not observer.diag['spoon_supported'][0] and not observer.diag['in_spoon_head'][0]
        window = float(update_window(window, lost, task.dt))
        if confirmed is None and window + 1e-12 >= cfg['support_confirm_s']:
            confirmed = dict(bean_id='bean_000', dropped=True, time_s=state['time'], absent_and_outside_s=window,
                        final_relative_position=observer.diag['bean_positions_tcp'][0],
                        command=command, diagnostic_limits=cfg['acceleration_diagnostic'] if kind == 'acceleration' else None)
            if kind == 'tilt':
                return confirmed
    if confirmed is not None:
        confirmed['observation_end_s'] = float(task.data.time)
        return confirmed
    raise AssertionError(f'bean_000 did not detach under {kind}')


def reset_check(task, cfg, observer):
    signatures, settle = [], []
    for iteration in range(cfg['reset_count']):
        task.adapter.set_twist(np.ones(6)*.01, task.data.time, task.data.time+1.)
        task.data.xfrc_applied[:] = 2.
        task.data.qfrc_applied[:] = 3.
        task.monitor.update([dict(group1='food', group2='spoon', force_n=6.)], task.dt, task.data.time)
        old_command = task.adapter.command
        task.adapter.stop('injected', fault=True)
        task.adapter.command = old_command
        task.adapter.velocity[:] = .01
        task.adapter.last_ik_velocity[:] = .02
        task.adapter.ik_failures = 2
        task.adapter.error_detail = 'injected'
        task.scenario_state['future_events'] = [dict(time_s=99.)]
        task.terminated, task.failure_reason = True, 'injected'
        task.external_wrench = np.ones((3, 3))
        task.reset(seed=7, preset='beans_in_bowl')
        assert task.seed == 7 and not task.terminated and task.failure_reason is None
        assert task.reset_diagnostics['status'] == 'passed'
        assert not task.monitor.events and not task.monitor.peak_n and not task.monitor.pair_peaks
        assert not task.monitor.impulse_ns and not task.monitor.over_limit_s
        assert task.external_wrench is None and task.adapter.command is None and task.adapter.fault is None
        assert not task.data.xfrc_applied.any() and not task.data.qfrc_applied.any()
        assert not task.adapter.velocity.any() and not task.adapter.last_ik_velocity.any()
        assert not task.adapter.ik_failures and task.adapter.error_detail is None
        assert task.tick == 0 and task.data.time == 0
        assert task.adapter.status == 'idle' and task.adapter.last_command_time == -np.inf
        assert task.logic is None and task.scenario_state['future_events'] == []
        assert task.substep_contact_peak_n == 0. and task.substep_wrist_peak_n == 0.
        assert not any(w.number for w in task.data.warning)
        state = task.get_state()
        settle.append({key: task.reset_diagnostics[key] for key in
                       ('bean_settle_s', 'bean_settle_wall_s', 'robot_settle_wall_s', 'robot_performance', 'performance', 'max_penetration_m')})
        if observer.initial is None:
            observer.initial = state
        if task.reset_diagnostics['max_penetration_m'] > observer.max_penetration_m:
            observer.max_penetration_m = task.reset_diagnostics['max_penetration_m']
            observer.penetration_peak = dict(task.reset_diagnostics['penetration_peak'], reset_iteration=iteration+1)
        signatures.append(np.r_[state['physics'], state['adapter']['reference_q'], state['adapter']['target']])
        if iteration:
            np.testing.assert_allclose(signatures[-1], signatures[0], rtol=0, atol=cfg['reset_atol'])
        if (iteration+1) % 10 == 0:
            print(task.robot_id, 'reset', iteration+1, '/', cfg['reset_count'], flush=True)
    return dict(resets=len(signatures), seed=7, max_difference=float(np.max(np.abs(np.array(signatures)-signatures[0]))),
                final_bean_velocities_preserved=bool(np.any(task.data.qvel[task.index.bean_dofs])),
                reset_performance=settle)


def head(task, cfg, observer):
    task.scene_config['head_fixed'] = False
    task.reset(preset='beans_in_bowl')
    samples = []
    for _ in range(round(2./task.dt)):
        task.step_physics()
        samples.append(task.data.qpos[task.model.jnt_qposadr[task.index.head_joints]].copy())
    excursion = np.ptp(samples, axis=0)
    assert np.linalg.norm(excursion[:2]) > .001 and excursion[4] > .001
    return dict(excursion=excursion)


def reachability(task, cfg, observer):
    task.reset(preset='empty')
    path = sweep_path(task, cfg)
    results = {}
    for phase, position, rotation, speed in path:
        observer.phase = phase
        state = move_pose(task, observer, position, rotation, cfg, speed=speed)
        results[phase] = dict(position_error_m=float(np.linalg.norm(state['tcp_position']-position)), q=state['q'])
    mouth = task.data.site_xpos[task.model.site('mouth_entry').id].copy()
    state = move_pose(task, observer, mouth+[-.06,0,0], np.eye(3), cfg)
    results['mouth_wait'] = dict(position_error_m=float(np.linalg.norm(state['tcp_position']-mouth-[-.06,0,0])), q=state['q'])
    return results


def guards(task, cfg, observer):
    tests = ['tests/test_contracts.py', 'tests/test_guard_physics.py', 'tests/test_tableware.py',
             'tests/test_m1c_native.py']
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(tests=tests, output=result.stdout, return_code=result.returncode)


def validate(robot, output, selected=None, *, numerical=None, replay=None):
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_json('configs/acceptance.json')
    cases = dict(hold=holding, tracking=tracking, wrench=wrench_check, faults=fault_checks,
                 reset=reset_check, head=head, carry=carry,
                 tilt=lambda t,c,o: drop(t,c,o,'tilt'), acceleration=lambda t,c,o: drop(t,c,o,'acceleration'),
                 reachability=reachability, guards=guards)
    for seed in cfg['seeds']:
        cases[f'sweep_seed_{seed}'] = lambda t,c,o,seed=seed: sweep(t,c,o,seed)
    if selected is not None and set(selected)-cases.keys():
        raise ValueError(f'Unknown cases: {set(selected)-cases.keys()}')
    inputs = asset_files(robot) + list((ROOT/'src/feedingrobot').rglob('*.py')) + list((ROOT/'tests').glob('test_*native*.py'))
    inputs += [ROOT/'tests/test_contracts.py', ROOT/'tests/test_guard_physics.py', ROOT/'tests/test_tableware.py',
               ROOT/'requirements.lock.txt', ROOT/'third_party_manifest.json']
    report = dict(schema_version=1, model_version='single_bean_native_v1', stage='M1-C', robot_id=robot,
                  status='incomplete', m1_status='incomplete', stages={'M1-C':'not_verified','M1-D':'not_verified'},
                  reset_layout='fixed position/quaternion; seeds test repetition, not layout coverage',
                  deferred=['control_numerical_comparison', 'parameter_freeze', 'M3', 'M4'],
                  acceptance=cfg, input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, cases={name: dict(status='not_verified') for name in cases})
    for name, check in cases.items():
        if selected is not None and name not in selected:
            report['cases'][name] = dict(status='not_verified')
            continue
        observer = None
        started = time.monotonic()
        try:
            loading_started = time.perf_counter()
            task = FeedingTask(robot, timestep=numerical[0] if numerical else None)
            load_wall_s = time.perf_counter() - loading_started
            if numerical:
                task.model.opt.iterations, task.model.opt.tolerance = numerical[1:]
            task.scene_config['head_fixed'] = True
            observer = Evidence(task)
            original_reset = task.reset
            def episode_reset(*args, **kwargs):
                original_reset(*args, **kwargs)
                if replay is not None:
                    from feedingrobot.scripts.validate_m1 import restore_numerical_state
                    with (replay / f'{name}_states.pkl').open('rb') as file:
                        reference = pickle.load(file)
                    restore_numerical_state(task, reference['episode_reset'])
                observer.episode_reset = task.get_state()
                return task.snapshot()
            task.reset = episode_reset
            if 'solver' not in report:
                report.update(mujoco_version=mujoco.__version__, model_signature=task.state_signature(),
                    solver=dict(timestep_s=task.dt, iterations=int(task.model.opt.iterations), tolerance=float(task.model.opt.tolerance),
                                ccd_tolerance_m=float(task.model.opt.ccd_tolerance), ccd_iterations=int(task.model.opt.ccd_iterations),
                                integrator=int(task.model.opt.integrator), solver=int(task.model.opt.solver), cone=int(task.model.opt.cone)),
                    bean_index=dict(ids=task.index.bean_ids, bodies=task.index.bean_bodies, joints=task.index.bean_joints,
                                    qpos=task.index.bean_qpos, dofs=task.index.bean_dofs,
                                    collision_geoms=task.index.bean_collision_geoms, masses_kg=task.model.body_mass[task.index.bean_bodies]),
                    contact_parameters=task.scene_config['beans'])
            if name == 'faults':
                # The original check deliberately latches this fault and verifies
                # cancellation; all physical guards and observations stay active.
                observer.expected_fault = 'workspace_limit'
            detail = check(task, cfg, None if name in ('hold','tracking','wrench','faults') else observer)
            report['cases'][name] = dict(status='passed', metrics=detail)
        except Exception:
            report['cases'][name] = dict(status='failed', error=traceback.format_exc())
        case = report['cases'][name]
        case['wall_s'] = time.monotonic()-started
        if observer is not None:
            case['final_tcp_position'] = task.snapshot()['tcp_position']
            case['final_tcp_rotation'] = task.snapshot()['tcp_rotation']
            case['semantic_pair_peaks_n'] = dict(task.monitor.pair_peaks)
            case['semantic_pair_impulses_ns'] = dict(task.monitor.pair_impulses)
            case.update(observer.result(), load_wall_s=load_wall_s, model_load_wall_s=task.model_load_wall_s, failure_reason=task.failure_reason, adapter_fault=task.adapter.fault,
                        actual_control_parameters=control_parameters(task))
            if case['status'] == 'failed':
                case['failure_time_s'] = float(task.data.time)
                case['failure_phase'] = observer.phase
                case['failure_bean_ids'] = (observer.diag['penetration_ids'] if observer.diag is not None else [])
                if name in ('carry', 'tilt', 'acceleration'):
                    case['failure_bean_ids'] = ['bean_000']
                elif task.failure_reason == 'contact_limit':
                    case['failure_bean_ids'] = sorted({b for row in task.contacts+task.applied_contacts
                        if 'spoon' in (row['group1'], row['group2']) for b in (row['bean1_id'], row['bean2_id']) if b})
                elif 'No real pickup' in case['error']:
                    case['failure_bean_ids'] = list(task.index.bean_ids)
            # Complete trusted local replay snapshots retain -inf timestamp
            # sentinels and all arrays; JSON is the human-readable evidence.
            logging_started = time.perf_counter()
            if observer.initial is None:
                observer.initial = task.get_state()
            with (output/f'{name}_states.pkl').open('wb') as file:
                pickle.dump(dict(initial=observer.initial, final=task.get_state(),
                    episode_reset=getattr(observer, 'episode_reset', None),
                    initial_parameters=observer.initial_parameters or control_parameters(task),
                    final_parameters=control_parameters(task)), file)
            if getattr(observer, 'pickup_state', None) is not None:
                with (output/f'{name}_pickup.pkl').open('wb') as file:
                    pickle.dump(observer.pickup_state, file)
            (output/f'{name}_trajectory.json').write_text(json.dumps(serializable(observer.trace), allow_nan=False)+'\n')
            case['state_file'] = f'{name}_states.pkl'
            case['trajectory_file'] = f'{name}_trajectory.json'
            case['performance']['logging_s'] = time.perf_counter() - logging_started
            episode_wall = case['performance']['episode_wall_s']
            case['performance']['control_with_logging_rtf'] = (
                case['performance']['steps']*task.dt/(episode_wall+case['performance']['logging_s']) if episode_wall else None)
        print(robot, name, case['status'], flush=True)
        report['status'] = 'failed' if any(c['status']=='failed' for c in report['cases'].values()) else 'incomplete'
        (output/'m1c_report.json').write_text(json.dumps(serializable(report), indent=2, allow_nan=False)+'\n')
    if all(c['status']=='passed' for c in report['cases'].values()):
        report['status'] = report['stages']['M1-C'] = 'passed'
    elif any(c['status']=='failed' for c in report['cases'].values()):
        report['stages']['M1-C'] = 'failed'
    (output/'m1c_report.json').write_text(json.dumps(serializable(report), indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=['panda','ur5e'], default='panda')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--output')
    args = parser.parse_args()
    output = ROOT/(args.output or f'outputs/single_bean/v1/m1/{args.robot}/m1c')
    report = validate(args.robot, output, args.cases)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
