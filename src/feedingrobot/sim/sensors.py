"""Raw wrist F/T and frame transforms.

MuJoCo force/torque sensors report the parent-on-child interaction in the site
frame. This project publishes the wrench of the outside world on the tool.
The fixed sign s is not chosen from the current sample.
"""

from __future__ import annotations

import numpy as np
import mujoco

# Sensor parent-on-child is opposite the external wrench on the tool.
WRENCH_SIGN_EXTERNAL_ON_TOOL = -1.0


def site_rotation(data, site_id: int) -> np.ndarray:
    """Rotation that maps site-frame vectors into world."""
    return np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()


def site_position(data, site_id: int) -> np.ndarray:
    return np.array(data.site_xpos[site_id], dtype=float).copy()


def read_raw_wrench(model, data, force_adr: int, torque_adr: int) -> np.ndarray:
    force = np.array(data.sensordata[force_adr : force_adr + 3], dtype=float)
    torque = np.array(data.sensordata[torque_adr : torque_adr + 3], dtype=float)
    return np.concatenate([force, torque])


def rotate_wrench(force_s: np.ndarray, torque_s: np.ndarray, rot_ws: np.ndarray, sign: float):
    """Map a site wrench into world. Moment stays at the site origin."""
    sign = float(sign)
    force_w = sign * (rot_ws @ np.asarray(force_s, dtype=float))
    torque_w = sign * (rot_ws @ np.asarray(torque_s, dtype=float))
    return force_w, torque_w


def shift_torque(torque_w: np.ndarray, force_w: np.ndarray, p_from: np.ndarray, p_to: np.ndarray):
    """Move the moment origin from p_from to p_to. Force is unchanged."""
    return np.asarray(torque_w, dtype=float) + np.cross(
        np.asarray(p_from, dtype=float) - np.asarray(p_to, dtype=float),
        np.asarray(force_w, dtype=float),
    )


def world_and_tcp_wrench(raw_sensor: np.ndarray, rot_ws: np.ndarray, p_site: np.ndarray, p_tcp: np.ndarray, sign: float):
    force_w, torque_w = rotate_wrench(raw_sensor[:3], raw_sensor[3:], rot_ws, sign)
    torque_tcp = shift_torque(torque_w, force_w, p_site, p_tcp)
    return np.concatenate([force_w, torque_w]), np.concatenate([force_w, torque_tcp])


def wrist_state(model, data, index):
    """External-on-tool wrench, plus rigid tool gravity/inertia compensation.

    MuJoCo object acceleration includes the -gravity term. BODY coordinates
    refer to the inertial centre. No contact-solver force is used here.
    """
    mujoco.mj_rnePostConstraint(model, data)
    addresses = [int(model.sensor_adr[s]) for s in index.sensors]
    raw = read_raw_wrench(model, data, *addresses)
    ft = site_position(data, index.ft)
    tcp = site_position(data, index.tcp)
    world, at_tcp = world_and_tcp_wrench(raw, site_rotation(data, index.ft), ft, tcp, -1.)
    predicted = np.zeros(6)
    for body in index.tool_bodies:
        if model.body_mass[body] == 0:
            continue
        acceleration, velocity = np.zeros(6), np.zeros(6)
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, body, acceleration, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        rotation = data.ximat[body].reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
        force = -model.body_mass[body] * acceleration[3:]
        moment = -inertia @ acceleration[:3] - np.cross(velocity[:3], inertia @ velocity[:3])
        moment += np.cross(data.xipos[body] - tcp, force)
        predicted += np.r_[force, moment]
    return dict(raw_wrench_sensor=raw, wrench_world_at_ft=world, wrench_world_at_tcp=at_tcp,
                tool_gravity_inertia_wrench=predicted, compensated_wrench=at_tcp - predicted)
