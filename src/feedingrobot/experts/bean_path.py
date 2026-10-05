"""Single-bean pickup targets from measured, immutable collision geometry."""

import mink
import numpy as np


def pickup_path(geometry, parameters=None):
    g, c = geometry, dict(geometry['acquisition_config'])
    parameters = parameters or {}
    c['entry_pitch_deg'] += np.rad2deg(parameters.get('entry_pitch_offset_rad', 0.))
    centre = g['bowl_position']
    yaw = mink.SO3.exp(np.array([0., 0., np.deg2rad(c['entry_yaw_deg'])]))
    rotation = (yaw @ mink.SO3.exp(np.array([0., np.deg2rad(c['entry_pitch_deg']), 0.]))).as_matrix()
    tip = g['scoop_points'][np.argmin((g['scoop_points'] @ rotation.T)[:, 2])]
    z = -float((rotation @ tip)[2]) + c['contact_clearance_m']
    x, y = c['sweep_start_xy_m']
    path = [('above', centre + [x, y, .14], rotation, .03),
            ('pre_entry', centre + [x, y, z + .025], rotation, .03),
            ('entry', centre + [x, y, z], rotation, c['sweep_speed_m_s'])]
    wall_yaw = mink.SO3.exp(np.array([0., 0., np.deg2rad(-10.)]))
    last_forward = x
    for phase, height, angle, side, stage_yaw, gap in [
            ('sweep', c['contact_clearance_m'], c['entry_pitch_deg'], y, yaw, .003),
            ('wall_align', .0085, 60., .006, wall_yaw, .001),
            ('wall_45', .028, 45., .006, wall_yaw, .001),
            ('wall_30', .045, 30., .006, wall_yaw, .001),
            ('wall_15', .080, 15., .006, wall_yaw, .001),
            ('wall_level', .110, 0., .006, wall_yaw, .001)]:
        r = (stage_yaw @ mink.SO3.exp(np.array([0., np.deg2rad(angle), 0.]))).as_matrix()
        vertical = height - (r @ tip)[2]
        vertices = g['tool_points'] @ r.T
        active = vertices[centre[2] + vertical + vertices[:, 2] <= g['rim_z']]
        upper, lower = np.inf, -np.inf
        if len(active):
            for normal, position, half_size in zip(g['wall_normals'], g['wall_positions'], g['wall_half_sizes']):
                denominator = normal[0]
                if abs(denominator) < 1e-8:
                    continue
                value = (normal @ (position-centre) - half_size - c['contact_clearance_m']
                         - normal @ np.array([0., side, vertical]) - np.max(active @ normal)) / denominator
                if denominator > 0:
                    upper = min(upper, value)
                else:
                    lower = max(lower, value)
            if lower > upper:
                raise ValueError(f'No whole-tool wall clearance at {phase}: {lower}, {upper}')
            last_forward = upper - gap
        path.append((phase, centre + [last_forward, side, vertical], r, c['sweep_speed_m_s']))
    clearance = np.r_[path[-1][1][:2], centre[2]+.18+parameters.get('lift_height_offset_m', 0.)]
    path.append(('clearance', clearance, wall_yaw.as_matrix(), .03))
    path.append(('wall_seat_tip', clearance,
                 (wall_yaw @ mink.SO3.exp(np.array([0., np.deg2rad(-5.), 0.]))).as_matrix(), .005))
    path.append(('wall_seat_level', clearance, wall_yaw.as_matrix(), .005))
    return path
