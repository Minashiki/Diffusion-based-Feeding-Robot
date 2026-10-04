"""Native Bean placement and read-only M1 physical evidence."""

import mujoco
import numpy as np

from feedingrobot.sim.model import named_id


def bean_state(model, data, index):
    velocities = np.zeros((len(index.bean_ids), 6))
    for i, body in enumerate(index.bean_bodies):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body), velocities[i], 0)
    return dict(bean_ids=list(index.bean_ids), bean_positions=data.xpos[index.bean_bodies].copy(),
                bean_quaternions=data.xquat[index.bean_bodies].copy(),
                bean_linear_velocities_world=velocities[:, 3:].copy(),
                bean_angular_velocities_world=velocities[:, :3].copy())


def spoon_frame_state(task, bounds=None):
    """Read-only head-region and moving-frame velocities; proximity is not support."""
    m, d, idx = task.model, task.data, task.index
    state = bean_state(m, d, idx)
    rotation = d.site_xmat[idx.tcp].reshape(3, 3)
    delta = state['bean_positions'] - d.site_xpos[idx.tcp]
    local = delta @ rotation
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, idx.tcp, velocity, 0)
    if bounds is None:
        points = []
        for geom in idx.scoop_geoms:
            mesh = m.geom_dataid[geom]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            world = m.mesh_vert[start:start + count] @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
            points.append((world - d.site_xpos[idx.tcp]) @ rotation)
        points = np.concatenate(points)
        lo, hi = points.min(axis=0), points.max(axis=0)
        # A centre must project onto the head and stay within one bean diameter
        # above it. Airborne proximity alone cannot establish support.
        radius = float(m.geom_size[idx.bean_collision_geoms].max())
        lo[2] -= radius
        hi[2] += 2 * radius
    else:
        lo, hi = (np.array(value, copy=True) for value in bounds)
    region = np.all((local >= lo) & (local <= hi), axis=1)
    for i in np.flatnonzero(region):
        region[i] = any(mujoco.mj_rayMesh(m, d, g, state['bean_positions'][i], -rotation[:, 2]) >= 0
                        for g in idx.scoop_geoms)
    return dict(bean_positions_tcp=local,
                bean_linear_velocities_tcp=(state['bean_linear_velocities_world'] - velocity[3:]
                                             - np.cross(velocity[:3], delta)) @ rotation,
                bean_angular_velocities_tcp=(state['bean_angular_velocities_world'] - velocity[:3]) @ rotation,
                in_spoon_head=region, spoon_region_min_tcp=lo, spoon_region_max_tcp=hi)


def place_beans(task, preset, seed):
    m, d, idx = task.model, task.data, task.index
    count = len(idx.bean_ids)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    quats = rng.normal(size=(count, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    origin = np.array(task.scene_config['bowl_frame_position_m'])
    positions = []
    for i in range(count):
        layer = int(i >= 8)
        angle = phase + 2 * np.pi * (i if layer == 0 else i - 8) / (8 if layer == 0 else 7)
        positions.append(origin + [.022 * np.cos(angle), .022 * np.sin(angle), (.0075, .022)[layer]])
    if count == 1:
        positions = [origin + task.scene_config['beans']['reset_position_bowl_m']]
        quats[:] = task.scene_config['beans']['reset_quaternion_wxyz']
    if preset == 'empty':
        positions = np.c_[2. + .03 * np.arange(count), np.full(count, 2.), np.full(count, -.2455)]
        quats[:] = [1, 0, 0, 0]
    d.qpos[idx.bean_qpos] = np.c_[positions, quats]
    d.qvel[idx.bean_dofs] = 0
    mujoco.mj_forward(m, d)
    if preset == 'beans_on_spoon':
        rotation = d.site_xmat[idx.tcp].reshape(3, 3)
        position = d.site_xpos[idx.tcp].copy()
        point = position + .05 * rotation[:, 2]
        hits = [mujoco.mj_rayMesh(m, d, g, point, -rotation[:, 2]) for g in idx.scoop_geoms]
        hits = [hit for hit in hits if hit >= 0]
        if not hits:
            raise ValueError('TCP has no real scoop support surface')
        position += (.05 - min(hits) + .004 + .0005) * rotation[:, 2]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        d.qpos[idx.bean_qpos[0]] = np.r_[position, quat]
        mujoco.mj_forward(m, d)
        # Raise only during reset until the actual convex scoop has the spawn gap.
        for _ in range(100):
            distance = min(mujoco.mj_geomDistance(m, d, int(idx.bean_collision_geoms[0]), g, .01, None)
                           for g in idx.scoop_geoms)
            if distance >= .0005 - 1e-9:
                break
            d.qpos[idx.bean_qpos[0, :3]] += (.0005 - distance + 1e-8) * rotation[:, 2]
            mujoco.mj_forward(m, d)
        else:
            raise ValueError('Unable to place bean_000 above the actual scoop')


def spawn_clearance(task, preset):
    m, d, idx = task.model, task.data, task.index
    positions = d.xpos[idx.bean_bodies]
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    pairs = distances[np.triu_indices(len(idx.bean_ids), 1)]
    minimum = float(pairs.min() - .014) if len(pairs) else float("inf")
    for i, geom in enumerate(idx.bean_collision_geoms):
        supports = [g for g in range(m.ngeom) if g != geom and g not in idx.bean_id_by_geom
                    and ((int(m.geom_contype[geom]) & int(m.geom_conaffinity[g]))
                         or (int(m.geom_conaffinity[geom]) & int(m.geom_contype[g])))]
        for other in supports:
            minimum = min(minimum, mujoco.mj_geomDistance(m, d, int(geom), int(other), .1, None))
        if preset != 'empty' and not (preset == 'beans_on_spoon' and i == 0):
            # Circumscribed sphere against the real finite bowl, not infinite wall planes.
            for other in idx.bowl_geoms:
                point = positions[i]
                rotation = d.geom_xmat[other].reshape(3, 3)
                local = rotation.T @ (point - d.geom_xpos[other])
                if m.geom_type[other] == mujoco.mjtGeom.mjGEOM_BOX:
                    delta = np.maximum(np.abs(local) - m.geom_size[other], 0)
                    minimum = min(minimum, float(np.linalg.norm(delta)) - .007)
                else:
                    delta = [max(np.linalg.norm(local[:2]) - m.geom_size[other, 0], 0),
                             max(abs(local[2]) - m.geom_size[other, 1], 0)]
                    minimum = min(minimum, float(np.linalg.norm(delta)) - .007)
    return minimum


def bean_diagnostics(task, *, spoon_frame=False, spoon_bounds=None):
    m, d, idx = task.model, task.data, task.index
    state = bean_state(m, d, idx)
    bottom = named_id(m, mujoco.mjtObj.mjOBJ_GEOM, 'collision_bowl_fast_bottom_disk')
    walls = [g for g in idx.bowl_geoms if g != bottom]
    floor_z = d.geom_xpos[bottom, 2] + m.geom_size[bottom, 1]
    rim_z = max(d.geom_xpos[g, 2] + np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g] for g in walls)
    limit = task.bean_acceptance['penetration_limit_m']
    bean_geoms = idx.bean_collision_geoms
    bean_centres = d.geom_xpos[bean_geoms]
    bean_rotations = d.geom_xmat[bean_geoms].reshape(-1, 3, 3)
    axes = m.geom_size[bean_geoms]
    vertical = np.linalg.norm(bean_rotations[:, 2, :] * axes, axis=1)
    inside = ((bean_centres[:, 2] - vertical >= floor_z - limit)
              & (bean_centres[:, 2] + vertical <= rim_z + limit))
    boundary_violations = []
    for reason, distances in (('bottom', bean_centres[:, 2] - vertical - floor_z),
                              ('rim', rim_z - bean_centres[:, 2] - vertical)):
        for i in np.flatnonzero(distances < -limit):
            boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason=reason,
                                            distance_m=float(distances[i])))
    for wall in walls:
        rotation = d.geom_xmat[wall].reshape(3, 3)
        local = (bean_centres - d.geom_xpos[wall]) @ rotation
        directions = np.einsum('bij,jk->bik', bean_rotations.transpose(0, 2, 1), rotation)
        radii = np.linalg.norm(directions * axes[:, :, None], axis=1)
        size = m.geom_size[wall]
        overlaps = np.all(np.abs(local[:, 1:]) <= size[1:] + radii[:, 1:], axis=1)
        outward = overlaps & (local[:, 0] > -size[0] + limit)
        inside[outward] = False
        for i in np.flatnonzero(outward):
            boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason='wall_outer_side',
                                            geom_name=m.geom(int(wall)).name,
                                            distance_m=float(-size[0] - local[i, 0])))
        candidates = np.flatnonzero(overlaps & ~outward & (local[:, 0] + radii[:, 0] > -size[0] + limit))
        for i in candidates:
            # Projection overlap is only a broad phase; the lower edge is finite.
            distance = mujoco.mj_geomDistance(m, d, int(bean_geoms[i]), int(wall), .0005, None)
            if distance < -limit:
                inside[i] = False
                boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason='wall_penetration',
                                                geom_name=m.geom(int(wall)).name, distance_m=float(distance)))
    outer_radius = max(np.linalg.norm(d.geom_xpos[g, :2] - d.geom_xpos[bottom, :2])
                       + np.linalg.norm(np.abs(d.geom_xmat[g].reshape(3, 3)[:2]) @ m.geom_size[g]) for g in walls)
    radial_clearance = outer_radius - np.linalg.norm(bean_centres[:, :2] - d.geom_xpos[bottom, :2], axis=1)
    inside &= radial_clearance >= 0
    for i in np.flatnonzero(radial_clearance < 0):
        boundary_violations.append(dict(bean_id=idx.bean_ids[i], reason='outer_radius',
                                        distance_m=float(radial_clearance[i])))
    physical = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
    rotations = d.geom_xmat.reshape(-1, 3, 3)
    centres = d.geom_xpos + np.einsum('gij,gj->gi', rotations, m.geom_aabb[:, :3])
    extents = np.einsum('gij,gj->gi', np.abs(rotations), m.geom_aabb[:, 3:])
    penetration = 0.
    penetration_peak = None
    penetration_ids = set()
    planes = set(physical[m.geom_type[physical] == mujoco.mjtGeom.mjGEOM_PLANE])
    for geom in idx.bean_collision_geoms:
        candidates = physical[np.all(np.abs(centres[physical] - centres[geom]) <= extents[physical] + extents[geom] + .0005, axis=1)]
        candidates = set(candidates) | planes
        for other in candidates:
            if other == geom or (other in idx.bean_id_by_geom and other < geom):
                continue
            distance = mujoco.mj_geomDistance(m, d, int(geom), int(other), .0005, None)
            if -distance > penetration:
                penetration = -distance
                penetration_peak = dict(geom_names=[m.geom(int(g)).name for g in (geom, other)],
                                        bean_ids=[idx.bean_id_by_geom[g] for g in (geom, other)
                                                  if g in idx.bean_id_by_geom],
                                        distance_m=float(distance), source='geometry')
            if distance < -limit:
                penetration_ids.add(idx.bean_id_by_geom[geom])
                if other in idx.bean_id_by_geom:
                    penetration_ids.add(idx.bean_id_by_geom[other])
    for source, rows in (('boundary_contact', task.contacts),
                         ('applied_contact', getattr(task, 'applied_contacts', []))):
        for row in rows:
            if row.get('bean1_id') or row.get('bean2_id'):
                if -row['distance'] > penetration:
                    penetration = -row['distance']
                    penetration_peak = dict(geom_names=[m.geom(row[k]).name for k in ('geom1', 'geom2')],
                                            bean_ids=[b for b in (row['bean1_id'], row['bean2_id']) if b],
                                            distance_m=row['distance'], source=source)
                if row['distance'] < -limit:
                    penetration_ids.update(b for b in (row['bean1_id'], row['bean2_id']) if b)
    roots = {'bowl': set(), 'spoon': set()}
    edges = []
    for row in task.contacts:
        g1, g2 = row['geom1'], row['geom2']
        for geom, other, sign in ((g1, g2, -1), (g2, g1, 1)):
            bean = idx.bean_id_by_geom.get(geom)
            if not bean or sign * row['force_on_geom2_world'][2] <= task.bean_acceptance['support_min_force_n']:
                continue
            if other == bottom:
                roots['bowl'].add(bean)
            elif other in idx.scoop_geoms:
                rotation = d.site_xmat[idx.tcp].reshape(3, 3)
                if any(mujoco.mj_rayMesh(m, d, g, d.geom_xpos[geom], -rotation[:, 2]) >= 0 for g in idx.scoop_geoms):
                    roots['spoon'].add(bean)
            elif other in idx.bean_id_by_geom and d.geom_xpos[geom, 2] > d.geom_xpos[other, 2]:
                edges.append((idx.bean_id_by_geom[other], bean))
    direct = {key: sorted(value) for key, value in roots.items()}
    for supported in roots.values():
        for _ in range(len(idx.bean_ids)):
            before = len(supported)
            supported.update(upper for lower, upper in edges if lower in supported)
            if len(supported) == before:
                break
    return dict(in_bowl=inside, bowl_supported=np.array([b in roots['bowl'] for b in idx.bean_ids]),
                spoon_supported=np.array([b in roots['spoon'] for b in idx.bean_ids]), support_edges=edges,
                linear_speed_m_s=np.linalg.norm(state['bean_linear_velocities_world'], axis=1),
                angular_speed_rad_s=np.linalg.norm(state['bean_angular_velocities_world'], axis=1),
                max_penetration_m=float(penetration), penetration_ids=sorted(penetration_ids),
                penetration_peak=penetration_peak, boundary_violations=boundary_violations,
                direct_support=direct, **(spoon_frame_state(task, spoon_bounds) if spoon_frame else {}))
