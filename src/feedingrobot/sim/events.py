"""P0 geometry evidence and task events, evaluated at every physical boundary."""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from feedingrobot.sim.model import named_id

PHASES = ("SELECT", "ACQUIRE", "TRANSPORT", "WAIT_READY", "APPROACH", "TRANSFER", "RETRACT", "RECOVER")
SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))


def geom_corners(model, data, geom):
    """World OBB corners; cylinders use their enclosing box (conservative)."""
    size = model.geom_size[geom].copy()
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
        return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        size = np.array([size[0], size[0], size[1]])
    return (SIGNS * size) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def rotation_error(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1., 1.)))


def ellipsoid_bounds(model, data, geom, origin, rotation):
    """Exact extrema in a frame, including the ellipsoid's current orientation."""
    centre = (data.geom_xpos[geom] - origin) @ rotation
    axes = data.geom_xmat[geom].reshape(3, 3).T @ rotation
    radius = np.linalg.norm(model.geom_size[geom, :, None] * axes, axis=0)
    return centre - radius, centre + radius


def evidence(task):
    """Current single-bean geometry/contact truth; no simulation state writes."""
    m, d, idx, cfg = task.model, task.data, task.index, task.task_config
    mouth = named_id(m, mujoco.mjtObj.mjOBJ_SITE, "mouth_entry")
    receiver = named_id(m, mujoco.mjtObj.mjOBJ_SITE, "mouth_receiver")
    bean = int(idx.bean_collision_geoms[0])
    mr, jr, tr = [d.site_xmat[g].reshape(3, 3) for g in (mouth, receiver, idx.tcp)]
    tcp, food = d.site_xpos[idx.tcp], d.geom_xpos[bean]
    # Tool geometry is rigid. Cache local vertices, never world-space head/jaw state.
    if getattr(task, "_event_geometry_model", None) is not m:
        task._event_geometry_model = m
        tool_points = [(geom_corners(m, d, g) - tcp) @ tr for g in idx.spoon_geoms]
        task._event_tool_starts = np.r_[0, np.cumsum([len(p) for p in tool_points])[:-1]]
        task._event_tool_points = np.concatenate(tool_points)
        task._event_scoop_points = np.concatenate([(geom_corners(m, d, g) - tcp) @ tr for g in idx.scoop_geoms])
        bottom = named_id(m, mujoco.mjtObj.mjOBJ_GEOM, "collision_bowl_fast_bottom_disk")
        task._event_rim = max(geom_corners(m, d, g)[:, 2].max() for g in idx.bowl_geoms if g != bottom)
    scoop_local = task._event_scoop_points
    rel = (food - tcp) @ tr
    rows = [r for r in task.contacts if r["force_n"] > cfg["contact_min_force_n"]]
    pairs = {frozenset((r["group1"], r["group2"])) for r in rows}
    contact = lambda a, b: frozenset((a, b)) in pairs
    loads = [r["force_on_geom2_world"] * (1 if r["geom2"] == bean else -1)
             for r in rows if (r["geom2"] == bean and r["geom1"] in idx.scoop_geoms)
             or (r["geom1"] == bean and r["geom2"] in idx.scoop_geoms)]
    load = np.sum(loads, axis=0) if loads else np.zeros(3)
    support_force = float(load @ tr[:, 2])
    supported = (load[2] > task.bean_acceptance["support_min_force_n"]
                 and support_force > cfg["contact_min_force_n"]
                 and np.all(rel[:2] >= scoop_local[:, :2].min(0))
                 and np.all(rel[:2] <= scoop_local[:, :2].max(0))
                 and any(mujoco.mj_rayMesh(m, d, g, food, -tr[:, 2]) >= 0 for g in idx.scoop_geoms))
    world_min, world_max = ellipsoid_bounds(m, d, bean, np.zeros(3), np.eye(3))
    on_bowl = contact("food", "bowl")
    off_bowl = not on_bowl and world_min[2] > task._event_rim
    fc_min, fc_max = ellipsoid_bounds(m, d, bean, d.site_xpos[mouth], mr)
    fj_min, _ = ellipsoid_bounds(m, d, bean, d.site_xpos[receiver], jr)
    tolerance = cfg["receiver_xy_tolerance_m"]
    in_receiver = (np.all(fc_min[:2] >= np.asarray(cfg["receiver_min_xy_m"]) - tolerance)
                   and np.all(fc_max[:2] <= np.asarray(cfg["receiver_max_xy_m"]) + tolerance)
                   and fc_max[2] <= cfg["receiver_top_m"]
                   and fj_min[2] >= -cfg["receiver_floor_tolerance_m"])
    mouth_supported = bool(in_receiver and contact("food", "mouth"))
    tool = task._event_tool_points @ (tr.T @ mr) + (tcp-d.site_xpos[mouth]) @ mr
    tool_min = np.minimum.reduceat(tool, task._event_tool_starts, axis=0)
    tool_max = np.maximum.reduceat(tool, task._event_tool_starts, axis=0)
    inside = bool(np.any(np.all(tool_max >= cfg["interaction_min_m"], axis=1)
                         & np.all(tool_min <= cfg["interaction_max_m"], axis=1)))
    floor = geom_corners(m, d, named_id(m, mujoco.mjtObj.mjOBJ_GEOM, "jaw_floor"))
    upper = geom_corners(m, d, named_id(m, mujoco.mjtObj.mjOBJ_GEOM, "mouth_upper"))
    aperture = float(((upper - d.site_xpos[mouth]) @ mr)[:, 2].min()
                     - ((floor - d.site_xpos[mouth]) @ mr)[:, 2].max())
    projected = (scoop_local @ tr.T + tcp - d.site_xpos[mouth]) @ mr
    required = np.maximum(projected.max(0), fc_max) - np.minimum(projected.min(0), fc_min)
    required_height = float(required[2]) + cfg["clearance_margin_m"]
    required_width = float(required[1]) + cfg["clearance_margin_m"]
    wait = d.site_xpos[mouth] - mr[:, 0] * cfg["wait_offset_m"]
    aligned = rotation_error(tr, mr) <= cfg["orientation_tolerance_rad"]
    at_wait = np.linalg.norm(tcp - wait) <= cfg["position_tolerance_m"] and aligned
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(idx.bean_bodies[0]), velocity, 0)
    low_speed = (np.linalg.norm(velocity[3:]) < task.bean_acceptance["linear_speed_m_s"]
                 and np.linalg.norm(velocity[:3]) < task.bean_acceptance["angular_speed_rad_s"])
    bean_penetration = any((r.get("bean1_id") or r.get("bean2_id"))
                           and r["distance"] < -task.bean_acceptance["penetration_limit_m"]
                           for r in task.contacts + task.applied_contacts)
    return dict(supported=bool(supported), off_bowl=bool(off_bowl), on_bowl=on_bowl,
                bowl_clearance_m=float(world_min[2] - task._event_rim),
                pickup_eligible=bool(supported and off_bowl and low_speed),
                spoon_support_force_n=support_force, bean_penetration=bean_penetration,
                required_height_m=required_height, required_width_m=required_width,
                mouth_supported=mouth_supported, released=not contact("food", "spoon"),
                tool_inside=inside, tool_mouth_contact=contact("spoon", "mouth"),
                at_wait=bool(at_wait), aligned=bool(aligned), aperture_m=aperture,
                ready=bool(aperture >= required_height and
                           cfg["receiver_max_xy_m"][1] - cfg["receiver_min_xy_m"][1] >= required_width and aligned),
                food_ground_contact=contact("food", "table") or contact("food", "floor"),
                food_valid=bool(np.linalg.norm(food - task.scene_config["bowl_frame_position_m"]) < 1.),
                penetration=any(r["distance"] < -cfg["penetration_limit_m"] for r in task.contacts),
                mouth_position=d.site_xpos[mouth].copy(), mouth_rotation=mr.copy(),
                wait_position=wait, bean_position=food.copy(), tcp_position=tcp.copy())


class TaskEvents:
    def __init__(self, config):
        self.config = config
        self.phase = "SELECT"
        self.timers = dict(pickup=0., delivery=0., retract=0., ready=0., unsupported=0., penetration=0.)
        self.events = []
        self.awarded = set()
        self.acquired = False
        self.delivered = False
        self.left_bowl = False
        self.success = False
        self.failure_reason = None

    def emit(self, name, time, **detail):
        self.events.append(dict(name=name, time=float(time), **detail))

    def switch(self, phase, time):
        if phase != self.phase:
            self.emit("phase", time, previous=self.phase, phase=phase)
            self.phase = phase
            self.timers["ready"] = 0.

    def duration(self, name, condition, dt, limit):
        self.timers[name] = self.timers[name] + dt if condition else 0.
        return condition and self.timers[name] + 1e-12 >= limit

    def update(self, e, dt, time, failure=None, penetration=False):
        c = self.config
        pickup = self.duration("pickup", e["supported"] and e["off_bowl"] and e["pickup_eligible"], dt, c["support_confirm_s"])
        delivery = self.duration("delivery", self.acquired and e["mouth_supported"] and e["released"],
                                 dt, c["delivery_confirm_s"])
        retract = self.duration("retract", self.delivered and e["mouth_supported"] and e["released"]
                                and not e["tool_inside"] and not e["tool_mouth_contact"],
                                dt, c["retract_confirm_s"])
        self.left_bowl |= e["off_bowl"]
        unsupported = self.duration("unsupported", (self.acquired or self.left_bowl)
                                    and not e["supported"] and not e["mouth_supported"] and not e["on_bowl"],
                                    dt, c["unsupported_grace_s"])
        deep = self.duration("penetration", penetration or e["penetration"], dt, c["penetration_confirm_s"])
        ready = self.duration("ready", self.phase == "WAIT_READY" and e["ready"] and e["at_wait"],
                              dt, c["ready_confirm_s"])
        reasons = []
        if failure:
            reasons.append(failure)
        if deep or e["bean_penetration"]:
            reasons.append("model_penetration")
        if self.delivered and (unsupported or e["supported"] or e["on_bowl"] or e["food_ground_contact"]):
            reasons.append("food_lost_after_delivery")
        elif e["food_ground_contact"] or unsupported or (self.acquired and e["on_bowl"]):
            reasons.append("food_dropped")
        if self.phase == "TRANSFER" and not self.delivered and not e["tool_inside"] and not e["released"]:
            reasons.append("withdrawal_before_release")
        if self.phase == "SELECT" and not e["food_valid"]:
            reasons.append("food_missing")
        # Keep simultaneous evidence, but never award success on a failed boundary.
        candidates = [("pickup", pickup and not self.acquired and self.phase == "ACQUIRE"),
                      ("delivery", delivery and not self.delivered and self.phase == "TRANSFER"),
                      ("success", retract and not self.success and self.phase == "RETRACT")]
        for name, condition in candidates:
            if condition:
                self.emit(name + "_candidate", time)
        if reasons:
            for reason in dict.fromkeys(reasons):
                self.emit("failure", time, reason=reason)
            self.failure_reason = reasons[0]
            return
        if self.phase == "SELECT":
            self.switch("ACQUIRE", time)
        elif self.phase == "ACQUIRE" and pickup:
            self.acquired = True
            self.emit("pickup", time)
            self.switch("TRANSPORT", time)
        elif self.phase == "TRANSPORT" and e["at_wait"]:
            self.switch("WAIT_READY", time)
        elif self.phase == "WAIT_READY" and ready:
            self.switch("APPROACH", time)
        elif self.phase == "APPROACH":
            if not e["ready"]:
                self.switch("RECOVER", time)
            elif e["tool_inside"]:
                self.switch("TRANSFER", time)
        elif self.phase == "TRANSFER" and delivery:
            self.delivered = True
            self.emit("delivery", time)
            self.switch("RETRACT", time)
        elif self.phase == "RETRACT" and retract:
            self.success = True
            self.emit("success", time)
        elif self.phase == "RECOVER" and e["at_wait"] and not e["tool_inside"] and not e["tool_mouth_contact"]:
            self.switch("WAIT_READY", time)

    def distance(self, e):
        target = {"ACQUIRE": "bean_position", "TRANSPORT": "wait_position", "RECOVER": "wait_position",
                  "APPROACH": "mouth_position", "RETRACT": "wait_position"}.get(self.phase)
        return None if target is None else float(np.linalg.norm(e["tcp_position"] - e[target]))
