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


def evidence(task):
    """Read current geometry/contact truth; never change simulation state."""
    m, d, idx, cfg = task.model, task.data, task.index, task.task_config
    site = lambda name: named_id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    geom = lambda name: named_id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    mouth, receiver, plate = (site(n) for n in ("mouth_entry", "mouth_receiver", "plate_frame"))
    mr = d.site_xmat[mouth].reshape(3, 3)
    jr = d.site_xmat[receiver].reshape(3, 3)
    tr = d.site_xmat[idx.tcp].reshape(3, 3)
    tcp = d.site_xpos[idx.tcp]
    food = d.xpos[idx.food_body]
    food_corners = geom_corners(m, d, geom("food_box"))
    fc = (food_corners - d.site_xpos[mouth]) @ mr
    fj = (food_corners - d.site_xpos[receiver]) @ jr
    rel = tr.T @ (food - tcp)
    pairs = {frozenset((r["group1"], r["group2"])) for r in task.contacts
             if r["force_n"] > cfg["contact_min_force_n"]}
    contact = lambda a, b: frozenset((a, b)) in pairs
    scoop = np.concatenate([geom_corners(m, d, g) for g in idx.scoop_geoms])
    scoop_local = (scoop - tcp) @ tr
    support_force = sum(float(np.dot(r["force_on_geom2_world"], tr[:, 2]))
                        * (1 if r["group2"] == "food" else -1)
                        for r in task.contacts
                        if ((r["geom1"] in idx.scoop_geoms and r["group2"] == "food")
                            or (r["geom2"] in idx.scoop_geoms and r["group1"] == "food")))
    # A loaded side/handle or the empty corners of the rounded mesh AABB are
    # not a carrying surface. Ray-test the actual 130 convex scoop meshes.
    over_scoop = (np.all(rel[:2] >= scoop_local[:, :2].min(axis=0))
                  and np.all(rel[:2] <= scoop_local[:, :2].max(axis=0)))
    supported = (support_force > cfg["contact_min_force_n"] and over_scoop
                 and any(mujoco.mj_rayMesh(m, d, g, food, -tr[:, 2]) >= 0
                         for g in idx.scoop_geoms))
    on_plate = contact("food", "plate")
    plate_corners = (food_corners - d.site_xpos[plate]) @ d.site_xmat[plate].reshape(3, 3)
    plate_height = float(plate_corners[:, 2].min())
    off_plate = not on_plate and plate_height > cfg["plate_clearance_m"]
    tolerance = cfg["receiver_xy_tolerance_m"]
    in_receiver = (np.all(fc[:, :2] >= np.asarray(cfg["receiver_min_xy_m"]) - tolerance)
                   and np.all(fc[:, :2] <= np.asarray(cfg["receiver_max_xy_m"]) + tolerance)
                   and fc[:, 2].max() <= cfg["receiver_top_m"]
                   and fj[:, 2].min() >= -cfg["receiver_floor_tolerance_m"])
    mouth_supported = bool(in_receiver and contact("food", "mouth"))
    inside = False
    for g in idx.spoon_geoms:
        corners = (geom_corners(m, d, g) - d.site_xpos[mouth]) @ mr
        inside |= bool(np.all(corners.max(axis=0) >= cfg["interaction_min_m"])
                       and np.all(corners.min(axis=0) <= cfg["interaction_max_m"]))
    # Current jaw plane at the entry and back of the receiver, in mouth coordinates.
    floor = geom_corners(m, d, geom("jaw_floor"))
    upper = geom_corners(m, d, geom("mouth_upper"))
    aperture = float(((upper - d.site_xpos[mouth]) @ mr)[:, 2].min()
                     - ((floor - d.site_xpos[mouth]) @ mr)[:, 2].max())
    width = cfg["receiver_max_xy_m"][1] - cfg["receiver_min_xy_m"][1]
    carried = np.concatenate([scoop, food_corners])
    # Height required by the present spoon/food arrangement, not a future jaw target.
    projected = carried @ mr
    required_height = float(np.ptp(projected[:, 2])) + cfg["clearance_margin_m"]
    required_width = float(np.ptp(projected[:, 1])) + cfg["clearance_margin_m"]
    wait = d.site_xpos[mouth] - mr[:, 0] * cfg["wait_offset_m"]
    aligned = rotation_error(tr, mr) <= cfg["orientation_tolerance_rad"]
    at_wait = np.linalg.norm(tcp - wait) <= cfg["position_tolerance_m"] and aligned
    return dict(supported=bool(supported), off_plate=bool(off_plate), on_plate=on_plate,
                plate_height_m=plate_height, spoon_support_force_n=support_force,
                required_height_m=required_height, required_width_m=required_width,
                mouth_supported=mouth_supported, released=not contact("food", "spoon"),
                tool_inside=inside, tool_mouth_contact=contact("spoon", "mouth"),
                at_wait=bool(at_wait), aligned=bool(aligned), aperture_m=aperture,
                ready=bool(aperture >= required_height and width >= required_width and aligned),
                food_ground_contact=contact("food", "table") or contact("food", "floor"),
                food_valid=bool(np.linalg.norm(food - d.site_xpos[plate]) < 1.),
                penetration=any(r["distance"] < -cfg["penetration_limit_m"] for r in task.contacts),
                mouth_position=d.site_xpos[mouth].copy(), mouth_rotation=mr.copy(),
                wait_position=wait, food_position=food.copy(), tcp_position=tcp.copy())


class TaskEvents:
    def __init__(self, config):
        self.config = config
        self.phase = "SELECT"
        self.timers = dict(pickup=0., delivery=0., retract=0., ready=0., unsupported=0., penetration=0.)
        self.events = []
        self.awarded = set()
        self.acquired = False
        self.delivered = False
        self.left_plate = False
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
        pickup = self.duration("pickup", e["supported"] and e["off_plate"], dt, c["support_confirm_s"])
        delivery = self.duration("delivery", self.acquired and e["mouth_supported"] and e["released"],
                                 dt, c["delivery_confirm_s"])
        retract = self.duration("retract", self.delivered and e["mouth_supported"] and e["released"]
                                and not e["tool_inside"] and not e["tool_mouth_contact"],
                                dt, c["retract_confirm_s"])
        self.left_plate |= e["off_plate"]
        unsupported = self.duration("unsupported", (self.acquired or self.left_plate)
                                    and not e["supported"] and not e["mouth_supported"] and not e["on_plate"],
                                    dt, c["unsupported_grace_s"])
        deep = self.duration("penetration", penetration or e["penetration"], dt, c["penetration_confirm_s"])
        ready = self.duration("ready", self.phase == "WAIT_READY" and e["ready"] and e["at_wait"],
                              dt, c["ready_confirm_s"])
        reasons = []
        if failure:
            reasons.append(failure)
        if deep:
            reasons.append("model_penetration")
        if self.delivered and (unsupported or e["supported"] or e["on_plate"] or e["food_ground_contact"]):
            reasons.append("food_lost_after_delivery")
        elif e["food_ground_contact"] or unsupported or (self.acquired and e["on_plate"]):
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
        target = {"ACQUIRE": "food_position", "TRANSPORT": "wait_position", "RECOVER": "wait_position",
                  "APPROACH": "mouth_position", "RETRACT": "wait_position"}.get(self.phase)
        return None if target is None else float(np.linalg.norm(e["tcp_position"] - e[target]))
