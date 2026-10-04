"""Compose a configured robot, rigid tool, fixed bowl and native Beans."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_json(path):
    return json.loads((ROOT / path).read_text())


def named_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing {kind.name}: {name}")
    return index


def asset_files(robot_id=None):
    """Runtime inputs, including mesh bytes; the inactive plate is excluded."""
    paths = set((ROOT / "configs").glob("*.json"))
    scene_cfg = load_json("configs/scene.json")
    tableware_paths = {ROOT / scene_cfg["bowl_asset"]}
    robots = [robot_id] if robot_id else ["panda", "ur5e"]
    for robot in robots:
        cfg = load_json(f"configs/robots/{robot}.json")
        tableware_paths.add(ROOT / cfg["tool_asset"])
        paths.add(ROOT / f"configs/robots/{robot}.json")
        path = ROOT / cfg["model"]
        paths.add(path)
        xml = ET.parse(path).getroot()
        compiler = xml.find("compiler")
        meshdir = path.parent / compiler.get("meshdir", ".")
        paths.update((meshdir / mesh.get("file")).resolve() for mesh in xml.findall("asset/mesh"))
    paths.add(ROOT / scene_cfg["model"])
    paths.add(ROOT / scene_cfg["beans"]["visual_mesh"])
    tableware = ROOT / "assets/task/tableware"
    paths.add(tableware / "contact_pairs.xml")
    for path in sorted(tableware_paths):
        paths.add(path)
        paths.update((path.parent / mesh.get("file")).resolve()
                     for mesh in ET.parse(path).findall("asset/mesh"))
    return sorted(paths)


def _merge_asset(root, path, namespace=None):
    source = ET.parse(path).getroot()
    compiler = source.find("compiler")
    meshdir = path.parent / (compiler.get("meshdir", ".") if compiler is not None else ".")
    for mesh in source.findall("asset/mesh"):
        mesh.set("file", str((meshdir / mesh.get("file")).resolve()))
    if namespace:
        # Scope standalone defaults instead of overwriting robot/task defaults.
        defaults = ET.SubElement(root.find("default"), "default", {"class": namespace})
        for child in source.find("default"):
            defaults.append(copy.deepcopy(child))
        source.find("worldbody/body").set("childclass", namespace)
        for material in source.findall("asset/material"):
            old = material.get("name")
            new = f"{namespace}_{old}"
            material.set("name", new)
            for geom in source.iter("geom"):
                if geom.get("material") == old:
                    geom.set("material", new)
    for child in source:
        if child.tag in ("compiler", "worldbody", "option", "keyframe"):
            continue
        if child.tag == "default":
            if not namespace:
                root.find("default").extend(copy.deepcopy(list(child)))
            continue
        section = root.find(child.tag)
        if section is None:
            section = ET.SubElement(root, child.tag)
        section.extend(copy.deepcopy(list(child)))
    return copy.deepcopy(source.find("worldbody"))


def _numbers(values):
    return " ".join(map(str, values))


def load_model(robot_id="panda", timestep=None):
    cfg = load_json(f"configs/robots/{robot_id}.json")
    scene_cfg = load_json("configs/scene.json")
    root = ET.parse(ROOT / scene_cfg["model"]).getroot()
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="auto")
    world = root.find("worldbody")
    base = ET.SubElement(world, "body", name="robot_base", pos=_numbers(cfg["base_position"]),
                         quat=_numbers(cfg["base_quaternion"]))
    ET.SubElement(base, "site", name="robot_base_frame", size="0.001")
    base.extend(_merge_asset(root, ROOT / cfg["model"]))
    mount = base.find(f".//body[@name='{cfg['mount_body']}']")
    if mount is None:
        raise ValueError(f"Missing tool mount: {cfg['mount_body']}")
    tool = ET.SubElement(mount, "body", name=cfg["tool_body"], pos=_numbers(cfg["tool_position"]),
                        quat=_numbers(cfg["tool_quaternion"]))
    ET.SubElement(tool, "site", name=cfg["ft_site"], size="0.002")
    spoon = _merge_asset(root, ROOT / cfg["tool_asset"], "new_spoon")[0]
    spoon.remove(spoon.find("freejoint"))
    spoon.set("pos", _numbers(cfg["spoon_position"]))
    tool.append(spoon)
    ET.SubElement(spoon, "site", name=cfg["tcp_site"], pos=_numbers(scene_cfg["spoon_tcp_m"]),
                  size="0.002", rgba="0 1 0 1")
    bowl = _merge_asset(root, ROOT / scene_cfg["bowl_asset"], "new_bowl")[0]
    bowl.remove(bowl.find("freejoint"))
    internal = bowl.find("body")
    disk = internal.find("geom[@name='collision_bowl_fast_bottom_disk']")
    surface = np.fromstring(disk.get("pos"), sep=" ")
    surface[2] += float(disk.get("size").split()[1])
    # Cancel the source internal rotation without changing its geometry.
    quat = np.fromstring(internal.get("quat"), sep=" ")
    quat /= np.linalg.norm(quat)
    quat[1:] *= -1
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    origin = np.array(scene_cfg["bowl_frame_position_m"])
    position = origin - rotation.reshape(3, 3) @ np.fromstring(internal.get("pos"), sep=" ") - surface
    bowl.set("pos", _numbers(position))
    bowl.set("quat", _numbers(quat))
    ET.SubElement(internal, "site", name="bowl_frame", pos=_numbers(surface), size="0.002")
    world.append(bowl)
    beans = scene_cfg["beans"]
    ET.SubElement(root.find("asset"), "mesh", name="bean_visual_mesh",
                  file=str((ROOT / beans["visual_mesh"]).resolve()))
    a, b, c = beans["semi_axes_m"]
    mass = beans["density_kg_m3"] * 4 * np.pi * a * b * c / 3
    inertia = mass / 5 * np.array([b*b + c*c, a*a + c*c, a*a + b*b])
    # Episode placement and natural settling belong to reset.
    for i in range(beans["count"]):
        layer = int(i >= 8)
        angle = 2 * np.pi * (i if layer == 0 else i - 8) / (8 if layer == 0 else 7)
        radius = beans["compile_ring_radius_m"]
        position = origin + [radius * np.cos(angle), radius * np.sin(angle),
                             beans["compile_layer_heights_m"][layer]]
        if beans["count"] == 1:
            position = origin + beans["reset_position_bowl_m"]
        name = f"bean_{i:03d}"
        body = ET.SubElement(world, "body", name=name, pos=_numbers(position))
        ET.SubElement(body, "freejoint", name=f"{name}_joint")
        ET.SubElement(body, "inertial", pos="0 0 0", quat="1 0 0 0",
                      mass=str(mass), diaginertia=_numbers(inertia))
        ET.SubElement(body, "geom", name=f"{name}_visual", type="mesh", mesh="bean_visual_mesh",
                      pos="0 0 0", quat="1 0 0 0", group="1", contype="0", conaffinity="0",
                      density="0", rgba="0.45 0.18 0.07 1")
        collision = dict(name=f"{name}_collision", type="ellipsoid", size=_numbers([a, b, c]),
                         pos="0 0 0", quat="1 0 0 0", group="3", contype="2", conaffinity="3",
                         density="0", rgba="0 0.55 1 0.25")
        collision.update({key: _numbers(beans[key]) for key in ("friction", "solref", "solimp")})
        collision.update({key: str(beans[key]) for key in ("condim", "priority", "margin", "gap")})
        ET.SubElement(body, "geom", collision)
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for kind, name in zip(("force", "torque"), cfg["ft_sensors"]):
        ET.SubElement(sensor, kind, name=name, site=cfg["ft_site"])
    for body in base.iter("body"):
        if body.get("name") in cfg["arm_bodies"]:
            body.set("gravcomp", str(cfg["gravcomp"]))
    contact = root.find("contact")
    names = {g.get("name") for g in root.iter("geom")}
    for pair in ET.parse(ROOT / "assets/task/tableware/contact_pairs.xml").findall("contact/pair"):
        if pair.get("geom1") in names and pair.get("geom2") in names:
            contact.append(copy.deepcopy(pair))
    if len(contact.findall("pair")) != 145:
        raise ValueError("Expected all 145 source spoon/bowl pairs")
    ET.SubElement(contact, "exclude", body1=cfg["mount_body"], body2="dynamic_spoon2")
    root.find("option").set("timestep", str(timestep if timestep is not None else scene_cfg["timestep"]))
    root.find("option").set("iterations", str(scene_cfg["solver_iterations"]))
    root.find("option").set("ccd_tolerance", str(scene_cfg["ccd_tolerance"]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    index = RobotIndex(model, cfg)
    scale = cfg.get("servo_gain_scale", 1.)
    model.actuator_gainprm[index.actuators, 0] *= scale
    model.actuator_biasprm[index.actuators, 1] *= scale
    model.actuator_biasprm[index.actuators, 2] *= np.sqrt(scale)
    return model, index, cfg, scene_cfg


class RobotIndex:
    def __init__(self, model, config):
        self.config = config
        jid = lambda n: named_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: named_id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        sid = lambda n: named_id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        self.joints = np.array([jid(n) for n in config["joints"]])
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in self.joints):
            raise ValueError("Arm joints must be scalar revolute joints")
        self.qpos = model.jnt_qposadr[self.joints].copy()
        self.dofs = model.jnt_dofadr[self.joints].copy()
        self.actuators = np.array([named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in config["actuators"]])
        if len(self.joints) != len(self.actuators) or len(set(self.actuators)) != len(self.joints):
            raise ValueError("Expected one unique position actuator per arm joint")
        for joint, act in zip(self.joints, self.actuators):
            if (model.actuator_trntype[act] != mujoco.mjtTrn.mjTRN_JOINT
                    or model.actuator_trnid[act, 0] != joint
                    or model.actuator_gear[act, 0] != 1
                    or model.actuator_dyntype[act] != mujoco.mjtDyn.mjDYN_NONE
                    or model.actuator_biastype[act] != mujoco.mjtBias.mjBIAS_AFFINE
                    or not np.isclose(model.actuator_gainprm[act, 0], -model.actuator_biasprm[act, 1])
                    or model.actuator_gainprm[act, 0] <= 0
                    or model.actuator_biasprm[act, 2] >= 0
                    or not model.actuator_forcelimited[act]):
                raise ValueError("Arm actuator must be a unit-gear, force-limited position PD servo")
        self.tcp = sid(config["tcp_site"])
        self.ft = sid(config["ft_site"])
        self.base = sid("robot_base_frame")
        self.tool_body = bid(config["tool_body"])
        gid = lambda n: named_id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        bean_count = int(np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self.bean_ids = tuple(f"bean_{i:03d}" for i in range(bean_count))
        self.bean_bodies = np.array([bid(n) for n in self.bean_ids])
        self.bean_joints = np.array([jid(f"{n}_joint") for n in self.bean_ids])
        self.bean_qpos = model.jnt_qposadr[self.bean_joints, None] + np.arange(7)
        self.bean_dofs = model.jnt_dofadr[self.bean_joints, None] + np.arange(6)
        self.bean_visual_geoms = np.array([gid(f"{n}_visual") for n in self.bean_ids])
        self.bean_collision_geoms = np.array([gid(f"{n}_collision") for n in self.bean_ids])
        self.bean_id_by_geom = dict(zip(self.bean_collision_geoms, self.bean_ids))
        self.sensors = [named_id(model, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in config["ft_sensors"]]
        if any(model.sensor_dim[s] != 3 for s in self.sensors):
            raise ValueError("Wrist F/T sensors must each have dimension 3")
        self.arm_bodies = {bid(n) for n in config["arm_bodies"]}
        self.arm_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in self.arm_bodies
                          and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.tool_bodies = {self.tool_body}
        for b in range(self.tool_body + 1, model.nbody):
            if model.body_parentid[b] in self.tool_bodies:
                self.tool_bodies.add(b)
        physical = lambda g: bool(model.geom_contype[g] or model.geom_conaffinity[g])
        self.spoon_geoms = [g for g in range(model.ngeom)
                            if model.geom_bodyid[g] in self.tool_bodies and physical(g)]
        scoop = bid("spoon_scoop_part")
        self.scoop_geoms = [g for g in self.spoon_geoms if model.geom_bodyid[g] == scoop]
        self.handle_geoms = [g for g in self.spoon_geoms if g not in self.scoop_geoms]
        self.bowl_geoms = [g for g in range(model.ngeom)
                           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("collision_bowl_")]
        self.tool_mass = float(model.body_mass[list(self.tool_bodies)].sum())
        self.head_joints = np.array([jid(n) for n in ["head_x", "head_y", "head_z", "head_yaw", "jaw"]])
        self.head_actuators = np.array([named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                                       for n in ["head_fx", "head_fy", "head_fz", "head_tau_yaw", "jaw_tau"]])
        self.n = len(self.joints)

    def group(self, model, geom):
        if geom in self.arm_geoms:
            return "arm"
        if geom in self.spoon_geoms:
            return "spoon"
        if geom in self.bowl_geoms:
            return "bowl"
        if geom in self.bean_id_by_geom:
            return "food"
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        for group, prefixes in [("mouth", ("mouth", "jaw")),
                                ("table", ("table",)), ("floor", ("floor",))]:
            if name.startswith(prefixes):
                return group
        return "other"
