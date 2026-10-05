"""Static copies of the active collision geometry, measured after reset."""

import copy
import hashlib
import json

import mujoco
import numpy as np

from feedingrobot.sim.events import geom_corners
from feedingrobot.sim.model import named_id


def teacher_geometry(task):
    model, data, index = task.model, task.data, task.index
    bowl = named_id(model, mujoco.mjtObj.mjOBJ_SITE, "bowl_frame")
    tcp = data.site_xpos[index.tcp]
    rotation = data.site_xmat[index.tcp].reshape(3, 3)
    bowl_position = data.site_xpos[bowl].copy()
    bowl_rotation = data.site_xmat[bowl].reshape(3, 3).copy()
    points = lambda geoms: np.unique(np.round(np.concatenate([
        (geom_corners(model, data, geom) - tcp) @ rotation for geom in geoms]), 10), axis=0)
    scoop, tool = points(index.scoop_geoms), points(index.spoon_geoms)
    front = scoop[scoop[:, 0] >= scoop[:, 0].max() - .001]
    bowl_points = np.concatenate([geom_corners(model, data, geom) for geom in index.bowl_geoms])
    walls = [geom for geom in index.bowl_geoms
             if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX]
    geometry = dict(schema_version=2, model_version="single_bean_native_v1",
                    wall_positions=data.geom_xpos[walls].copy(),
                    wall_normals=data.geom_xmat[walls].reshape(-1, 3, 3)[:, :, 0].copy(),
                    wall_half_sizes=model.geom_size[walls, 0].copy(),
                    rim_z=max(geom_corners(model, data, geom)[:, 2].max() for geom in walls),
                    pickup_linear_speed_m_s=task.bean_acceptance["linear_speed_m_s"],
                    acquisition_config=copy.deepcopy(task.bean_acceptance["m1c"]), bowl_position=bowl_position, bowl_rotation=bowl_rotation,
                    bowl_points=(bowl_points - bowl_position) @ bowl_rotation,
                    scoop_points=scoop, tool_points=tool, front_lip=front,
                    bean_half_size=model.geom_size[index.bean_collision_geoms[0]].copy(),
                    joint_ranges=model.jnt_range[index.joints].copy(),
                    task_config=copy.deepcopy(task.task_config))
    # Pure values: no simulator references, material parameters or driver plans.
    encoded = json.dumps(geometry, sort_keys=True, default=lambda value: value.tolist()).encode()
    geometry["sha256"] = hashlib.sha256(encoded).hexdigest()
    for value in geometry.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return geometry
