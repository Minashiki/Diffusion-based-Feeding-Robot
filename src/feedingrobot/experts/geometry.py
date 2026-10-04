"""Static copies of the active collision geometry, measured after reset."""

import copy
import hashlib
import json

import mujoco
import numpy as np

from feedingrobot.sim.events import geom_corners
from feedingrobot.sim.model import named_id


def teacher_geometry(task):
    if task.index.bean_ids:
        raise NotImplementedError("Single-bean M4 teacher migration is pending; use the M3 proof driver for task validation")
    model, data, index = task.model, task.data, task.index
    plate = named_id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
    tcp = data.site_xpos[index.tcp]
    rotation = data.site_xmat[index.tcp].reshape(3, 3)
    plate_position = data.site_xpos[plate].copy()
    plate_rotation = data.site_xmat[plate].reshape(3, 3).copy()
    points = lambda geoms: np.unique(np.round(np.concatenate([
        (geom_corners(model, data, geom) - tcp) @ rotation for geom in geoms]), 10), axis=0)
    scoop, tool = points(index.scoop_geoms), points(index.spoon_geoms)
    front = scoop[scoop[:, 0] >= scoop[:, 0].max() - .001]
    plate_points = np.concatenate([geom_corners(model, data, geom) for geom in index.plate_geoms])
    geometry = dict(schema_version=1, plate_position=plate_position, plate_rotation=plate_rotation,
                    plate_points=(plate_points - plate_position) @ plate_rotation,
                    scoop_points=scoop, tool_points=tool, front_lip=front,
                    food_half_size=model.geom_size[task.food_geom].copy(),
                    joint_ranges=model.jnt_range[index.joints].copy(),
                    task_config=copy.deepcopy(task.task_config))
    # Pure values: no simulator references, material parameters or driver plans.
    encoded = json.dumps(geometry, sort_keys=True, default=lambda value: value.tolist()).encode()
    geometry["sha256"] = hashlib.sha256(encoded).hexdigest()
    for value in geometry.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return geometry
