"""Geometry unit fixtures; physical receipt is checked by episode validation."""

import mujoco
import numpy as np
import pytest

from feedingrobot.sim import events
from feedingrobot.sim.events import evidence, geom_corners
from feedingrobot.sim.model import named_id
from feedingrobot.sim.task import FeedingTask


@pytest.fixture
def receiver():
    task = FeedingTask(task_mode=True)
    task.reset()
    for name in ("mouth_entry", "mouth_receiver"):
        site = named_id(task.model, mujoco.mjtObj.mjOBJ_SITE, name)
        task.data.site_xpos[site] = 0.
        task.data.site_xmat[site] = np.eye(3).ravel()
    task.contacts = [dict(geom1=task.food_geom, geom2=named_id(task.model, mujoco.mjtObj.mjOBJ_GEOM, "jaw_floor"),
                          group1="food", group2="mouth", force_n=.01, distance=0.)]
    return task


def place_food(task, center, angle=0.):
    address = task.index.food_qpos
    task.data.qpos[address:address + 3] = center
    task.data.xpos[task.index.food_body] = center
    task.data.geom_xpos[task.food_geom] = center
    c, s = np.cos(angle), np.sin(angle)
    task.data.geom_xmat[task.food_geom] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]).ravel()


@pytest.mark.parametrize("axis,side", [(0, -1), (0, 1), (1, -1), (1, 1)])
@pytest.mark.parametrize("angle", [0., .3])
@pytest.mark.parametrize("excess,accepted", [(-.001, True), (.0003, True), (.0005, True), (.000501, False)])
def test_receiver_xy_boundary(receiver, monkeypatch, axis, side, angle, excess, accepted):
    cfg = receiver.task_config
    place_food(receiver, np.zeros(3), angle)
    corners = geom_corners(receiver.model, receiver.data, receiver.food_geom)
    bounds = cfg["receiver_min_xy_m"] if side < 0 else cfg["receiver_max_xy_m"]
    extreme = corners[:, axis].min() if side < 0 else corners[:, axis].max()
    center = np.array([.02, 0., .005])
    center[axis] = bounds[axis] + side * excess - extreme
    place_food(receiver, center, angle)
    if excess == cfg["receiver_xy_tolerance_m"]:
        # Supply the exact floating-point boundary, avoiding cancellation in placement.
        corners = geom_corners(receiver.model, receiver.data, receiver.food_geom)
        edge = corners[:, axis].min() if side < 0 else corners[:, axis].max()
        corners[corners[:, axis] == edge, axis] = bounds[axis] + side * excess
        monkeypatch.setattr(events, "geom_corners", lambda m, d, g:
                            corners if g == receiver.food_geom else geom_corners(m, d, g))
    assert evidence(receiver)["mouth_supported"] == accepted


@pytest.mark.parametrize("force", [0., 1e-6, 1e-5])
def test_receiver_requires_contact_above_original_threshold(receiver, force):
    place_food(receiver, [.02, 0., .005])
    receiver.contacts[0]["force_n"] = force
    assert not evidence(receiver)["mouth_supported"]
    receiver.contacts = []
    assert not evidence(receiver)["mouth_supported"]


@pytest.mark.parametrize("height", [.022, -.003])
def test_receiver_xy_tolerance_does_not_relax_top_or_floor(receiver, height):
    place_food(receiver, [.02, 0., height])
    assert not evidence(receiver)["mouth_supported"]


def test_receiver_tolerance_does_not_expand_readiness_width(receiver, monkeypatch):
    place_food(receiver, [.02, 0., .005])
    receiver.data.site_xmat[receiver.index.tcp] = np.eye(3).ravel()
    wide_food = geom_corners(receiver.model, receiver.data, receiver.food_geom)
    wide_food[:, 1] = np.sign(wide_food[:, 1]) * .03
    scoop_ids = set(receiver.index.scoop_geoms)
    monkeypatch.setattr(events, "geom_corners", lambda m, d, g:
                        wide_food if g == receiver.food_geom or g in scoop_ids else geom_corners(m, d, g))
    before = evidence(receiver)
    assert before["aligned"] and before["aperture_m"] > .014
    assert not before["ready"]
    receiver.task_config["receiver_xy_tolerance_m"] = .01
    after = evidence(receiver)
    assert after["ready"] == before["ready"]
    assert after["at_wait"] == before["at_wait"]
