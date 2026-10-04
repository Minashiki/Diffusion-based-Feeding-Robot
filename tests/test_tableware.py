"""New tableware contracts and real contact evidence."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from feedingrobot.data.episodes import input_hashes
from feedingrobot.sim.beans import bean_diagnostics
from feedingrobot.sim.model import ROOT, load_model
from feedingrobot.sim.task import FeedingTask


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_source_collision_parameters_and_pair_membership(robot, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model, index, _, _ = load_model(robot)
    assert not any("plate" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                   for g in range(model.ngeom))
    for name in ("dynamic_spoon2_freejoint", "dynamic_bowl2_freejoint"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1
    for kind in ("spoon", "bowl"):
        source = ET.parse(ROOT / f"assets/task/tableware/{kind}/{kind}.xml")
        for geom in source.findall("worldbody//geom"):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom.get("name"))
            assert g >= 0
            if geom.get("group") == "1":
                assert not model.geom_contype[g] and not model.geom_conaffinity[g]
            else:
                assert index.group(model, g) == kind
                for key in ("contype", "conaffinity", "condim"):
                    assert getattr(model, f"geom_{key}")[g] == int(geom.get(key))
                np.testing.assert_allclose(model.geom_friction[g], np.fromstring(geom.get("friction"), sep=" "))
    assert set(model.pair_geom1).issubset(index.spoon_geoms)
    assert set(model.pair_geom2).issubset(index.bowl_geoms)


def test_episode_hashes_cover_new_meshes_and_exclude_inactive_plate():
    hashes = input_hashes()
    assert "assets/task/tableware/spoon/meshes/spoon_visual.obj" in hashes
    assert "assets/task/tableware/bowl/bowl.xml" in hashes
    assert "assets/task/tableware/bowl/meshes/bowl2_exact.obj" in hashes
    assert "assets/task/foods/beans/meshes/bean_visual.obj" in hashes
    assert "assets/task/tableware/contact_pairs.xml" in hashes
    assert not any("/plate/" in path or "extract/" in path for path in hashes)


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
@pytest.mark.parametrize("preset", ["beans_in_bowl", "beans_on_spoon"])
def test_reset_footprint_and_physical_support(robot, preset):
    task = FeedingTask(robot)
    task.reset(seed=2, preset=preset)
    state = task.snapshot()
    assert np.isfinite(state["bean_positions"]).all()
    assert task.reset_diagnostics["spawn_clearance_m"] >= .0005 - 1e-9
    assert task.reset_diagnostics["status"] in {"passed", "failed"}
    diag = bean_diagnostics(task)
    if preset == 'beans_in_bowl':
        assert diag['bowl_supported'].all()
    else:
        assert not diag['bowl_supported'].any()
    if preset == "beans_on_spoon":
        assert diag["spoon_supported"][0]
    if task.reset_diagnostics["status"] == "failed":
        assert task.terminated and task.failure_reason == "bean_reset_settling_failed"


def test_handle_contact_is_not_food_support():
    task = FeedingTask()
    task.reset(preset="empty")
    rotation = task.data.site_xmat[task.index.tcp].reshape(3, 3)
    # Keep the handle fixture clear of the wrist after moving the grip 10 mm.
    point = task.data.site_xpos[task.index.tcp] + rotation @ [-.035, 0., .05]
    hits = [mujoco.mj_rayMesh(task.model, task.data, g, point, -rotation[:, 2])
            for g in task.index.handle_geoms if task.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    height = .05 - min(hit for hit in hits if hit >= 0)
    address = task.index.bean_qpos[0, 0]
    task.data.qpos[address:address + 3] = task.data.site_xpos[task.index.tcp] + rotation @ [-.035, 0., height + .0045]
    mujoco.mju_mat2Quat(task.data.qpos[address + 3:address + 7], rotation.ravel())
    mujoco.mj_forward(task.model, task.data)
    # The ray hits the source mesh; the native convex hull can extend above it.
    for _ in range(100):
        distance = min(mujoco.mj_geomDistance(task.model, task.data, int(task.index.bean_collision_geoms[0]),
                                             g, .01, None) for g in task.index.handle_geoms)
        if distance >= .0005 - 1e-9:
            break
        task.data.qpos[address:address + 3] += (.0005 - distance + 1e-8) * rotation[:, 2]
        mujoco.mj_forward(task.model, task.data)
    assert distance >= .0005 - 1e-9
    assert not any(int(task.index.bean_collision_geoms[0]) in (int(c.geom1), int(c.geom2))
                   and bool({int(c.geom1), int(c.geom2)} & set(task.index.arm_geoms)) for c in task.data.contact)
    touched_handle = False
    handle_geoms = set(task.index.handle_geoms)
    for _ in range(100):
        task.step_physics()
        touched_handle |= any({row["group1"], row["group2"]} == {"food", "spoon"}
                              and bool({row["geom1"], row["geom2"]} & handle_geoms)
                              and row["force_n"] > 0
                              for row in task.applied_contacts + task.contacts)
        assert not bean_diagnostics(task)["spoon_supported"][0]
        if touched_handle:
            break
    assert touched_handle


def test_wrench_compensation_includes_mass_on_fixed_child():
    from feedingrobot.sim.model import named_id
    task = FeedingTask()
    body = named_id(task.model, mujoco.mjtObj.mjOBJ_BODY, "spoon_scoop_part")
    task.model.body_mass[body] = .005
    task.model.body_inertia[body] = [1e-6, 1e-6, 1e-6]
    task.reset(preset="empty")
    now = task.data.time
    task.adapter.set_twist([.01, 0., 0., .03, -.02, 0.], now, now + .5)
    for _ in range(400):
        state = task.step_physics()
        assert np.max(np.abs(state["compensated_wrench"][:3])) < .02
        assert np.max(np.abs(state["compensated_wrench"][3:])) < .002


def test_snapshot_rejects_changed_installation():
    task = FeedingTask()
    state = task.get_state()
    task.robot_config["spoon_position"][0] += .001
    with pytest.raises(ValueError, match="Incompatible"):
        task.set_state(state)
