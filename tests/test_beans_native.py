"""M1-A resource/compile evidence, independent of the pending task migration."""

from collections import Counter
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from feedingrobot.sim.model import ROOT, asset_files, load_model
from feedingrobot.scripts.validate_m1a import beans_check, masks_check, tableware_check


@pytest.fixture(params=["panda", "ur5e"])
def compiled(request):
    model, index, cfg, scene = load_model(request.param)
    data = mujoco.MjData(model)
    data.qpos[index.qpos] = cfg["reset_q"]
    data.ctrl[index.actuators] = cfg["reset_q"]
    mujoco.mj_forward(model, data)
    return model, data, index, cfg, scene


def test_independent_free_bodies_and_mass(compiled):
    model, data, index, cfg, scene = compiled
    assert (model.nq, model.nv) == {"panda": (19, 18), "ur5e": (18, 17)}[cfg["robot_id"]]
    assert index.bean_qpos.shape == (1, 7) and index.bean_dofs.shape == (1, 6)
    assert not hasattr(index, "food_body") and not hasattr(index, "food_qpos")
    assert not hasattr(index, "plate_geoms")
    np.testing.assert_allclose(model.body_mass[index.bean_bodies], .0005277875658030852, rtol=1e-12)
    np.testing.assert_array_equal(data.qpos[index.bean_qpos[:, 3:]], np.tile([1, 0, 0, 0], (1, 1)))
    before = data.qpos.copy()
    data.qpos[index.bean_qpos[0, 0]] += .01
    changed = np.flatnonzero(before != data.qpos)
    np.testing.assert_array_equal(changed, [index.bean_qpos[0, 0]])
    mujoco.mj_forward(model, data)
    assert not set(index.bean_bodies).intersection(index.tool_bodies)
    beans_check(model, data, index, scene)


def test_visual_alignment_after_independent_rotations(compiled):
    model, data, index, _, scene = compiled
    rng = np.random.default_rng(12)
    positions = rng.uniform(-.5, .5, (1, 3))
    quats = rng.normal(size=(1, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    data.qpos[index.bean_qpos] = np.c_[positions, quats]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[index.bean_bodies], positions, atol=1e-12)
    beans_check(model, data, index, scene)
    # Detect an actual visual/collision offset, rather than comparing body IDs alone.
    data.geom_xpos[index.bean_visual_geoms[0], 0] += .001
    with pytest.raises(AssertionError):
        beans_check(model, data, index, scene)


def test_fixed_bowl_source_pairs_and_masks(compiled):
    model, data, index, _, scene = compiled
    tableware_check(model, data, index, scene)
    masks_check(model, index)
    source = ET.parse(ROOT / scene["bowl_asset"]).find("worldbody/body/body")
    internal = model.body("dynamic_bowl2_usd_internal").id
    np.testing.assert_allclose(model.body_pos[internal], np.fromstring(source.get("pos"), sep=" "), atol=1e-12)
    q = np.fromstring(source.get("quat"), sep=" ")
    np.testing.assert_allclose(model.body_quat[internal], q / np.linalg.norm(q), atol=1e-12)
    assert data.ncon == 0
    masks = model.geom_conaffinity[index.bowl_geoms].copy()
    model.geom_contype[index.bowl_geoms] = 0
    model.geom_conaffinity[index.bowl_geoms] = 0
    with pytest.raises(AssertionError, match="bowl"):
        masks_check(model, index)
    assert np.all(masks == 7)


def test_compile_layout_and_timestep_override(compiled):
    model, data, index, cfg, scene = compiled
    centre = np.array([.45, -.18, -.0156])
    local = data.xpos[index.bean_bodies] - centre
    np.testing.assert_allclose(local[0], scene['beans']['reset_position_bowl_m'], atol=1e-12)
    assert index.bean_ids == ('bean_000',)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bean_001') == -1
    assert local[:, 2].min() - .007 >= .0005 - 1e-12
    override, _, _, _ = load_model(cfg["robot_id"], timestep=.0005)
    assert override.opt.timestep == .0005
    assert model.opt.iterations == 100
    assert model.opt.ccd_tolerance == scene["ccd_tolerance"]
    assert override.opt.ccd_tolerance == scene["ccd_tolerance"]
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON


def test_obj_closed_outward_and_smooth():
    lines = (ROOT / "assets/task/foods/beans/meshes/bean_visual.obj").read_text().splitlines()
    vertices = np.array([[float(v) for v in line.split()[1:]] for line in lines if line.startswith("v ")])
    normals = np.array([[float(v) for v in line.split()[1:]] for line in lines if line.startswith("vn ")])
    face_tokens = [line.split()[1:] for line in lines if line.startswith("f ")]
    faces = np.array([[int(v.split("//")[0]) - 1 for v in face] for face in face_tokens])
    assert vertices.shape == normals.shape == (482, 3) and faces.shape == (960, 3)
    assert all(int(v.split("//")[0]) == int(v.split("//")[1]) for face in face_tokens for v in face)
    axes = np.array([.007, .0045, .004])
    np.testing.assert_allclose(np.ptp(vertices, axis=0), 2 * axes, atol=1e-12)
    np.testing.assert_allclose(vertices.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.sum((vertices / axes)**2, axis=1), 1, atol=1e-10)
    expected = vertices / axes**2
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(normals, expected, atol=1e-10)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.einsum("ij,ij->i", face_normals, triangles.mean(axis=1)) > 0)
    edges = Counter((int(face[j]), int(face[(j + 1) % 3])) for face in faces for j in range(3))
    assert all(n == 1 and edges[(b, a)] == 1 for (a, b), n in edges.items())


def test_manifest_tracks_generated_mesh_and_preserves_source_assets():
    manifest = json.loads((ROOT / "third_party_manifest.json").read_text())
    source = next(s for s in manifest["sources"] if s.get("repository") == "https://github.com/EBiM-Benchmark/benchmark")
    assert source["active_assets"] == ["spoon", "bowl"] and source["inactive_assets"] == ["plate"]
    path = "assets/task/foods/beans/meshes/bean_visual.obj"
    generated = next(f for f in manifest["derived_files"] if f["path"] == path)
    assert generated["origin"] == "project_generated"
    assert not any(f["path"] == path for f in source["files"])
    for record in source["files"] + [generated]:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    for record in manifest["derived_files"]:
        if record["path"] in ("configs/scene.json", "assets/task/scene.xml"):
            assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_standalone_acceptance_from_other_cwd(robot, tmp_path):
    output = tmp_path / robot
    result = subprocess.run([sys.executable, "-m", "feedingrobot.scripts.validate_m1a", "--robot", robot,
                             "--output", str(output)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "m1a_report.json").read_text())
    assert report["stage"] == "M1-A" and report["status"] == "passed"
    assert report["m1_status"] == "incomplete"
    assert all(report["stages"][stage] == "not_verified" for stage in ("M1-B", "M1-C", "M1-D"))
    assert report["cases"]["smoke"]["metrics"]["steps"] == 100
    assert report["cases"]["smoke"]["metrics"]["warning_counts"] == [0] * int(mujoco.mjtWarning.mjNWARNING)
    assert np.asarray(report["cases"]["smoke"]["metrics"]["bean_positions_trace_m"]).shape == (101, 1, 3)
    assert all(case["status"] == "passed" for case in report["cases"].values())
    assert "src/feedingrobot/sim/model.py" in report["input_hashes"]
    assert set(str(p.relative_to(ROOT)) for p in asset_files(robot)).issubset(report["input_hashes"])
