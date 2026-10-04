"""Model-only M1-A acceptance; does not use the pending FeedingTask API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from feedingrobot.sim.model import ROOT, asset_files, load_model, named_id


def beans_check(model, data, index, scene):
    beans = scene["beans"]
    bean_count = beans["count"]
    assert bean_count == len(index.bean_ids)
    assert index.bean_ids == tuple(f"bean_{i:03d}" for i in range(bean_count))
    bodies, joints = index.bean_bodies, index.bean_joints
    visual, collision = index.bean_visual_geoms, index.bean_collision_geoms
    for ids in (bodies, joints, visual, collision):
        assert ids.shape == (bean_count,) and len(set(ids)) == bean_count
    assert index.bean_qpos.shape == (bean_count, 7) and index.bean_dofs.shape == (bean_count, 6)
    assert len(set(index.bean_qpos.ravel())) == 7 * bean_count
    assert len(set(index.bean_dofs.ravel())) == 6 * bean_count
    np.testing.assert_array_equal(index.bean_qpos, model.jnt_qposadr[joints, None] + np.arange(7))
    np.testing.assert_array_equal(index.bean_dofs, model.jnt_dofadr[joints, None] + np.arange(6))
    assert not set(index.bean_dofs.ravel()).intersection(index.dofs)
    assert np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == bean_count
    assert np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_FREE)
    np.testing.assert_array_equal(model.jnt_bodyid[joints], bodies)
    np.testing.assert_array_equal(model.body_parentid[bodies], 0)
    np.testing.assert_array_equal(model.body_geomnum[bodies], 2)
    for field in ("dof_damping", "dof_armature", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(model, field)[index.bean_dofs], 0)
    np.testing.assert_array_equal(model.geom_bodyid[visual], bodies)
    np.testing.assert_array_equal(model.geom_bodyid[collision], bodies)
    assert np.all(model.geom_type[visual] == mujoco.mjtGeom.mjGEOM_MESH)
    assert np.all(model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    assert np.count_nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID) == bean_count
    assert len(set(model.geom_dataid[visual])) == 1
    mesh = named_id(model, mujoco.mjtObj.mjOBJ_MESH, "bean_visual_mesh")
    np.testing.assert_array_equal(model.geom_dataid[visual], mesh)
    np.testing.assert_array_equal(model.geom_contype[visual], 0)
    np.testing.assert_array_equal(model.geom_conaffinity[visual], 0)
    np.testing.assert_array_equal(model.geom_group[visual], 1)
    np.testing.assert_array_equal(model.geom_group[collision], 3)
    np.testing.assert_array_equal(model.geom_contype[collision], 2)
    np.testing.assert_array_equal(model.geom_conaffinity[collision], 3)
    np.testing.assert_allclose(model.geom_pos[collision], 0, atol=1e-12)
    np.testing.assert_allclose(model.geom_quat[collision], np.tile([1, 0, 0, 0], (bean_count, 1)), atol=1e-12)
    axes = np.array(beans["semi_axes_m"])
    a, b, c = axes
    mass = beans["density_kg_m3"] * 4 * np.pi * a * b * c / 3
    inertia = mass / 5 * np.array([b*b + c*c, a*a + c*c, a*a + b*b])
    np.testing.assert_allclose(model.body_mass[bodies], mass, rtol=1e-8)
    np.testing.assert_allclose(model.body_inertia[bodies], np.tile(inertia, (bean_count, 1)), rtol=1e-8)
    np.testing.assert_allclose(model.body_ipos[bodies], 0, atol=1e-12)
    np.testing.assert_allclose(model.body_iquat[bodies], np.tile([1, 0, 0, 0], (bean_count, 1)), atol=1e-12)
    np.testing.assert_allclose(model.geom_size[collision], np.tile(axes, (bean_count, 1)), atol=1e-12)
    for key in ("friction", "solref", "solimp", "condim", "priority", "margin", "gap"):
        actual = getattr(model, f"geom_{key}")[collision]
        np.testing.assert_allclose(actual, np.broadcast_to(beans[key], actual.shape), rtol=1e-8)
    assert not set(bodies).intersection(index.tool_bodies)
    assert all(index.group(model, g) == "food" for g in collision)
    assert all(index.group(model, g) == "other" for g in visual)
    assert index.bean_id_by_geom == dict(zip(collision, index.bean_ids))
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start + count]
    assert count == 482 and model.mesh_facenum[mesh] == 960
    max_error = 0.
    for name, body, geom in zip(index.bean_ids, bodies, visual):
        # geom_xmat includes MuJoCo's mesh reorientation/recentring transform.
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        error = float(np.max(np.abs(np.ptp(local, axis=0) - 2 * axes)))
        max_error = max(max_error, error)
        assert error <= 1e-7, (name, error)
        np.testing.assert_allclose(local.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(np.sum((local / axes)**2, axis=1), 1, atol=1e-6, rtol=0)
    return dict(count=bean_count, mass_per_bean_kg=float(mass), total_mass_kg=float(bean_count * mass),
                inertia_kg_m2=inertia.tolist(), shared_mesh_id=mesh,
                mesh_vertices=count, mesh_faces=960, visual_size_error_m=max_error)


def tableware_check(model, data, index, scene):
    assert len(index.spoon_geoms) == 145 and len(index.scoop_geoms) == 130
    assert len(index.handle_geoms) == 15 and len(index.bowl_geoms) == 17
    np.testing.assert_allclose(index.tool_mass, .035, atol=1e-12, rtol=0)
    assert model.nplugin == 0
    for kind in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                 mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_MESH):
        count = {mujoco.mjtObj.mjOBJ_BODY: model.nbody, mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
                 mujoco.mjtObj.mjOBJ_GEOM: model.ngeom, mujoco.mjtObj.mjOBJ_SITE: model.nsite,
                 mujoco.mjtObj.mjOBJ_MESH: model.nmesh}[kind]
        names = [mujoco.mj_id2name(model, kind, i) or "" for i in range(count)]
        assert not any("plate" in n or n in {"food", "food_joint", "food_box"} for n in names)
    for name in ("dynamic_spoon2_freejoint", "dynamic_bowl2_freejoint"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1
    bowl = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_bowl2")
    assert model.body_parentid[bowl] == 0 and model.body_jntnum[bowl] == 0
    internal = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_bowl2_usd_internal")
    assert model.body_parentid[internal] == bowl and model.body_jntnum[internal] == 0
    frame = named_id(model, mujoco.mjtObj.mjOBJ_SITE, "bowl_frame")
    np.testing.assert_allclose(data.site_xpos[frame], scene["bowl_frame_position_m"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(data.site_xpos[frame], [.45, -.18, -.0156], atol=1e-9, rtol=0)
    np.testing.assert_allclose(data.site_xmat[frame].reshape(3, 3), np.eye(3), atol=1e-9)
    disk = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision_bowl_fast_bottom_disk")
    table = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    np.testing.assert_allclose(data.geom_xpos[disk, 2] - model.geom_size[disk, 1],
                               data.geom_xpos[table, 2] + model.geom_size[table, 2], atol=1e-9)
    for kind in ("spoon", "bowl"):
        source = ET.parse(ROOT / f"assets/task/tableware/{kind}/{kind}.xml")
        for geom in source.findall("worldbody//geom"):
            g = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom.get("name"))
            if geom.get("group") == "1":
                assert not model.geom_contype[g] and not model.geom_conaffinity[g]
                continue
            assert index.group(model, g) == kind
            for key in ("contype", "conaffinity", "condim"):
                assert getattr(model, f"geom_{key}")[g] == int(geom.get(key))
            keys = ["friction", "solref", "solimp"]
            if geom.get("type") != "mesh":
                keys += ["pos", "size"]
            for key in keys:
                if geom.get(key) is not None:
                    expected = np.fromstring(geom.get(key), sep=" ")
                    np.testing.assert_allclose(getattr(model, f"geom_{key}")[g, :len(expected)], expected)
    pairs = [p for p in ET.parse(ROOT / "assets/task/tableware/contact_pairs.xml").findall("contact/pair")
             if p.get("geom2").startswith("collision_bowl_")]
    assert len(pairs) == model.npair == 145
    assert set(model.pair_geom1).issubset(index.spoon_geoms)
    assert set(model.pair_geom2).issubset(index.bowl_geoms)
    for pair in pairs:
        p = named_id(model, mujoco.mjtObj.mjOBJ_PAIR, pair.get("name"))
        assert model.pair_geom1[p] == named_id(model, mujoco.mjtObj.mjOBJ_GEOM, pair.get("geom1"))
        assert model.pair_geom2[p] == named_id(model, mujoco.mjtObj.mjOBJ_GEOM, pair.get("geom2"))
        for key in ("dim", "friction", "margin", "gap", "solref", "solimp"):
            value = pair.get("condim" if key == "dim" else key)
            actual = getattr(model, f"pair_{key}")[p]
            expected = np.fromstring(value, sep=" ")
            np.testing.assert_allclose(actual, expected if np.ndim(actual) else expected[0])
    return dict(spoon_collision_count=145, scoop_collision_count=130, handle_collision_count=15,
                bowl_collision_count=17, spoon_bowl_pairs=145, tool_mass_kg=index.tool_mass,
                bowl_frame_position_m=data.site_xpos[frame].tolist())


def masks_check(model, index):
    groups = {"food": index.bean_collision_geoms, "bowl": index.bowl_geoms, "spoon": index.scoop_geoms}
    for group in ("table", "floor", "mouth"):
        groups[group] = [g for g in range(model.ngeom) if index.group(model, g) == group]
    beans = index.bean_collision_geoms
    for group, geoms in groups.items():
        assert len(geoms) > 0
        matches = ((model.geom_contype[beans, None] & model.geom_conaffinity[geoms])
                   | (model.geom_conaffinity[beans, None] & model.geom_contype[geoms]))
        assert np.all(matches != 0), group
    return {f"bean_{group}": "mask_matches" for group in groups}


def integration_state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, state, kind)
    return state.tolist()


def smoke_check(model, data, index, scene):
    initial = integration_state(model, data)
    assert all(c.geom1 not in index.bean_visual_geoms and c.geom2 not in index.bean_visual_geoms
               for c in data.contact[:data.ncon])
    assert all(c.dist >= 0 for c in data.contact[:data.ncon]), "Compile layout starts penetrated"
    trace = [data.xpos[index.bean_bodies].copy()]
    started = time.monotonic()
    for _ in range(100):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc, data.ctrl, data.sensordata))
        assert not any(w.number for w in data.warning)
        assert all(c.geom1 not in index.bean_visual_geoms and c.geom2 not in index.bean_visual_geoms
                   for c in data.contact[:data.ncon])
        trace.append(data.xpos[index.bean_bodies].copy())
    wall_seconds = time.monotonic() - started
    np.testing.assert_allclose(data.time, 100 * model.opt.timestep, atol=1e-12)
    # Also check visual alignment after the bodies have moved independently.
    beans_check(model, data, index, scene)
    return dict(steps=100, simulation_seconds=float(data.time), wall_seconds=wall_seconds,
                warning_counts=[int(w.number) for w in data.warning],
                state_kind="mjSTATE_INTEGRATION", initial_state=initial,
                final_state=integration_state(model, data), bean_positions_trace_m=np.array(trace).tolist())


def validate(robot):
    report = dict(schema_version=1, model_version="single_bean_native_v1", stage="M1-A", robot_id=robot,
                  mujoco_version=mujoco.__version__, status="failed", m1_status="incomplete",
                  stages={"M1-A": "failed", "M1-B": "not_verified", "M1-C": "not_verified", "M1-D": "not_verified"},
                  cases={})
    try:
        paths = asset_files(robot) + list((ROOT / "src/feedingrobot").rglob("*.py"))
        paths += [ROOT / "tests/test_beans_native.py", ROOT / "tests/test_tableware.py",
                  ROOT / "third_party_manifest.json", ROOT / "requirements.lock.txt"]
        report["input_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        model, index, cfg, scene = load_model(robot)
        data = mujoco.MjData(model)
        data.qpos[index.qpos] = cfg["reset_q"]
        data.ctrl[index.actuators] = cfg["reset_q"]
        mujoco.mj_forward(model, data)
        report["cases"]["compile_forward"] = dict(status="passed", nq=model.nq, nv=model.nv, plugins=model.nplugin)
        report["solver"] = dict(timestep=float(model.opt.timestep), iterations=int(model.opt.iterations),
                                ccd_tolerance_m=float(model.opt.ccd_tolerance), ccd_iterations=int(model.opt.ccd_iterations),
                                tolerance=float(model.opt.tolerance), impratio=float(model.opt.impratio),
                                integrator=mujoco.mjtIntegrator(int(model.opt.integrator)).name,
                                solver=mujoco.mjtSolver(int(model.opt.solver)).name,
                                cone=mujoco.mjtCone(int(model.opt.cone)).name)
        assert scene["beans"]["count"] == 1, "Formal M1 requires exactly one bean"
        report["bean_index"] = dict(ids=list(index.bean_ids), bodies=index.bean_bodies.tolist(),
                                    joints=index.bean_joints.tolist(), qpos=index.bean_qpos.tolist(),
                                    dofs=index.bean_dofs.tolist(), visual_geoms=index.bean_visual_geoms.tolist(),
                                    collision_geoms=index.bean_collision_geoms.tolist())
    except Exception:
        report["cases"]["compile_forward"] = dict(status="failed", error=traceback.format_exc())
        return report
    checks = dict(beans=lambda: beans_check(model, data, index, scene),
                  tableware=lambda: tableware_check(model, data, index, scene),
                  masks=lambda: masks_check(model, index),
                  smoke=lambda: smoke_check(model, data, index, scene))
    for name, check in checks.items():
        started = time.monotonic()
        try:
            report["cases"][name] = dict(status="passed", metrics=check())
        except Exception:
            report["cases"][name] = dict(status="failed", error=traceback.format_exc())
        report["cases"][name]["wall_seconds"] = time.monotonic() - started
    if all(case["status"] == "passed" for case in report["cases"].values()):
        report["status"] = report["stages"]["M1-A"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--output", help="Report directory, relative to the repository or absolute")
    args = parser.parse_args()
    report = validate(args.robot)
    output = ROOT / (args.output or f"outputs/single_bean/v1/m1/{args.robot}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "m1a_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, result in report["cases"].items():
        print(args.robot, name, result["status"])
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
