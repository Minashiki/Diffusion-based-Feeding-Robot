"""Physical M1 acceptance with frozen thresholds and machine-readable evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import subprocess
import sys
import traceback

import mink
import mujoco
import numpy as np

from feedingrobot.sim.model import ROOT, asset_files, load_json, named_id
from feedingrobot.sim.task import FeedingTask


def angle_error(a, b):
    return float(np.linalg.norm((mink.SO3.from_matrix(a) @ mink.SO3.from_matrix(b).inverse()).log()))


def advance(task, seconds, trace=None):
    state = task.snapshot()
    for _ in range(round(seconds / task.dt)):
        state = task.step_physics()
        if trace is not None:
            trace.append({"time": state["time"], "tcp_x": state["tcp_position"][0],
                          "tcp_y": state["tcp_position"][1], "tcp_z": state["tcp_position"][2],
                          "peak_force_n": state["contact_peak_n"],
                          "impulse_ns": state["contact_impulse_ns"],
                          "max_joint_speed": float(np.max(np.abs(state["dq"]))),
                          "wrist_force_n": float(np.linalg.norm(state["compensated_wrench"][:3])),
                          "status": state["execution_status"]})
        assert not task.terminated, task.failure_reason
        assert task.adapter.fault is None, (task.adapter.fault, getattr(task.adapter, "error_detail", ""))
    return state


def supported(task):
    from feedingrobot.sim.events import evidence
    return evidence(task)["supported"]


def plate(task, cfg, trace):
    task.reset(preset="food_on_plate")
    advance(task, .3, trace)
    start = task.snapshot()["food_position"].copy()
    advance(task, cfg["support_hold_s"], trace)
    assert any({row["group1"], row["group2"]} == {"food", "plate"} for row in task.contacts)
    drift = float(np.linalg.norm(task.snapshot()["food_position"] - start))
    assert drift < cfg["support_drift_m"], drift
    return dict(duration_s=cfg["support_hold_s"], food_drift_m=drift)


def holding(task, cfg, trace):
    task.reset(preset="empty")
    start = task.snapshot()
    final = advance(task, cfg["hold_duration_s"], trace)
    pos = float(np.linalg.norm(final["tcp_position"] - start["tcp_position"]))
    rot = angle_error(final["tcp_rotation"], start["tcp_rotation"])
    assert pos < cfg["hold_position_error_m"], pos
    assert rot < cfg["hold_orientation_error_rad"], rot
    return dict(position_drift_m=pos, rotation_drift_rad=rot,
                gravity_joint_offset_rad=(final["q"] - task.robot_config["reset_q"]).tolist(),
                gravcomp=task.robot_config["gravcomp"])


def tracking(task, cfg, trace):
    task.reset(preset="empty")
    errors, rotation_errors, speeds, saturation = [], [], [], 0
    # Each of the six action axes has its own excitation.
    for axis in range(6):
        twist = np.zeros(6)
        twist[axis] = .01 if axis < 3 else .05
        now = task.data.time
        task.adapter.set_twist(twist, now, now + .45)
        for _ in range(round(.4 / task.dt)):
            state = advance(task, task.dt, trace)
            errors.append(float(np.linalg.norm(state["tcp_position"] - task.adapter.target.translation())))
            rotation_errors.append(angle_error(state["tcp_rotation"], task.adapter.target.rotation().as_matrix()))
            speeds.append(float(np.max(np.abs(state["dq"]))))
            force = state["actuator_force"]
            limits = task.model.actuator_forcerange[task.index.actuators]
            saturation += int(np.any(np.isclose(force, limits[:, 0], atol=1e-5) | np.isclose(force, limits[:, 1], atol=1e-5)))
    task.adapter.stop()
    final = advance(task, cfg["stop_settle_s"], trace)
    assert max(errors) < cfg["tracking_position_error_m"], max(errors)
    assert max(rotation_errors) < cfg["tracking_orientation_error_rad"], max(rotation_errors)
    assert np.linalg.norm(final["tcp_twist_world"][:3]) < cfg["stop_linear_speed_m_s"]
    assert np.linalg.norm(final["tcp_twist_world"][3:]) < cfg["stop_angular_speed_rad_s"]
    return dict(max_tracking_error_m=max(errors), max_rotation_error_rad=max(rotation_errors),
                max_joint_speed_rad_s=max(speeds), actuator_saturation_ticks=saturation,
                final_twist=final["tcp_twist_world"].tolist())


def reset_check(task, cfg, trace):
    signatures = []
    for iteration in range(cfg["reset_count"]):
        task.data.xfrc_applied[:] = 2
        task.data.qfrc_applied[:] = 3
        task.monitor.events.append({"stale": True})
        task.monitor.peak_n = 999
        task.adapter.stop("injected", fault=True)
        task.external_wrench = np.ones((3, 3))
        state = task.reset(seed=7, preset="food_on_plate")
        assert not task.monitor.events and task.monitor.peak_n == 0
        assert task.external_wrench is None and task.adapter.command is None and task.adapter.fault is None
        assert not task.data.xfrc_applied.any() and not task.data.qfrc_applied.any()
        assert not task.adapter.velocity.any() and task.tick == 0 and task.data.time == 0
        signatures.append(np.r_[task.data.qpos, task.data.qvel, task.data.ctrl])
        if iteration:
            np.testing.assert_allclose(signatures[-1], signatures[0], rtol=0, atol=cfg["reset_atol"])
    return dict(resets=len(signatures), max_difference=float(np.max(np.abs(np.array(signatures) - signatures[0]))))


def wrench_check(task, cfg, trace):
    task.reset(preset="empty")
    force_errors, torque_errors, residuals = [], [], []
    for direction in np.r_[np.eye(3), -np.eye(3)]:
        point = task.data.site_xpos[task.index.tcp].copy() + np.array([.03, -.02, .01])
        torque = np.array([.01, -.02, .03])
        task.set_external_wrench(direction, torque, point)
        state = advance(task, .02, trace)
        expected_torque = torque + np.cross(point - state["tcp_position"], direction)
        force_errors.append(float(np.max(np.abs(state["compensated_wrench"][:3] - direction))))
        torque_errors.append(float(np.max(np.abs(state["compensated_wrench"][3:] - expected_torque))))
    task.clear_external_wrench()
    now = task.data.time
    task.adapter.set_twist([.01, -.01, .01, .03, -.02, .01], now, now + .5)
    for _ in range(round(.4 / task.dt)):
        state = advance(task, task.dt, trace)
        residuals.append(state["compensated_wrench"].copy())
    maximum = np.max(np.abs(residuals), axis=0)
    assert max(force_errors) < cfg["wrench_force_error_n"], force_errors
    assert max(torque_errors) < cfg["wrench_torque_error_nm"], torque_errors
    assert np.max(maximum[:3]) < cfg["wrench_force_error_n"], maximum
    assert np.max(maximum[3:]) < cfg["wrench_torque_error_nm"], maximum
    return dict(force_error_n=max(force_errors), torque_error_nm=max(torque_errors), dynamic_residual=maximum.tolist())


def carry(task, cfg, trace, seed=0):
    task.reset(seed=seed, preset="food_on_spoon")
    advance(task, .3, trace)
    assert supported(task), "Food did not settle on spoon"
    advance(task, cfg["support_hold_s"], trace)
    assert supported(task), "Food lost during static support"
    now = task.data.time
    task.adapter.set_twist([.008, 0, .003, 0, 0, 0], now, now + 1.1)
    advance(task, 1., trace)
    assert supported(task), "Food lost during gentle carry"
    return dict(food_supported=True, peak_force_n=task.monitor.peak_n, impulse_ns=task.monitor.impulse_ns,
                tcp_position=task.snapshot()["tcp_position"].tolist(),
                tcp_rotation=task.snapshot()["tcp_rotation"].tolist())


def drop(task, cfg, trace, kind):
    task.reset(preset="food_on_spoon")
    advance(task, .3, trace)
    assert supported(task)
    rotation = task.snapshot()["tcp_rotation"]
    base = task.data.site_xmat[task.index.base].reshape(3, 3)
    if kind == "tilt":
        command = np.r_[np.zeros(3), base.T @ rotation[:, 1] * .4]
        duration = 4.
    else:
        # Recreate limits before running the explicitly labelled drop diagnostic.
        task.robot_config.update({k: v for k, v in cfg["acceleration_diagnostic"].items()
                                  if k in task.robot_config})
        task.scene_config["joint_speed_fault_rad_s"] = cfg["acceleration_diagnostic"]["joint_speed_fault_rad_s"]
        from feedingrobot.control.adapter import RobotAdapter
        task.adapter = RobotAdapter(task.model, task.data, task.index, task.robot_config)
        gain = cfg["acceleration_diagnostic"]["servo_gain_scale"]
        ids = task.index.actuators
        task.model.actuator_gainprm[ids, 0] *= gain
        task.model.actuator_biasprm[ids, 1] *= gain
        task.model.actuator_biasprm[ids, 2] *= np.sqrt(gain)
        command = np.r_[-base.T @ rotation[:, 1] * cfg["acceleration_diagnostic"]["linear_speed_limit"], np.zeros(3)]
        duration = .65
    now = task.data.time
    task.adapter.set_twist(command, now, now + duration + .01)
    absence = 0.
    max_accel, previous_velocity = 0., task.snapshot()["tcp_twist_world"][:3]
    max_speed = 0.
    for _ in range(round(duration / task.dt)):
        state = advance(task, task.dt, trace)
        max_accel = max(max_accel, float(np.linalg.norm(state["tcp_twist_world"][:3] - previous_velocity) / task.dt))
        previous_velocity = state["tcp_twist_world"][:3]
        max_speed = max(max_speed, float(np.max(np.abs(state["dq"]))))
        absence = 0 if supported(task) else absence + task.dt
        if absence >= cfg["support_confirm_s"]:
            break
    assert absence >= cfg["support_confirm_s"], f"Food did not drop under {kind}"
    final_relative = state["tcp_rotation"].T @ (state["food_position"] - state["tcp_position"])
    lo, hi = np.array(task.task_config["spoon_support_min_m"]), np.array(task.task_config["spoon_support_max_m"])
    assert np.any(final_relative < lo) or np.any(final_relative > hi), final_relative
    return dict(dropped=True, absent_s=absence, final_relative_position=final_relative.tolist(), max_tcp_acceleration_m_s2=max_accel,
                max_joint_speed_rad_s=max_speed, diagnostic_limits=cfg["acceleration_diagnostic"] if kind == "acceleration" else None)


def assembly(task, cfg, trace):
    idx, model = task.index, task.model
    assert len(idx.spoon_geoms) == 145 and len(idx.scoop_geoms) == 130 and len(idx.handle_geoms) == 15
    assert len(idx.plate_geoms) == 17 and model.npair == 145
    assert all(idx.group(model, g) == "spoon" for g in idx.spoon_geoms)
    assert all(idx.group(model, g) == "plate" for g in idx.plate_geoms)
    assert set(model.pair_geom1).issubset(idx.spoon_geoms)
    assert set(model.pair_geom2).issubset(idx.plate_geoms)
    assert np.isclose(idx.tool_mass, .035) and model.body_mass[idx.tool_body] == 0
    assert not any("bowl" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                   for g in range(model.ngeom))
    for name in ("dynamic_spoon2_freejoint", "dynamic_plate2_freejoint"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1
    advance(task, .02, trace)
    assert all(row["distance"] >= -.001 for row in task.contacts)
    return dict(spoon_collision_count=145, scoop_collision_count=130, plate_collision_count=17,
                source_pair_count=145, tool_mass_kg=idx.tool_mass,
                robot_config=task.robot_config, scene_config=task.scene_config)


def move_to(task, position, cfg, trace):
    base = task.data.site_xmat[task.index.base].reshape(3, 3)
    for tick in range(round(cfg["reachability_timeout_s"] / task.dt)):
        state = task.snapshot()
        delta = position - state["tcp_position"]
        angular = mink.SO3.from_matrix(state["tcp_rotation"]).inverse().log()
        if np.linalg.norm(delta) < .004 and np.linalg.norm(angular) < .03:
            task.adapter.stop("stopped", hold_reference=True)
            return advance(task, .5, trace)
        if tick % 20 == 0:
            now = task.data.time
            task.adapter.set_twist(np.r_[base.T @ delta * 3., base.T @ angular * 3.], now, now + .04)
        advance(task, task.dt, trace)
        assert not any("arm" in (row["group1"], row["group2"]) and row["force_n"] > 1e-5
                       for row in task.contacts), "Forbidden arm contact on the path"
    raise AssertionError(f"Actual servo motion could not reach {position}: {task.snapshot()['tcp_position']}")


def reachability(task, cfg, trace):
    task.reset(preset="empty")
    results = {}
    plate = task.data.site_xpos[named_id(task.model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")].copy()
    mouth = task.data.site_xpos[named_id(task.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_entry")].copy()
    for name, position in [("plate_above", plate + [0, 0, .14]),
                           ("plate_low", plate + [0, 0, .05]),
                           ("carry_clearance", plate + [0, 0, .18]),
                           ("mouth_wait", mouth + [-.06, 0, 0])]:
        final = move_to(task, position, cfg, trace)
        error = float(np.linalg.norm(final["tcp_position"] - position))
        assert error < cfg["tracking_position_error_m"], (name, error)
        assert angle_error(final["tcp_rotation"], np.eye(3)) < cfg["tracking_orientation_error_rad"]
        forbidden = [row for row in task.contacts if "arm" in (row["group1"], row["group2"]) and row["force_n"] > 1e-5]
        assert not forbidden, forbidden
        results[name] = dict(position_error_m=error, actual_q=final["q"].tolist(),
                             kind="actual IK/position-servo/physics path; not full feeding success")
    return results


def dynamic_head(task, cfg, trace):
    task.scene_config["head_fixed"] = False
    task.reset(preset="food_on_plate")
    samples = []
    for _ in range(round(2. / task.dt)):
        advance(task, task.dt, trace)
        samples.append(task.data.qpos[task.model.jnt_qposadr[task.index.head_joints]].copy())
    excursion = np.ptp(np.array(samples), axis=0)
    assert np.linalg.norm(excursion[:2]) > .001
    assert excursion[4] > .001
    return dict(excursion_rad_or_m=excursion.tolist(), driver_force_limits=
                task.model.actuator_forcerange[task.index.head_actuators].tolist())


def viewer_session(robot):
    from feedingrobot.sim.viewer import passive_viewer
    task = FeedingTask(robot)
    task.reset(preset="food_on_spoon")
    with passive_viewer(task.model, task.data) as viewer:
        viewer.opt.geomgroup[3] = 0
        for _ in range(100):
            task.step_physics()
            viewer.sync()
            time.sleep(.005)
        assert viewer.is_running(), "Viewer closed before verification"


def viewer_check(task, cfg, trace):
    result = subprocess.run([sys.executable, "-c",
        "from feedingrobot.scripts.validate_m1 import viewer_session; import sys; viewer_session(sys.argv[1])",
        task.robot_id], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(open_sync_close=True, stderr=result.stderr)


def convergence(robot, cfg, trace):
    runs = []
    for seed in cfg["seeds"]:
        variants = []
        baseline = load_json("configs/scene.json")["solver_iterations"]
        for dt, iterations in [(.001, baseline), (.0005, baseline), (.001, 2 * baseline)]:
            task = FeedingTask(robot, timestep=dt)
            task.scene_config["head_fixed"] = True
            task.model.opt.iterations = iterations
            if iterations == 2 * baseline:
                task.model.opt.tolerance /= 10
            result = carry(task, cfg, [], seed)
            result.update(dt=dt, iterations=iterations, seed=seed)
            variants.append(result)
        base = variants[0]
        for other in variants[1:]:
            pos = np.linalg.norm(np.array(base["tcp_position"]) - other["tcp_position"])
            rotation = angle_error(np.array(base["tcp_rotation"]), np.array(other["tcp_rotation"]))
            force = abs(base["peak_force_n"] - other["peak_force_n"])
            impulse = abs(base["impulse_ns"] - other["impulse_ns"])
            tolerance = max(cfg["convergence_force_absolute_n"], cfg["convergence_force_relative"] * base["peak_force_n"])
            assert pos <= cfg["convergence_position_m"], pos
            assert rotation <= cfg["convergence_rotation_rad"], rotation
            assert force <= tolerance, (force, tolerance)
            assert impulse <= max(cfg["convergence_impulse_absolute_ns"], cfg["convergence_impulse_relative"] * base["impulse_ns"])
        runs.extend(variants)
    return dict(runs=runs)


def fault_checks(task, cfg, trace):
    task.reset(preset="empty")
    # Unreachable workspace request must clear and latch the reference.
    base_p = task.data.site_xpos[task.index.base]
    base_r = task.data.site_xmat[task.index.base].reshape(3, 3)
    p = base_r.T @ (task.data.site_xpos[task.index.tcp] - base_p)
    task.robot_config["workspace_max"] = (p + [.0001, 1, 1]).tolist()
    now = task.data.time
    task.adapter.set_twist([.05, 0, 0, 0, 0, 0], now, now + 1)
    for _ in range(200):
        task.step_physics()
        if task.adapter.fault:
            break
    assert task.adapter.fault == "workspace_limit" and task.adapter.command is None
    reference = task.data.ctrl[task.index.actuators].copy()
    for _ in range(20):
        task.step_physics()
    np.testing.assert_array_equal(reference, task.data.ctrl[task.index.actuators])
    return {"unreachable_cancelled": True}


def guard_regressions(task, cfg, trace):
    tests = ["tests/test_contracts.py", "tests/test_guard_physics.py", "tests/test_tableware.py"]
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(tests=tests, return_code=result.returncode, output=result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_json("configs/acceptance.json")
    cases = dict(assembly=assembly, head=dynamic_head, viewer=viewer_check, plate=plate, hold=holding, tracking=tracking, reset=reset_check, wrench=wrench_check, carry=carry,
                 tilt=lambda t, c, tr: drop(t, c, tr, "tilt"),
                 acceleration=lambda t, c, tr: drop(t, c, tr, "acceleration"),
                 reachability=reachability, faults=fault_checks, guards=guard_regressions)
    cases["convergence"] = lambda t, c, tr: convergence(args.robot, c, tr)
    output = ROOT / (args.output or f"outputs/new_tableware/v1/m1/{args.robot}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = asset_files(args.robot) + list((ROOT / "src/feedingrobot").rglob("*.py"))
    inputs += [ROOT / name for name in ("tests/test_contracts.py", "tests/test_guard_physics.py", "tests/test_tableware.py",
                                       "requirements.lock.txt", "third_party_manifest.json")]
    report = {"robot_id": args.robot, "model_version": "new_tableware_v1", "acceptance": cfg,
              "input_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "cases": {}}
    for name, check in cases.items():
        if args.cases is not None and name not in args.cases:
            report["cases"][name] = {"status": "not_verified"}
            continue
        trace = []
        started = time.monotonic()
        try:
            task = FeedingTask(args.robot)
            task.scene_config["head_fixed"] = name != "head"
            task.reset()
            detail = check(task, cfg, trace)
            report["cases"][name] = {"status": "passed", "head_fixed": task.scene_config["head_fixed"], "metrics": detail}
        except Exception:
            report["cases"][name] = {"status": "failed", "error": traceback.format_exc()}
        report["cases"][name]["wall_seconds"] = time.monotonic() - started
        print(args.robot, name, report["cases"][name]["status"], flush=True)
        if trace:
            with (output / f"{name}.csv").open("w") as file:
                writer = csv.DictWriter(file, fieldnames=trace[0].keys())
                writer.writeheader()
                writer.writerows(trace)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"].values()) else "incomplete"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
