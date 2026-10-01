"""Teacher baseline, physical replay and M4 evidence, never synthetic success."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
import subprocess
import sys
import traceback

import mink
import numpy as np

from feedingrobot.data.episodes import input_hashes, load_episode, write_json
from feedingrobot.data.recipes import recipe
from feedingrobot.data.replay import replay_episode
from feedingrobot.data.rollout import run_episode
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts.feasibility import check_waypoints
from feedingrobot.scripts.collect import dataset_statistics
from feedingrobot.sim.model import ROOT, load_json

CASES = ("feasibility", "teacher", "replay", "convergence", "viewer", "regressions", "dataset")


def physical_success(result):
    if not result["success"]:
        return False
    rows = {row["event"]["name"]: row for row in result["evidence"]}
    assert set(rows) == {"pickup", "delivery", "success"}
    assert rows["pickup"]["supported"] and rows["pickup"]["off_plate"] and not rows["pickup"]["on_plate"]
    assert rows["delivery"]["mouth_supported"] and rows["delivery"]["released"] and not rows["delivery"]["supported"]
    assert rows["success"]["mouth_supported"] and rows["success"]["released"]
    assert not rows["success"]["tool_inside"] and not rows["success"]["tool_mouth_contact"]
    assert rows["pickup"]["event"]["time"] < rows["delivery"]["event"]["time"] < rows["success"]["event"]["time"]
    return True


def acceptance_trial(robot, config, output, index):
    seed, scenario, parameters, group = recipe(config, "acceptance", index)
    path = output / "episodes" / str(seed)
    if path.exists():
        result, _ = load_episode(path)
        if result["input_hashes"] != input_hashes() or result["teacher_config"] != config:
            raise ValueError("Acceptance directory contains a different teacher/source version")
    else:
        result = run_episode(robot, seed, config, path, scenario=scenario, teacher_parameters=parameters,
                             split="acceptance", group_id=group)
    physical_success(result)
    summary = {key: result[key] for key in ("seed", "success", "failure_reason", "truncated", "time_s",
                                           "contact_peak_n", "contact_impulse_ns", "wrist_peak_n")}
    summary["milestones"] = [e["name"] for e in result["events"] if e["name"] in ("pickup", "delivery", "success")]
    summary["phases_visited"] = ["SELECT"] + [e["phase"] for e in result["events"] if e["name"] == "phase"]
    return summary


def baseline(robot, config, output, trials, workers=1):
    completed = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(acceptance_trial, robot, config, output, i): i for i in range(trials)}
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            completed[index] = result
            results = [completed[k] for k in sorted(completed)]
            write_json(output / "baseline_progress.json", results)
            print(f"{robot}: completed {len(completed)}/{trials}, seed={result['seed']}, "
                  f"success={result['success']}, failure={result['failure_reason']}", flush=True)
    results = [completed[k] for k in sorted(completed)]
    successes = sum(r["success"] for r in results)
    required = 100 if robot == "panda" else 10
    passed = trials == required and successes >= (95 if robot == "panda" else 10)
    return dict(status="passed" if passed else "failed" if trials == required else "incomplete", attempts=trials,
                successes=successes, success_rate=successes/trials,
                failure_counts=dict(Counter(r["failure_reason"] or "time_limit" for r in results if not r["success"])),
                milestone_counts=dict(Counter(name for r in results for name in r["milestones"])),
                trials=results, threshold="95/100" if robot == "panda" else "10/10")


def convergence(robot, config, output):
    tolerances = load_json("configs/acceptance_m3.json")
    results = []
    for index in range(10):
        seed, scenario, parameters, group = recipe(config, "acceptance", index)
        path = output / "episodes" / str(seed)
        original, arrays = load_episode(path)
        if not physical_success(original):
            raise AssertionError("Full-flow convergence requires successful baseline episodes")
        for name, dt, iterations in (("half_dt", .0005, 50), ("double_iterations", .001, 100)):
            fine_path = output / "convergence" / f"{seed}_{name}"
            fine = run_episode(robot, seed, config, fine_path, scenario=scenario, teacher_parameters=parameters,
                               timestep=dt, iterations=iterations, split="acceptance", group_id=group)
            assert physical_success(fine), fine["failure_reason"]
            original_events = [e for e in original["events"] if e["name"] in ("pickup", "delivery", "success")]
            fine_events = [e for e in fine["events"] if e["name"] in ("pickup", "delivery", "success")]
            event_error = max(abs(a["time"]-b["time"]) for a, b in zip(original_events, fine_events))
            assert event_error <= tolerances["event_time_tolerance_s"], event_error
            for key, absolute, relative in (("contact_peak_n", "force_absolute_tolerance_n", "force_relative_tolerance"),
                                            ("contact_impulse_ns", "impulse_absolute_tolerance_ns", "impulse_relative_tolerance")):
                assert abs(original[key]-fine[key]) <= max(tolerances[absolute], tolerances[relative]*original[key])
            tcp_error = float(np.max(np.abs(np.asarray(original["evidence"][-1]["tcp_position"])
                                           - np.asarray(fine["evidence"][-1]["tcp_position"]))))
            assert tcp_error <= tolerances["tcp_position_tolerance_m"]
            results.append(dict(seed=seed, variant=name, event_error_s=event_error, tcp_error_m=tcp_error,
                                contact_peak_n=fine["contact_peak_n"], contact_impulse_ns=fine["contact_impulse_ns"]))
    return dict(status="passed", trials=results, criteria=tolerances)


def run_logged(command, path, timeout=None):
    with path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise AssertionError(f"Command exited {result.returncode}; see {path}")
    return dict(command=command, log=str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--config", default="configs/collect.json")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--trials", type=int, help="Partial checks are never a formal teacher gate")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    trials = args.trials if args.trials is not None else (100 if args.robot == "panda" else 10)
    if trials < 1 or args.workers < 1:
        parser.error("--trials and --workers must be positive")
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / (args.output or f"outputs/m4/{args.robot}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if report and (report.get("input_hashes") != input_hashes() or report.get("teacher_config") != config):
        raise ValueError("Report directory belongs to different source/teacher; use a new --output")
    report.update(robot_id=args.robot, teacher_config=config, input_hashes=input_hashes(),
                  scope="P0 teacher and data only; no learned policies", status="incomplete")
    report.setdefault("cases", {name: dict(status="not_verified") for name in CASES})
    report.setdefault("teacher_gate", "not_verified")
    selected = args.cases or CASES
    for name in selected:
        write_json(report_path, report)
        try:
            if name == "feasibility":
                env = FeedingGymEnv(args.robot)
                try:
                    env.reset(seed=0, options={"scenario": config["scene"]})
                    task, p = env.task, config["teacher"]
                    food = task.snapshot()["food_position"]
                    mouth = task.provider.observe()["policy_obs"]
                    mr = mouth["mouth_rotation"]
                    mp = mouth["tcp_position"] + mouth["mouth_relative_world"]
                    yaw = mink.SO3.exp(np.array([0., 0., p["scoop_yaw_rad"]])).as_matrix()
                    scoop = food-yaw[:,0]*p["scoop_start_offset_m"]
                    scoop[2] = p["scoop_height_m"]
                    before = task.data.qpos.copy()
                    checks = check_waypoints(task, [("scoop_entry", scoop,
                                      yaw @ mink.SO3.exp(np.array([0., p["scoop_pitch_rad"],0.])).as_matrix()),
                                      ("wait", mp-mr[:,0]*p["wait_offset_m"], mr),
                                      ("insert", mp+mr@np.array([p["insert_depth_m"],0.,p["insert_height_m"]]), mr)])
                    np.testing.assert_array_equal(before, task.data.qpos)
                    result = dict(status="passed" if all(c["reachable"] for c in checks) else "failed", checks=checks,
                                  simulation_qpos_unchanged=True)
                finally:
                    env.close()
            elif name == "teacher":
                result = baseline(args.robot, config, output, trials, args.workers)
                report["baseline"] = result
                report["teacher_gate"] = result["status"]
            elif name == "replay":
                paths = sorted((output / "episodes").glob("*/manifest.json"))
                if not paths:
                    raise ValueError("No baseline episodes to replay")
                results = []
                for path in paths:
                    results.append(replay_episode(path.parent))
                    write_json(output / "replay_progress.json", results)
                result = dict(status="passed", episodes=results)
            elif name == "convergence":
                if report["teacher_gate"] != "passed":
                    result = dict(status="not_verified", reason="Full-flow teacher gate has not passed")
                else:
                    result = convergence(args.robot, config, output)
            elif name == "viewer":
                if not os.environ.get("DISPLAY"):
                    result = dict(status="not_verified", reason="DISPLAY unavailable")
                else:
                    successful = [p.parent for p in (output / "episodes").glob("*/manifest.json")
                                  if json.loads(p.read_text())["success"]]
                    if not successful:
                        result = dict(status="not_verified", reason="Full-flow viewer needs a successful baseline")
                    else:
                        command = [sys.executable, "-m", "feedingrobot.scripts.replay", str(successful[0]),
                                   "--robot", args.robot, "--viewer"]
                        result = dict(status="passed", **run_logged(command, output / "viewer.log", timeout=180))
            elif name == "regressions":
                commands = [[sys.executable, "-m", "pytest", "-q", f"--junitxml={output / 'tests.xml'}"],
                            [sys.executable, "-m", "feedingrobot.scripts.validate_m1", "--robot", args.robot,
                             "--output", str(output / "m1_regression")],
                            [sys.executable, "-m", "feedingrobot.scripts.validate_m3", "--robot", args.robot,
                             "--output", str(output / "m3_regression")]]
                result = dict(status="passed", checks=[run_logged(c, output / f"regression_{i}.log") for i,c in enumerate(commands)])
            else:
                if args.robot == "ur5e":
                    result = dict(status="passed", scope="UR5e smoke only; dataset intentionally excluded")
                else:
                    if report["teacher_gate"] != "passed":
                        result = dict(status="not_verified", reason="Teacher gate failed or has not passed; formal collection prohibited")
                    else:
                        stats = dataset_statistics(ROOT / "datasets/m4" / args.robot)
                        passed = all(stats["counts"].get(s,0) >= n and stats["recovery_counts"].get(s,0) >= n
                                     for s,n in config["quotas"].items())
                        result = dict(status="passed" if passed else "incomplete", statistics=stats)
            report["cases"][name] = result
        except Exception as exc:
            report["cases"][name] = dict(status="failed", reason=str(exc), traceback=traceback.format_exc())
            if name == "teacher":
                report["teacher_gate"] = "failed"
        write_json(report_path, report)
    statuses = [c["status"] for c in report["cases"].values()]
    report["status"] = "passed" if all(s=="passed" for s in statuses) else "failed" if "failed" in statuses else "incomplete"
    write_json(report_path, report)
    print(f"M4 {args.robot}: {report['status']}; teacher gate: {report['teacher_gate']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
