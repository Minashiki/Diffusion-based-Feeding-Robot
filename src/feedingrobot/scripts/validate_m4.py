"""Frozen new-tableware teacher, physical replay, convergence and data gates."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import multiprocessing
import os
import subprocess
import sys
import traceback

import numpy as np

from feedingrobot.data.episodes import input_hashes, load_episode, write_json
from feedingrobot.data.recipes import recipe
from feedingrobot.data.replay import replay_episode
from feedingrobot.data.rollout import run_episode
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.experts.gate import PRECOLLECTION_CASES, local_teacher_passed, matching_teachers_passed
from feedingrobot.experts.feasibility import check_waypoints
from feedingrobot.scripts.collect import dataset_statistics
from feedingrobot.sim.model import ROOT, load_json

CASES = PRECOLLECTION_CASES + ("dataset",)


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


def saved_or_run(robot, config, path, seed, scenario, parameters, group, *, split="acceptance", **kwargs):
    if path.exists():
        result, _ = load_episode(path)
        if result["input_hashes"] != input_hashes() or result["teacher_config"] != config:
            raise ValueError("Directory contains a different frozen teacher/source version")
        return result
    return run_episode(robot, seed, config, path, scenario=scenario, teacher_parameters=parameters,
                       split=split, group_id=group, **kwargs)


def acceptance_trial(robot, config, output, index, recover=False, viewer=False):
    seed, scenario, parameters, group = recipe(config, "acceptance", index, recover=recover)
    path = output / "episodes" / f"{'recovery' if recover else 'normal'}_{seed}"
    result = saved_or_run(robot, config, path, seed, scenario, parameters, group, viewer=viewer)
    complete = physical_success(result)
    recovered = any(s["recovery_valid"] for s in result["segments"])
    summary = {key: result[key] for key in ("seed", "success", "failure_reason", "truncated", "time_s",
                                           "contact_peak_n", "contact_impulse_ns", "wrist_peak_n")}
    summary.update(success=bool(complete and (not recover or recovered)), recovery_completed=recovered,
                   episode=str(path), milestones=[e["name"] for e in result["events"]
                                                 if e["name"] in ("pickup", "delivery", "success")])
    return summary


def baseline(robot, config, output, trials, workers=1, *, recover=False, viewer=False):
    completed = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(acceptance_trial, robot, config, output, i, recover, viewer and i == 0): i
                   for i in range(trials)}
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            completed[index] = result
            write_json(output / ("recovery_progress.json" if recover else "baseline_progress.json"),
                       [completed[k] for k in sorted(completed)])
            print(f"{robot} {'recovery' if recover else 'normal'} {len(completed)}/{trials}: "
                  f"seed={result['seed']}, success={result['success']}, failure={result['failure_reason']}", flush=True)
    results = [completed[k] for k in sorted(completed)]
    successes = sum(r["success"] for r in results)
    required = (10 if recover else 100) if robot == "panda" else 5
    threshold = 95 if robot == "panda" and not recover else required
    passed = trials == required and successes >= threshold
    return dict(status="passed" if passed else "failed" if trials == required else "incomplete", attempts=trials,
                successes=successes, success_rate=successes/trials, threshold=f"{threshold}/{required}", trials=results,
                failure_counts=dict(Counter(r["failure_reason"] or "time_limit" for r in results if not r["success"])))


def convergence_cases(config):
    # Named before any results are observed, never replace failed cases.
    return [dict(index=i, recover=recover, seed=recipe(config, "acceptance", i, recover=recover)[0])
            for recover in (False, True) for i in range(5)]


def calibration(config, output, *, viewer=False):
    """Calibrate only on training-fold scenes, stop on the first failed step."""
    results = []
    for mode, count in (("fixed",3),("static",10),("dynamic",10),("recovery",10)):
        for index in range(count):
            if mode == "fixed":
                seed, parameters, group = 0, {}, "fixed-calibration"
                scenario = dict(config["scene"], food_offset_m=[0.,0.], head_freq_hz=0., head_phase_rad=0.)
            else:
                seed, scenario, parameters, group = recipe(config,"calibration",index,recover=mode=="recovery")
                if mode == "static":
                    scenario.update(head_freq_hz=0.,head_phase_rad=0.)
            path = output/"calibration"/f"{mode}_{index}"
            result = saved_or_run("panda",config,path,seed,scenario,parameters,group,
                                  split="calibration",viewer=viewer and index==0)
            complete = physical_success(result)
            if mode == "recovery":
                complete &= any(s["recovery_valid"] for s in result["segments"])
            milestones = [e["name"] for e in result["events"] if e["name"] in ("pickup","delivery","success")]
            results.append(dict(mode=mode,index=index,episode=str(path),success=bool(complete),
                                milestones=milestones,failure_reason=result["failure_reason"]))
            write_json(output/"calibration_progress.json",results)
            if not complete:
                failed_step = ("pickup" if "pickup" not in milestones else "transport"
                               if not any(e.get("phase")=="WAIT_READY" for e in result["events"])
                               else "delivery_retract_recovery")
                return dict(status="failed",failed_step=failed_step,trials=results,
                            reason="Calibration failed; later steps and independent acceptance are prohibited")
    return dict(status="passed",trials=results,fixed_repeats=3,independent_static=10,
                dynamic=10,recovery=10)


def convergence(robot, config, output):
    tolerances = load_json("configs/acceptance_m3.json")
    results = []
    for item in convergence_cases(config):
        seed, scenario, parameters, group = recipe(config, "acceptance", item["index"], recover=item["recover"])
        path = output / "episodes" / f"{'recovery' if item['recover'] else 'normal'}_{seed}"
        original, original_arrays = load_episode(path)
        assert physical_success(original), "A preassigned convergence baseline failed; cannot substitute another seed"
        base_iterations = original["solver_iterations"]
        base_tolerance = original["solver_tolerance"]
        for name, dt, iterations, tolerance in (("half_dt", original["dt"]/2, base_iterations, base_tolerance),
                ("refined_solver", original["dt"], base_iterations*2, base_tolerance/10)):
            fine_path = output / "convergence" / f"{seed}_{name}"
            fine = saved_or_run(robot, config, fine_path, seed, scenario, parameters, group,
                               timestep=dt, iterations=iterations, solver_tolerance=tolerance)
            assert physical_success(fine), fine["failure_reason"]
            names = ("pickup", "delivery", "success", "phase")
            old_events = [e for e in original["events"] if e["name"] in names]
            new_events = [e for e in fine["events"] if e["name"] in names]
            assert [{k:v for k,v in e.items() if k!='time'} for e in old_events] == [
                {k:v for k,v in e.items() if k!='time'} for e in new_events], "Event sequence differs"
            event_error = max(abs(a["time"]-b["time"]) for a,b in zip(old_events,new_events))
            assert event_error <= tolerances["event_time_tolerance_s"], event_error
            for key, absolute, relative in (("contact_peak_n", "force_absolute_tolerance_n", "force_relative_tolerance"),
                                            ("wrist_peak_n", "force_absolute_tolerance_n", "force_relative_tolerance"),
                                            ("contact_impulse_ns", "impulse_absolute_tolerance_ns", "impulse_relative_tolerance")):
                assert abs(original[key]-fine[key]) <= max(tolerances[absolute], tolerances[relative]*original[key]), key
            tcp_error = max(float(np.linalg.norm(np.asarray(a["tcp_position"])-b["tcp_position"]))
                            for a,b in zip(original["evidence"],fine["evidence"]))
            assert tcp_error <= tolerances["tcp_position_tolerance_m"], tcp_error
            for field, absolute, relative in (("contact_group_peaks_n", "force_absolute_tolerance_n", "force_relative_tolerance"),
                                             ("contact_group_impulses_ns", "impulse_absolute_tolerance_ns", "impulse_relative_tolerance")):
                for group in original[field].keys() | fine[field].keys():
                    old, new = original[field].get(group,0.), fine[field].get(group,0.)
                    assert abs(old-new) <= max(tolerances[absolute], tolerances[relative]*old), (field,group,old,new)
            _, fine_arrays = load_episode(fine_path)
            old_ticks, new_ticks = original_arrays["observation_ticks"], fine_arrays["observation_ticks"]
            old_grid = old_ticks % round(.02/original["dt"]) == 0
            new_grid = new_ticks % round(.02/fine["dt"]) == 0
            # TCP follows q/dq in the unchanged Gym observation schema.
            offset = 2*original["observation_schema"]["fields"][0][1]
            old_path = original_arrays["observations"][old_grid, offset:offset+3]
            new_path = fine_arrays["observations"][new_grid, offset:offset+3]
            length = min(len(old_path),len(new_path))
            path_error = float(np.max(np.linalg.norm(old_path[:length]-new_path[:length],axis=1)))
            assert path_error <= tolerances["tcp_position_tolerance_m"], path_error
            if item["recover"]:
                assert any(s["recovery_valid"] for s in fine["segments"])
            results.append(dict(seed=seed, recover=item["recover"], variant=name, event_error_s=event_error,
                                tcp_error_m=tcp_error, path_error_m=path_error, iterations=iterations, solver_tolerance=tolerance))
    return dict(status="passed", trials=results, criteria=tolerances)


def run_logged(command, path, timeout=None):
    with path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise AssertionError(f"Command exited {result.returncode}; see {path}")
    return dict(command=command, log=str(path))


def update_gate(report, companion):
    passed = matching_teachers_passed(report, companion)
    report["teacher_gate"] = "passed" if passed else "not_verified"
    if passed:
        report["compatibility"] = {k: companion[k] for k in
            ("robot_id", "cases", "baseline", "recovery_baseline", "input_hashes", "teacher_config")}
    else:
        report.pop("compatibility", None)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--config", default="configs/collect.json")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--trials", type=int, help="Partial checks cannot release the teacher")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--companion-report")
    parser.add_argument("--dataset")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.trials is not None and args.trials < 1):
        parser.error("--trials and --workers must be positive")
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / (args.output or f"outputs/new_tableware/v3/m4/{args.robot}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    companion_path = ROOT / args.companion_report if args.companion_report else output.parent / (
        "ur5e" if args.robot == "panda" else "panda") / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if report and (report.get("input_hashes") != input_hashes() or report.get("teacher_config") != config):
        raise ValueError("Report belongs to different source/teacher; use a new --output")
    report.update(robot_id=args.robot, teacher_config=config, input_hashes=input_hashes(),
                  scope="P0 new-tableware teacher/data; no learned policies", status="incomplete", teacher_gate="not_verified")
    report.setdefault("cases", {name: dict(status="not_verified") for name in CASES})
    report["convergence_cases"] = convergence_cases(config)
    selected = args.cases or CASES
    for name in selected:
        write_json(report_path, report)
        try:
            if name == "feasibility":
                env = FeedingGymEnv(args.robot)
                try:
                    env.reset(seed=0, options={"scenario": config["scene"]})
                    task = env.task
                    teacher = Teacher(task.robot_config, config)
                    teacher.reset({}, geometry=teacher_geometry(task))
                    obs = task.provider.observe()["policy_obs"]
                    teacher.act(obs)
                    r, points = teacher.acquisition_waypoints(obs)
                    mp = obs["tcp_position"]+obs["mouth_relative_world"]
                    mr = obs["mouth_rotation"]
                    before = task.get_state()["physics"]
                    checks = check_waypoints(task, [(f"acquire_{i}",point,r) for i,point in enumerate(points)] + [
                        ("wait",mp-mr[:,0]*task.task_config["wait_offset_m"],mr),
                        ("insert",mp+mr@np.array([config["teacher"]["insert_depth_m"],0.,config["teacher"]["insert_height_m"]]),mr)])
                    np.testing.assert_array_equal(before, task.get_state()["physics"])
                    result = dict(status="passed" if all(c["reachable"] for c in checks) else "failed", checks=checks)
                finally:
                    env.close()
            elif name == "calibration":
                if report["cases"]["feasibility"]["status"] != "passed":
                    result = dict(status="not_verified",reason="Geometry waypoints must be feasible first")
                elif args.robot == "ur5e":
                    result = dict(status="passed",scope="Compatibility checked by five normal and five recovery full episodes")
                else:
                    result = calibration(config,output,viewer=not args.headless)
            elif name in ("teacher", "recovery"):
                if any(report["cases"][k]["status"] != "passed" for k in ("feasibility","calibration")):
                    report["cases"][name] = dict(status="not_verified",reason="Calibration has not passed")
                    continue
                recover = name == "recovery"
                trials = args.trials or ((10 if recover else 100) if args.robot == "panda" else 5)
                result = baseline(args.robot, config, output, trials, args.workers,
                                  recover=recover, viewer=not args.headless)
                report["recovery_baseline" if recover else "baseline"] = result
            elif name == "convergence":
                if all(report["cases"][k]["status"] == "passed" for k in ("teacher", "recovery")):
                    result = convergence(args.robot, config, output)
                else:
                    result = dict(status="not_verified", reason="Normal and recovery baselines must pass")
            elif name == "replay":
                paths = (sorted((output / "episodes").glob("*/manifest.json"))
                         + sorted((output / "convergence").glob("*/manifest.json"))
                         + sorted((output / "calibration").glob("*/manifest.json")))
                expected = sum(report.get(k, {}).get("attempts", 0) for k in ("baseline", "recovery_baseline"))
                if not expected or len(list((output/"episodes").glob("*/manifest.json"))) != expected:
                    raise ValueError("Incomplete baseline episode coverage")
                results = []
                for path in paths:
                    results.append(replay_episode(path.parent))
                    write_json(output / "replay_progress.json", results)
                result = dict(status="passed", episodes=results)
            elif name == "viewer":
                if not os.environ.get("DISPLAY"):
                    result = dict(status="not_verified", reason="DISPLAY unavailable")
                else:
                    checks = []
                    for recover in (False, True):
                        successful = [p.parent for p in (output / "episodes").glob("*/manifest.json")
                                      if (m:=json.loads(p.read_text()))["success"] and bool(m["scenario"].get("recover"))==recover]
                        if not successful:
                            raise ValueError("Viewer requires a successful normal and recovery episode")
                        command = [sys.executable, "-m", "feedingrobot.scripts.replay", str(successful[0]), "--viewer"]
                        checks.append(run_logged(command, output / f"viewer_{int(recover)}.log", timeout=300))
                    result = dict(status="passed", checks=checks)
            elif name == "regressions":
                commands = [[sys.executable,"-m","pytest","-q",f"--junitxml={output/'tests.xml'}"],
                    [sys.executable,"-m","feedingrobot.scripts.validate_m1","--robot",args.robot,"--output",str(output/'m1_regression')],
                    [sys.executable,"-m","feedingrobot.scripts.validate_m3","--robot",args.robot,"--output",str(output/'m3_regression')]]
                result = dict(status="passed", checks=[run_logged(c,output/f"regression_{i}.log") for i,c in enumerate(commands)])
            else:
                companion = json.loads(companion_path.read_text()) if companion_path.exists() else {}
                if args.robot == "ur5e":
                    result = dict(status="passed", scope="Compatibility only; no UR5e formal data")
                elif not matching_teachers_passed(report, companion):
                    result = dict(status="not_verified", reason="Both teachers require all precollection checks")
                else:
                    directory = ROOT / (args.dataset or "datasets/new_tableware/v3/m4/panda")
                    stats = dataset_statistics(directory, config=config, robot=args.robot, replay=True)
                    passed = stats["replay_status"] == "passed" and all(stats["counts"].get(s,0)>=n
                        and stats["recovery_counts"].get(s,0)>=n for s,n in config["quotas"].items())
                    result = dict(status="passed" if passed else "incomplete", statistics=stats)
            report["cases"][name] = result
        except Exception as exc:
            report["cases"][name] = dict(status="failed", reason=str(exc), traceback=traceback.format_exc())
        write_json(report_path, report)
    companion = json.loads(companion_path.read_text()) if companion_path.exists() else {}
    if update_gate(report, companion):
        update_gate(companion, report)
        companion["status"] = "passed" if all(c["status"] == "passed" for c in companion["cases"].values()) else "incomplete"
        write_json(companion_path, companion)
    statuses = [c["status"] for c in report["cases"].values()]
    report["status"] = ("passed" if report["teacher_gate"] == "passed" and all(s=="passed" for s in statuses)
                        else "failed" if "failed" in statuses else "incomplete")
    write_json(report_path, report)
    print(f"M4 {args.robot}: {report['status']}; teacher gate: {report['teacher_gate']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
