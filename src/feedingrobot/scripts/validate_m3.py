"""M3 checks, directed physical evidence, traces and reproducibility artifacts."""

import argparse
import csv
import hashlib
import gzip
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from feedingrobot.envs import FeedingGymEnv
from feedingrobot.scripts.m3_cases import physical_case, PHYSICAL_CASES
from feedingrobot.sim.model import ROOT, asset_files, load_json

CASES = ("event_logic", "environment", "snapshot") + PHYSICAL_CASES + ("convergence", "viewer")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")


def pytest_case(robot, name, output):
    test = "test_m3_events.py" if name == "event_logic" else "test_m3_env.py"
    command = [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / test),
               f"--junitxml={output / (name + '.xml')}"]
    if name == "environment":
        command += ["-k", "not " + ("ur5e" if robot == "panda" else "panda")]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / (name + ".log")).write_text(result.stdout)
    if result.returncode:
        raise AssertionError(result.stdout)
    suite = ET.parse(output / (name + ".xml")).getroot().find("testsuite")
    return dict(command=command, tests=int(suite.get("tests")), failures=int(suite.get("failures")),
                evidence="synthetic criteria" if name == "event_logic" else "Gym/physics interface and replay tests")


def snapshot_case(robot, output):
    env = FeedingGymEnv(robot)
    try:
        env.reset(seed=17, options={"preset": "food_on_spoon"})
        for _ in range(8):
            env.step(np.array([.05, 0, 0, 0, 0, 0]))
        # Trusted local binary artifact, including all numpy and generator state.
        path = output / "snapshot.pkl"
        path.write_bytes(pickle.dumps(env.get_state(), protocol=5))
        expected = [env.step(np.array([0, .05, 0, 0, 0, 0])) for _ in range(5)]
        env.set_state(pickle.loads(path.read_bytes()))
        actual = [env.step(np.array([0, .05, 0, 0, 0, 0])) for _ in range(5)]
        max_error = 0.
        for a, b in zip(expected, actual):
            max_error = max(max_error, float(np.max(np.abs(a[0] - b[0]))))
            np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-7)
            np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-12)
            assert a[2:4] == b[2:4]
            for key in ("time", "phase", "events", "reward_terms", "failure_reason"):
                assert a[4][key] == b[4][key]
        return dict(steps=5, max_observation_error=max_error, file=path.name,
                    signature=env.task.state_signature())
    finally:
        env.close()


def save_physics(output, result, trace):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    with gzip.open(output / "physics.jsonl.gz", "wt") as file:
        for row in trace:
            file.write(json.dumps(row, default=json_value, allow_nan=False) + "\n")
    fields = ("time", "phase", "failure_reason", "supported", "off_plate", "on_plate", "mouth_supported",
              "released", "tool_inside", "ready", "plate_height_m", "spoon_support_force_n",
              "contact_peak_n", "contact_impulse_ns", "minimum_contact_distance_m")
    with (output / "trajectory.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields + ("tcp_x", "tcp_y", "tcp_z"))
        writer.writeheader()
        for row in trace:
            writer.writerow(dict({k: row[k] for k in fields}, **dict(zip(
                ("tcp_x", "tcp_y", "tcp_z"), row["tcp_position"]))))


def compare_runs(base, other, cfg):
    event_key = lambda e: (e["name"], e.get("previous"), e.get("phase"), e.get("reason"))
    same_events = [event_key(e) for e in base["events"]] == [event_key(e) for e in other["events"]]
    event_error = (max((abs(a["time"] - b["time"]) for a, b in zip(base["events"], other["events"])), default=0.)
                   if same_events else None)
    comparisons = dict(outcome=(base["success"], base["failure_reason"], base["phase"])
                               == (other["success"], other["failure_reason"], other["phase"]),
                       event_sequence=same_events,
                       event_time=same_events and event_error <= cfg["event_time_tolerance_s"],
                       end_time=abs(base["time"] - other["time"]) <= cfg["event_time_tolerance_s"],
                       tcp_position=np.linalg.norm(np.array(base["tcp_position"]) - other["tcp_position"])
                                    <= cfg["tcp_position_tolerance_m"])
    errors = dict(event_time_s=event_error,
                  tcp_position_m=float(np.linalg.norm(np.array(base["tcp_position"]) - other["tcp_position"])))
    first = {round(s["time"], 8): np.array(s["position"]) for s in base["tcp_samples"]}
    second = {round(s["time"], 8): np.array(s["position"]) for s in other["tcp_samples"]}
    path_error = max((float(np.linalg.norm(first[t] - second[t])) for t in first.keys() & second.keys()), default=0.)
    comparisons["tcp_path"] = path_error <= cfg["tcp_position_tolerance_m"]
    errors["tcp_path_m"] = path_error
    if same_events:
        event_position_error = max((float(np.linalg.norm(np.array(a) - b)) for a, b in
                                    zip(base["event_tcp_positions"], other["event_tcp_positions"])), default=0.)
        comparisons["event_tcp_position"] = event_position_error <= cfg["tcp_position_tolerance_m"]
        errors["event_tcp_position_m"] = event_position_error
    for metric, absolute, relative in (("peak_force_n", "force_absolute_tolerance_n", "force_relative_tolerance"),
                                       ("impulse_ns", "impulse_absolute_tolerance_ns", "impulse_relative_tolerance")):
        error = abs(base[metric] - other[metric])
        comparisons[metric] = error <= max(cfg[absolute], cfg[relative] * base[metric])
        errors[metric] = error
    for metric, absolute, relative in (("contact_pair_peaks_n", "force_absolute_tolerance_n", "force_relative_tolerance"),
                                       ("contact_pair_impulses_ns", "impulse_absolute_tolerance_ns", "impulse_relative_tolerance")):
        for pair in base[metric].keys() | other[metric].keys():
            first, second = base[metric].get(pair, 0.), other[metric].get(pair, 0.)
            key = metric + ":" + pair
            comparisons[key] = abs(first - second) <= max(cfg[absolute], cfg[relative] * first)
            errors[key] = abs(first - second)
    return dict(passed=all(comparisons.values()), checks=comparisons, errors=errors)


def convergence_case(robot, output, baselines=None):
    cfg = load_json("configs/acceptance_m3.json")
    runs, baselines = [], baselines or {}
    for scenario in PHYSICAL_CASES:
        for seed in cfg["seeds"]:
            base = baselines.get((scenario, seed))
            if base is None:
                base, trace = physical_case(robot, scenario, seed=seed)
                save_physics(output / scenario / str(seed) / "baseline", base, trace)
            for name, kwargs in (("half_dt", dict(timestep=.0005)),
                                 ("refined_solver", dict(iterations=200, refined=True))):
                other, trace = physical_case(robot, scenario, seed=seed, **kwargs)
                save_physics(output / scenario / str(seed) / name, other, trace)
                runs.append(dict(scenario=scenario, seed=seed, variant=name,
                                 baseline=base, alternative=other, comparison=compare_runs(base, other, cfg)))
                print(robot, "convergence", scenario, seed, name,
                      "passed" if runs[-1]["comparison"]["passed"] else "failed", flush=True)
    result = dict(runs=runs, criteria=cfg, comparisons=len(runs))
    if not all(r["comparison"]["passed"] for r in runs):
        error = AssertionError("Convergence mismatch; inspect comparison checks and physical traces")
        error.metrics = result
        raise error
    return result


def viewer_session(robot, output=None):
    results = []
    for scenario in ("entry", "receiver"):
        result, trace = physical_case(robot, scenario, viewer=True,
                                     frame_output=output / scenario if output is not None else None)
        if output is not None:
            assert ("entry" in result["frames"] if scenario == "entry" else
                    {"delivery", "retracted"}.issubset(result["frames"]))
            save_physics(output / scenario, result, trace)
        results.append(result)
    return dict(open_sync_close=True, directed_cases=results, training_claim=False)


def viewer_case(robot, output):
    # GLFW may terminate the process rather than raising. Keep report ownership
    # in the parent so an unavailable desktop never discards other case results.
    command = [sys.executable, "-c",
               "from feedingrobot.scripts.validate_m3 import viewer_session; "
               "import sys; from pathlib import Path; viewer_session(sys.argv[1], Path(sys.argv[2]))", robot, str(output / "viewer")]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60)
    (output / "viewer.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Viewer subprocess exited {result.returncode}: {result.stdout}")
    return dict(open_sync_close=True, directed_cases=["entry", "receiver"], training_claim=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--cases", nargs="*", choices=CASES)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = ROOT / (args.output or f"outputs/new_tableware/v2/m3/{args.robot}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = sorted(set(list((ROOT / "src/feedingrobot").rglob("*.py"))
                        + list((ROOT / "tests").glob("*.py")) + list((ROOT / "configs").rglob("*.json"))
                        + asset_files(args.robot)
                        + [ROOT / f"assets/robots/{args.robot}.xml", ROOT / "requirements.lock.txt",
                           ROOT / "third_party_manifest.json", ROOT / "pyproject.toml", ROOT / "environment.yml"]))
    env = FeedingGymEnv(args.robot)
    write_json(output / "state_schema.json", env.schema)
    report = dict(status="incomplete", model_version="new_tableware_v2", robot_id=args.robot,
                  scope="M3 only; directed fixtures are not full feeding/teacher evidence",
                  task_config=load_json("configs/task.json"), signature=env.task.state_signature(),
                  input_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  cases={name: dict(status="not_verified") for name in CASES})
    env.close()
    baselines = {}
    acceptance = load_json("configs/acceptance_m3.json")
    for name in CASES:
        if args.cases is not None and name not in args.cases:
            continue
        if name == "viewer" and not os.environ.get("DISPLAY"):
            report["cases"][name]["reason"] = "DISPLAY unavailable; run in desktop session"
            continue
        started = time.monotonic()
        try:
            if name in ("event_logic", "environment"):
                result = pytest_case(args.robot, name, output)
            elif name == "snapshot":
                result = snapshot_case(args.robot, output)
            elif name == "convergence":
                result = convergence_case(args.robot, output, baselines)
            elif name == "viewer":
                result = viewer_case(args.robot, output)
            else:
                results = []
                for seed in acceptance["seeds"]:
                    result, trace = physical_case(args.robot, name, seed=seed)
                    save_physics(output / name / str(seed) / "baseline", result, trace)
                    baselines[name, seed] = result
                    results.append(result)
                result = dict(runs=results, seeds=acceptance["seeds"])
            report["cases"][name] = dict(status="passed", metrics=result)
        except Exception as exc:
            if hasattr(exc, "trace"):
                save_physics(output / name / "failed", exc.metrics, exc.trace)
            report["cases"][name] = dict(status="failed", error=traceback.format_exc(),
                                        **(dict(metrics=exc.metrics) if hasattr(exc, "metrics") else {}))
        report["cases"][name]["wall_seconds"] = time.monotonic() - started
        print(args.robot, name, report["cases"][name]["status"], flush=True)
        write_json(output / "report.json", report)
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"].values()) else "incomplete"
    write_json(output / "report.json", report)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
