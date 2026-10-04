"""M3 checks, directed physical evidence, traces and reproducibility artifacts."""

import argparse
import csv
import hashlib
import gzip
import json
import pickle
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from feedingrobot.envs import FeedingGymEnv
from feedingrobot.scripts.m3_cases import physical_case, PHYSICAL_CASES
from feedingrobot.sim.model import ROOT, load_json
from feedingrobot.scripts.validate_m1 import input_hashes, validate as validate_m1

FULL_CASES = ("full_static", "full_dynamic")
CASES = ("event_logic", "environment", "geometry", "snapshot") + PHYSICAL_CASES + FULL_CASES + ("convergence", "viewer", "m1_regression", "manifest")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")


def pytest_case(robot, name, output):
    tests = {"event_logic": ["test_m3_events.py", "test_m3_acceptance.py"], "environment": ["test_m3_env.py"],
             "geometry": ["test_m3_geometry.py", "test_receiver_boundary.py"]}[name]
    command = [sys.executable, "-m", "pytest", "-q", *[str(ROOT / "tests" / test) for test in tests],
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
        env.reset(seed=17, options={"preset": "beans_on_spoon"})
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
    if 'initial_state' in result:
        (output / 'initial_state.pkl').write_bytes(pickle.dumps(result['initial_state'], protocol=5))
    if result.get('transfer_state') is not None:
        (output / 'transfer_state.pkl').write_bytes(pickle.dumps(result['transfer_state'], protocol=5))
    write_json(output / "metrics.json", {k:v for k,v in result.items() if k not in ('initial_state','transfer_state')})
    with gzip.open(output / "physics.jsonl.gz", "wt") as file:
        for row in trace:
            file.write(json.dumps(row, default=json_value, allow_nan=False) + "\n")
    fields = ("time", "phase", "failure_reason", "supported", "off_bowl", "on_bowl", "mouth_supported",
              "released", "tool_inside", "ready", "bowl_clearance_m", "spoon_support_force_n",
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
    common = sorted(first.keys() & second.keys())
    expected_count = int(np.floor(min(base["time"], other["time"])/.02+1e-8))
    comparisons["common_path_present"] = (len(common) == expected_count and
        np.allclose(common, np.arange(1, expected_count+1)*.02, rtol=0, atol=1e-8))
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
    def run(scenario, seed, folder, **kwargs):
        try:
            result, trace = physical_case(robot, scenario, seed=seed, **kwargs)
            passed = True
        except AssertionError as exc:
            if not hasattr(exc, "metrics"):
                raise
            result, trace, passed = exc.metrics, exc.trace, False
        save_physics(folder, result, trace)
        return result, passed
    for scenario in PHYSICAL_CASES + FULL_CASES:
        for seed in cfg["seeds"]:
            base = baselines.get((scenario, seed))
            base_passed = True
            if base is None:
                base, base_passed = run(scenario, seed, output / scenario / str(seed) / "baseline")
            for name, kwargs in (("half_dt", dict(timestep=.0005)),
                                 ("refined_solver", dict(iterations=200, refined=True))):
                other, other_passed = run(scenario, seed, output / scenario / str(seed) / name,
                                         initial_state=base["initial_state"], **kwargs)
                comparison = compare_runs(base, other, cfg)
                comparison["checks"]["physical_cases_passed"] = base_passed and other_passed
                comparison["passed"] = all(comparison["checks"].values())
                runs.append(dict(scenario=scenario, seed=seed, variant=name,
                                 baseline={k:v for k,v in base.items() if k not in ("initial_state", "transfer_state")},
                                 alternative={k:v for k,v in other.items() if k not in ("initial_state", "transfer_state")},
                                 comparison=comparison))
                write_json(output / "comparisons.json", dict(runs=runs, criteria=cfg, comparisons=len(runs)))
                print(robot, "convergence", scenario, seed, name,
                      "passed" if comparison["passed"] else "failed", flush=True)
    result = dict(runs=runs, criteria=cfg, comparisons=len(runs))
    if not all(r["comparison"]["passed"] for r in runs):
        error = AssertionError("Convergence mismatch; inspect all comparison checks and physical traces")
        error.metrics = result
        raise error
    return result


def viewer_session(robot, output):
    results = []
    for scenario in FULL_CASES:
        result, trace = physical_case(robot, scenario, viewer=True, frame_output=output / scenario)
        assert {"pickup", "delivery", "retracted"}.issubset(result["frames"])
        save_physics(output / scenario, result, trace)
        results.append({k:v for k,v in result.items() if k not in ("initial_state", "transfer_state")})
    return dict(open_sync_close=True, full_flows=results, training_claim=False)


def viewer_case(robot, output):
    command = [sys.executable, "-c",
               "from feedingrobot.scripts.validate_m3 import viewer_session; "
               "import sys; from pathlib import Path; viewer_session(sys.argv[1], Path(sys.argv[2]))",
               robot, str(output / "viewer")]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=300)
    (output / "viewer.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Viewer subprocess exited {result.returncode}: {result.stdout}")
    return dict(open_sync_close=True, full_flows=FULL_CASES, training_claim=False)


def parent_freeze_check():
    path = ROOT / "outputs/single_bean/v1/m1/freeze_manifest.json"
    parent = json.loads(path.read_text())
    assert parent["status"] == "frozen" and parent["model_version"] == "single_bean_native_v1"
    # Task code evolves; assets, initial pose, physics and M1 limits stay frozen.
    fixed = {p:h for p,h in parent["input_sha256"].items()
             if p.startswith(("assets/", "configs/robots/"))
             or p in ("configs/scene.json", "configs/acceptance.json")}
    for file, digest in fixed.items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file
    for file, digest in parent["evidence_sha256"].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == digest, file
    return dict(status="passed", manifest=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                physical_inputs=len(fixed), evidence_files=len(parent["evidence_sha256"]))


def validate(robot, output, selected=None, expected_hashes=None):
    output.mkdir(parents=True, exist_ok=True)
    chosen = set(CASES if selected is None else selected)
    start_hashes = input_hashes()
    parent = parent_freeze_check()
    env = FeedingGymEnv(robot)
    write_json(output / "state_schema.json", env.schema)
    report = dict(schema_version=2, status="incomplete", model_version="single_bean_native_v1",
                  task_version="single_bean_m3_v1", robot_id=robot,
                  scope="Fixed-layout single-bean M3; natural static/dynamic full flow; M4 not verified",
                  stages={"M3":"not_verified", "M4":"not_verified"}, parent_m1=parent,
                  task_config=load_json("configs/task.json"), signature=env.task.state_signature(),
                  input_hashes=start_hashes, cases={name:dict(status="not_verified") for name in CASES})
    env.close()
    write_json(output / "report.json", report)
    baselines = {}
    for name in CASES:
        if name not in chosen:
            continue
        started = time.monotonic()
        try:
            if name in ("event_logic", "environment", "geometry"):
                result = pytest_case(robot, name, output)
            elif name == "snapshot":
                result = snapshot_case(robot, output)
            elif name == "convergence":
                result = convergence_case(robot, output / "convergence", baselines)
            elif name == "viewer":
                result = viewer_case(robot, output)
            elif name == "m1_regression":
                result = validate_m1(robot, output / "m1_regression", expected_hashes=start_hashes)
                assert result["status"] == "passed", "M1 regression failed; inspect m1_regression/m1d_report.json"
            elif name == "manifest":
                assert input_hashes() == start_hashes == (expected_hashes or start_hashes), "Input hash drift"
                result = parent_freeze_check()
            else:
                results = []
                for seed in load_json("configs/acceptance_m3.json")["seeds"]:
                    try:
                        result, trace = physical_case(robot, name, seed=seed)
                        passed = True
                        baselines[name, seed] = result
                    except AssertionError as exc:
                        if not hasattr(exc, "metrics"):
                            raise
                        result, trace, passed = exc.metrics, exc.trace, False
                    save_physics(output / name / str(seed) / "baseline", result, trace)
                    results.append(dict(physical_passed=passed,
                        **{k:v for k,v in result.items() if k not in ("initial_state", "transfer_state")}))
                result = dict(runs=results)
                if not all(r["physical_passed"] for r in results):
                    error = AssertionError("Physical case failed; all seeds retained")
                    error.metrics = result
                    raise error
            report["cases"][name] = dict(status="passed", metrics=result)
        except Exception as exc:
            if hasattr(exc, "trace"):
                save_physics(output / name / "failed", exc.metrics, exc.trace)
            report["cases"][name] = dict(status="failed", error=traceback.format_exc(),
                **(dict(metrics={k:v for k,v in exc.metrics.items() if k not in ("initial_state", "transfer_state")})
                   if hasattr(exc, "metrics") else {}))
        report["cases"][name]["wall_seconds"] = time.monotonic()-started
        print(robot, name, report["cases"][name]["status"], flush=True)
        write_json(output / "report.json", report)
    report["final_input_hashes"] = input_hashes()
    report["hashes_unchanged"] = start_hashes == report["final_input_hashes"] == (expected_hashes or start_hashes)
    if not report["hashes_unchanged"]:
        report["cases"]["manifest"] = dict(status="failed", error="Input hash drift")
    statuses = [c["status"] for c in report["cases"].values()]
    report["status"] = "passed" if all(s == "passed" for s in statuses) else "failed" if "failed" in statuses else "incomplete"
    report["stages"]["M3"] = report["status"]
    write_json(output / "report.json", report)
    return report


def publish_freeze(reports, output, expected):
    assert set(reports) == {"panda", "ur5e"}
    assert input_hashes() == expected
    parent = parent_freeze_check()
    for report in reports.values():
        assert report["status"] == "passed" and report["hashes_unchanged"]
        assert report["input_hashes"] == report["final_input_hashes"] == expected
        assert all(report["cases"][n]["status"] == "passed" for n in CASES)
        assert report["stages"]["M4"] == "not_verified"
    files = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for robot in reports for p in sorted((output/robot).rglob('*')) if p.is_file()}
    manifest = dict(schema_version=1, status="frozen", model_version="single_bean_native_v1",
                    task_version="single_bean_m3_v1", parent_m1=parent, input_sha256=expected,
                    evidence_sha256=files, acceptance=load_json("configs/acceptance_m3.json"),
                    reports={r:str((output/r/'report.json').relative_to(ROOT)) for r in reports},
                    stages={"M1-A":"passed", "M1-B":"passed", "M1-C":"passed", "M1-D":"passed",
                            "M3":"passed", "M4":"not_verified"})
    assert input_hashes() == expected
    write_json(output/'freeze_manifest.json', manifest)
    for file, digest in files.items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == digest, file
    write_json(output/'freeze_audit.json', dict(status="passed", input_files=len(expected),
               verified_evidence_files=len(files), parent_m1=parent, stages=manifest["stages"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["panda", "ur5e", "all"], default="panda")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = ROOT / (args.output or "outputs/single_bean/v1/m3")
    expected = input_hashes()
    robots = ("panda", "ur5e") if args.robot == "all" else (args.robot,)
    reports = {r:validate(r, output/r, args.cases, expected) for r in robots}
    if all(r["status"] == "passed" for r in reports.values()):
        if args.robot == "all" and args.cases is None:
            publish_freeze(reports, output, expected)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
