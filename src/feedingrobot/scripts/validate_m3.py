"""M3 checks, directed physical evidence, traces and reproducibility artifacts."""

import argparse
import csv
import hashlib
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
from feedingrobot.scripts.m3_cases import physical_case
from feedingrobot.sim.model import ROOT, asset_files, load_json

CASES = ("event_logic", "environment", "snapshot", "plate", "carry", "receiver", "unsupported", "force",
         "penetration", "convergence", "viewer")


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


def convergence_case(robot, output):
    cfg = load_json("configs/acceptance_m3.json")
    runs = []
    for scenario in ("plate", "carry", "receiver", "unsupported", "force", "penetration"):
        base, _ = physical_case(robot, scenario, .001)
        fine, trace = physical_case(robot, scenario, .0005)
        write_json(output / f"{scenario}_fine_trace.json", trace)
        assert (base["success"], base["failure_reason"], base["phase"]) == (fine["success"], fine["failure_reason"], fine["phase"])
        assert abs(base["time"] - fine["time"]) <= cfg["event_time_tolerance_s"]
        assert abs(base["peak_force_n"] - fine["peak_force_n"]) <= max(
            cfg["force_absolute_tolerance_n"], cfg["force_relative_tolerance"] * base["peak_force_n"])
        assert abs(base["impulse_ns"] - fine["impulse_ns"]) <= max(
            cfg["impulse_absolute_tolerance_ns"], cfg["impulse_relative_tolerance"] * base["impulse_ns"])
        np.testing.assert_allclose(base["tcp_position"], fine["tcp_position"], rtol=0,
                                   atol=cfg["tcp_position_tolerance_m"])
        runs.append(dict(scenario=scenario, baseline=base, half_dt=fine))
    return dict(runs=runs, criteria=cfg)


def viewer_session(robot):
    env = FeedingGymEnv(robot, render_mode="human")
    try:
        env.reset(seed=0)
        for _ in range(5):
            env.step(np.zeros(6))
            time.sleep(.02)
        assert env.viewer.is_running(), "Viewer closed before smoke test completed"
        return dict(open_sync_close=True, training_claim=False)
    finally:
        env.close()


def viewer_case(robot, output):
    # GLFW may terminate the process rather than raising. Keep report ownership
    # in the parent so an unavailable desktop never discards other case results.
    command = [sys.executable, "-c",
               "from feedingrobot.scripts.validate_m3 import viewer_session; "
               "import sys; viewer_session(sys.argv[1])", robot]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30)
    (output / "viewer.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Viewer subprocess exited {result.returncode}: {result.stdout}")
    return dict(open_sync_close=True, training_claim=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--cases", nargs="*", choices=CASES)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = ROOT / (args.output or f"outputs/m3/{args.robot}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = sorted(set(list((ROOT / "src/feedingrobot").rglob("*.py"))
                        + list((ROOT / "tests").glob("*.py")) + list((ROOT / "configs").rglob("*.json"))
                        + asset_files(args.robot)
                        + [ROOT / f"assets/robots/{args.robot}.xml", ROOT / "requirements.lock.txt",
                           ROOT / "third_party_manifest.json", ROOT / "pyproject.toml", ROOT / "environment.yml"]))
    env = FeedingGymEnv(args.robot)
    write_json(output / "state_schema.json", env.schema)
    report = dict(status="incomplete", robot_id=args.robot, scope="M3 only; directed fixtures are not full feeding/teacher evidence",
                  task_config=load_json("configs/task.json"), signature=env.task.state_signature(),
                  input_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  cases={name: dict(status="not_verified") for name in CASES})
    env.close()
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
                result = convergence_case(args.robot, output)
            elif name == "viewer":
                result = viewer_case(args.robot, output)
            else:
                result, trace = physical_case(args.robot, name)
                write_json(output / f"{name}_trace.json", trace)
                with (output / f"{name}.csv").open("w") as file:
                    writer = csv.DictWriter(file, fieldnames=[k for k in trace[0] if k != "events"])
                    writer.writeheader()
                    writer.writerows({k: v for k, v in row.items() if k != "events"} for row in trace)
            report["cases"][name] = dict(status="passed", metrics=result)
        except Exception:
            report["cases"][name] = dict(status="failed", error=traceback.format_exc())
        report["cases"][name]["wall_seconds"] = time.monotonic() - started
        print(args.robot, name, report["cases"][name]["status"], flush=True)
        write_json(output / "report.json", report)
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"].values()) else "incomplete"
    write_json(output / "report.json", report)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
