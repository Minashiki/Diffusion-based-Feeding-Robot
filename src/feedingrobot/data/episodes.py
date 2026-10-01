"""Versioned mmap episode logs. Binary snapshots are trusted-local artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np

from feedingrobot.sim.model import ROOT, asset_files


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")
    temporary.replace(path)


def input_hashes(config_path="configs/collect.json"):
    paths = sorted(set(list((ROOT / "src/feedingrobot").rglob("*.py"))
                       + list((ROOT / "configs").rglob("*.json"))
                       + asset_files()
                       + [ROOT / "requirements.lock.txt", ROOT / config_path]))
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def physics_row(task, state, fields):
    if "q" not in state:
        state = task.snapshot()
    values = dict(time=state["time"], reference_tcp_wxyz_xyz=task.adapter.target.wxyz_xyz,
                  reference_q=task.adapter.reference.q[task.index.qpos], shaped_twist_base=task.adapter.velocity,
                  contact_peak_n=task.substep_contact_peak_n, wrist_peak_n=task.substep_wrist_peak_n,
                  contact_impulse_ns=task.monitor.impulse_ns, contact_over_limit_s=task.monitor.over_limit_s,
                  measured_tcp_twist_world=state["tcp_twist_world"], **{key: state[key] for key in
                  ("q", "dq", "raw_wrench_sensor", "wrench_world_at_tcp", "compensated_wrench")})
    return np.concatenate([np.asarray(values[key]).reshape(-1) for key, _ in fields])


class EpisodeWriter:
    def __init__(self, directory, schema, dt, max_s):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        n = schema["fields"][0][1]
        self.fields = [("time", 1), ("reference_tcp_wxyz_xyz", 7), ("reference_q", n),
                       ("shaped_twist_base", 6), ("measured_tcp_twist_world", 6),
                       ("q", n), ("dq", n), ("raw_wrench_sensor", 6),
                       ("wrench_world_at_tcp", 6), ("compensated_wrench", 6),
                       ("contact_peak_n", 1), ("wrist_peak_n", 1),
                       ("contact_impulse_ns", 1), ("contact_over_limit_s", 1)]
        self.physics = np.lib.format.open_memmap(self.directory / "physics.npy", mode="w+", dtype="float64",
                                                shape=(int(round(max_s / dt)) + 2, sum(s for _, s in self.fields)))
        self.physics_rows = 0
        self.obs = np.lib.format.open_memmap(self.directory / "observations.npy", mode="w+", dtype="float32",
                                            shape=(int(round(max_s / .02)) + 3, sum(s for _, s, _ in schema["fields"])))
        self.obs_rows = 0
        self.obs_ticks, self.obs_phases = [], []
        self.obs_valid = []
        self.commands, self.actions = [], []
        self.dt = dt
        self.schema = schema
        write_json(self.directory / "manifest.json", dict(schema_version=1, status="recording", robot_id=schema["robot_id"]))

    def record_physics(self, task, state):
        self.physics[self.physics_rows] = physics_row(task, state, self.fields)
        self.physics_rows += 1

    def record_observation(self, tick, phase, observation, valid=True):
        self.obs[self.obs_rows] = observation
        self.obs_rows += 1
        self.obs_ticks.append(tick)
        self.obs_phases.append(phase)
        self.obs_valid.append(valid)

    def command(self, tick, twist=None, valid_until=None, *, hold_reference=False):
        # Stop is a distinct operation, not an artificial zero action label.
        self.commands.append(dict(tick=tick, kind="stop" if twist is None else "twist",
                                  twist=None if twist is None else np.asarray(twist).tolist(),
                                  valid_until=valid_until))
        if hold_reference:
            self.commands[-1]["hold_reference"] = True

    def action(self, tick, phase, proposal, command, observation, end_tick):
        self.actions.append(dict(tick=tick, phase=phase, proposal=np.asarray(proposal).copy(),
                                 command=np.asarray(command).copy(), observation=np.asarray(observation).copy(),
                                 end_tick=end_tick, valid=True))

    def interrupt(self, tick):
        if self.actions and tick < self.actions[-1]["end_tick"]:
            self.actions[-1]["end_tick"] = tick
            self.actions[-1]["valid"] = False

    def finish(self, metadata):
        self.physics.flush()
        self.obs.flush()
        np.save(self.directory / "observation_ticks.npy", np.asarray(self.obs_ticks, dtype=np.int64))
        np.save(self.directory / "observation_phases.npy", np.asarray(self.obs_phases, dtype=np.int8))
        np.save(self.directory / "observation_valid.npy", np.asarray(self.obs_valid, dtype=bool))
        np.save(self.directory / "action_ticks.npy", np.asarray([a["tick"] for a in self.actions], dtype=np.int64))
        np.save(self.directory / "action_end_ticks.npy", np.asarray([a["end_tick"] for a in self.actions], dtype=np.int64))
        np.save(self.directory / "action_phases.npy", np.asarray([a["phase"] for a in self.actions], dtype=np.int8))
        np.save(self.directory / "action_mask.npy", np.asarray([a["valid"] for a in self.actions], dtype=bool))
        for name, key in (("actions", "command"), ("proposals", "proposal"), ("action_observations", "observation")):
            width = 6 if key != "observation" else self.obs.shape[1]
            np.save(self.directory / f"{name}.npy", np.asarray([a[key] for a in self.actions], dtype=np.float64).reshape(-1, width))
        write_json(self.directory / "commands.json", self.commands)
        metadata.update(schema_version=1, status="complete", observation_schema=self.schema,
                        physics_fields=self.fields, dt=self.dt, physics_rows=self.physics_rows,
                        observation_rows=self.obs_rows, action_rows=len(self.actions))
        write_json(self.directory / "manifest.json", metadata)
        self.physics = self.obs = None
        metadata["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(self.directory.iterdir()) if p.name != "manifest.json"}
        write_json(self.directory / "manifest.json", metadata)
        return metadata


def load_episode(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise ValueError("Incomplete or incompatible episode")
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Episode file hash mismatch: {name}")
    arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in directory.glob("*.npy")}
    arrays["physics"] = arrays["physics"][:manifest["physics_rows"]]
    arrays["observations"] = arrays["observations"][:manifest["observation_rows"]]
    return manifest, arrays


def annotate(events):
    segments = []
    for i, event in enumerate(events):
        if event["name"] != "phase":
            continue
        end = next((e for e in events[i + 1:] if e["name"] in ("phase", "success", "failure", "time_limit")), None)
        if end:
            recovered = event["phase"] == "RECOVER" and end["name"] == "phase" and end["phase"] == "WAIT_READY"
            segments.append(dict(phase=event["phase"], start_s=event["time"], end_s=end["time"],
                                 recovery_valid=recovered, end_event=end["name"]))
    return segments
