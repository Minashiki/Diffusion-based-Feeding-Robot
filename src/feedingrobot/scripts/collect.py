"""Collect only after a frozen teacher passes independent acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np

from feedingrobot.data.episodes import annotate, input_hashes, load_episode, write_json
from feedingrobot.data.recipes import recipe
from feedingrobot.data.rollout import run_episode
from feedingrobot.data.replay import replay_episode
from feedingrobot.experts.gate import matching_teachers_passed
from feedingrobot.sim.model import ROOT


def check_gate(report, config, robot):
    if (robot != "panda" or report.get("teacher_gate") != "passed" or report.get("robot_id") != robot
            or report.get("baseline", {}).get("attempts") != 100
            or report.get("baseline", {}).get("successes", 0) < 95
            or report.get("input_hashes") != input_hashes()
            or report.get("teacher_config") != config
            or not matching_teachers_passed(report, report.get("compatibility", {}))):
        raise ValueError("Formal collection requires matching frozen 100-seed acceptance and all physical teacher checks for both robots")


def dataset_statistics(directory, *, config=None, robot=None, replay=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counts, recovery_counts, groups, seeds = {}, {}, {}, {}
    mean, m2, count = None, None, 0
    manifests = sorted(directory.glob("*/*/manifest.json"))
    episodes = []
    version = None
    for path in manifests:
        m, a = load_episode(path.parent)
        current = (m["robot_id"], m["observation_schema"], m["teacher_config"], m["input_hashes"])
        if version is None:
            version = current
        elif current != version:
            raise ValueError("Dataset contains mixed robot/schema/teacher/input versions")
        if config is not None and (m["teacher_config"] != config or m["input_hashes"] != input_hashes()
                                   or m["robot_id"] != robot):
            raise ValueError("Dataset does not match frozen teacher inputs")
        if m["segments"] != annotate(m["events"]):
            raise ValueError("Episode recovery annotations do not match physical events")
        if (m["accepted_normal"] != m["success"]
                or m["accepted_recovery"] != any(s["recovery_valid"] for s in m["segments"])):
            raise ValueError("Episode selection flags do not match outcomes")
        split = m["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("Formal dataset contains a non-dataset split")
        group, seed = m["group_id"], m["seed"]
        if (group in groups and groups[group] != split) or (seed in seeds and seeds[seed] != split):
            raise ValueError("Episode recipe/seed leaked across splits")
        groups[group], seeds[seed] = split, split
        normal = bool(m["accepted_normal"] and not m["scenario"].get("recover", False))
        recovery = bool(m["accepted_recovery"])
        counts[split] = counts.get(split, 0) + int(normal)
        recovery_counts[split] = recovery_counts.get(split, 0) + int(recovery)
        expected_ticks = round(.05 / m["dt"])
        assert np.all(a["action_ticks"] % expected_ticks == 0)
        assert np.all(a["action_end_ticks"] <= a["action_ticks"] + expected_ticks)
        assert np.all(a["action_end_ticks"][a["action_mask"]] == a["action_ticks"][a["action_mask"]] + expected_ticks)
        for index in np.flatnonzero(a["action_mask"]):
            start, end = a["action_ticks"][index] * m["dt"], a["action_end_ticks"][index] * m["dt"]
            assert not any(e["name"] == "phase" and start < e["time"] < end - 1e-10 for e in m["events"])
        eligible = a["action_mask"].copy()
        if not normal:
            eligible &= a["action_phases"] == 7
            eligible &= np.array([any(s["recovery_valid"] and s["start_s"] <= tick*m["dt"] < s["end_s"]
                                     for s in m["segments"]) for tick in a["action_ticks"]])
        if split == "train":
            x = a["action_observations"][eligible]
            if len(x):
                batch_mean, batch_m2 = x.mean(0), ((x-x.mean(0))**2).sum(0)
                if mean is None:
                    mean, m2 = batch_mean, batch_m2
                else:
                    delta = batch_mean - mean
                    m2 += batch_m2 + delta**2 * count * len(x) / (count + len(x))
                    mean += delta * len(x) / (count + len(x))
                count += len(x)
        episodes.append(dict(path=str(path.parent.relative_to(directory)), seed=seed, split=split,
                             normal=normal, recovery=recovery, failure_reason=m["failure_reason"]))
        if replay:
            episodes[-1]["replay"] = replay_episode(path.parent)
    summary = dict(schema_version=1, status="incomplete", counts=counts, recovery_counts=recovery_counts, attempts=len(episodes), episodes=episodes,
                   replay_status="passed" if replay and episodes else "not_verified")
    write_json(directory / "statistics.json", summary)
    if count:
        raw_std = np.sqrt(m2/count)
        write_json(directory / "normalization.json", dict(source_split="train", count=count,
                   mean=mean, std=np.where(raw_std > 1e-8, raw_std, 1.), raw_std=raw_std,
                   observation_schema=m["observation_schema"]))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--config", default="configs/collect.json")
    parser.add_argument("--gate", default="outputs/new_tableware/v3/m4/panda/report.json")
    parser.add_argument("--output")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--viewer", dest="viewer", action="store_true")
    display.add_argument("--headless", dest="viewer", action="store_false")
    parser.set_defaults(viewer=True)
    parser.add_argument("--train-episodes", type=int, help="Override train quota, e.g. 1000; held-out quotas unchanged")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    check_gate(json.loads((ROOT / args.gate).read_text()), config, args.robot)
    directory = ROOT / (args.output or f"datasets/new_tableware/v3/m4/{args.robot}")
    directory.mkdir(parents=True, exist_ok=True)
    quotas = dict(config["quotas"])
    if args.train_episodes is not None:
        if args.train_episodes < quotas["train"]:
            raise ValueError("Train quota cannot be reduced below startup budget")
        quotas["train"] = args.train_episodes
    for split, quota in quotas.items():
        for recover in (False, True):
            accepted = 0
            for index in range(quota * config["max_attempt_multiplier"]):
                seed, scenario, parameters, group = recipe(config, split, index, recover=recover)
                episode = directory / split / f"{'recovery' if recover else 'normal'}_{seed}"
                if episode.exists():
                    result, _ = load_episode(episode)
                    if (result["input_hashes"] != input_hashes() or result["teacher_config"] != config
                            or result["robot_id"] != args.robot):
                        raise ValueError("Cannot resume collection with changed teacher/source")
                else:
                    result = run_episode(args.robot, seed, config, episode, scenario=scenario,
                                         teacher_parameters=parameters, split=split, group_id=group,
                                         viewer=args.viewer)
                accepted += int(result["accepted_recovery"] if recover else result["accepted_normal"])
                print(f"{split} {'recovery' if recover else 'normal'}: {accepted}/{quota} (seed={seed})", flush=True)
                if accepted >= quota:
                    break
            if accepted < quota:
                dataset_statistics(directory, config=config, robot=args.robot, replay=True)
                raise RuntimeError(f"Quota unmet: {split}, recovery={recover}, {accepted}/{quota}")
    summary = dataset_statistics(directory, config=config, robot=args.robot, replay=True)
    summary["status"] = "passed"
    summary["quotas"] = quotas
    write_json(directory / "statistics.json", summary)


if __name__ == "__main__":
    main()
