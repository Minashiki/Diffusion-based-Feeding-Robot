"""Collect only after a frozen teacher passes independent acceptance."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path

import numpy as np

from feedingrobot.data.episodes import annotate, input_hashes, load_episode, recovery_action_mask, write_json
from feedingrobot.data.recipes import recipe
from feedingrobot.data.rollout import run_episode
from feedingrobot.data.replay import replay_many
from feedingrobot.experts.gate import panda_teacher_passed
from feedingrobot.sim.model import ROOT


def check_gate(report, config, robot):
    from feedingrobot.experts.freeze import parent_m3_check
    parent = parent_m3_check()
    if (robot != "panda" or report.get("teacher_gate") != "passed" or report.get("robot_id") != robot
            or report.get("baseline", {}).get("attempts") != 100
            or report.get("baseline", {}).get("successes", 0) < 95
            or report.get("input_hashes") != input_hashes()
            or report.get("teacher_config") != config
            or report.get("parent_m3") != parent
            or not panda_teacher_passed(report)):
        raise ValueError("Formal collection requires matching frozen 100-seed acceptance and all physical Panda teacher checks")
    if report.get("evidence_origin"):
        from feedingrobot.experts.reuse import verify_reuse_audit
        verify_reuse_audit(report)


def dataset_statistics(directory, *, config=None, robot=None, replay=False, workers=1):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    statistics = directory / "statistics.json"
    if replay and statistics.exists():
        saved = json.loads(statistics.read_text())
        if saved.get("validation_sha256") and saved.get("status") == "passed":
            return verified_statistics(directory, config=config, robot=robot)
    counts, recovery_counts, groups, seeds = {}, {}, {}, {}
    mean, m2, count = None, None, 0
    manifests = sorted(directory.glob("*/*/manifest.json"))
    episodes = []
    version = None
    expected_schema = None
    if robot is not None:
        from feedingrobot.envs import FeedingGymEnv
        env = FeedingGymEnv(robot)
        expected_schema = json.loads(json.dumps(env.schema))
        env.close()
    for path in manifests:
        m, a = load_episode(path.parent)
        current = (m["robot_id"], m["observation_schema"], m["teacher_config"], m["input_hashes"])
        if version is None:
            version = current
        elif current != version:
            raise ValueError("Dataset contains mixed robot/schema/teacher/input versions")
        if config is not None and (m["teacher_config"] != config or m["input_hashes"] != input_hashes()
                                   or m["robot_id"] != robot or m["observation_schema"] != expected_schema):
            raise ValueError("Dataset does not match frozen teacher inputs")
        if m["segments"] != annotate(m["events"]):
            raise ValueError("Episode recovery annotations do not match physical events")
        recovery_mask = recovery_action_mask(m['segments'], a['action_phases'], a['action_ticks'],
                                             a['action_end_ticks'], a['action_mask'], m['dt'])
        if (m["accepted_normal"] != m["success"]
                or m["accepted_recovery"] != bool(np.any(recovery_mask))
                or m["recovery_action_rows"] != int(np.count_nonzero(recovery_mask))):
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
            eligible &= recovery_mask
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
        checks = replay_many([p.parent for p in manifests], workers=workers, progress=directory / 'replay_progress.json')
        for episode, check in zip(episodes, checks):
            episode['replay'] = check
    summary = dict(schema_version=1, status="incomplete", counts=counts, recovery_counts=recovery_counts, attempts=len(episodes), episodes=episodes,
                   replay_status="passed" if replay and episodes else "not_verified")
    write_json(directory / "statistics.json", summary)
    if count:
        raw_std = np.sqrt(m2/count)
        write_json(directory / "normalization.json", dict(source_split="train", count=count,
                   mean=mean, std=np.where(raw_std > 1e-8, raw_std, 1.), raw_std=raw_std,
                   observation_schema=m["observation_schema"]))
    if replay and episodes:
        from feedingrobot.experts.freeze import sha256
        summary.update(input_hashes=input_hashes(), teacher_config=config, robot_id=robot,
                       evidence_sha256={str(p.relative_to(directory)): sha256(p)
                           for p in sorted(directory.rglob("*")) if p.is_file()
                           and p.name not in ("statistics.json", "replay_progress.json")})
        summary["validation_sha256"] = statistics_digest(summary)
        write_json(directory / "statistics.json", summary)
    return summary


def statistics_digest(summary):
    import hashlib
    bound = {k: v for k, v in summary.items() if k not in ("status", "quotas", "validation_sha256")}
    return hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verified_statistics(directory, *, config, robot):
    """Verify the collected statistics and replay evidence without rerunning physics."""
    from feedingrobot.experts.freeze import sha256
    directory = Path(directory)
    summary = json.loads((directory / "statistics.json").read_text())
    if (summary.get("validation_sha256") != statistics_digest(summary)
            or summary.get("input_hashes") != input_hashes()
            or summary.get("teacher_config") != config or summary.get("robot_id") != robot
            or summary.get("status") != "passed" or summary.get("replay_status") != "passed"):
        raise ValueError("Dataset statistics/replay do not match frozen inputs")
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
             and p.name not in ("statistics.json", "replay_progress.json")}
    if files != set(summary.get("evidence_sha256", {})):
        raise ValueError("Dataset evidence coverage changed")
    for relative, digest in summary["evidence_sha256"].items():
        if sha256(directory / relative) != digest:
            raise ValueError(f"Dataset evidence changed: {relative}")
    manifests = {str(p.parent.relative_to(directory)) for p in directory.glob("*/*/manifest.json")}
    if (manifests != {e["path"] for e in summary["episodes"]}
            or len(manifests) != summary["attempts"]
            or any(e["replay"]["status"] != "passed" for e in summary["episodes"])
            or any(summary[key].get(split, 0) < quota for key in ("counts", "recovery_counts")
                   for split, quota in config["quotas"].items())):
        raise ValueError("Dataset requires complete quotas and replay coverage")
    return summary


def collection_trial(robot, config, directory, split, index, recover, viewer):
    seed, scenario, parameters, group = recipe(config, split, index, recover=recover)
    episode = directory / split / f"{'recovery' if recover else 'normal'}_{seed}"
    if episode.exists():
        result, _ = load_episode(episode)
        if (result['input_hashes'] != input_hashes() or result['teacher_config'] != config
                or result['robot_id'] != robot):
            raise ValueError('Cannot resume collection with changed teacher/source')
    else:
        result = run_episode(robot, seed, config, episode, scenario=scenario,
                             teacher_parameters=parameters, split=split, group_id=group, viewer=viewer)
    return dict(seed=seed, accepted=bool(result['accepted_recovery'] if recover else result['accepted_normal']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--config", default="configs/collect.json")
    parser.add_argument("--gate", default="outputs/single_bean/v1/m4/revision_4/panda/report.json")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=1)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--viewer", dest="viewer", action="store_true")
    display.add_argument("--headless", dest="viewer", action="store_false")
    parser.set_defaults(viewer=True)
    parser.add_argument("--train-episodes", type=int, help="Override train quota, e.g. 1000; held-out quotas unchanged")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = json.loads((ROOT / args.config).read_text())
    check_gate(json.loads((ROOT / args.gate).read_text()), config, args.robot)
    directory = ROOT / (args.output or f"datasets/single_bean/v1/m4/{args.robot}")
    directory.mkdir(parents=True, exist_ok=True)
    quotas = dict(config["quotas"])
    if args.train_episodes is not None:
        if args.train_episodes < quotas["train"]:
            raise ValueError("Train quota cannot be reduced below startup budget")
        quotas["train"] = args.train_episodes
    for split, quota in quotas.items():
        for recover in (False, True):
            accepted = 0
            index, limit = 0, quota*config['max_attempt_multiplier']
            while accepted < quota and index < limit:
                count = min(args.workers, quota-accepted, limit-index)
                inputs = [(args.robot, config, directory, split, i, recover, args.viewer and i == index)
                          for i in range(index, index+count)]
                if args.workers == 1:
                    batch = [collection_trial(*inputs[0])]
                else:
                    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) as pool:
                        batch = [f.result() for f in [pool.submit(collection_trial, *values) for values in inputs]]
                for result in batch:
                    accepted += int(result['accepted'])
                    print(f"{split} {'recovery' if recover else 'normal'}: {accepted}/{quota} (seed={result['seed']})", flush=True)
                index += count
            if accepted < quota:
                dataset_statistics(directory, config=config, robot=args.robot, replay=True, workers=args.workers)
                raise RuntimeError(f"Quota unmet: {split}, recovery={recover}, {accepted}/{quota}")
    summary = dataset_statistics(directory, config=config, robot=args.robot, replay=True, workers=args.workers)
    summary["status"] = "passed"
    summary["quotas"] = quotas
    write_json(directory / "statistics.json", summary)


if __name__ == "__main__":
    main()
