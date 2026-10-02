"""Regression tests for new geometry and the physical precollection gate."""

import numpy as np
import pytest

from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.experts.gate import PRECOLLECTION_CASES
from feedingrobot.data.episodes import input_hashes
from feedingrobot.scripts.collect import check_gate
from feedingrobot.scripts.validate_m4 import convergence_cases
from feedingrobot.sim.model import load_json
from feedingrobot.sim.observer_viewer import benchmark_visualization


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_geometry_is_static_copy_and_retains_curved_collision_surface(robot):
    env = FeedingGymEnv(robot)
    env.reset(seed=0)
    before = env.task.get_state()["physics"].copy()
    g = teacher_geometry(env.task)
    np.testing.assert_array_equal(before, env.task.get_state()["physics"])
    assert not {"seed", "parameters", "future_events", "food_mass_kg", "food_friction"}.intersection(g)
    assert g["front_lip"][:, 2].min() > .006
    assert g["scoop_points"][:, 2].min() < 0
    assert np.ptp(g["scoop_points"][:, 2]) > .009
    assert g["tool_points"][:, 0].min() < -.09
    assert not g["scoop_points"].flags.writeable
    original = g["plate_position"].copy()
    env.reset(seed=7)
    np.testing.assert_array_equal(g["plate_position"], original)
    assert g["sha256"] == teacher_geometry(env.task)["sha256"]
    env.close()


def test_teacher_requires_new_config_and_measured_geometry():
    env = FeedingGymEnv()
    old = load_json("configs/collect.json")
    old["schema_version"] = 1
    with pytest.raises(ValueError, match="schema 2"):
        Teacher(env.task.robot_config, old)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    with pytest.raises(ValueError, match="measured geometry"):
        teacher.act(env.task.provider.observe()["policy_obs"])
    with pytest.raises(TypeError, match="geometry"):
        teacher.reset({})
    env.close()


def test_arc_turns_spoon_and_preserves_clearance_of_actual_orientation():
    env = FeedingGymEnv()
    env.reset(seed=0)
    teacher = Teacher(env.task.robot_config, load_json("configs/collect.json"))
    teacher.reset({}, geometry=teacher_geometry(env.task))
    obs = env.task.provider.observe()["policy_obs"]
    teacher.act(obs)
    teacher.part = 3
    p, g = teacher.parameters, teacher.geometry
    obs["tcp_position"] = teacher.food_start-g["plate_rotation"][:,0]*p["arc_start_offset_m"]
    initial_r, initial_points = teacher.acquisition_waypoints(obs)
    obs["tcp_rotation"] = initial_r.copy()
    obs["tcp_position"] += g["plate_rotation"][:,0]*p["arc_length_m"]
    final_r, final_points = teacher.acquisition_waypoints(obs)
    assert np.linalg.norm(final_r-initial_r) > .5
    lowest = ((g["scoop_points"] @ initial_r.T) @ g["plate_rotation"])[:,2].min()
    target_z = g["plate_rotation"][:,2] @ (final_points[2]-g["plate_position"])
    assert target_z+lowest >= p["plate_gap_m"]-1e-12
    env.close()


def passed_reports(config):
    hashes = input_hashes()
    def report(robot, normal, recovery):
        return dict(robot_id=robot, teacher_gate="passed", teacher_config=config, input_hashes=hashes,
                    baseline=dict(attempts=normal, successes=normal),
                    recovery_baseline=dict(attempts=recovery, successes=recovery),
                    cases={name:dict(status="passed") for name in PRECOLLECTION_CASES})
    panda, ur5e = report("panda",100,10), report("ur5e",5,5)
    panda["compatibility"] = ur5e
    return panda


@pytest.mark.parametrize("robot,missing", [(r,c) for r in ("panda","ur5e") for c in PRECOLLECTION_CASES])
def test_partial_physical_gate_cannot_collect(robot, missing):
    config = load_json("configs/collect.json")
    report = passed_reports(config)
    target = report if robot=="panda" else report["compatibility"]
    target["cases"][missing]["status"] = "not_verified"
    with pytest.raises(ValueError, match="all physical"):
        check_gate(report,config,"panda")


def test_gate_accepts_completed_checks_but_rejects_changed_inputs():
    config = load_json("configs/collect.json")
    report = passed_reports(config)
    check_gate(report,config,"panda")
    report["compatibility"]["input_hashes"] = {}
    with pytest.raises(ValueError):
        check_gate(report,config,"panda")


def test_convergence_cases_are_preassigned_normal_and_recovery():
    config = load_json("configs/collect.json")
    cases = convergence_cases(config)
    assert len(cases)==10 and len({c["seed"] for c in cases})==10
    assert [c["index"] for c in cases]==list(range(5))*2
    assert sum(c["recover"] for c in cases)==5


@pytest.mark.parametrize("factor,enabled", [(1.5,True),(1.51,False)])
def test_visualization_budget_uses_three_equal_work_windows(monkeypatch, factor, enabled):
    import feedingrobot.sim.observer_viewer as module
    elapsed = [0.]
    calls = []
    def run_window(*, viewer, warmup):
        calls.append((viewer,warmup))
        elapsed[0] += factor if viewer else 1.
    monkeypatch.setattr(module.time,"monotonic",lambda: elapsed[0])
    report = benchmark_visualization(run_window)
    assert report["enabled"] is enabled
    assert report["time_ratio"]==pytest.approx(factor)
    assert len(report["headless_windows_s"])==len(report["visible_windows_s"])==3
    assert calls[:2]==[(False,True),(True,True)]
    assert len(calls)==8


def test_unmet_collection_quota_replays_all_attempts_before_reporting_failure(tmp_path, monkeypatch):
    import sys
    from feedingrobot.scripts import collect
    config = load_json("configs/collect.json")
    config["quotas"] = {"train": 1, "validation": 1, "test": 1}
    path = tmp_path/"config.json"
    import json
    path.write_text(json.dumps(config))
    gate = tmp_path/"gate.json"
    gate.write_text("{}")
    attempts, checks = [], []
    monkeypatch.setattr(collect,"check_gate",lambda *args: None)
    def failed_episode(*args, **kwargs):
        attempts.append(args[1])
        return dict(accepted_normal=False,accepted_recovery=False)
    monkeypatch.setattr(collect,"run_episode",failed_episode)
    monkeypatch.setattr(collect,"dataset_statistics",lambda directory,**kwargs: checks.append(kwargs))
    monkeypatch.setattr(sys,"argv",["collect","--config",str(path),"--gate",str(gate),
                                    "--output",str(tmp_path/"dataset"),"--headless"])
    with pytest.raises(RuntimeError,match="Quota unmet"):
        collect.main()
    assert len(attempts)==3
    assert checks==[dict(config=config,robot="panda",replay=True)]
