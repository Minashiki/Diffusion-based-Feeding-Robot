"""Real P0 integration evidence, intentionally separate from synthetic phase tests."""

import numpy as np
import pytest

from feedingrobot.scripts.m3_cases import physical_case
from feedingrobot.sim.model import load_json


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
@pytest.mark.parametrize("scenario", ["plate", "carry", "receiver", "unsupported", "force", "penetration"])
def test_directed_physics(robot, scenario):
    result, trace = physical_case(robot, scenario)
    assert trace
    if scenario == "carry":
        assert sum(e["name"] == "pickup" for e in result["events"]) == 1
        assert result["peak_force_n"] < 5
    elif scenario == "receiver":
        assert trace[-1]["mouth_supported"] and not trace[-1]["tool_inside"]
        assert trace[-1]["released"]
        assert [e["name"] for e in result["events"] if e["name"] in {"delivery", "success"}] == ["delivery", "success"]


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
@pytest.mark.parametrize("scenario", ["plate", "carry", "receiver", "unsupported", "force", "penetration"])
def test_physics_timestep_convergence(robot, scenario):
    cfg = load_json("configs/acceptance_m3.json")
    base, _ = physical_case(robot, scenario, .001)
    fine, _ = physical_case(robot, scenario, .0005)
    assert (base["success"], base["failure_reason"], base["phase"]) == (fine["success"], fine["failure_reason"], fine["phase"])
    assert abs(base["time"] - fine["time"]) <= cfg["event_time_tolerance_s"]
    assert abs(base["peak_force_n"] - fine["peak_force_n"]) <= max(
        cfg["force_absolute_tolerance_n"], cfg["force_relative_tolerance"] * base["peak_force_n"])
    assert abs(base["impulse_ns"] - fine["impulse_ns"]) <= max(
        cfg["impulse_absolute_tolerance_ns"], cfg["impulse_relative_tolerance"] * base["impulse_ns"])
    np.testing.assert_allclose(base["tcp_position"], fine["tcp_position"], rtol=0,
                               atol=cfg["tcp_position_tolerance_m"])
