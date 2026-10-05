"""Reset-time P0 scenario recipes; future driver choices never enter policy_obs."""

import copy
from numbers import Real

import numpy as np


def validate_scenario(scenario):
    if scenario is not None and not isinstance(scenario, dict):
        raise ValueError("Scenario must be a parameter dictionary")
    scenario = copy.deepcopy(scenario or {})
    ranges = {"recover_departure_m": (0., .015), "food_mass_kg": (.001, .01), "food_friction": (.05, .8),
              "head_amp_m": (0., .012), "head_freq_hz": (0., .3),
              "head_phase_rad": (-np.pi, np.pi)}
    vectors = {"food_offset_m": (2, .01), "head_offset_m": (3, .005)}
    if set(scenario) - set(ranges) - set(vectors) - {"recover", "head_origin_m"}:
        raise ValueError("Unknown scenario parameter")
    for key, (low, high) in ranges.items():
        if key in scenario and (not isinstance(scenario[key], Real) or isinstance(scenario[key], bool)
                                or not np.isfinite(scenario[key]) or not low <= scenario[key] <= high):
            raise ValueError(f"Invalid scenario parameter: {key}")
    for key, (size, limit) in vectors.items():
        if key in scenario:
            value = np.asarray(scenario[key], dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all() or np.any(np.abs(value) > limit):
                raise ValueError(f"Invalid scenario parameter: {key}")
            scenario[key] = value.tolist()
    if "head_origin_m" in scenario:
        origin = np.asarray(scenario["head_origin_m"], dtype=float)
        if (origin.shape != (3,) or not np.isfinite(origin).all()
                or np.any(origin < [.5, .08, .3]) or np.any(origin > [.75, .2, .4])):
            raise ValueError("Invalid scenario parameter: head_origin_m")
        scenario["head_origin_m"] = origin.tolist()
    if "recover" in scenario and not isinstance(scenario["recover"], bool):
        raise ValueError("recover must be boolean")
    return scenario
