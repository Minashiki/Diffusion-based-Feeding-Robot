"""Deterministic, disjoint driver groups for whole single-bean episodes."""

import hashlib
import json

import numpy as np


FOLDS = {"train": set(range(1, 16)), "validation": {16, 17}, "test": {18, 19}, "acceptance": {0}}
FOLDS["calibration"] = FOLDS["train"]


def recipe(config, split, index, *, recover=False):
    for attempt in range(10000):
        seed = config["seed_bases"][split]*100000000 + index*10000 + attempt + (10000000000000 if recover else 0)
        rng = np.random.default_rng(seed)
        scenario = {key: float(rng.uniform(*bounds)) for key, bounds in config["scenario_ranges"].items()}
        scenario.update(head_offset_m=rng.uniform(-.003, .003, 3).tolist(),
                        head_phase_rad=float(rng.uniform(-np.pi, np.pi)), recover=bool(recover))
        scenario.update(config["scene"])
        bins = {key: min(2, int((scenario[key]-low)/(high-low)*3)) if high > low else 0
                for key, (low, high) in config["scenario_ranges"].items()}
        bins.update(food_layout=[min(3, int((x+.008)/.016*4)) for x in scenario.get("food_offset_m", [0., 0.])],
                    head_layout=[min(2, int((x+.003)/.006*3)) for x in scenario["head_offset_m"]],
                    head_phase=min(2, int((scenario["head_phase_rad"]+np.pi)/(2*np.pi)*3)),
                    origin=scenario["head_origin_m"])
        group = hashlib.sha256(json.dumps(bins, sort_keys=True).encode()).hexdigest()
        if int(group[:8], 16) % 20 in FOLDS[split]:
            parameters = {key: float(rng.uniform(*bounds)) for key, bounds in config["teacher_ranges"].items()}
            return seed, scenario, parameters, group
    raise ValueError("Scenario ranges cannot supply the requested disjoint split groups")
