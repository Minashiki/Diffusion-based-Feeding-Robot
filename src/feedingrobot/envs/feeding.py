"""50 Hz Gymnasium task, sharing the M1 physical execution path."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

from feedingrobot.sim.events import PHASES, evidence
from feedingrobot.sim.task import FeedingTask

EXECUTION_STATES = ("idle", "active", "expired", "stopped", "success", "blocked", "ik_failure",
                    "workspace_limit", "invalid_command", "contact_limit", "joint_speed_limit",
                    "nonfinite_state", "model_penetration", "food_dropped", "food_lost_after_delivery",
                    "withdrawal_before_release", "food_missing")


def observation_schema(robot_id, n):
    fields = [("q", n, "rad"), ("dq", n, "rad/s"), ("tcp_position", 3, "m"),
              ("tcp_rotation", 9, "rotation matrix row-major"), ("tcp_twist_world", 6, "m/s, rad/s"),
              ("bean_relative_world", 3, "m"), ("mouth_relative_world", 3, "m"),
              ("mouth_rotation", 9, "rotation matrix row-major"), ("mouth_aperture_m", 1, "m"),
              ("raw_wrench_sensor", 6, "N, Nm"), ("wrench_world_at_tcp", 6, "N, Nm"),
              ("compensated_wrench", 6, "N, Nm"), ("stage", len(PHASES), "one-hot"),
              ("interaction", 4, "boolean"), ("execution_status", len(EXECUTION_STATES), "one-hot"),
              ("frame_age_s", 1, "s"), ("receiver_relative_world", 3, "m"),
              ("receiver_rotation", 9, "rotation matrix row-major")]
    return dict(version=3, robot_id=robot_id, fields=fields, phases=PHASES,
                execution_states=EXECUTION_STATES,
                interaction=("spoon_support", "mouth_support", "tool_mouth_contact", "ready"))


class FeedingGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, robot_id="panda", render_mode=None, timestep=None, max_episode_s=None):
        if render_mode not in (None, "human"):
            raise ValueError("Supported render modes: None, human")
        self.render_mode = render_mode
        self.task = FeedingTask(robot_id, timestep, task_mode=True)
        self.config = self.task.task_config
        self.control_dt = self.config["control_dt_s"]
        self.substeps = round(self.control_dt / self.task.dt)
        if self.substeps < 1 or not np.isclose(self.substeps * self.task.dt, self.control_dt, rtol=0, atol=1e-12):
            raise ValueError("Physics dt must divide the 20 ms control period")
        self.max_episode_s = self.config["max_episode_s"] if max_episode_s is None else float(max_episode_s)
        if not np.isfinite(self.max_episode_s) or self.max_episode_s <= 0:
            raise ValueError("Episode time limit must be finite and positive")
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
        n = self.task.index.n
        self.schema = observation_schema(robot_id, n)
        self.fields = self.schema['fields']
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (sum(f[1] for f in self.fields),), dtype=np.float32)
        self.viewer_context = self.viewer = None
        self.done = True
        self.steps = 0
        self.last_observation = None
        self.step_metrics = {}

    def _observation(self):
        p = self.task.provider.observe()["policy_obs"]
        p["stage"] = np.eye(len(PHASES))[PHASES.index(p["stage"])]
        p["execution_status"] = np.eye(len(EXECUTION_STATES))[EXECUTION_STATES.index(p["execution_status"])]
        return np.concatenate([np.asarray(p[name]).reshape(-1) for name, _, _ in self.fields]).astype(np.float32)

    def observe_policy(self):
        """Current policy vector for shared teacher/data tooling; excludes oracle state."""
        return self._observation().copy()

    def _info(self, events, reward_terms, elapsed, valid=True):
        task = self.task
        return dict(robot_id=task.robot_id, schema_version=self.schema["version"], time=float(task.data.time),
                    elapsed_s=float(elapsed), phase=task.logic.phase, success=task.logic.success,
                    failure_reason=task.failure_reason, observation_valid=valid,
                    events=copy.deepcopy(events), reward_terms=reward_terms,
                    step_metrics=self.step_metrics.copy(),
                    oracle_info=dict(timers=copy.deepcopy(task.logic.timers), contacts=copy.deepcopy(task.contacts),
                                     contact_peak_n=task.monitor.peak_n, contact_impulse_ns=task.monitor.impulse_ns,
                                     contact_over_limit_s=task.monitor.over_limit_s))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"preset", "scenario"}:
            raise ValueError("Only reset options 'preset' and 'scenario' are supported")
        actual_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31))
        if "scenario" in options:
            self.task.reset(actual_seed, options.get("preset", "beans_in_bowl"), scenario=options["scenario"])
        else:
            self.task.reset(actual_seed, options.get("preset", "beans_in_bowl"))
        self.steps, self.done = 0, False
        self.step_metrics = {}
        self.last_observation = self._observation()
        if self.render_mode:
            self.render()
        return self.last_observation.copy(), self._info([], {}, 0.)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished or not initialized; reset required")
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            self.task.adapter.stop("invalid_command", fault=True)
            self.task._terminate("invalid_command")
            self.task.logic.failure_reason = "invalid_command"
            self.task.logic.emit("failure", self.task.data.time, reason="invalid_command")
            self.done = True
            raise ValueError("Action must be a finite six-vector within [-1, 1]; reset required")
        task, logic, reward = self.task, self.task.logic, self.config["reward"]
        start = float(task.data.time)
        phase = logic.phase
        before = logic.distance(evidence(task))
        event_start = len(logic.events)
        impulse = task.monitor.impulse_ns
        over_limit = task.monitor.over_limit_s
        self.step_metrics = dict(contact_peak_n=0., wrist_peak_n=0.)
        scale = np.array([task.robot_config["linear_speed_limit"]] * 3
                         + [task.robot_config["angular_speed_limit"]] * 3)
        task.adapter.set_twist(action * scale, start, start + self.control_dt)
        for _ in range(self.substeps):
            task.step_physics()
            self.step_metrics["contact_peak_n"] = max(self.step_metrics["contact_peak_n"], task.substep_contact_peak_n)
            self.step_metrics["wrist_peak_n"] = max(self.step_metrics["wrist_peak_n"], task.substep_wrist_peak_n)
            if task.terminated or task.tick * task.dt + 1e-12 >= self.max_episode_s:
                break
        elapsed = float(task.data.time) - start
        self.step_metrics.update(contact_impulse_ns=task.monitor.impulse_ns - impulse,
                                 contact_over_limit_s=task.monitor.over_limit_s - over_limit)
        if task.terminated and task.failure_reason and logic.failure_reason is None:
            logic.failure_reason = task.failure_reason
            logic.emit("failure", task.data.time, reason=task.failure_reason)
        valid = task.failure_reason != "nonfinite_state"
        if valid:
            obs = self._observation()
            valid = bool(np.isfinite(obs).all())
            if not valid:
                task._terminate("nonfinite_state")
        if not valid:
            # A finite last valid observation is explicit, never presented as fresh state.
            obs = self.last_observation.copy()
        self.last_observation = obs.copy()
        progress = 0.
        if valid and not task.terminated and phase == logic.phase and before is not None:
            after = logic.distance(evidence(task))
            progress = float(np.clip((before - after) / reward["progress_scale_m"],
                                     -reward["progress_clip_per_step"], reward["progress_clip_per_step"]))
        terms = dict(progress=progress, time=reward["time_per_s"] * elapsed,
                     contact=reward["contact_per_ns"] * (task.monitor.impulse_ns - impulse),
                     pickup=0., delivery=0., success=0., failure=0.)
        for event in logic.events[event_start:]:
            name = event["name"]
            if name in ("pickup", "delivery", "success", "failure") and name not in logic.awarded:
                terms[name] = reward[name]
                logic.awarded.add(name)
        self.steps += 1
        terminated = bool(task.terminated)
        truncated = bool(not terminated and task.tick * task.dt + 1e-12 >= self.max_episode_s)
        self.done = terminated or truncated
        if truncated:
            task.adapter.stop()
            logic.emit("time_limit", task.data.time)
            obs = self._observation()
            self.last_observation = obs.copy()
        info = self._info(logic.events[event_start:], terms, elapsed, valid)
        if self.render_mode:
            self.render()
        return obs, float(sum(terms.values())), terminated, truncated, info

    def get_state(self):
        return copy.deepcopy(dict(schema_version=3, observation_schema=copy.deepcopy(self.schema), task=self.task.get_state(), steps=self.steps, done=self.done,
                                  max_episode_s=self.max_episode_s, last_observation=self.last_observation,
                                  step_metrics=self.step_metrics, rng_seed=self.np_random_seed,
                                  rng=self.np_random.bit_generator.state,
                                  action_rng=self.action_space.np_random.bit_generator.state))

    def set_state(self, state):
        if (state["schema_version"] != 3 or state.get("observation_schema") != self.schema
                or state["max_episode_s"] != self.max_episode_s):
            raise ValueError("Incompatible environment snapshot")
        self.task.set_state(state["task"])
        self.steps, self.done = state["steps"], state["done"]
        self.last_observation = copy.deepcopy(state["last_observation"])
        self.step_metrics = copy.deepcopy(state["step_metrics"])
        self._np_random_seed = state["rng_seed"]
        self.np_random.bit_generator.state = copy.deepcopy(state["rng"])
        self.action_space.np_random.bit_generator.state = copy.deepcopy(state["action_rng"])

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from feedingrobot.sim.viewer import passive_viewer
                self.viewer_context = passive_viewer(self.task.model, self.task.data)
                self.viewer = self.viewer_context.__enter__()
            self.viewer.sync()

    def close(self):
        if self.viewer_context:
            self.viewer_context.__exit__(None, None, None)
            self.viewer_context = self.viewer = None
