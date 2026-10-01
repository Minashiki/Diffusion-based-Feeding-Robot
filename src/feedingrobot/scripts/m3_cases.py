"""Reset-only diagnostic fixtures. These are NOT a teacher or a full episode demonstration."""

import mujoco
import numpy as np

from feedingrobot.envs import FeedingGymEnv
from feedingrobot.sim.contacts import read_contacts
from feedingrobot.sim.events import evidence
from feedingrobot.sim.model import named_id
from feedingrobot.sim.task import FeedingTask


class DiagnosticTask(FeedingTask):
    def __init__(self, robot_id, scenario, timestep=None):
        self.scenario = scenario
        super().__init__(robot_id, timestep, task_mode=True)

    def reset(self, seed=0, preset="food_on_plate"):
        super().reset(seed, "food_on_spoon" if self.scenario == "carry" else "food_on_plate")
        if self.scenario in ("receiver", "unsupported", "penetration"):
            address = self.index.food_qpos
            if self.scenario == "receiver":
                site = named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_receiver")
                rotation = self.data.site_xmat[site].reshape(3, 3)
                self.data.qpos[address:address + 3] = self.data.site_xpos[site] + rotation @ [0, 0, .008]
                mujoco.mju_mat2Quat(self.data.qpos[address + 3:address + 7], rotation.ravel())
                # Explicit acquired prehistory to isolate real receipt/retraction evidence.
                self.logic.acquired = True
                self.logic.phase = "TRANSFER"
            elif self.scenario == "unsupported":
                site = named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
                self.data.qpos[address:address + 3] = self.data.site_xpos[site] + [0., -.13, .18]
                self.logic.acquired = True
                self.logic.phase = "TRANSPORT"
            else:
                site = named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "plate_frame")
                self.data.qpos[address:address + 3] = self.data.site_xpos[site]
            mujoco.mj_forward(self.model, self.data)
            self.contacts = read_contacts(self.model, self.data, self.index)
        return self.snapshot()


def physical_case(robot, scenario, timestep=.001):
    env = FeedingGymEnv(robot, timestep=timestep)
    if scenario in ("carry", "receiver", "unsupported", "penetration"):
        env.task = DiagnosticTask(robot, scenario, timestep)
        env.config = env.task.task_config
    env.reset(seed=0)
    if scenario == "force":
        env.task.set_external_wrench([0, 0, 12], [0, 0, 0], env.task.data.site_xpos[env.task.index.tcp])
    trace = []
    try:
        for _ in range(25):
            action = np.zeros(6)
            if scenario == "carry":
                action[0] = .1
            _, _, terminated, truncated, info = env.step(action)
            e = evidence(env.task)
            trace.append(dict(time=info["time"], phase=info["phase"], success=info["success"],
                              failure_reason=info["failure_reason"], events=info["events"],
                              supported=e["supported"], mouth_supported=e["mouth_supported"],
                              released=e["released"], tool_inside=e["tool_inside"],
                              contact_peak_n=env.task.monitor.peak_n,
                              contact_impulse_ns=env.task.monitor.impulse_ns,
                              minimum_contact_distance_m=min((r["distance"] for r in env.task.contacts), default=0.)))
            if terminated or truncated:
                break
        result = dict(scenario=scenario, fixture="reset-only directed P0 diagnostic, not full feeding",
                      dt=timestep, time=float(env.task.data.time), success=env.task.logic.success,
                      failure_reason=env.task.failure_reason, phase=env.task.logic.phase,
                      peak_force_n=env.task.monitor.peak_n,
                      impulse_ns=env.task.monitor.impulse_ns,
                      tcp_position=env.task.data.site_xpos[env.task.index.tcp].tolist(),
                      events=env.task.logic.events)
        if scenario == "carry":
            assert env.task.logic.acquired and e["supported"] and not terminated, result
        elif scenario == "receiver":
            assert terminated and env.task.logic.success and e["mouth_supported"] and e["released"], result
        elif scenario == "unsupported":
            assert terminated and env.task.failure_reason == "food_dropped", result
        elif scenario == "force":
            assert terminated and env.task.failure_reason == "contact_limit", result
            assert env.task.data.time <= timestep + 1e-12
        elif scenario == "plate":
            assert not terminated and not env.task.logic.acquired and e["on_plate"], result
        elif scenario == "penetration":
            assert terminated and env.task.failure_reason == "model_penetration", result
        return result, trace
    finally:
        env.close()
