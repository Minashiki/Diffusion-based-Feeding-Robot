"""Short physical twist demo for either configured robot."""

import argparse
from contextlib import nullcontext
import json
import time

import numpy as np

from feedingrobot.sim.task import FeedingTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=4.)
    args = parser.parse_args()
    task = FeedingTask(args.robot)
    task.reset(preset="food_on_spoon")
    if args.headless:
        context = nullcontext(None)
    else:
        from feedingrobot.sim.viewer import passive_viewer
        context = passive_viewer(task.model, task.data)
    with context as viewer:
        started = time.monotonic()
        for tick in range(round(args.seconds / task.dt)):
            if tick % 20 == 0:
                t = task.data.time
                task.adapter.set_twist([.01 * np.sin(2 * np.pi * t / 2), 0, 0, 0, 0, .03 * np.cos(t)], t, t + .04)
            state = task.step_physics()
            if state["terminated"] or task.adapter.fault:
                break
            if viewer:
                viewer.sync()
                if not viewer.is_running():
                    break
                time.sleep(max(0., started + (tick + 1) * task.dt - time.monotonic()))
        task.adapter.stop()
    print(json.dumps({k: state[k] for k in ["robot_id", "time", "execution_status", "terminated", "failure_reason", "contact_peak_n"]}, indent=2))
    if task.terminated or task.adapter.fault:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
