"""Gym interface smoke demo; no teacher and no full feeding claim."""

import argparse
import json
import time

import numpy as np

from feedingrobot.envs import FeedingGymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--preset", choices=["beans_in_bowl", "beans_on_spoon"], default="beans_in_bowl")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=2.)
    args = parser.parse_args()
    env = FeedingGymEnv(args.robot, render_mode=None if args.headless else "human", max_episode_s=args.seconds)
    try:
        env.reset(seed=0, options={"preset": args.preset})
        started = time.monotonic()
        done = False
        while not done:
            _, reward, terminated, truncated, info = env.step(np.zeros(6))
            done = terminated or truncated
            if info["events"]:
                print(json.dumps(dict(time=info["time"], phase=info["phase"], reward=reward, events=info["events"])))
            if env.viewer:
                if not env.viewer.is_running():
                    break
                time.sleep(max(0., started + info["time"] - time.monotonic()))
        print(json.dumps(dict(robot=args.robot, time=info["time"], phase=info["phase"],
                              failure_reason=info["failure_reason"], success=info["success"],
                              scope="zero-action Gym smoke demo, not a feeding teacher")))
        if info["failure_reason"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
