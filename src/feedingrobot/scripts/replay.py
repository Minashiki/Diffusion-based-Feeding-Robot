"""Replay a trusted local M4 episode through the same physical execution path."""

import argparse
import json

from feedingrobot.data.replay import replay_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    from feedingrobot.data.episodes import load_episode
    manifest, _ = load_episode(args.episode)
    if manifest["robot_id"] != args.robot:
        raise ValueError("Episode robot does not match --robot")
    print(json.dumps(replay_episode(args.episode, viewer=args.viewer), indent=2))


if __name__ == "__main__":
    main()
