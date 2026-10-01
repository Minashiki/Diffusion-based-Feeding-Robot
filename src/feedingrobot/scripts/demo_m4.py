"""Full physical teacher demonstration, with explicit failure exit status."""

import argparse
import json
from datetime import datetime, timezone

from feedingrobot.data.rollout import run_episode
from feedingrobot.sim.model import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="panda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/collect.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / (args.output or f"outputs/calibration/m4/demo_{args.robot}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}")
    result = run_episode(args.robot, args.seed, config, output,
                         scenario=dict(config["scene"], recover=args.recover), viewer=not args.headless)
    print(json.dumps({key: result[key] for key in ("success", "time_s", "phase", "failure_reason", "contact_peak_n")}, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
