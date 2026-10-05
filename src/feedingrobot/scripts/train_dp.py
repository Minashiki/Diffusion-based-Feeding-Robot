"""Train the M5 action DiT; full runs are explicitly started by the user."""

import argparse
from pathlib import Path

from feedingrobot.policies.audit import read_json
from feedingrobot.sim.model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dp_dit.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--resume')
    parser.add_argument('--updates', type=int, help='Total target updates, including the resumed steps')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    from feedingrobot.policies.training import train
    train(read_json(ROOT / args.config), Path(args.output), device=args.device, resume=args.resume,
          updates=args.updates, headless=args.headless)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
