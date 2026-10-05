"""Evaluate an EMA DiT checkpoint without running the teacher or replaying M4."""

import argparse
from pathlib import Path

import torch

from feedingrobot.data.episodes import write_json
from feedingrobot.policies.audit import checkpoint_check, parent_check, sha256
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.policies.evaluation import evaluate
from feedingrobot.policies.runtime import setup


def evaluate_checkpoint(path, split, output, device='cuda', headless=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    device, hardware = setup(config, device)
    parent = parent_check(config)
    checkpoint_check(checkpoint, config, parent)
    model = ActionDiT(config).to(device)
    model.load_state_dict(checkpoint['ema'])
    report = evaluate(model, config, checkpoint['normalization'], device, split, output, viewer=not headless)
    report.update(checkpoint_sha256=sha256(path), parent=parent, hardware=hardware,
                  training_step=checkpoint['step'], diagnostic=checkpoint.get('diagnostic', False))
    write_json(Path(output) / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.split, args.output, args.device, args.headless)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
