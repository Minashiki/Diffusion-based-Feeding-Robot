"""Train the M5 action DiT; full runs are explicitly started by the user."""

import argparse
from pathlib import Path

from feedingrobot.policies.audit import read_json
from feedingrobot.policies.models import MODEL_NAMES, model_name, resolve_config
from feedingrobot.sim.model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--model', choices=MODEL_NAMES, help='Fresh runs default to MinaDiTV3')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--resume')
    parser.add_argument('--updates', type=int, help='Total target updates, including the resumed steps')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    config = resolve_config(args.config, args.model) if not args.resume else None
    if args.resume:
        import torch
        from feedingrobot.data.episodes import input_hashes
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        config = checkpoint['config']
        if args.model is not None and args.model != model_name(config):
            parser.error('Selected model and resume checkpoint differ')
        if args.config is not None and read_json(ROOT / args.config) != config:
            parser.error('Selected configuration and resume checkpoint differ')
        if model_name(config) == 'MinaDiTv1' and checkpoint['source_hashes'] != input_hashes():
            from feedingrobot.policies.legacy import run
            if checkpoint['schema_version'] != 1:
                parser.error('Historical correction forks resume through feedingrobot.policies.legacy and their original training entry')
            arguments = ['--resume', str(Path(args.resume).resolve()), '--output', str(Path(args.output).resolve()),
                         '--device', args.device]
            if args.config is not None:
                arguments += ['--config', str((ROOT / args.config).resolve())]
            if args.updates is not None:
                arguments += ['--updates', str(args.updates)]
            if args.headless:
                arguments.append('--headless')
            return run('feedingrobot.scripts.train_dp', arguments, expected=checkpoint['source_hashes'])
    from feedingrobot.policies.training import train
    train(config, Path(args.output), device=args.device, resume=args.resume,
          updates=args.updates, headless=args.headless)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
