"""Evaluate an EMA DiT checkpoint without running the teacher or replaying M4."""

import argparse
from pathlib import Path

import torch

from feedingrobot.data.episodes import write_json
from feedingrobot.policies.audit import checkpoint_check, parent_check, read_json, sha256
from feedingrobot.policies.models import MODEL_NAMES, check_normalization, create_model, model_name
from feedingrobot.policies.corpus import correction_check
from feedingrobot.policies.evaluation import evaluate
from feedingrobot.policies.runtime import setup


def evaluate_checkpoint(path, split, output, device='cuda', headless=False, selected_model=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = checkpoint['config']
    if selected_model is not None and selected_model != model_name(config):
        raise ValueError('Selected model and evaluation checkpoint differ')
    from feedingrobot.data.episodes import input_hashes
    if model_name(config) == 'MinaDiTv1' and checkpoint['source_hashes'] != input_hashes():
        from feedingrobot.policies.legacy import run
        if checkpoint['schema_version'] != 1:
            raise ValueError('Historical correction forks evaluate through feedingrobot.policies.legacy and their original evaluation entry')
        arguments = ['--checkpoint', str(Path(path).resolve()), '--split', split,
                     '--output', str(Path(output).resolve()), '--device', device]
        if headless:
            arguments.append('--headless')
        code = run('feedingrobot.scripts.eval_dp', arguments, expected=checkpoint['source_hashes'])
        report_path = Path(output) / 'report.json'
        if code not in (0, 1) or not report_path.exists():
            raise RuntimeError('Historical evaluation failed')
        return read_json(report_path)
    device, hardware = setup(config, device)
    parent = parent_check(config)
    checkpoint_check(checkpoint, config, parent)
    if checkpoint.get('correction') != correction_check(config):
        raise ValueError('Checkpoint correction data binding differs')
    check_normalization(config, checkpoint['normalization'])
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint['ema'])
    report = evaluate(model, config, checkpoint['normalization'], device, split, output, viewer=not headless)
    report.update(checkpoint_sha256=sha256(path), parent=parent, hardware=hardware,
                  training_step=checkpoint['step'], model_name=model_name(config), diagnostic=checkpoint.get('diagnostic', False))
    write_json(Path(output) / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', choices=MODEL_NAMES, help='Optional checkpoint model assertion')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.split, args.output, args.device, args.headless, args.model)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
