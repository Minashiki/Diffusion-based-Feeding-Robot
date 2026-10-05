"""Small M5 engineering validation, or trained-policy acceptance and freezing."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies.audit import parent_check, read_json, sha256
from feedingrobot.policies.data import ActionWindows, fit_normalization
from feedingrobot.policies.dit import ActionDiT, noise_loss, sample_actions, schedulers
from feedingrobot.policies.evaluation import run_policy
from feedingrobot.policies.runtime import precision, rng_state, save_checkpoint, setup, to_device
from feedingrobot.sim.model import ROOT


def engineering(config, output, device='cuda', updates=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device, hardware = setup(config, device)
    print('Checking frozen M4 files; no physical replay...', flush=True)
    parent = parent_check(config)
    print('Building legal train windows and train-only normalization...', flush=True)
    dataset = ActionWindows(ROOT / config['dataset'], 'train', config['model']['horizon'])
    normalization = fit_normalization(dataset)
    dataset.normalization = normalization
    rng = np.random.default_rng(config['training']['seed'])
    batch = to_device(default_collate([dataset[i] for i in dataset.sample_indices(rng, 8)]), device)
    model = ActionDiT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                 weight_decay=config['training']['weight_decay'])
    scheduler, _ = schedulers(config)
    noise = torch.randn_like(batch['actions'])
    steps = torch.full((8,), 50, device=device, dtype=torch.long)
    losses = []
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            loss = noise_loss(model, scheduler, batch, noise=noise, timesteps=steps)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite diagnostic loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 20 == 0:
            print(f'fixed batch {step+1}/{updates}: loss={losses[-1]:.6f}', flush=True)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(1)
    with precision(device):
        samples = sample_actions(model, config, batch, normalization, generator)
    if samples.shape != (8, config['model']['horizon'], 6) or not torch.isfinite(samples).all():
        raise ValueError('Invalid DDIM samples')
    count = min(10, max(1, updates // 4))
    initial, final = float(np.mean(losses[:count])), float(np.mean(losses[-count:]))
    passed = updates >= 20 and final <= initial * .5
    state = model.state_dict()
    checkpoint = dict(schema_version=1, diagnostic=True, config=config, parent=parent, source_hashes=input_hashes(),
                      normalization=normalization, model=state, ema=state, optimizer=optimizer.state_dict(),
                      step=updates, best_score=-1., rng=rng_state(rng), hardware=hardware)
    save_checkpoint(output / 'diagnostic.pt', checkpoint)
    # A short prefix validates the real physics integration, never policy success.
    episode = dataset.episodes[0][0]
    prefix = run_policy(model, config, normalization, episode, device, max_ticks=100)
    write_json(output / 'physical_prefix.json', prefix)
    write_json(output / 'normalization.json', normalization)
    report = dict(status='ready_for_training' if passed else 'failed', implementation_check='passed' if passed else 'failed',
                  formal_training='not_run', closed_loop_acceptance='not_verified', dp_v1='not_frozen',
                  config=config, parent=parent, hardware=hardware, updates=updates, batch_size=8,
                  initial_loss=initial, final_loss=final, losses=losses,
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_windows=len(dataset), legal_phase_counts={str(k): sum(map(len, v.values())) for k, v in dataset.by_phase.items()},
                  short_physics_ticks=round(prefix['simulated_s']/.001), source_hashes=input_hashes())
    write_json(output / 'engineering_report.json', report)
    return report


def freeze_policy(checkpoint_path, output, *, device='cuda', headless=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('diagnostic') or checkpoint['step'] < 1:
        raise ValueError('Diagnostic or untrained weights cannot be released')
    from feedingrobot.scripts.eval_dp import evaluate_checkpoint
    report = evaluate_checkpoint(checkpoint_path, 'test', output, device, headless)
    if report['status'] != 'passed':
        return report
    output = Path(output)
    write_json(output / 'freeze_manifest.json', dict(schema_version=1, status='frozen', policy_version='DP_v1',
               backbone='dit_adaln_cross_attention', robot_id='panda', checkpoint=str(Path(checkpoint_path).resolve()),
               checkpoint_sha256=sha256(checkpoint_path), config=checkpoint['config'], normalization=checkpoint['normalization'],
               parent=report['parent'], source_hashes=checkpoint['source_hashes'], training_step=checkpoint['step'],
               report_sha256=sha256(output / 'report.json'),
               evidence_sha256={p.name: sha256(p) for p in output.glob('*.json') if p.name != 'freeze_manifest.json'}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dp_dit.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--updates', type=int, default=200, help='Fixed-batch diagnostic updates, maximum 200')
    parser.add_argument('--checkpoint')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        if not args.checkpoint:
            parser.error('--freeze requires --checkpoint')
        report = freeze_policy(args.checkpoint, args.output, device=args.device, headless=args.headless)
        return 0 if report['status'] == 'passed' else 1
    if args.checkpoint or not 1 <= args.updates <= 200:
        parser.error('Engineering mode uses 1..200 updates and no --checkpoint')
    report = engineering(read_json(ROOT / args.config), args.output, args.device, args.updates)
    return 0 if report['implementation_check'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
