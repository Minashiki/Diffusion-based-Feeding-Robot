"""EMA diffusion training and checkpointing with exact phase sampler resume."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies.audit import checkpoint_check, parent_check
from feedingrobot.policies.data import ActionWindows, fit_normalization
from feedingrobot.policies.dit import ActionDiT, noise_loss, schedulers
from feedingrobot.policies.runtime import precision, restore_rng, rng_state, save_checkpoint, setup, to_device
from feedingrobot.sim.model import ROOT


class PhaseBatches:
    """Prefetch uses a separate RNG; checkpoints track consumed batches only."""
    def __init__(self, dataset, rng, count, batch_size):
        self.dataset, self.rng = dataset, copy.deepcopy(rng)
        self.count, self.batch_size = count, batch_size

    def __iter__(self):
        for _ in range(self.count):
            yield self.dataset.sample_indices(self.rng, self.batch_size)

    def __len__(self):
        return self.count


@torch.no_grad()
def update_ema(ema, model, decay):
    for average, parameter in zip(ema.parameters(), model.parameters()):
        average.lerp_(parameter.detach(), 1-decay)


@torch.no_grad()
def validation_loss(model, dataset, scheduler, config, device):
    # Fixed validation noise/samples; this must not consume training RNG state.
    rng = np.random.default_rng(123)
    generator = torch.Generator(device=device).manual_seed(123)
    total = 0.
    model.eval()
    for _ in range(config['training']['validation_batches']):
        indices = dataset.sample_indices(rng, config['training']['batch_size'])
        batch = to_device(default_collate([dataset[i] for i in indices]), device)
        noise = torch.randn(batch['actions'].shape, device=device, generator=generator)
        steps = torch.randint(scheduler.config.num_train_timesteps, (len(indices),), device=device, generator=generator)
        with precision(device):
            total += float(noise_loss(model, scheduler, batch, noise=noise, timesteps=steps))
    return total / config['training']['validation_batches']


def train(config, output, *, device='cuda', resume=None, updates=None, headless=False):
    device, hardware = setup(config, device)
    output = Path(output)
    parent = parent_check(config)
    source = input_hashes()
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if checkpoint:
        checkpoint_check(checkpoint, config, parent)
        if checkpoint.get('diagnostic'):
            raise ValueError('Diagnostic overfit is not a formal training checkpoint')
        if output.resolve() != Path(resume).resolve().parent:
            raise ValueError('Resume must use the original run directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
    raw = ActionWindows(ROOT / config['dataset'], 'train', config['model']['horizon'])
    normalization = checkpoint['normalization'] if checkpoint else fit_normalization(raw)
    raw.normalization = normalization
    validation = ActionWindows(ROOT / config['dataset'], 'validation', config['model']['horizon'], normalization)
    model = ActionDiT(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    settings = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler, _ = schedulers(config)
    rng = np.random.default_rng(settings['seed'])
    step, best = 0, -1.
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        step, best = checkpoint['step'], checkpoint['best_score']
        restore_rng(checkpoint['rng'], rng)
    write_json(output / 'config.json', config)
    write_json(output / 'normalization.json', normalization)
    write_json(output / 'parent_audit.json', parent)
    write_json(output / 'hardware.json', hardware)
    write_json(output / 'data_summary.json', dict(windows=len(raw),
               phases={str(phase): sum(map(len, episodes.values())) for phase, episodes in raw.by_phase.items()},
               normalization_count=normalization['count'], parameters=sum(p.numel() for p in model.parameters())))
    total = settings['updates'] if updates is None else updates
    if total <= step:
        raise ValueError('Target updates must exceed the resumed step')
    loader_generator = torch.Generator().manual_seed(settings['seed'])
    loader = DataLoader(raw, batch_sampler=PhaseBatches(raw, rng, total-step, settings['batch_size']),
                        num_workers=settings['workers'], pin_memory=device.type == 'cuda',
                        generator=loader_generator, persistent_workers=bool(settings['workers']),
                        multiprocessing_context='spawn' if settings['workers'] else None)
    def payload():
        return dict(schema_version=1, config=config, parent=parent, source_hashes=source, normalization=normalization,
                    model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, best_score=best, rng=rng_state(rng), diagnostic=False, hardware=hardware)
    write_json(output / 'status.json', dict(status='training', step=step))
    with (output / 'metrics.jsonl').open('a') as metrics:
        for batch in loader:
            raw.sample_indices(rng, settings['batch_size'])
            batch = to_device(batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with precision(device):
                loss = noise_loss(model, scheduler, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite diffusion loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            update_ema(ema, model, settings['ema_decay'])
            step += 1
            record = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm))
            if step % settings['validation_every'] == 0:
                record['validation_loss'] = validation_loss(ema, validation, scheduler, config, device)
            if step % settings['evaluation_every'] == 0:
                from feedingrobot.policies.evaluation import evaluate
                report = evaluate(ema, config, normalization, device, 'validation', output / f'validation_{step}', viewer=not headless)
                record['closed_loop_score'] = report['score']
                if report['score'] > best:
                    best = report['score']
                    save_checkpoint(output / 'best.pt', payload())
            metrics.write(json.dumps(record) + '\n')
            if step % 100 == 0 or step == total:
                print(f'update={step}/{total} loss={record["loss"]:.6f}', flush=True)
                metrics.flush()
            if step % settings['checkpoint_every'] == 0:
                if input_hashes() != source:
                    raise ValueError('Runtime source changed during training')
                save_checkpoint(output / f'step_{step}.pt', payload())
                save_checkpoint(output / 'last.pt', payload())
        if input_hashes() != source:
            raise ValueError('Runtime source changed during training')
        save_checkpoint(output / 'last.pt', payload())
    write_json(output / 'status.json', dict(status='trained_not_released', step=step, best_validation_score=best,
                                          m5_status='incomplete', test_status='not_verified'))
    return output
