"""EMA diffusion training and checkpointing with exact phase sampler resume."""

import copy
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies.audit import checkpoint_check, parent_check
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import noise_loss, schedulers
from feedingrobot.policies.dp_normalizer import fit_dp_limits
from feedingrobot.policies.corpus import TrainingWindows, correction_check, start_pose_check
from feedingrobot.policies.models import check_normalization, create_model, model_name
from feedingrobot.policies.normalization import denormalize_actions, fit_limits
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
        with precision(device, config):
            total += float(noise_loss(model, scheduler, batch, noise=noise, timesteps=steps))
    return total / config['training']['validation_batches']


@torch.no_grad()
def validation_actions(model, dataset, config, normalization, device):
    """Fixed expert observations; score physical sampled actions before control clipping."""
    from feedingrobot.policies.dit import sample_actions
    from feedingrobot.sim.model import load_json
    count = config['training']['validation_action_samples']
    rng = np.random.default_rng(123)
    first = {}
    for index, (episode, _, _, _) in enumerate(dataset.windows):
        first.setdefault(episode, index)
    starts = [first[episode] for episode in rng.permutation(list(first))[:max(1, count//2)]]
    indices = starts + dataset.sample_indices(rng, count-len(starts))
    batch = to_device(default_collate([dataset[i] for i in indices]), device)
    model.eval()
    prediction = sample_actions(model, config, batch, normalization, torch.Generator(device=device).manual_seed(123))
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite sampled validation actions')
    truth = denormalize_actions(batch['actions'], normalization)
    robot = load_json(f'configs/robots/{config["robot_id"]}.json')
    result = {}
    for name in ('episode_start', 'first_action', 'first_four', 'legal_horizon'):
        mask = batch['action_mask'].clone()
        if name == 'episode_start':
            mask[len(starts):] = False
            mask[:, 1:] = False
        elif name == 'first_action':
            mask[:, 1:] = False
        elif name == 'first_four':
            mask[:, 4:] = False
        metrics = dict(actions=int(mask.sum()))
        for field, columns in (('linear', slice(0, 3)), ('angular', slice(3, 6))):
            error = (prediction[..., columns] - truth[..., columns])[mask]
            speed = prediction[..., columns][mask].norm(dim=-1)
            metrics[field] = dict(vector_rmse=float(error.square().sum(-1).mean().sqrt()),
                                 zero_action_vector_rmse=float(truth[..., columns][mask].square().sum(-1).mean().sqrt()),
                                 over_limit_fraction=float((speed > robot[f'{field}_speed_limit']).float().mean()))
        result[name] = metrics
    return result


def train(config, output, *, device='cuda', resume=None, updates=None, headless=False):
    device, hardware = setup(config, device)
    output = Path(output)
    parent = parent_check(config)
    source = input_hashes()
    correction = correction_check(config)
    extension = start_pose_check(config)
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if checkpoint:
        checkpoint_check(checkpoint, config, parent)
        if checkpoint.get('diagnostic'):
            raise ValueError('Diagnostic overfit is not a formal training checkpoint')
        if output.resolve() != Path(resume).resolve().parent:
            raise ValueError('Resume must use the original run directory')
        if checkpoint.get('correction') != correction:
            raise ValueError('Checkpoint correction data binding differs')
        if checkpoint.get('train_extension') != extension:
            raise ValueError('Checkpoint start-pose data binding differs')
        if checkpoint['normalization'] != json.loads((output / 'normalization.json').read_text()):
            raise ValueError('Checkpoint normalization differs from its run')
    else:
        output.mkdir(parents=True, exist_ok=False)
    raw = TrainingWindows(config, correction, extension=extension)
    normalization = (checkpoint['normalization'] if checkpoint else
                     fit_dp_limits(raw) if model_name(config) == 'MinaDiTV5' else
                     fit_limits(raw))
    if not checkpoint and model_name(config) == 'MinaDiTV4':
        normalization['observation_timing'] = 'pre_action'
    if not checkpoint:
        normalization['source_binding'] = dict(m4_dataset_binding=parent['dataset_binding'],
                                               correction_report_sha256=correction['report_sha256'] if correction else None)
        if extension:
            normalization['source_binding']['train_extension_report_sha256'] = extension['report_sha256']
    check_normalization(config, normalization)
    raw.set_normalization(normalization)
    validation = ActionWindows(ROOT / config['dataset'], 'validation', config['model']['horizon'], normalization)
    model = create_model(config).to(device)
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
               normalization_count=normalization['count'], model_name=model_name(config),
               parameters=sum(p.numel() for p in model.parameters()), **raw.summary()))
    write_json(output / 'correction_audit.json', correction)
    if extension:
        write_json(output / 'train_extension_audit.json', extension)
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
                    model_name=model_name(config), correction=correction, train_extension=extension,
                    model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, best_score=best, rng=rng_state(rng), diagnostic=False, hardware=hardware)
    write_json(output / 'status.json', dict(status='training', step=step))
    with (output / 'metrics.jsonl').open('a') as metrics:
        for batch in loader:
            raw.sample_indices(rng, settings['batch_size'])
            batch = to_device(batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with precision(device, config):
                loss = noise_loss(model, scheduler, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite diffusion loss')
            with precision(device, config) if settings.get('precision') == 'fp32' else contextlib.nullcontext():
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            update_ema(ema, model, settings['ema_decay'])
            step += 1
            record = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm))
            if step % settings['validation_every'] == 0:
                record['validation_loss'] = validation_loss(ema, validation, scheduler, config, device)
                if settings.get('validation_action_samples'):
                    record['validation_actions'] = validation_actions(ema, validation, config, normalization, device)
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
        if correction_check(config) != correction:
            raise ValueError('Correction inputs changed during training')
        if start_pose_check(config) != extension:
            raise ValueError('Start-pose inputs changed during training')
        save_checkpoint(output / 'last.pt', payload())
    write_json(output / 'status.json', dict(status='trained_not_released', step=step, best_validation_score=best,
                                          m5_status='incomplete', test_status='not_verified'))
    return output
