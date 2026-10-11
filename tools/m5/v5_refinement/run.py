"""Independent, audited low-LR refinement of the V5 90k checkpoint."""

import argparse
import contextlib
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies.audit import checkpoint_check, parent_check, read_json, sha256
from feedingrobot.policies.corpus import TrainingWindows, correction_check, start_pose_check
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import noise_loss, schedulers
from feedingrobot.policies.evaluation import evaluate
from feedingrobot.policies.models import create_model, model_name
from feedingrobot.policies.runtime import precision, restore_rng, rng_state, save_checkpoint, setup, to_device
from feedingrobot.policies.training import PhaseBatches, update_ema, validation_actions, validation_loss
from feedingrobot.sim.model import ROOT

START_STEP, SWITCH_STEP, FINAL_STEP = 90000, 100000, 110000
VALIDATION_EVERY, CHECKPOINT_EVERY, EVALUATION_EVERY = 1000, 5000, 5000


def learning_rate(step):
    return 2e-5 if step < SWITCH_STEP else 1e-5


def refinement_config(original):
    config = copy.deepcopy(original)
    config['training'].update(lr=2e-5, ema_decay=.9998, updates=FINAL_STEP,
                              validation_every=VALIDATION_EVERY,
                              checkpoint_every=CHECKPOINT_EVERY, evaluation_every=EVALUATION_EVERY)
    return config


def closed_loop_counts(directory):
    report = read_json(directory / 'report.json')
    if any(report['full'][kind]['attempts'] != 15 for kind in ('normal', 'recovery')):
        raise ValueError('Refinement requires the fixed 15 normal + 15 recovery validation cases')
    cases = [read_json(path) for path in sorted(directory.glob('*.json')) if path.name != 'report.json']
    for kind, recover in (('normal', False), ('recovery', True)):
        selected = [case for case in cases if case['recover'] == recover]
        if len(selected) != 15 or sum(case['success'] for case in selected) != report['full'][kind]['successes']:
            raise ValueError('Validation report differs from its episode results')
    if sum(case['pickup'] for case in cases) != report['stages']['pickup']['successes']:
        raise ValueError('Validation pickup count differs from its episode results')
    return dict(full_successes=sum(report['full'][kind]['successes'] for kind in ('normal', 'recovery')),
                pickup_successes=report['stages']['pickup']['successes'],
                normal_pickups=sum(case['pickup'] for case in cases if not case['recover']))


def validation_stats(directory, actions):
    first_four = actions['first_four']
    error = sum(first_four[kind]['vector_rmse'] / first_four[kind]['zero_action_vector_rmse']
                for kind in ('linear', 'angular'))
    return dict(**closed_loop_counts(directory), action_error=error)


def selection_key(stats):
    return stats['full_successes'], stats['pickup_successes'], -stats['action_error']


def gate_passed(stats):
    return stats['full_successes'] > 0 or (stats['pickup_successes'] >= 5 and stats['normal_pickups'] > 0)


def prepare(init_from, resume, output):
    path = Path(resume or init_from).resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if resume:
        if path.name != 'last.pt' or path.parent != output:
            raise ValueError('Resume must use last.pt in the refinement output directory')
        provenance = checkpoint['refinement']
        original_path = Path(provenance['parent_checkpoint'])
        original = torch.load(original_path, map_location='cpu', weights_only=False, mmap=True)
    else:
        original_path, original = path, checkpoint
        if output.exists():
            raise ValueError('A new refinement output directory must not already exist')
    if (output.is_relative_to(original_path.parent) or original_path.parent.is_relative_to(output)):
        raise ValueError('Refinement output must be separate from the original run')
    if original.get('diagnostic') or original.get('refinement') or model_name(original['config']) != 'MinaDiTV5':
        raise ValueError('Requires an original, formal MinaDiTV5 checkpoint')
    if original['step'] != START_STEP:
        raise ValueError(f'Refinement must start at step {START_STEP}')
    parent = parent_check(original['config'])
    checkpoint_check(original, original['config'], parent)
    correction, extension = correction_check(original['config']), start_pose_check(original['config'])
    if original.get('correction') != correction or original.get('train_extension') != extension:
        raise ValueError('Parent training data binding differs')
    if original['normalization'] != read_json(original_path.parent / 'normalization.json'):
        raise ValueError('Parent normalization differs from its run')
    config = refinement_config(original['config'])
    if resume:
        if (provenance['parent_sha256'] != sha256(original_path)
                or provenance['parent_config'] != original['config']
                or provenance['tool_sha256'] != sha256(__file__)
                or provenance['switch_step'] != SWITCH_STEP
                or provenance != read_json(output / 'refinement.json')):
            raise ValueError('Refinement parent, tool or provenance changed')
        for evidence, expected in provenance['baseline_evidence'].items():
            if sha256(evidence) != expected:
                raise ValueError('Parent validation evidence changed')
        checkpoint_check(checkpoint, config, parent)
        if (checkpoint['normalization'] != original['normalization']
                or checkpoint.get('correction') != correction or checkpoint.get('train_extension') != extension
                or read_json(output / 'config.json') != config
                or read_json(output / 'normalization.json') != original['normalization']):
            raise ValueError('Refinement configuration, normalization or data binding differs')
        selection, gate = checkpoint['selection'], checkpoint['progress_gate']
        if gate is not None:
            for evidence, expected in gate['evidence'].items():
                if sha256(evidence) != expected:
                    raise ValueError('Refinement gate evidence changed')
            counts = closed_loop_counts(Path(gate['directory']))
            if (gate['passed'] != gate_passed(counts)
                    or any(gate['stats'][key] != value for key, value in counts.items())):
                raise ValueError('Refinement gate decision differs from its validation evidence')
        if checkpoint['step'] >= SWITCH_STEP and gate is None:
            raise ValueError('Missing 100k progress gate')
    else:
        metrics_path = original_path.parent / 'metrics.jsonl'
        with metrics_path.open() as stream:
            record = next(row for row in map(json.loads, stream) if row['step'] == START_STEP)
        directory = original_path.parent / f'validation_{START_STEP}'
        selection = dict(step=START_STEP, **validation_stats(directory, record['validation_actions']))
        evidence = [metrics_path, *sorted(directory.glob('*.json'))]
        provenance = dict(schema_version=1, parent_checkpoint=str(original_path), parent_step=START_STEP,
                          parent_sha256=sha256(original_path), parent_config=original['config'],
                          tool_sha256=sha256(__file__), switch_step=SWITCH_STEP,
                          baseline=selection, baseline_evidence={str(p): sha256(p) for p in evidence})
        gate = None
    return checkpoint, config, parent, correction, extension, provenance, selection, gate


def refine(output, *, init_from=None, resume=None, updates=None, device='cuda', headless=False, check_only=False):
    output = Path(output).resolve()
    if bool(init_from) == bool(resume):
        raise ValueError('Select exactly one of init_from or resume')
    checkpoint, config, parent, correction, extension, provenance, selection, gate = prepare(init_from, resume, output)
    step = checkpoint['step']
    total = FINAL_STEP if updates is None else updates
    if not step < total <= FINAL_STEP:
        raise ValueError(f'Target updates must exceed {step} and be at most {FINAL_STEP}')
    if check_only:
        return dict(status='checked', step=step, target=total, lr=learning_rate(step), ema_decay=.9998,
                    progress_gate=gate, parent_checkpoint=provenance['parent_checkpoint'], output=str(output))
    if gate is not None and not gate['passed']:
        raise ValueError('The 100k progress gate failed; this refinement cannot continue')
    device, hardware = setup(config, device)
    source = input_hashes()
    normalization = checkpoint['normalization']
    raw = TrainingWindows(config, correction, extension=extension)
    raw.set_normalization(normalization)
    validation = ActionWindows(ROOT / config['dataset'], 'validation', config['model']['horizon'], normalization)
    model = create_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    ema.load_state_dict(checkpoint['ema'])
    settings = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(step), weight_decay=settings['weight_decay'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(step)
    scheduler, _ = schedulers(config)
    rng = np.random.default_rng(settings['seed'])
    restore_rng(checkpoint['rng'], rng)

    def payload():
        return dict(schema_version=1, config=config, parent=parent, source_hashes=source,
                    normalization=normalization, model_name=model_name(config), correction=correction,
                    train_extension=extension, model=model.state_dict(), ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(), step=step, best_score=selection['full_successes']/30,
                    rng=rng_state(rng), diagnostic=False, hardware=hardware, refinement=provenance,
                    selection=selection, progress_gate=gate)

    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (('config', config), ('normalization', normalization), ('refinement', provenance),
                            ('parent_audit', parent), ('hardware', hardware), ('correction_audit', correction)):
            write_json(output / f'{name}.json', value)
        if extension:
            write_json(output / 'train_extension_audit.json', extension)
        save_checkpoint(output / f'step_{step}.pt', payload())
        save_checkpoint(output / 'last.pt', payload())
    else:
        # Discard only uncommitted metric rows from this branch after an interruption.
        path = output / 'metrics.jsonl'
        if path.exists():
            lines, kept = path.read_text().splitlines(), []
            for index, line in enumerate(lines):
                try:
                    recorded_step = json.loads(line)['step']
                except json.JSONDecodeError:
                    if index != len(lines)-1:
                        raise
                    continue  # An interrupted write can leave an incomplete trailing row.
                if recorded_step <= step:
                    kept.append(line)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(''.join(line + '\n' for line in kept))
            temporary.replace(path)
    best_source = output / f'step_{selection["step"]}.pt'
    temporary = output / 'best.tmp'
    shutil.copyfile(best_source, temporary)
    temporary.replace(output / 'best.pt')
    write_json(output / 'selection.json', selection)
    loader = DataLoader(raw, batch_sampler=PhaseBatches(raw, rng, total-step, settings['batch_size']),
                        num_workers=settings['workers'], pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(settings['seed']),
                        persistent_workers=bool(settings['workers']),
                        multiprocessing_context='spawn' if settings['workers'] else None)
    write_json(output / 'status.json', dict(status='training', step=step))
    with (output / 'metrics.jsonl').open('a') as metrics:
        for batch in loader:
            raw.sample_indices(rng, settings['batch_size'])
            batch = to_device(batch, device)
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group['lr'] = lr
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
            record = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), lr=lr,
                          ema_decay=settings['ema_decay'])
            improved = False
            if step % settings['validation_every'] == 0:
                record['validation_loss'] = validation_loss(ema, validation, scheduler, config, device)
                record['validation_actions'] = validation_actions(ema, validation, config, normalization, device)
            if step % settings['evaluation_every'] == 0:
                directory = output / f'validation_{step}'
                attempt = 0
                while directory.exists():
                    attempt += 1
                    directory = output / f'validation_{step}_retry_{attempt}'
                report = evaluate(ema, config, normalization, device, 'validation', directory, viewer=not headless)
                stats = validation_stats(directory, record['validation_actions'])
                record.update(closed_loop_score=report['score'], closed_loop=stats, validation_directory=str(directory))
                if selection_key(stats) > selection_key(selection):
                    selection, improved = dict(step=step, **stats), True
                if step == SWITCH_STEP:
                    gate = dict(passed=gate_passed(stats), stats=stats, directory=str(directory),
                                evidence={str(p): sha256(p) for p in sorted(directory.glob('*.json'))})
                    record['progress_gate'] = gate
            metrics.write(json.dumps(record) + '\n')
            if step % 100 == 0 or step == total:
                metrics.flush()
                print(f'update={step}/{total} lr={lr:g} loss={record["loss"]:.6f}', flush=True)
            if step % settings['checkpoint_every'] == 0 or step == total:
                if input_hashes() != source or sha256(__file__) != provenance['tool_sha256']:
                    raise ValueError('Runtime source or refinement tool changed')
                save_checkpoint(output / f'step_{step}.pt', payload())
                save_checkpoint(output / 'last.pt', payload())
                if improved:
                    save_checkpoint(output / 'best.pt', payload())
                write_json(output / 'selection.json', selection)
            if gate is not None and not gate['passed']:
                print('Stopped: 100k progress gate failed.', flush=True)
                break
    if correction_check(config) != correction or start_pose_check(config) != extension:
        raise ValueError('Training data changed during refinement')
    if input_hashes() != source or sha256(__file__) != provenance['tool_sha256']:
        raise ValueError('Runtime source or refinement tool changed')
    status = dict(status='trained_not_released', step=step, best_validation_score=selection['full_successes']/30,
                  best_selection=selection, progress_gate=gate,
                  stop_reason='progress_gate_failed' if gate is not None and not gate['passed'] else 'target_reached',
                  m5_status='incomplete', test_status='not_verified')
    write_json(output / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument('--init-from', help='Original step_90000.pt; create a separate refinement run')
    start.add_argument('--resume', help='This refinement run\'s last.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, default=FINAL_STEP, help='Cumulative target, at most 110000')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Read-only provenance check; no training or GPU needed')
    args = parser.parse_args()
    result = refine(args.output, init_from=args.init_from, resume=args.resume, updates=args.updates,
                    device=args.device, headless=args.headless, check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
