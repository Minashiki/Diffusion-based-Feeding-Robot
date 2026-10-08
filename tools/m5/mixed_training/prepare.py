"""Audit 400 mixed episodes and fit train-only scaling, with zero optimizer updates."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies.audit import parent_check, read_json, sha256
from feedingrobot.policies.corpus import TrainingWindows, start_pose_check
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import noise_loss, schedulers
from feedingrobot.policies.models import check_normalization, create_model
from feedingrobot.policies.normalization import fit_minmax
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/single_bean/mixed_v1_start_pose_v2/preparation_001')
    args = parser.parse_args()
    config_path = Path(__file__).with_name('config.json')
    config = read_json(config_path)
    device, hardware = setup(config, 'cpu')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(status='preparing', optimizer_created=False, optimizer_updates=0))
    source = input_hashes()
    parent = parent_check(config)
    extension = start_pose_check(config)
    raw = TrainingWindows(config, extension=extension)
    episodes = raw.base.episodes + raw.extension.episodes
    counts = Counter('recovery' if m['scenario'].get('recover') else 'normal' for _, m in episodes)
    if counts != dict(normal=300, recovery=100) or len(episodes) != 400:
        raise ValueError('Requires exactly 300 normal and 100 recovery training episodes')
    # Exercise at least one legal window from every episode before fitting.
    for reader in (raw.base, raw.extension):
        represented = set()
        for index, window in enumerate(reader.windows):
            episode = window[0]
            if episode not in represented:
                item = reader[index]
                if not item['action_mask'].any() or not all(torch.isfinite(v).all() for v in item.values()):
                    raise ValueError('Invalid training window')
                represented.add(episode)
        if len(represented) != len(reader.episodes):
            raise ValueError('Missing training labels for an episode')
    print(f'Checked {len(episodes)} episodes, {len(raw)} windows; fitting joint train-only Min-Max', flush=True)
    normalization = fit_minmax(raw)
    normalization['source_binding'] = dict(m4_dataset_binding=parent['dataset_binding'],
                                           correction_report_sha256=None,
                                           train_extension_report_sha256=extension['report_sha256'])
    check_normalization(config, normalization)
    raw.set_normalization(normalization)
    validation = ActionWindows(ROOT / config['dataset'], 'validation', config['model']['horizon'], normalization)
    test = ActionWindows(ROOT / config['dataset'], 'test', config['model']['horizon'], normalization)
    indices = raw.sample_indices(np.random.default_rng(0), 4)
    # Include an extension window explicitly in the model smoke check.
    indices[-1] = len(raw.base)
    batch = default_collate([raw[i] for i in indices])
    model = create_model(config).to(device).eval()
    scheduler, _ = schedulers(config)
    with torch.inference_mode():
        loss = noise_loss(model, scheduler, batch)
    if not torch.isfinite(loss) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Zero-update forward check failed')
    if input_hashes() != source or start_pose_check(config) != extension or parent_check(config) != parent:
        raise ValueError('Inputs changed during preparation')
    write_json(output / 'config.json', config)
    write_json(output / 'normalization.json', normalization)
    write_json(output / 'parent_audit.json', parent)
    write_json(output / 'train_extension_audit.json', extension)
    write_json(output / 'hardware.json', hardware)
    report = dict(status='ready_for_training_not_started', episodes=len(episodes), counts=dict(counts),
                  windows=len(raw), **raw.summary(), validation_episodes=len(validation.episodes),
                  test_episodes=len(test.episodes), heldout_dataset=config['dataset'],
                  normalization_source='combined_train_only', normalization_count=normalization['count'],
                  normalization_sha256=sha256(output / 'normalization.json'),
                  config_sha256=sha256(config_path), preparation_sha256=sha256(Path(__file__)),
                  source_hashes=source, sampler='uniform_phase_then_episode_then_window',
                  optimizer_created=False, optimizer_updates=0, backward_run=False,
                  forward_check_loss=float(loss), parameters=sum(p.numel() for p in model.parameters()),
                  model_training_run=False, checkpoint_written=False, v1_unchanged=True,
                  excluded=['correction', 'rejected', 'pending'])
    write_json(output / 'report.json', report)
    write_json(output / 'status.json', dict(status=report['status'], optimizer_created=False, optimizer_updates=0))
    print({key: report[key] for key in ('status', 'episodes', 'counts', 'windows', 'optimizer_updates')}, flush=True)


if __name__ == '__main__':
    main()
