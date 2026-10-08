"""Mixed-source indexing, recovery preservation, leakage and joint scaling."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from feedingrobot.data.episodes import write_json
from feedingrobot.policies.audit import read_json
from feedingrobot.policies.corpus import TrainingWindows
from feedingrobot.policies.normalization import fit_minmax
from feedingrobot.sim.model import ROOT

sys.path.insert(0, str(ROOT / 'tests'))
from test_m5 import compute_budget, config, dataset, episode


def mixed(config, dataset, tmp_path):
    directory = tmp_path / 'extension'
    directory.mkdir()
    write_json(directory / 'normalization.json', read_json(dataset.directory / 'normalization.json'))
    path = episode(directory)
    m = read_json(path / 'manifest.json')
    m.update(seed=987654321, group_id='new-group', base_group_id='new-base-group')
    write_json(path / 'manifest.json', m)
    config['dataset'] = str(dataset.directory)
    return TrainingWindows(config, extension=dict(dataset=str(directory))), path


def test_mixed_indices_recovery_and_sampling(config, dataset, tmp_path):
    raw, _ = mixed(config, dataset, tmp_path)
    assert len(raw) == len(raw.base) + len(raw.extension)
    assert len(raw.base.episodes) == 2 and len(raw.extension.episodes) == 1
    assert raw.base.episodes[1][1]['scenario']['recover']
    for key in raw[0]:
        torch.testing.assert_close(raw[len(raw.base)][key], raw.extension[0][key])
    indices = raw.sample_indices(np.random.default_rng(4), 2000)
    assert any(i >= len(raw.base) for i in indices)
    assert any(raw.base.windows[i][3] == 7 for i in indices if i < len(raw.base))
    assert raw.summary()['correction_windows'] == 0
    for phase, episodes in raw.by_phase.items():
        for indices in episodes.values():
            for i in indices:
                reader, local = (raw.base, i) if i < len(raw.base) else (raw.extension, i-len(raw.base))
                assert reader.windows[local][3] == phase


@pytest.mark.parametrize('key', ['seed', 'group_id', 'base_group_id'])
def test_cross_source_leakage_rejected(config, dataset, tmp_path, key):
    raw, path = mixed(config, dataset, tmp_path)
    m = read_json(path / 'manifest.json')
    old = raw.base.episodes[0][1]
    if key == 'base_group_id':
        heldout = episode(dataset.directory, split='validation')
        old = read_json(heldout / 'manifest.json')
    m[key] = old.get(key, old['group_id'])
    write_json(path / 'manifest.json', m)
    with pytest.raises(ValueError, match='leakage'):
        TrainingWindows(config, extension=dict(dataset=str(path.parent.parent)))


def test_combined_scaling_uses_extension_excludes_rejected(config, dataset, tmp_path):
    raw, path = mixed(config, dataset, tmp_path)
    actions = np.load(path / 'actions.npy')
    actions[:, 0] = 3.
    np.save(path / 'actions.npy', actions)
    rejected = tmp_path / 'rejected'
    episode(rejected, group='rejected')
    norm = fit_minmax(raw)
    assert norm['count'] == len(raw)
    assert norm['action_offset'][0] == 0 and norm['action_scale'][0] == 3
    raw.set_normalization(norm)
    assert raw.base.normalization is norm and raw.extension.normalization is norm
    assert len(raw.extension.episodes) == 1


def test_correction_cannot_mix_with_extension(config, dataset, tmp_path):
    raw, _ = mixed(config, dataset, tmp_path)
    with pytest.raises(ValueError, match='excludes correction'):
        TrainingWindows(config, correction={'dataset': 'unused'},
                        extension=dict(dataset=str(raw.extension.directory)))
