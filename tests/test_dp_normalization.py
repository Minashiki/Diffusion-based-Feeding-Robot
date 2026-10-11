"""DP action limits and valid training rows."""

import json
import shutil

import numpy as np
import pytest
import torch

from test_m5 import config, dataset, episode
from feedingrobot.policies.corpus import TrainingWindows
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.normalization import denormalize_actions, fit_limits, normalize_actions


class Windows:
    def __init__(self, actions, mask=None, split='train', normalization=None):
        self.split, self.normalization = split, normalization
        self.schema = {'fields': [('action', 6, 'velocity')]}
        self.sample = dict(actions=torch.as_tensor(actions, dtype=torch.float64),
                           action_mask=torch.ones(len(actions), dtype=torch.bool) if mask is None
                           else torch.as_tensor(mask, dtype=torch.bool))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def test_limits_endpoints_constants_masks_and_roundtrip():
    actions = np.array([[2., -4., 1., 0., 3., -2.], [6., 8., 1., 5e-5, 5., 4.], [1e6] * 6])
    norm = fit_limits(Windows(actions, [True, True, False]))
    assert norm['schema_version'] == 4 and norm['kind'] == 'limits'
    assert norm['source_split'] == 'train' and norm['clipping'] is False
    assert norm['action_constant_mask'] == [False, False, True, True, False, False]
    np.testing.assert_array_equal(norm['action_min'], actions[0])
    np.testing.assert_array_equal(norm['action_max'], actions[1])
    np.testing.assert_array_equal(np.array(norm['action_offset'])[[2, 3]], [1., 0.])
    np.testing.assert_array_equal(np.array(norm['action_scale'])[[2, 3]], [1., 1.])
    normalized = normalize_actions(actions[:2], norm)
    np.testing.assert_array_equal(normalized[:, [0, 1, 4, 5]], [[-1.] * 4, [1.] * 4])
    np.testing.assert_array_equal(normalized[:, 2], 0.)
    np.testing.assert_array_equal(normalized[:, 3], [0., 5e-5])
    np.testing.assert_allclose(denormalize_actions(normalized, norm), actions[:2], rtol=0, atol=1e-15)
    outside = np.array(norm['action_offset']) + 2 * np.array(norm['action_scale'])
    np.testing.assert_array_equal(normalize_actions(outside, norm), 2.)
    np.testing.assert_array_equal(denormalize_actions(normalize_actions(outside, norm), norm), outside)


def test_limits_requires_raw_train_and_finite_valid_rows():
    actions = [[0.] * 6, [1.] * 6]
    for windows in (Windows(actions, split='validation'), Windows(actions, split='test'),
                    Windows(actions, normalization={'kind': 'none'})):
        with pytest.raises(ValueError, match='raw train'):
            fit_limits(windows)
    for windows in (Windows(actions, [False, False]), Windows([[np.nan] * 6])):
        with pytest.raises(ValueError, match='finite valid'):
            fit_limits(windows)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device_name', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_limits_tensor_dtype_device_and_inverse(dtype, device_name):
    norm = fit_limits(Windows([[-2.] * 6, [6.] * 6]))
    actions = torch.tensor([[0., 1., 2., 3., 4., 5.]], dtype=dtype, device=device_name)
    normalized = normalize_actions(actions, norm)
    decoded = denormalize_actions(normalized, norm)
    assert normalized.dtype == decoded.dtype == actions.dtype
    assert normalized.device == decoded.device == actions.device
    torch.testing.assert_close(decoded, actions, rtol=0, atol=0)


def test_limits_training_windows_include_extension_without_held_out(config, dataset, tmp_path):
    before = fit_limits(dataset)
    episode(dataset.directory, split='validation')
    validation = ActionWindows(dataset.directory, 'validation')
    held_out_path = validation.episodes[0][0] / 'actions.npy'
    held_out = np.load(held_out_path)
    np.save(held_out_path, np.full_like(held_out, 10000.))
    assert fit_limits(dataset) == before
    with pytest.raises(ValueError, match='raw train'):
        fit_limits(validation)
    extension = tmp_path / 'extension'
    extension.mkdir()
    shutil.copyfile(dataset.directory / 'normalization.json', extension / 'normalization.json')
    path = episode(extension, group='extension')
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['seed'] = 100
    manifest_path.write_text(json.dumps(manifest))
    sample = ActionWindows(extension, 'train')
    path = sample.episodes[0][0] / 'actions.npy'
    values = np.load(path)
    values[:, 0] = .03
    np.save(path, values)
    settings = dict(config, dataset=str(dataset.directory))
    combined = TrainingWindows(settings, extension={'dataset': str(extension)})
    norm = fit_limits(combined)
    rows = np.concatenate([combined[index]['actions'][combined[index]['action_mask']].numpy()
                           for index in range(len(combined))])
    reference = fit_limits(Windows(rows))
    reference.update(count=len(combined), observation_schema=norm['observation_schema'])
    assert norm == reference
    assert norm['action_max'][0] == pytest.approx(.03)
    assert norm['count'] == len(dataset) + len(combined.extension)
    assert set(norm) == {'schema_version', 'kind', 'source_split', 'count', 'observation_schema', 'clipping',
                         'action_min', 'action_max', 'action_constant_mask', 'action_offset', 'action_scale'}


def test_limits_episode_fast_path_matches_windows_without_reading_features(dataset, monkeypatch):
    directory = dataset.episodes[0][0]
    path = directory / 'actions.npy'
    actions = np.load(path)
    actions[7] = 1e6  # Invalid source row must stay out of both fitting paths.
    np.save(path, actions)
    rows = np.concatenate([dataset[index]['actions'][dataset[index]['action_mask']].numpy()
                           for index in range(len(dataset))])
    reference = fit_limits(Windows(rows))
    reference.update(count=len(dataset), observation_schema=json.loads(json.dumps(dataset.schema)))
    def fail(*args):
        raise AssertionError('Action fitting must not compute observation features')
    monkeypatch.setattr(ActionWindows, '__getitem__', fail)
    monkeypatch.setattr(ActionWindows, 'arrays', fail)
    assert fit_limits(dataset) == reference


def test_raw_actions_remain_unscaled():
    actions = np.array([[-3., .01, -.1, .2, -.5, .6]])
    raw = {'kind': 'none'}
    assert normalize_actions(actions, raw) is actions
    assert denormalize_actions(actions, raw) is actions
