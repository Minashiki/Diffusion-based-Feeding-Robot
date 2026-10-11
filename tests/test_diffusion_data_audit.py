"""Action-time causality checks use saved arrays and never step physics."""

import json
import shutil

import numpy as np
import pytest

from tools.m5.diffusion_diagnostics.data_audit import audit_data, audit_episode, pre_action_status
from feedingrobot.envs.feeding import EXECUTION_STATES
from feedingrobot.policies.causality import pre_action_observations
from test_m5 import compute_budget, config, dataset, episode
from feedingrobot.policies.corpus import TrainingWindows
from feedingrobot.policies.normalization import fit_limits


STATES = ['idle', 'active', 'expired', 'stopped']


def test_status_recovery_respects_cancellations_and_expiry_boundary():
    commands = [dict(tick=0, kind='twist', valid_until=.05),
                dict(tick=1, kind='stop', after_physics=True),
                dict(tick=50, kind='twist', valid_until=.1),
                dict(tick=150, kind='stop', after_physics=True),
                dict(tick=150, kind='stop', after_physics=False),
                dict(tick=150, kind='twist', valid_until=.2)]
    ticks = [0, 1, 50, 51, 100, 101, 150, 151, 200, 201]
    actual = pre_action_status(ticks, commands, 'idle', STATES, .001)
    assert [STATES[i] for i in actual] == [
        'idle', 'stopped', 'stopped', 'active', 'active', 'expired',
        'stopped', 'active', 'active', 'expired']


def test_status_recovery_rejects_ambiguous_stop_order():
    with pytest.raises(ValueError, match='explicit stop ordering'):
        pre_action_status([50], [dict(tick=1, kind='stop')], 'idle', STATES, .001)


def make_episode(directory):
    directory.mkdir(parents=True)
    schema = dict(fields=[['q', 1, 'rad'], ['stage', 2, 'one-hot'],
                          ['execution_status', 4, 'one-hot']], execution_states=STATES)
    ot = np.array([0, 20, 40, 60, 80, 100])
    observations = np.zeros((6, 7))
    observations[:, 1] = 1
    observations[0, 3] = 1
    observations[1:3, 6] = 1
    observations[3:, 4] = 1
    action_observations = np.zeros((3, 7))
    action_observations[:, 1] = 1
    action_observations[:, 4] = 1
    arrays = dict(action_ticks=np.array([0, 50, 100]), action_end_ticks=np.array([1, 100, 150]),
                  action_mask=np.array([False, True, True]), actions=np.zeros((3, 6)),
                  action_observations=action_observations, observation_ticks=ot,
                  observation_valid=np.ones(6, bool), observations=observations)
    for name, array in arrays.items():
        np.save(directory / f'{name}.npy', array)
    manifest = dict(observation_schema=schema, observation_rows=6, dt=.001, split='train')
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    commands = [dict(tick=0, kind='twist', valid_until=.05, twist=[0.]*6),
                dict(tick=1, kind='stop', after_physics=True),
                dict(tick=50, kind='twist', valid_until=.1, twist=[0.]*6),
                dict(tick=100, kind='twist', valid_until=.15, twist=[0.]*6)]
    (directory / 'commands.json').write_text(json.dumps(commands))
    return arrays


def test_audit_detects_post_action_status_and_keeps_inputs_unchanged(tmp_path):
    directory = tmp_path / 'train' / 'episode'
    make_episode(directory)
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = audit_episode(directory)
    assert result['status'] == 'passed'
    assert result['observation_status_mismatches'] == 0
    assert result['first_action'] == dict(tick=50, before='stopped', after='active')
    assert result['status_pairs'] == {'stopped -> active': 1, 'active -> active': 1}
    assert result['ordinary_only_age_ticks'] == {'10': 1, '0': 1}
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
    report = audit_data([tmp_path], ['train'])
    assert report['episode_count'] == 1 and report['status'] == 'passed'


def test_audit_fails_when_a_physical_field_changes_at_same_tick(tmp_path):
    directory = tmp_path / 'train' / 'episode'
    arrays = make_episode(directory)
    arrays['action_observations'][2, 0] = .01
    np.save(directory / 'action_observations.npy', arrays['action_observations'])
    result = audit_episode(directory)
    assert result['status'] == 'failed'
    assert result['common_tick_field_differences']['q'] == 1


def test_test_split_is_excluded(tmp_path):
    with pytest.raises(ValueError, match='excludes the test'):
        audit_data([tmp_path], ['test'])


def test_pre_action_copy_changes_only_status_and_rejects_unverified_logs(tmp_path):
    directory = tmp_path / 'train' / 'episode'
    arrays = make_episode(directory)
    # The production helper uses the complete policy schema's status ordering.
    states = np.zeros((6, len(EXECUTION_STATES)))
    states[0, EXECUTION_STATES.index('idle')] = 1
    states[1:3, EXECUTION_STATES.index('stopped')] = 1
    states[3:, EXECUTION_STATES.index('active')] = 1
    arrays['observations'] = np.concatenate([arrays['observations'][:, :3], states], axis=1)
    action_states = np.zeros((3, len(EXECUTION_STATES)))
    action_states[:, EXECUTION_STATES.index('active')] = 1
    arrays['action_observations'] = np.concatenate([arrays['action_observations'][:, :3], action_states], axis=1)
    original = arrays['action_observations'].copy()
    fields = {'execution_status': slice(3, 3 + len(EXECUTION_STATES))}
    corrected = pre_action_observations(directory, arrays, fields)
    assert not np.shares_memory(corrected, arrays['action_observations'])
    np.testing.assert_array_equal(corrected[:, :3], original[:, :3])
    assert corrected[:, fields['execution_status']].argmax(-1).tolist() == [
        EXECUTION_STATES.index(s) for s in ['idle', 'stopped', 'active']]
    np.testing.assert_array_equal(arrays['action_observations'], original)
    arrays['observations'][1, fields['execution_status']] = 0
    arrays['observations'][1, 3 + EXECUTION_STATES.index('active')] = 1
    with pytest.raises(ValueError, match='does not recover'):
        pre_action_observations(directory, arrays, fields)


def test_training_windows_timing_switch_clears_base_and_extension_caches(config, dataset, tmp_path):
    extension = tmp_path / 'extension'
    extension.mkdir()
    shutil.copyfile(dataset.directory / 'normalization.json', extension / 'normalization.json')
    extra = episode(extension, group='unique-extension')
    manifest_path = extra / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['seed'] = 100
    manifest_path.write_text(json.dumps(manifest))
    # Supply ordinary pre-command and exact post-command frames on the actual
    # ActionWindows fixture, including a cancellation before the t50 action.
    for directory in (dataset.directory, extension):
        for path in directory.glob('train/*/manifest.json'):
            ticks = np.load(path.parent / 'action_ticks.npy')
            commands = [dict(tick=int(t), kind='twist', valid_until=(int(t)+50)*.001)
                        for t in ticks]
            commands.insert(1, dict(tick=1, kind='stop', after_physics=True))
            (path.parent / 'commands.json').write_text(json.dumps(commands))
            for name in ('observations', 'action_observations'):
                values = np.load(path.parent / f'{name}.npy')
                status = dataset.fields['execution_status']
                values[:, status] = 0
                if name == 'observations':
                    observation_ticks = np.load(path.parent / 'observation_ticks.npy')
                    indices = pre_action_status(observation_ticks, commands, 'idle', EXECUTION_STATES, .001)
                else:
                    indices = np.full(len(ticks), EXECUTION_STATES.index('active'))
                values[np.arange(len(values)), status.start + indices] = 1
                np.save(path.parent / f'{name}.npy', values)
    settings = dict(config, dataset=str(dataset.directory))
    combined = TrainingWindows(settings, extension={'dataset': str(extension)})
    norm = dict(fit_limits(combined), observation_timing='pre_action')
    base_index = next(i for i, (_, start, _, _) in enumerate(combined.base.windows) if start == 1)
    extension_index = next(i for i, (_, start, _, _) in enumerate(combined.extension.windows) if start == 1)
    before = [combined.base[base_index], combined.extension[extension_index]]
    assert combined.base.cache and combined.extension.cache
    original_files = {str(path): path.read_bytes() for directory in (dataset.directory, extension)
                      for path in directory.rglob('*') if path.is_file()}
    combined.set_normalization(norm)
    assert not combined.base.cache and not combined.extension.cache
    after = [combined[base_index], combined[len(combined.base) + extension_index]]
    status = combined.fields['execution_status']
    nonstatus = np.r_[0:status.start, status.stop:before[0]['states'].shape[1]]
    for old, new in zip(before, after):
        assert old['states'][:, status].argmax(-1).tolist() == [1, 1]
        assert new['states'][:, status].argmax(-1).tolist() == [0, 3]
        np.testing.assert_array_equal(old['states'][:, nonstatus], new['states'][:, nonstatus])
        for name in ('history', 'state_mask', 'history_mask', 'phase', 'interaction', 'action_mask'):
            np.testing.assert_array_equal(old[name], new[name])
    assert {str(path): path.read_bytes() for directory in (dataset.directory, extension)
            for path in directory.rglob('*') if path.is_file()} == original_files
    combined.set_normalization({'kind': 'none'})
    assert not combined.base.cache and not combined.extension.cache
    np.testing.assert_array_equal(combined[base_index]['states'], before[0]['states'])
    np.testing.assert_array_equal(combined[len(combined.base)+extension_index]['states'], before[1]['states'])
