"""Causal feature assembly and phase-bounded mmap action windows."""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from feedingrobot.data.episodes import annotate, recovery_action_mask
from feedingrobot.envs.feeding import observation_schema
from feedingrobot.policies.audit import read_json
from feedingrobot.sim.model import load_json


def field_slices(schema):
    offset, fields = 0, {}
    for name, size, _ in schema['fields']:
        fields[name] = slice(offset, offset + size)
        offset += size
    return fields


def asof(ticks, values, valid, queries):
    available = np.flatnonzero(valid)
    result = np.zeros((len(queries), values.shape[1]), dtype=np.float32)
    ages, mask = np.zeros(len(queries), dtype=np.float32), np.zeros(len(queries), dtype=bool)
    positions = np.searchsorted(ticks[available], queries, side='right') - 1
    for i, position in enumerate(positions):
        if position >= 0:
            j = available[position]
            result[i], ages[i], mask[i] = values[j], (queries[i] - ticks[j]) * .001, True
    return result, ages, mask


def features(ticks, observations, valid, tick, fields, normalization=None):
    states, state_age, state_mask = asof(ticks, observations, valid, np.array([tick - 50, tick]))
    history, age, history_mask = asof(ticks, observations, valid, tick + np.arange(-180, 1, 20))
    # Current target linear/angular velocity, obtained solely from the two past poses.
    velocity = np.zeros(12, dtype=np.float32)
    if state_mask.all():
        elapsed = .05 + state_age[0] - state_age[1]
        if elapsed > 0:
            for i, target in enumerate(('mouth', 'receiver')):
                positions = states[:, fields['tcp_position']] + states[:, fields[f'{target}_relative_world']]
                velocity[i*6:i*6+3] = np.diff(positions, axis=0)[0] / elapsed
                rotations = states[:, fields[f'{target}_rotation']].reshape(2, 3, 3)
                import mujoco
                quaternion, angular = np.empty(4), np.empty(3)
                mujoco.mju_mat2Quat(quaternion, (rotations[1] @ rotations[0].T).reshape(-1))
                mujoco.mju_quat2Vel(angular, quaternion, elapsed)
                velocity[i*6+3:i*6+6] = angular
    phase = states[-1, fields['stage']].copy()
    interaction = states[-1, fields['interaction']].copy()
    if normalization:
        states = (states - np.array(normalization['observation_mean'])) / np.array(normalization['observation_std'])
        history = (history - np.array(normalization['observation_mean'])) / np.array(normalization['observation_std'])
        velocity = (velocity - np.array(normalization['velocity_mean'])) / np.array(normalization['velocity_std'])
    state_input = np.concatenate([states, np.tile(velocity, (2, 1)), state_age[:, None], state_mask[:, None]], axis=1)
    columns = np.r_[tuple(np.arange(fields[name].start, fields[name].stop) for name in
                         ('compensated_wrench', 'q', 'dq', 'tcp_twist_world'))]
    history_input = np.concatenate([history[:, columns], age[:, None], history_mask[:, None]], axis=1)
    state_input[~state_mask] = 0
    history_input[~history_mask] = 0
    return dict(states=state_input.astype(np.float32), history=history_input.astype(np.float32),
                phase=np.int64(np.argmax(phase)), interaction=interaction.astype(np.float32),
                state_mask=state_mask, history_mask=history_mask)


class ActionWindows(Dataset):
    def __init__(self, directory, split, horizon=16, normalization=None):
        self.directory, self.split, self.horizon = Path(directory), split, horizon
        self.schema = observation_schema('panda', len(load_json('configs/robots/panda.json')['joints']))
        self.fields = field_slices(self.schema)
        self.normalization = normalization
        self.episodes, self.windows, self.by_phase = [], [], {}
        self.cache = OrderedDict()
        groups, seeds = {}, {}
        expected = json.loads(json.dumps(self.schema))
        if read_json(self.directory / 'normalization.json')['observation_schema'] != expected:
            raise ValueError('Normalization observation schema mismatch')
        # Whole-episode separation is checked across all splits, even for a train reader.
        for path in sorted(self.directory.glob('*/*/manifest.json')):
            m = read_json(path)
            if (m['robot_id'] != 'panda' or m['status'] != 'complete' or m['dt'] != .001
                    or m['observation_schema'] != expected or m['segments'] != annotate(m['events'])
                    or m['split'] != path.parent.parent.name):
                raise ValueError('Episode schema, timing or annotation mismatch')
            for key, table in (('group_id', groups), ('seed', seeds)):
                if m[key] in table and table[m[key]] != m['split']:
                    raise ValueError('Whole episode split leakage')
                table[m[key]] = m['split']
            if m['split'] != split:
                continue
            arrays = {name: np.load(path.parent / f'{name}.npy', mmap_mode='r') for name in
                      ('action_mask', 'action_phases', 'action_ticks', 'action_end_ticks')}
            eligible = arrays['action_mask'].copy()
            if m['scenario'].get('recover', False):
                eligible &= recovery_action_mask(m['segments'], arrays['action_phases'], arrays['action_ticks'],
                                                 arrays['action_end_ticks'], eligible, m['dt'])
            elif not m['accepted_normal'] or not m['success']:
                eligible[:] = False
            recovery = recovery_action_mask(m['segments'], arrays['action_phases'], arrays['action_ticks'],
                                            arrays['action_end_ticks'], arrays['action_mask'], m['dt'])
            if m['recovery_action_rows'] != int(recovery.sum()) or m['accepted_recovery'] != bool(recovery.any()):
                raise ValueError('Recovery labels differ from physical events')
            ticks, phases = arrays['action_ticks'], arrays['action_phases']
            if (np.any(ticks % 50) or np.any(np.diff(ticks) <= 0)
                    or np.any(arrays['action_end_ticks'][eligible] != ticks[eligible] + 50)):
                raise ValueError('Invalid 20 Hz action labels')
            episode = len(self.episodes)
            self.episodes.append((path.parent, m))
            for start in np.flatnonzero(eligible):
                end = start + 1
                while (end < len(ticks) and end - start < horizon and eligible[end]
                       and phases[end] == phases[start] and ticks[end] == ticks[end-1] + 50):
                    end += 1
                index = len(self.windows)
                self.windows.append((episode, int(start), int(end), int(phases[start])))
                self.by_phase.setdefault(int(phases[start]), {}).setdefault(episode, []).append(index)
        if not self.windows:
            raise ValueError(f'No legal action windows in {split}')

    def arrays(self, episode):
        if episode not in self.cache:
            path, m = self.episodes[episode]
            a = {name: np.load(path / f'{name}.npy', mmap_mode='r') for name in
                 ('actions', 'action_ticks', 'action_observations', 'observations', 'observation_ticks', 'observation_valid')}
            n = m['observation_rows']
            # Merge exact action-boundary observations with the 50 Hz history.
            ticks = np.r_[a['observation_ticks'][:n], a['action_ticks']]
            values = np.concatenate([a['observations'][:n], a['action_observations']])
            valid = np.r_[a['observation_valid'][:n], np.ones(len(a['action_ticks']), dtype=bool)]
            order = np.argsort(ticks, kind='stable')
            a['feature_ticks'], a['feature_values'], a['feature_valid'] = ticks[order], values[order], valid[order]
            self.cache[episode] = a
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        return self.cache[episode]

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = OrderedDict()
        return state

    def __getitem__(self, index):
        episode, start, end, _ = self.windows[index]
        a = self.arrays(episode)
        x = features(a['feature_ticks'], a['feature_values'], a['feature_valid'], int(a['action_ticks'][start]),
                     self.fields, self.normalization)
        actions, mask = np.zeros((self.horizon, 6), np.float32), np.zeros(self.horizon, bool)
        values = a['actions'][start:end]
        if self.normalization:
            values = (values - np.array(self.normalization['action_mean'])) / np.array(self.normalization['action_std'])
        actions[:end-start], mask[:end-start] = values, True
        return {k: torch.as_tensor(v) for k, v in dict(**x, actions=actions, action_mask=mask).items()}

    def sample_indices(self, rng, count):
        phases = sorted(self.by_phase)
        result = []
        for _ in range(count):
            phase = phases[int(rng.integers(len(phases)))]
            episodes = sorted(self.by_phase[phase])
            episode = episodes[int(rng.integers(len(episodes)))]
            windows = self.by_phase[phase][episode]
            result.append(windows[int(rng.integers(len(windows)))])
        return result


def fit_normalization(dataset):
    if dataset.split != 'train':
        raise ValueError('Normalization must use train only')
    observation = read_json(dataset.directory / 'normalization.json')
    mean, std = np.array(observation['mean']), np.array(observation['std'])
    for name in ('stage', 'interaction', 'execution_status'):
        mean[dataset.fields[name]], std[dataset.fields[name]] = 0, 1
    sums, squares, count = np.zeros(18), np.zeros(18), 0
    for episode, start, _, _ in dataset.windows:
        a = dataset.arrays(episode)
        x = features(a['feature_ticks'], a['feature_values'], a['feature_valid'], int(a['action_ticks'][start]), dataset.fields)
        width = sum(size for _, size, _ in dataset.schema['fields'])
        values = np.r_[a['actions'][start], x['states'][-1, width:width+12]]
        sums += values
        squares += values**2
        count += 1
    average = sums / count
    scale = np.sqrt(np.maximum(squares / count - average**2, 0))
    scale = np.where(scale > 1e-8, scale, 1.)
    return dict(schema_version=1, source_split='train', count=count, observation_schema=observation['observation_schema'],
                observation_mean=mean.tolist(), observation_std=std.tolist(),
                action_mean=average[:6].tolist(), action_std=scale[:6].tolist(),
                velocity_mean=average[6:].tolist(), velocity_std=scale[6:].tolist())
