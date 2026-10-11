"""Recover pre-command execution status without rewriting frozen episodes."""

import json
from pathlib import Path

import numpy as np

from feedingrobot.envs.feeding import EXECUTION_STATES


def pre_action_status(ticks, commands, initial_status, execution_states, dt):
    """Status after phase cancellations, before this tick's teacher commands.

    Physics updates the adapter before advancing the tick, so a command expiring
    at tick T remains active in the observation at T and expires at T + 1.
    Explicit after_physics stops at T precede that observation and are included.
    """
    ticks = np.asarray(ticks)
    initial = execution_states.index(initial_status)
    if not commands:
        return np.full(len(ticks), initial, dtype=np.int64)
    if any(c['kind'] == 'stop' and 'after_physics' not in c for c in commands):
        raise ValueError('Status recovery requires explicit stop ordering')
    if any(c['kind'] not in ('stop', 'twist') or
           (c['kind'] == 'twist' and c.get('after_physics', False)) for c in commands):
        raise ValueError('Unsupported command ordering')
    keys = np.array([2*c['tick'] + int(not c.get('after_physics', False)) for c in commands])
    if np.any(np.diff(keys) < 0):
        raise ValueError('Commands are not in chronological execution order')
    indices = np.searchsorted(keys, 2*ticks, side='right') - 1
    selected = indices.clip(0)
    values = np.array([execution_states.index('stopped' if c['kind'] == 'stop' else 'active')
                       for c in commands])[selected]
    expiry = np.array([c['valid_until'] if c['kind'] == 'twist' else np.inf for c in commands])[selected]
    values[(ticks - 1)*dt >= expiry - 1e-12] = execution_states.index('expired')
    values[indices < 0] = initial
    return values


def pre_action_observations(directory, arrays, fields, *, dt=.001):
    """Copy exact action-time physical observations and recover only status.

    The ordinary frames are captured before commands. Their status provides an
    independent check of log ordering, including expiry and phase cancellation.
    No disk arrays or supplied arrays are changed.
    """
    commands = json.loads((Path(directory) / 'commands.json').read_text())
    ticks, status = arrays['action_ticks'], fields['execution_status']
    initial = EXECUTION_STATES[int(arrays['observations'][0, status].argmax())]
    ordinary_ticks, valid = arrays['observation_ticks'], arrays['observation_valid']
    checked = valid & (ordinary_ticks <= ticks[-1])
    ordinary = arrays['observations'][:len(ordinary_ticks)]
    expected = pre_action_status(ordinary_ticks[checked], commands, initial, EXECUTION_STATES, dt)
    if np.any(expected != ordinary[checked, status].argmax(-1)):
        raise ValueError('Command log does not recover ordinary observation execution status')
    recovered = pre_action_status(ticks, commands, initial, EXECUTION_STATES, dt)
    result = arrays['action_observations'].copy()
    result[:, status] = 0
    result[np.arange(len(ticks)), status.start + recovered] = 1
    return result
