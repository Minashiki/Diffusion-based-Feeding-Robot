"""Read-only audit of action-boundary observation timing in frozen episodes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from feedingrobot.policies.causality import pre_action_status


def field_slices(schema):
    result, offset = {}, 0
    for name, size, _ in schema['fields']:
        result[name] = slice(offset, offset + size)
        offset += size
    return result


def audit_episode(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    commands = json.loads((directory / 'commands.json').read_text())
    filenames = ('action_ticks', 'action_end_ticks', 'action_mask', 'actions',
                 'action_observations', 'observation_ticks', 'observation_valid', 'observations')
    arrays = {name: np.load(directory / f'{name}.npy', mmap_mode='r') for name in filenames}
    ticks, mask = arrays['action_ticks'], arrays['action_mask']
    observations = arrays['observations'][:manifest['observation_rows']]
    observation_ticks, valid = arrays['observation_ticks'], arrays['observation_valid']
    fields = field_slices(manifest['observation_schema'])
    execution_states = manifest['observation_schema']['execution_states']
    status = fields['execution_status']
    initial = execution_states[int(observations[0, status].argmax())]
    recover = lambda query: pre_action_status(query, commands, initial, execution_states, manifest['dt'])
    before, after = recover(ticks), arrays['action_observations'][:, status].argmax(-1)
    checked = valid & (observation_ticks <= ticks[-1])
    actual = observations[checked, status].argmax(-1)
    status_mismatches = int(np.count_nonzero(recover(observation_ticks[checked]) != actual))
    rows = np.flatnonzero(mask)
    positions = np.searchsorted(observation_ticks[valid], ticks[rows], side='right') - 1
    if np.any(positions < 0):
        raise ValueError('An action precedes all valid ordinary observations')
    ordinary_rows = np.flatnonzero(valid)[positions]
    age_ticks = ticks[rows] - observation_ticks[ordinary_rows]
    common = rows[age_ticks == 0]
    common_observations = ordinary_rows[age_ticks == 0]
    delta = np.abs(arrays['action_observations'][common] - observations[common_observations])
    field_differences = {name: int(np.any(delta[:, field] > 1e-7, axis=1).sum())
                         for name, field in fields.items()}
    # Action labels are actual commands for [tick, end_tick), not later states.
    twists = {c['tick']: c for c in commands if c['kind'] == 'twist'}
    command_mismatches = sum(int(t) not in twists or not np.allclose(
        arrays['actions'][i], twists[int(t)]['twist'], rtol=0, atol=1e-12)
        for i, t in enumerate(ticks))
    label_timing_valid = bool(np.all(ticks % round(.05/manifest['dt']) == 0) and
                              np.all(np.diff(ticks) > 0) and np.all(
                                  arrays['action_end_ticks'][mask] == ticks[mask] + round(.05/manifest['dt'])))
    phase = fields['stage']
    phase_stale = int(np.count_nonzero(
        observations[ordinary_rows, phase].argmax(-1) != arrays['action_observations'][rows, phase].argmax(-1)))
    status_pairs = Counter(f'{execution_states[a]} -> {execution_states[b]}'
                           for a, b in zip(before[rows], after[rows]))
    first = None
    if len(rows):
        i = rows[0]
        first = dict(tick=int(ticks[i]), before=execution_states[before[i]], after=execution_states[after[i]])
    hashes = {}
    for name in ('manifest.json', 'commands.json', *(f'{n}.npy' for n in filenames)):
        with (directory / name).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        hashes[name] = digest
    passed = (label_timing_valid and not command_mismatches and not status_mismatches and
              not any(count for name, count in field_differences.items() if name != 'execution_status'))
    return dict(episode=str(directory.resolve()), split=manifest['split'], status='passed' if passed else 'failed',
                label_timing_valid=label_timing_valid, command_mismatches=int(command_mismatches),
                action_mask_rows=len(rows), observation_status_rows=int(checked.sum()),
                observation_status_mismatches=status_mismatches, common_tick_rows=len(common),
                common_tick_field_differences=field_differences, status_pairs=dict(status_pairs),
                first_action=first, ordinary_only_age_ticks={str(t): int(n) for t, n in Counter(age_ticks).items()},
                ordinary_only_phase_stale_rows=phase_stale, input_sha256=hashes)


def audit_data(directories, splits=('train', 'validation')):
    if any(split not in ('train', 'validation') for split in splits):
        raise ValueError('This diagnostic excludes the test split')
    episodes = [audit_episode(path.parent) for directory in directories for split in splits
                for path in sorted((Path(directory) / split).glob('*/manifest.json'))]
    if not episodes:
        raise ValueError('No episodes selected')
    counts = ('action_mask_rows', 'observation_status_rows', 'observation_status_mismatches',
              'common_tick_rows', 'command_mismatches', 'ordinary_only_phase_stale_rows')
    summary = {name: sum(e[name] for e in episodes) for name in counts}
    for name in ('status_pairs', 'ordinary_only_age_ticks', 'common_tick_field_differences'):
        combined = Counter()
        for episode in episodes:
            combined.update(episode[name])
        summary[name] = dict(combined)
    summary['first_action_status_pairs'] = dict(Counter(
        f"{e['first_action']['before']} -> {e['first_action']['after']}"
        for e in episodes if e['first_action']))
    return dict(schema_version=1, status='passed' if all(e['status'] == 'passed' for e in episodes) else 'failed',
                read_only=True, splits=list(splits), episode_count=len(episodes), summary=summary, episodes=episodes,
                observation_timing='pre_action_recovery_checked', interpretation=dict(
                    action_observations='After set_twist, before physics; only execution_status is directly affected.',
                    recovery='Command-log status reconstruction is checked against ordinary observations before the last action.',
                    limitation='Coincident-frame agreement supports status-only recovery; it does not prove the cause of policy failure.',
                    ordinary_only='Using ordinary 50 Hz frames removes exact odd-50-ms action boundaries and can age states or phases.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', type=Path, nargs='+')
    parser.add_argument('--splits', nargs='+', choices=('train', 'validation'), default=['train', 'validation'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_data(args.datasets, args.splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'episodes'}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
