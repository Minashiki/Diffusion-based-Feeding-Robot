"""M5 causality, masking, ancestry and resumable learning contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from feedingrobot.data.episodes import annotate, write_json
from feedingrobot.envs.feeding import observation_schema
from feedingrobot.policies import audit, training
from feedingrobot.policies.data import ActionWindows, asof, features, field_slices, fit_normalization
from feedingrobot.policies.dit import ActionDiT, noise_loss, sample_actions, schedulers
from feedingrobot.policies.evaluation import ActionChunk, summarize
from feedingrobot.policies.runtime import restore_rng, rng_state, save_checkpoint
from feedingrobot.sim.model import ROOT


@pytest.fixture(scope='session', autouse=True)
def compute_budget():
    from feedingrobot.policies.runtime import setup
    config = audit.read_json(ROOT / 'configs/dp_dit.json')
    setup(config, 'cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config():
    result = audit.read_json(ROOT / 'configs/dp_dit.json')
    result['model'].update(depth=2, hidden_size=32, heads=4, mlp_hidden=128)
    result['training'].update(batch_size=4, workers=0, validation_every=100, checkpoint_every=2, evaluation_every=100)
    return result


def observation(ticks, phases=None):
    schema = observation_schema('panda', 7)
    fields = field_slices(schema)
    x = np.zeros((len(ticks), 108), np.float32)
    for name in ('tcp_rotation', 'mouth_rotation', 'receiver_rotation'):
        x[:, fields[name]] = np.eye(3).reshape(1, 9)
    x[:, fields['mouth_relative_world']] = np.asarray(ticks)[:, None] * 1e-6 + .1
    x[:, fields['receiver_relative_world']] = .2
    x[:, fields['stage']] = np.eye(8)[np.ones(len(ticks), int) if phases is None else phases]
    return x


def episode(directory, recover=False, split='train', group=None):
    path = directory / split / ('recovery' if recover else 'normal')
    path.mkdir(parents=True)
    ticks = np.arange(8 if recover else 20) * 50
    phases = np.full(len(ticks), 7 if recover else 1)
    if not recover:
        phases[8:12], phases[12:] = 4, 5
    mask = np.ones(len(ticks), bool)
    ends = ticks + 50
    if not recover:
        mask[7], ends[7] = False, 370
    events = ([dict(name='phase', time=0., phase='RECOVER'), dict(name='phase', time=.4, phase='WAIT_READY'),
               dict(name='success', time=.5)] if recover else
              [dict(name='phase', time=0., phase='ACQUIRE'), dict(name='phase', time=.4, phase='APPROACH'),
               dict(name='phase', time=.6, phase='TRANSFER'), dict(name='success', time=1.)])
    obs_ticks = np.arange(0, 1000, 20)
    arrays = dict(action_ticks=ticks, action_end_ticks=ends, action_phases=phases, action_mask=mask,
                  actions=np.tile(np.arange(6, dtype=float), (len(ticks), 1)) * ticks[:, None] * 1e-5,
                  action_observations=observation(ticks, phases), observations=observation(obs_ticks),
                  observation_ticks=obs_ticks, observation_valid=np.ones(len(obs_ticks), bool))
    for name, a in arrays.items():
        np.save(path / f'{name}.npy', a)
    m = dict(robot_id='panda', status='complete', dt=.001,
             observation_schema=json.loads(json.dumps(observation_schema('panda', 7))),
             events=events, segments=annotate(events), scenario=dict(recover=recover),
             accepted_normal=True, accepted_recovery=recover, success=True,
             recovery_action_rows=8 if recover else 0, split=split,
             group_id=group or f'{split}-{recover}', seed=hash(f'{split}-{recover}'), observation_rows=len(obs_ticks))
    write_json(path / 'manifest.json', m)
    return path


@pytest.fixture
def dataset(tmp_path):
    write_json(tmp_path / 'normalization.json', dict(mean=[0.]*108, std=[1.]*108,
               observation_schema=observation_schema('panda', 7)))
    episode(tmp_path)
    episode(tmp_path, recover=True)
    return ActionWindows(tmp_path, 'train')


def test_asof_never_reads_future_and_masks_missing():
    ticks = np.array([0, 20, 40, 60])
    values = np.array([[0], [20], [40], [60]])
    x, ages, valid = asof(ticks, values, np.array([True, True, False, True]), np.array([-10, 30, 50]))
    np.testing.assert_array_equal(x[:, 0], [0, 20, 20])
    np.testing.assert_array_equal(valid, [False, True, True])
    np.testing.assert_allclose(ages, [0, .01, .03])


@pytest.mark.parametrize('tick', [50, 100, 150])
def test_features_action_grid_and_no_future_leak(tick):
    ticks = np.arange(0, 400, 20)
    x = observation(ticks)
    fields = field_slices(observation_schema('panda', 7))
    before = features(ticks, x, np.ones(len(x), bool), tick, fields)
    x[ticks > tick] = 999
    after = features(ticks, x, np.ones(len(x), bool), tick, fields)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert before['history'].shape == (10, 28)
    assert before['states'].shape == (2, 122)


def test_windows_start_at_current_action_and_stop_at_invalid_or_phase(dataset):
    first = dataset[0]
    assert first['action_mask'].sum() == 7
    assert not first['state_mask'][0] and first['state_mask'][1]
    np.testing.assert_array_equal(first['actions'][0], np.zeros(6))
    assert torch.count_nonzero(first['actions'][7:]) == 0
    last = next(i for i, (_, start, _, phase) in enumerate(dataset.windows) if start == 8 and phase == 4)
    assert dataset[last]['action_mask'].sum() == 4
    np.testing.assert_allclose(dataset[last]['actions'][0], np.arange(6) * 400 * 1e-5)


def test_recovery_contains_only_confirmed_segment(dataset):
    windows = [i for i, value in enumerate(dataset.windows) if value[3] == 7]
    assert len(windows) == 8
    assert dataset[windows[0]]['action_mask'].sum() == 8
    assert dataset[windows[-1]]['action_mask'].sum() == 1


def test_phase_episode_sampling_is_balanced(dataset):
    indices = dataset.sample_indices(np.random.default_rng(0), 4000)
    counts = np.bincount([dataset.windows[i][3] for i in indices])
    for phase in dataset.by_phase:
        assert 850 < counts[phase] < 1150


def test_normalization_train_only_and_constant_scale(dataset):
    norm = fit_normalization(dataset)
    assert norm['count'] == len(dataset)
    assert norm['action_std'][0] == 1
    fields = dataset.fields
    for name in ('stage', 'interaction', 'execution_status'):
        np.testing.assert_array_equal(np.array(norm['observation_std'])[fields[name]], 1)
    dataset.split = 'validation'
    with pytest.raises(ValueError, match='train only'):
        fit_normalization(dataset)


def test_episode_split_leakage_is_rejected(dataset):
    episode(dataset.directory, split='validation', group='train-False')
    with pytest.raises(ValueError, match='split leakage'):
        ActionWindows(dataset.directory, 'train')


@pytest.mark.parametrize('change', ['schema', 'recovery', 'ticks'])
def test_invalid_episode_is_rejected(dataset, change):
    path = dataset.directory / 'train/recovery'
    metadata = audit.read_json(path / 'manifest.json')
    if change == 'schema':
        metadata['observation_schema']['version'] = -1
    elif change == 'recovery':
        metadata['recovery_action_rows'] = 0
    else:
        np.save(path / 'action_end_ticks.npy', np.arange(8)*50 + 49)
    write_json(path / 'manifest.json', metadata)
    with pytest.raises(ValueError):
        ActionWindows(dataset.directory, 'train')


def test_dit_shapes_masks_gradients_and_sampling(config, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm = fit_normalization(dataset)
    dataset.normalization = norm
    batch = {k: v.to(device) for k, v in default_collate([dataset[0], dataset[-1]]).items()}
    model = ActionDiT(config).to(device)
    scheduler, _ = schedulers(config)
    condition = model.encoder(batch)
    assert condition.tokens.shape == (2, 13, 32)
    assert condition.valid[:, -1].all()
    noise, times = torch.randn_like(batch['actions']), torch.tensor([5, 20], device=device)
    # Zero output initialization is intentional; gradients reach inner blocks after output updates.
    loss = noise_loss(model, scheduler, batch, noise=noise, timesteps=times)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    torch.nn.init.normal_(model.output.weight, std=.02)
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight, std=.01)
    model.zero_grad()
    noise_loss(model, scheduler, batch, noise=noise, timesteps=times).backward()
    assert model.blocks[0].cross_attention.in_proj_weight.grad.abs().sum() > 0
    changed = batch['actions'].clone()
    changed[~batch['action_mask']] = 10000
    same = dict(batch, actions=changed)
    torch.testing.assert_close(noise_loss(model, scheduler, same, noise=noise, timesteps=times),
                               noise_loss(model, scheduler, batch, noise=noise, timesteps=times))
    model.eval()
    def sample(seed, inputs=batch):
        return sample_actions(model, config, inputs, norm, torch.Generator(device=device).manual_seed(seed))
    a, b = sample(10), sample(10)
    torch.testing.assert_close(a, b)
    assert a.shape == (2, 16, 6) and torch.isfinite(a).all()
    altered = {k: v.clone() for k, v in batch.items()}
    altered['states'] += .5
    assert not torch.allclose(a, sample(10, altered))
    invalid = torch.zeros_like(batch['action_mask'])
    with pytest.raises(ValueError, match='legal action'):
        model.denoiser(batch['actions'], times, model.encoder(batch), invalid)


def test_causal_history_encoder_does_not_use_later_frames(config, dataset):
    model = ActionDiT(config)
    batch = default_collate([dataset[-1]])
    before = model.encoder(batch).tokens[:, 2:7].detach()
    batch['history'][:, 5:] += 100
    after = model.encoder(batch).tokens[:, 2:7].detach()
    torch.testing.assert_close(before, after)


def test_chunk_phase_expiry_and_four_action_prefix():
    chunk = ActionChunk(50, 'TRANSFER', np.arange(96).reshape(16, 6))
    np.testing.assert_array_equal(chunk.current(100, 'TRANSFER'), np.arange(6, 12))
    assert chunk.current(250, 'TRANSFER') is None
    assert chunk.current(100, 'RETRACT') is None
    assert chunk.current(0, 'TRANSFER') is None


def test_missing_recovery_or_stage_coverage_cannot_pass(config):
    results = [dict(recover=i >= 15, success=True, physical_success=True, recovery_completed=i >= 15,
                    entered=['ACQUIRE', 'TRANSFER', 'RETRACT'] + (['RECOVER'] if i >= 15 else []),
                    pickup=True, delivery=True, contact_peak_n=1., wrist_peak_n=2.) for i in range(30)]
    assert summarize(results, config)['status'] == 'passed'
    results[-1]['entered'].remove('RECOVER')
    assert summarize(results, config)['status'] == 'failed'
    assert summarize(results[:15], config)['status'] == 'failed'


def test_diagnostic_weights_cannot_freeze(tmp_path):
    from feedingrobot.scripts.validate_m5 import freeze_policy
    path = tmp_path / 'diagnostic.pt'
    torch.save(dict(diagnostic=True, step=200), path)
    with pytest.raises(ValueError, match='Diagnostic'):
        freeze_policy(path, tmp_path / 'release')


def test_rng_and_atomic_checkpoint_resume(tmp_path):
    rng = np.random.default_rng(7)
    state = rng_state(rng)
    expected = rng.integers(100, size=10), torch.randn(5)
    restore_rng(state, rng)
    np.testing.assert_array_equal(rng.integers(100, size=10), expected[0])
    torch.testing.assert_close(torch.randn(5), expected[1])
    path = tmp_path / 'last.pt'
    save_checkpoint(path, dict(step=2))
    assert torch.load(path, weights_only=False)['step'] == 2 and not path.with_suffix('.tmp').exists()


def test_training_resume_matches_uninterrupted_updates(config, dataset, tmp_path, monkeypatch):
    episode(dataset.directory, split='validation')
    episode(dataset.directory, recover=True, split='validation')
    config['dataset'] = str(dataset.directory)
    parent = dict(parent_sha256='parent', dataset_binding='data')
    monkeypatch.setattr(training, 'parent_check', lambda _: parent)
    monkeypatch.setattr(training, 'input_hashes', lambda: {'code': 'fixed'})
    monkeypatch.setattr(audit, 'input_hashes', lambda: {'code': 'fixed'})
    def setup(config, device):
        torch.manual_seed(config['training']['seed'])
        torch.set_num_threads(1)
        return torch.device('cpu'), {'device': 'cpu'}
    monkeypatch.setattr(training, 'setup', setup)
    continuous, resumed = tmp_path / 'continuous', tmp_path / 'resumed'
    training.train(config, continuous, device='cpu', updates=4, headless=True)
    training.train(config, resumed, device='cpu', updates=2, headless=True)
    training.train(config, resumed, device='cpu', updates=4, resume=resumed / 'last.pt', headless=True)
    a, b = (torch.load(path / 'last.pt', weights_only=False) for path in (continuous, resumed))
    assert a['step'] == b['step'] == 4
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
        torch.testing.assert_close(a['ema'][name], b['ema'][name], rtol=0, atol=0)


@pytest.fixture
def parent(tmp_path):
    from feedingrobot.scripts.collect import statistics_digest
    config = dict(robot_id='panda', parent='parent', dataset='data')
    physical = tmp_path / 'src/physics.py'
    physical.parent.mkdir()
    physical.write_text('fixed physics')
    archived = tmp_path / 'parent/frozen_inputs/src/physics.py'
    archived.parent.mkdir(parents=True)
    archived.write_bytes(physical.read_bytes())
    hashes = {'src/physics.py': audit.sha256(physical)}
    files = {}
    for name in ('train/normal/manifest.json', 'train/recovery/manifest.json', 'normalization.json'):
        p = tmp_path / 'data' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
        files[name] = audit.sha256(p)
    stats = dict(status='passed', replay_status='passed', robot_id='panda', input_hashes=hashes,
                 counts={'train': 1}, recovery_counts={'train': 1}, attempts=2, evidence_sha256=files,
                 episodes=[dict(path=f'train/{kind}', replay={'status': 'passed'}) for kind in ('normal', 'recovery')])
    stats['validation_sha256'] = statistics_digest(stats)
    write_json(tmp_path / 'data/statistics.json', stats)
    evidence = {f'data/{name}': digest for name, digest in files.items()}
    evidence['data/statistics.json'] = audit.sha256(tmp_path / 'data/statistics.json')
    frozen = dict(status='frozen', acceptance_robots=['panda'], model_version='single_bean_native_v1',
                  observation_schema_version=3, dataset='data', input_sha256=hashes, evidence_sha256=evidence,
                  teacher_config={'quotas': {'train': 1}})
    write_json(tmp_path / 'parent/freeze_manifest.json', frozen)
    write_json(tmp_path / 'parent/acceptance_audit.json', dict(status='passed', teacher_gate='passed', dataset_gate='passed',
               input_sha256=hashes, evidence_sha256=evidence))
    return tmp_path, config


def test_parent_check_is_read_only_and_does_not_replay(parent, monkeypatch):
    root, config = parent
    from feedingrobot.data import replay
    monkeypatch.setattr(replay, 'replay_many', lambda *a, **k: pytest.fail('Old physics replay'))
    result = audit.parent_check(config, root=root)
    assert result['status'] == 'passed' and result['old_replay_reruns'] == 0


@pytest.mark.parametrize('relative', ['src/physics.py', 'parent/frozen_inputs/src/physics.py',
                                     'data/normalization.json', 'data/train/normal/manifest.json',
                                     'data/statistics.json'])
def test_parent_rejects_changed_inputs_and_evidence(parent, relative):
    root, config = parent
    (root / relative).write_text('{"changed":true}')
    with pytest.raises(ValueError):
        audit.parent_check(config, root=root)


def test_checkpoint_rejects_wrong_config_data_or_source(config, monkeypatch):
    parent = dict(parent_sha256='parent', dataset_binding='data')
    checkpoint = dict(schema_version=1, config=config, parent=parent, source_hashes={'code': 'old'})
    monkeypatch.setattr(audit, 'input_hashes', lambda: {'code': 'new'})
    with pytest.raises(ValueError, match='Checkpoint'):
        audit.checkpoint_check(checkpoint, config, parent)


def test_cuda_unavailable_does_not_silently_train_on_cpu(config, monkeypatch):
    from feedingrobot.policies.runtime import setup
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='No CPU fallback'):
        setup(config, 'cuda')


def test_cpu_budget_above_eight_is_rejected(config):
    from feedingrobot.policies.runtime import setup
    config['training']['cpu_budget'] = 9
    with pytest.raises(ValueError, match='CPU budget'):
        setup(config, 'cpu')


def test_spawn_workers_read_identical_mmap_windows(dataset):
    from torch.utils.data import DataLoader
    expected = default_collate([dataset[0], dataset[-1]])
    loader = DataLoader(dataset, batch_sampler=[[0, len(dataset)-1]], num_workers=2,
                        multiprocessing_context='spawn', generator=torch.Generator().manual_seed(1))
    actual = next(iter(loader))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
