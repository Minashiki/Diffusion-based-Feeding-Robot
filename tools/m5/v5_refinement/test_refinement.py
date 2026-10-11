"""CPU-only refinement regression checks; no dataset access or physics rollouts."""

import copy
import contextlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feedingrobot.policies import audit
from feedingrobot.policies.runtime import rng_state
from tools.m5.v5_refinement import run


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def assert_same(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def normalization():
    fields = [['q', 1, 'float'], ['stage', 1, 'float'],
              ['interaction', 1, 'float'], ['execution_status', 1, 'float']]

    def params(size):
        return dict(scale=[1.] * size, offset=[0.] * size,
                    input_stats=dict(min=[-1.] * size, max=[1.] * size,
                                     mean=[0.] * size, std=[.5] * size))

    return dict(schema_version=5, kind='dp_limits', source_split='train',
                observation_timing='pre_action', clipping=False,
                observation_schema=dict(fields=fields),
                params_dict={name: params(size) for name, size in
                             dict(observation=4, velocity=12, state_age=1, history_age=1, action=6).items()})


def write_validation(directory, pickups, normal_pickups, full=0):
    directory.mkdir(parents=True)
    report = dict(score=full / 30,
                  full=dict(normal=dict(attempts=15, successes=full),
                            recovery=dict(attempts=15, successes=0)),
                  stages=dict(pickup=dict(successes=pickups, attempts=30)))
    run.write_json(directory / 'report.json', report)
    for recover in (False, True):
        count = pickups - normal_pickups if recover else normal_pickups
        for index in range(15):
            run.write_json(directory / f'{int(recover)}_{index:02d}.json',
                           dict(recover=recover, pickup=index < count,
                                success=not recover and index < full))
    return report


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    for name, value in dict(START_STEP=4, SWITCH_STEP=8, FINAL_STEP=12,
                            VALIDATION_EVERY=2, CHECKPOINT_EVERY=2, EVALUATION_EVERY=4).items():
        monkeypatch.setattr(run, name, value)
    parent_audit = dict(parent_sha256='frozen-parent', dataset_binding='frozen-data')
    source = {'test-source.py': 'frozen-source'}
    monkeypatch.setattr(run, 'parent_check', lambda _: copy.deepcopy(parent_audit))
    monkeypatch.setattr(run, 'correction_check', lambda _: None)
    monkeypatch.setattr(run, 'start_pose_check', lambda _: None)
    monkeypatch.setattr(run, 'input_hashes', lambda: copy.deepcopy(source))
    monkeypatch.setattr(audit, 'input_hashes', lambda: copy.deepcopy(source))

    config = dict(dataset='unused', model=dict(name='MinaDiTV5', horizon=4),
                  training=dict(seed=123, lr=1e-4, ema_decay=.9999, updates=100000,
                                batch_size=3, weight_decay=.01, gradient_clip=1.,
                                workers=0, precision='fp32'))

    def setup(config, device):
        assert device == 'cpu'
        random.seed(config['training']['seed'])
        np.random.seed(config['training']['seed'])
        torch.manual_seed(config['training']['seed'])
        return torch.device('cpu'), {'device': 'mock-cpu'}

    datasets = []

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            self.fetched = []
            self.normalization = None
            datasets.append(self)

        def set_normalization(self, value):
            self.normalization = value

        def sample_indices(self, rng, count):
            return rng.integers(0, 13, size=count).tolist()

        def __len__(self):
            return 13

        def __getitem__(self, index):
            self.fetched.append(index)
            return dict(x=torch.tensor([index / 13], dtype=torch.float64),
                        y=torch.tensor([index / 31], dtype=torch.float64))

    def create_model(_):
        return torch.nn.Linear(1, 1).double()

    def loss(model, _, batch):
        noise = torch.randn_like(batch['y']) * .01 + np.random.normal() * .001 + random.random() * .001
        return ((model(batch['x']) - batch['y'] - noise) ** 2).mean()

    actions = dict(first_four={kind: dict(vector_rmse=.1, zero_action_vector_rmse=1.)
                               for kind in ('linear', 'angular')})
    control = SimpleNamespace(pickups=5, normal_pickups=1, full=0)

    def evaluate(model, config, normalization, device, split, directory, viewer):
        assert split == 'validation' and device.type == 'cpu' and viewer is False
        return write_validation(directory, control.pickups, control.normal_pickups, control.full)

    monkeypatch.setattr(run, 'setup', setup)
    monkeypatch.setattr(run, 'TrainingWindows', Dataset)
    monkeypatch.setattr(run, 'ActionWindows', Dataset)
    monkeypatch.setattr(run, 'create_model', create_model)
    monkeypatch.setattr(run, 'schedulers', lambda _: (None, None))
    monkeypatch.setattr(run, 'noise_loss', loss)
    monkeypatch.setattr(run, 'precision', lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(run, 'validation_loss', lambda *args: .1)
    monkeypatch.setattr(run, 'validation_actions', lambda *args: copy.deepcopy(actions))
    monkeypatch.setattr(run, 'evaluate', evaluate)

    setup(config, 'cpu')
    model = create_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    for _ in range(2):
        optimizer.zero_grad()
        (model(torch.tensor([[.25]], dtype=torch.float64)) - .75).square().mean().backward()
        optimizer.step()
    ema = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in ema.parameters():
            parameter.add_(.2)
    sampler = np.random.default_rng(543)
    sampler.integers(0, 13, size=7)
    original = dict(schema_version=1, step=run.START_STEP, config=config,
                    parent=parent_audit, source_hashes=source, normalization=normalization(),
                    model_name='MinaDiTV5', model=model.state_dict(), ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(), rng=rng_state(sampler), diagnostic=False,
                    correction=None, train_extension=None)
    parent = tmp_path / 'parent'
    parent.mkdir()
    run.write_json(parent / 'normalization.json', original['normalization'])
    (parent / 'metrics.jsonl').write_text(json.dumps(dict(step=run.START_STEP, validation_actions=actions)) + '\n')
    write_validation(parent / f'validation_{run.START_STEP}', 2, 0)
    checkpoint = parent / f'step_{run.START_STEP}.pt'
    torch.save(original, checkpoint)
    return SimpleNamespace(parent=checkpoint, original=load(checkpoint), control=control,
                           output=tmp_path / 'refined', datasets=datasets, source=source)


def train(experiment, target, *, resume=False, output=None):
    output = output or experiment.output
    return run.refine(output, updates=target, device='cpu', headless=True,
                      **({'resume': output / 'last.pt'} if resume else {'init_from': experiment.parent}))


def test_real_100k_learning_rate_boundary():
    assert run.learning_rate(90000) == run.learning_rate(99999) == 2e-5
    assert run.learning_rate(100000) == run.learning_rate(109999) == 1e-5


def test_initial_checkpoint_retains_parent_state_and_overrides_loaded_lr(experiment):
    parent_bytes = experiment.parent.read_bytes()
    train(experiment, 5)
    initial = load(experiment.output / 'step_4.pt')
    for key in ('model', 'ema', 'rng', 'normalization'):
        assert_same(initial[key], experiment.original[key])
    expected_optimizer = copy.deepcopy(experiment.original['optimizer'])
    assert expected_optimizer['param_groups'][0]['lr'] == 1e-4
    expected_optimizer['param_groups'][0]['lr'] = 2e-5
    assert_same(initial['optimizer'], expected_optimizer)
    assert initial['config']['training']['ema_decay'] == .9998
    updated = load(experiment.output / 'last.pt')
    assert updated['optimizer']['param_groups'][0]['lr'] == 2e-5
    for key, average in initial['ema'].items():
        expected = average.clone().lerp_(updated['model'][key], 1 - .9998)
        torch.testing.assert_close(updated['ema'][key], expected, rtol=0, atol=0)
    assert experiment.parent.read_bytes() == parent_bytes


def test_interrupted_resume_across_lr_switch_exactly_matches_continuous(experiment, monkeypatch):
    continuous = experiment.output.parent / 'continuous'
    train(experiment, 12, output=continuous)
    expected = load(continuous / 'last.pt')
    fetched = experiment.datasets[0].fetched.copy()
    experiment.datasets.clear()
    original_loss = run.noise_loss
    calls = 0

    def interrupted_loss(*args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt('Interrupt after uncommitted update 7')
        return original_loss(*args)

    monkeypatch.setattr(run, 'noise_loss', interrupted_loss)
    with pytest.raises(KeyboardInterrupt):
        train(experiment, 12)
    assert load(experiment.output / 'last.pt')['step'] == 6
    stale_metrics = [json.loads(line) for line in (experiment.output / 'metrics.jsonl').read_text().splitlines()]
    assert stale_metrics[-1]['step'] == 7
    committed_fetches = experiment.datasets[0].fetched[:6]
    experiment.datasets.clear()
    monkeypatch.setattr(run, 'noise_loss', original_loss)
    train(experiment, 8, resume=True)
    train(experiment, 12, resume=True)
    actual = load(experiment.output / 'last.pt')
    for key in ('model', 'ema', 'optimizer', 'rng', 'selection'):
        assert_same(actual[key], expected[key])
    assert fetched == committed_fetches + [index for dataset in experiment.datasets for index in dataset.fetched]
    metrics = [json.loads(line) for line in (experiment.output / 'metrics.jsonl').read_text().splitlines()]
    assert [record['step'] for record in metrics] == list(range(5, 13))
    assert [record['lr'] for record in metrics] == [2e-5] * 4 + [1e-5] * 4
    assert all(record['ema_decay'] == .9998 for record in metrics)
    assert actual['progress_gate']['passed'] is True


def test_interrupted_evaluation_retries_existing_directory_and_restores_training(experiment, monkeypatch):
    continuous = experiment.output.parent / 'continuous'
    train(experiment, 12, output=continuous)
    expected = load(continuous / 'last.pt')
    original_evaluate = run.evaluate

    def interrupted_evaluate(*args, **kwargs):
        original_evaluate(*args, **kwargs)
        raise RuntimeError('Interrupted after evaluation wrote its directory')

    monkeypatch.setattr(run, 'evaluate', interrupted_evaluate)
    with pytest.raises(RuntimeError, match='Interrupted after evaluation'):
        train(experiment, 12)
    assert load(experiment.output / 'last.pt')['step'] == 6
    assert (experiment.output / 'validation_8' / 'report.json').exists()
    monkeypatch.setattr(run, 'evaluate', original_evaluate)
    train(experiment, 12, resume=True)
    actual = load(experiment.output / 'last.pt')
    for key in ('model', 'ema', 'optimizer', 'rng', 'selection'):
        assert_same(actual[key], expected[key])
    assert actual['progress_gate']['passed'] is True
    assert (experiment.output / 'validation_8_retry_1' / 'report.json').exists()
    assert all('validation_8_retry_1' in path for path in actual['progress_gate']['evidence'])


def test_resume_discards_only_truncated_final_metric_row(experiment):
    train(experiment, 6)
    with (experiment.output / 'metrics.jsonl').open('a') as stream:
        stream.write('{"step":7,"loss":')
    train(experiment, 12, resume=True)
    rows = [json.loads(line) for line in (experiment.output / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(5, 13))


def test_resume_rejects_nonfinal_malformed_metric_row(experiment):
    train(experiment, 6)
    path = experiment.output / 'metrics.jsonl'
    final = path.read_text().splitlines()[-1]
    path.write_text('{"step":5,"loss":\n' + final + '\n')
    with pytest.raises(json.JSONDecodeError):
        train(experiment, 12, resume=True)


@pytest.mark.parametrize('pickups,normal_pickups,full,passed', [(2, 0, 0, False), (5, 0, 0, False),
                                                              (5, 1, 0, True), (1, 1, 1, True)])
def test_progress_gate_requires_meaningful_closed_loop_gain(pickups, normal_pickups, full, passed):
    assert run.gate_passed(dict(pickup_successes=pickups, normal_pickups=normal_pickups,
                                full_successes=full)) is passed


def test_failed_gate_stops_at_switch_and_resume_cannot_continue(experiment):
    experiment.control.pickups, experiment.control.normal_pickups = 2, 0
    status = train(experiment, 12)
    assert status['step'] == 8 and status['stop_reason'] == 'progress_gate_failed'
    assert status['progress_gate']['passed'] is False
    assert not (experiment.output / 'step_10.pt').exists()
    before = (experiment.output / 'last.pt').read_bytes()
    with pytest.raises(ValueError, match='progress gate failed'):
        train(experiment, 12, resume=True)
    assert (experiment.output / 'last.pt').read_bytes() == before


def test_editing_failed_gate_stats_and_decision_cannot_bypass_evidence(experiment):
    experiment.control.pickups, experiment.control.normal_pickups = 2, 0
    train(experiment, 12)
    path = experiment.output / 'last.pt'
    checkpoint = load(path)
    checkpoint['progress_gate']['stats'].update(pickup_successes=5, normal_pickups=1)
    checkpoint['progress_gate']['passed'] = True
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='gate decision differs from its validation evidence'):
        train(experiment, 12, resume=True)


@pytest.mark.parametrize('changed', ['parent', 'tool', 'config', 'normalization', 'baseline', 'source'])
def test_resume_rejects_lineage_tampering(experiment, changed):
    train(experiment, 6)
    if changed == 'parent':
        parent = load(experiment.parent)
        parent['model']['weight'].add_(1.)
        torch.save(parent, experiment.parent)
    elif changed == 'tool':
        checkpoint = load(experiment.output / 'last.pt')
        checkpoint['refinement']['tool_sha256'] = 'modified-tool'
        torch.save(checkpoint, experiment.output / 'last.pt')
        run.write_json(experiment.output / 'refinement.json', checkpoint['refinement'])
    elif changed in ('config', 'normalization'):
        path = experiment.output / f'{changed}.json'
        value = run.read_json(path)
        if changed == 'config':
            value['training']['lr'] = 1e-4
        else:
            value['params_dict']['action']['input_stats']['mean'][0] = .1
        run.write_json(path, value)
    elif changed == 'baseline':
        with (experiment.parent.parent / 'metrics.jsonl').open('a') as stream:
            stream.write('\n')
    else:
        experiment.source['test-source.py'] = 'modified-source'
    with pytest.raises(ValueError, match='changed|differs'):
        train(experiment, 12, resume=True)


def test_resume_rejects_tampered_gate_case(experiment):
    train(experiment, 8)
    case = experiment.output / 'validation_8' / '0_00.json'
    result = run.read_json(case)
    result['pickup'] = False
    run.write_json(case, result)
    with pytest.raises(ValueError, match='gate evidence changed'):
        train(experiment, 12, resume=True)


@pytest.mark.parametrize('changed', ['kind_count', 'full_count', 'pickup_count'])
def test_parent_validation_report_must_match_all_cases(experiment, changed):
    case = experiment.parent.parent / 'validation_4' / '0_00.json'
    result = run.read_json(case)
    if changed == 'kind_count':
        result['recover'] = True
    elif changed == 'full_count':
        result['success'] = True
    else:
        result['pickup'] = True
    run.write_json(case, result)
    with pytest.raises(ValueError, match='differs from its episode results'):
        run.refine(experiment.output, init_from=experiment.parent, check_only=True)


def test_check_only_is_read_only_and_needs_no_runtime_setup(experiment, monkeypatch):
    monkeypatch.setattr(run, 'setup', lambda *args: pytest.fail('check-only must not set up training'))
    before = experiment.parent.read_bytes()
    status = run.refine(experiment.output, init_from=experiment.parent, check_only=True)
    assert status['status'] == 'checked' and status['lr'] == 2e-5 and status['ema_decay'] == .9998
    assert not experiment.output.exists()
    assert experiment.parent.read_bytes() == before
