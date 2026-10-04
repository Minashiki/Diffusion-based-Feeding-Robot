"""M3 release must be complete, paired, and tied to its physical M1 parent."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from feedingrobot.scripts import validate_m3 as m3
from feedingrobot.scripts import m3_driver
from feedingrobot.sim.model import load_json


def reports(hashes):
    return {r:dict(status='passed', hashes_unchanged=True, input_hashes=hashes,
                   final_input_hashes=hashes, stages={'M3':'passed','M4':'not_verified'},
                   cases={n:dict(status='passed') for n in m3.CASES})
            for r in ('panda','ur5e')}


@pytest.mark.parametrize('damage', ['missing_robot', 'missing_case', 'failed_case', 'input_drift', 'm4_release'])
def test_freeze_rejects_incomplete_or_changed_evidence(tmp_path, monkeypatch, damage):
    hashes = {'input.py':'frozen'}
    data = reports(hashes)
    monkeypatch.setattr(m3, 'input_hashes', lambda: copy.deepcopy(hashes))
    monkeypatch.setattr(m3, 'parent_freeze_check', lambda: {'status':'passed'})
    if damage == 'missing_robot':
        del data['ur5e']
    elif damage == 'missing_case':
        del data['panda']['cases']['full_dynamic']
    elif damage == 'failed_case':
        data['ur5e']['cases']['viewer']['status'] = 'failed'
    elif damage == 'input_drift':
        data['panda']['final_input_hashes'] = {'input.py':'changed'}
    else:
        data['panda']['stages']['M4'] = 'passed'
    with pytest.raises((AssertionError, KeyError)):
        m3.publish_freeze(data, tmp_path, hashes)
    assert not (tmp_path/'freeze_manifest.json').exists()


def test_matrix_contains_natural_pickup_and_both_full_flows():
    assert len(m3.PHYSICAL_CASES) == 21
    assert set(m3.FULL_CASES) == {'full_static','full_dynamic'}
    assert {'pickup_lift','bowl','bowl_return','m1_regression','manifest','viewer'} <= set(m3.CASES)


def test_retraction_speed_limit_preserves_mouth_tracking(monkeypatch):
    rotation = np.eye(3)
    e = dict(tcp_position=np.zeros(3), mouth_position=np.zeros(3), mouth_rotation=rotation,
             wait_position=np.array([-.06,0.,0.]))
    monkeypatch.setattr(m3_driver, 'evidence', lambda task:e)
    def velocity(model,data,kind,index,result,local):
        result[:] = [0.,0.,0.,0.,.01,0.]
    monkeypatch.setattr(m3_driver.mujoco, 'mj_objectVelocity', velocity)
    task = SimpleNamespace(data=SimpleNamespace(time=2.,site_xmat=np.array([rotation.ravel()]*2)),
                           index=SimpleNamespace(tcp=0,base=1),
                           logic=SimpleNamespace(phase='RETRACT',acquired=True),
                           model=SimpleNamespace(site=lambda name:SimpleNamespace(id=0)),
                           robot_config=dict(linear_speed_limit=.05,angular_speed_limit=.5))
    driver = m3_driver.FeedingDriver.__new__(m3_driver.FeedingDriver)
    driver.path = [('unused',None,None,None)]
    driver.part, driver.lifted = 1, True
    driver.pickup_hold_start, driver.transport_start = 0., {}
    driver.release_rotation = np.zeros(3)
    driver.release_part = 3
    action = driver.action(task)
    np.testing.assert_allclose(action[:3]*.05,[-.025,.01,0.],rtol=0,atol=1e-12)
    assert np.linalg.norm(action[:3]*.05) < .05


def test_early_delivery_cannot_skip_release_pose_and_clearance(monkeypatch):
    e = dict(tcp_position=np.array([.008,-.008,-.014]),mouth_position=np.zeros(3),
             mouth_rotation=np.eye(3),wait_position=np.array([-.06,0.,0.]))
    monkeypatch.setattr(m3_driver,'evidence',lambda task:e)
    monkeypatch.setattr(m3_driver.mujoco,'mj_objectVelocity',lambda m,d,k,i,v,l:v.fill(0.))
    task = SimpleNamespace(data=SimpleNamespace(time=3.,site_xmat=np.array([np.eye(3).ravel()]*3),
                           site_xpos=np.array([[0.,0.,0.],[0.,0.,0.],[.02,0.,-.015]])),
                           index=SimpleNamespace(tcp=0,base=1),
                           logic=SimpleNamespace(phase='RETRACT',acquired=True,delivered=True),
                           model=SimpleNamespace(site=lambda name:SimpleNamespace(id=2)),
                           robot_config=dict(linear_speed_limit=.05,angular_speed_limit=.5),
                           _event_scoop_points=np.zeros((1,3)))
    driver = m3_driver.FeedingDriver.__new__(m3_driver.FeedingDriver)
    driver.path, driver.part, driver.lifted = [('unused',None,None,None)], 1, True
    driver.pickup_hold_start, driver.transport_start = 0., {}
    driver.transfer_start, driver.roll_start, driver.release_part = 1., 0., 1
    driver.action(task)
    assert driver.stage == 'release_roll' and driver.release_part == 1
    task.data.site_xmat[0] = m3_driver.mink.SO3.exp([-1.,.4,0.]).as_matrix().ravel()
    driver.action(task)
    assert driver.release_part == 2
    driver.action(task)
    assert driver.stage == 'release_clear' and driver.release_part == 2
    e['tcp_position'] += [-.003,-.003,.001]
    driver.action(task)
    assert driver.release_part == 3
    driver.action(task)
    assert driver.stage == 'retract'


def comparison_run(delay=0.):
    samples = []
    for time in np.arange(.02, .4+delay+.001, .02):
        moving = time > .2+delay+1e-9
        position = [.05*max(0.,time-.2-delay), 0., 0.]
        samples.append(dict(time=float(time), stage='move' if moving else 'hold', frame='world',
                            position=position, comparison_position=position.copy()))
    return dict(scenario='full_static', success=True, failure_reason=None, phase='RETRACT', time=.4+delay,
                events=[dict(name='pickup',time=.2+delay),dict(name='success',time=.4+delay)],
                action_starts=[dict(stage='hold',time=0.),dict(stage='move',time=.2+delay)],
                tcp_position=[.01,0.,0.], tcp_comparison_position=[.01,0.,0.], tcp_comparison_frame='world',
                tcp_samples=samples, event_tcp_positions=[[0.,0.,0.],[.01,0.,0.]],
                event_tcp_comparison_positions=[[0.,0.,0.],[.01,0.,0.]],
                peak_force_n=0., impulse_ns=0., contact_pair_peaks_n={}, contact_pair_impulses_ns={})


def test_comparison_separates_safe_wait_from_action_path():
    result = m3.compare_runs(comparison_run(), comparison_run(.12), load_json('configs/acceptance_m3.json'))
    assert result['passed']
    assert result['errors']['tcp_unaligned_world_path_m'] > .002
    assert result['errors']['tcp_path_m'] < 1e-12
    assert result['errors']['action_start_time_s'] == pytest.approx(.12)


def test_comparison_ignores_stationary_wait_at_end_of_movement():
    base, changed = comparison_run(), comparison_run()
    changed['time'] = changed['events'][-1]['time'] = .52
    for time in np.arange(.42,.521,.02):
        sample = copy.deepcopy(changed['tcp_samples'][-1])
        sample['time'] = float(time)
        changed['tcp_samples'].append(sample)
    assert m3.compare_runs(base,changed,load_json('configs/acceptance_m3.json'))['passed']


def test_contact_confirmation_delay_does_not_relax_entry_timing():
    base, delayed = comparison_run(), comparison_run(.6)
    cfg = load_json('configs/acceptance_m3.json')
    assert not m3.compare_runs(base,delayed,cfg)['passed']
    for run in (base,delayed):
        run['action_starts'][-1]['stage'] = 'retract'
        for sample in run['tcp_samples']:
            if sample['stage'] == 'move':
                sample['stage'] = 'retract'
    assert m3.compare_runs(base,delayed,cfg)['passed']
    delayed['events'][-1]['time'] += .42
    assert not m3.compare_runs(base,delayed,cfg)['passed']


def test_contact_event_positions_are_diagnostics_along_same_action_path():
    base, changed = comparison_run(), comparison_run()
    for run,time,position in ((base,.24,[.002,0.,0.]),(changed,.36,[.008,0.,0.])):
        run['events'].insert(1,dict(name='delivery',time=time))
        run['event_tcp_positions'].insert(1,position)
        run['event_tcp_comparison_positions'].insert(1,position)
    result = m3.compare_runs(base,changed,load_json('configs/acceptance_m3.json'))
    assert result['passed']
    assert result['errors']['event_tcp_position_m'] == pytest.approx(.006)
    assert result['errors']['pose_event_tcp_position_m'] == 0.


def test_timing_budget_fits_dynamic_motion_and_aperture_margin():
    head = load_json('configs/scene.json')['head']
    task = load_json('configs/task.json')
    cfg = load_json('configs/acceptance_m3.json')
    angular_frequency = 2*np.pi*head['freq_hz']
    mouth_speed = angular_frequency*(np.sqrt(2)*head['amp_m'] + task['wait_offset_m']*head['yaw_amp_rad'])
    aperture_speed = angular_frequency*.04*head['jaw_amp_rad']
    assert cfg['event_time_tolerance_s']*mouth_speed < task['position_tolerance_m']
    assert cfg['event_time_tolerance_s']*aperture_speed < task['clearance_margin_m']
    assert cfg['contact_event_time_tolerance_s'] < 1/(4*head['freq_hz'])
    assert cfg['failure_time_tolerance_s'] == .01


@pytest.mark.parametrize('damage', ['path', 'start', 'grid', 'frame', 'sequence', 'failure_time'])
def test_comparison_still_rejects_spatial_timing_and_evidence_errors(damage):
    base, changed = comparison_run(), comparison_run()
    if damage == 'path':
        changed['tcp_samples'][-4]['comparison_position'][1] = .003
    elif damage == 'start':
        changed['action_starts'][1]['time'] += .42
    elif damage == 'grid':
        del changed['tcp_samples'][2]
    elif damage == 'frame':
        changed['tcp_samples'][-1]['frame'] = 'mouth'
    elif damage == 'sequence':
        changed['events'][-1]['name'] = 'failure'
    else:
        base['events'][-1]['name'] = changed['events'][-1]['name'] = 'failure'
        changed['events'][-1]['time'] += .02
    assert not m3.compare_runs(base, changed, load_json('configs/acceptance_m3.json'))['passed']
