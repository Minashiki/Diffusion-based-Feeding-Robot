"""M1-D numerical replay and fail-closed freeze contracts."""

import copy
import json

import numpy as np
import pytest

from feedingrobot.sim.task import FeedingTask
from feedingrobot.sim.model import load_json
from feedingrobot.sim.contacts import ContactMonitor
from feedingrobot.scripts import validate_m1 as d


@pytest.fixture(scope='module')
def states():
    base = FeedingTask('panda')
    base.scene_config['head_fixed'] = True
    base.reset(preset='empty')
    base.adapter.set_twist([.01,0,0,0,0,0],0,.04)
    for _ in range(20):
        base.step_physics()
    other = FeedingTask('panda', timestep=.0005)
    other.scene_config['head_fixed'] = True
    return base, other, base.get_state()


def test_full_state_replay_and_strict_public_compatibility(states):
    base, other, state = states
    with pytest.raises(ValueError):
        other.set_state(state)
    d.restore_numerical_state(other,state)
    result = other.get_state()
    np.testing.assert_array_equal(result['physics'],state['physics'])
    np.testing.assert_array_equal(result['boundary']['sensordata'],state['boundary']['sensordata'])
    np.testing.assert_array_equal(result['boundary']['qacc'],state['boundary']['qacc'])
    np.testing.assert_array_equal(result['adapter']['reference_q'],state['adapter']['reference_q'])
    np.testing.assert_array_equal(result['adapter']['target'],state['adapter']['target'])
    assert result['monitor']==state['monitor']
    assert result['task']['scenario_state']==state['task']['scenario_state']
    assert other.tick==40 and other.data.time==base.data.time
    assert other.adapter.command[1]==base.adapter.command[1]
    assert [tick*.0005 for tick in range(0,81,round(.02/.0005))]==[tick*.001 for tick in range(0,41,round(.02/.001))]


def test_numerical_replay_rejects_other_parameter_changes(states):
    base, other, state = states
    other.model.geom_friction[other.index.bean_collision_geoms[0],0] += .01
    try:
        with pytest.raises(ValueError,match='non-numerical'):
            d.restore_numerical_state(other,state)
    finally:
        other.model.geom_friction[other.index.bean_collision_geoms[0],0] -= .01


@pytest.mark.parametrize('robot', ['panda', 'ur5e'])
def test_acceleration_drop_has_a_common_physical_endpoint(robot, tmp_path):
    from feedingrobot.scripts import validate_m1c as c
    reports = {}
    for name, setting in d.SETTINGS.items():
        reports[name] = c.validate(robot, tmp_path/name, ['acceleration'], numerical=setting,
                                  replay=tmp_path/'baseline' if name != 'baseline' else None)
        case = reports[name]['cases']['acceleration']
        assert case['status'] == 'passed', case.get('error')
        assert reports[name]['status'] == 'incomplete'
        assert case['metrics']['absent_and_outside_s'] >= .1 - 1e-12
        assert case['metrics']['time_s'] < case['metrics']['observation_end_s']
        assert case['metrics']['observation_end_s'] == pytest.approx(.5)
    left = json.loads((tmp_path/'baseline/acceleration_trajectory.json').read_text())
    for name in ('dt05ms', 'iterations200'):
        right = json.loads((tmp_path/name/'acceleration_trajectory.json').read_text())
        metrics = d.compare_motion(reports['baseline']['cases']['acceleration'],
                                   reports[name]['cases']['acceleration'], left, right,
                                   load_json('configs/acceptance.json'))
        assert metrics['duration_difference_s'] == pytest.approx(0)


def fixture_metrics():
    row=dict(time_s=.01,tcp_position=[0.,0.,0.],tcp_rotation=np.eye(3).tolist(),
             bean_positions=np.zeros((1,3)).tolist(),compensated_wrench=[0.]*6)
    result=dict(status='passed',final_tcp_position=[0.,0.,0.],final_tcp_rotation=np.eye(3).tolist(),
                max_wrist_force_n=.1,semantic_pair_peaks_n={'food|spoon':1.},
                semantic_pair_impulses_ns={'food|spoon':.01},metrics={})
    return result,[row]


@pytest.mark.parametrize('damage',['position','rotation','force','impulse','nan','missing','metric_nan','terminal_nan','terminal_rotation_nan','grid_gap'])
def test_convergence_rejects_invalid_or_excessive_evidence(damage):
    base,left=fixture_metrics(); other,right=copy.deepcopy((base,left))
    if damage=='position':
        right[0]['tcp_position'][0]=.003
    elif damage=='rotation':
        from feedingrobot.scripts.validate_m1c import mink
        right[0]['tcp_rotation']=mink.SO3.exp([.04,0,0]).as_matrix()
    elif damage=='force':
        other['semantic_pair_peaks_n']['food|spoon']=1.21
    elif damage=='impulse':
        other['semantic_pair_impulses_ns']['food|spoon']=.016
    elif damage=='nan':
        right[0]['tcp_position'][0]=np.nan
    elif damage=='metric_nan':
        other['max_wrist_force_n']=np.nan
    elif damage=='terminal_nan':
        other['final_tcp_position'][0]=np.nan
    elif damage=='terminal_rotation_nan':
        other['final_tcp_rotation'][0][0]=np.nan
    elif damage=='grid_gap':
        right.append(copy.deepcopy(right[0]));right[-1]['time_s']=.03
    else:
        right=[]
    with pytest.raises(AssertionError):
        d.compare_motion(base,other,left,right,load_json('configs/acceptance.json'))


def test_applied_impulse_not_integrated_at_boundary():
    monitor=ContactMonitor(5.)
    rows=[dict(group1='food',group2='spoon',force_n=2.)]
    monitor.update(rows,.001,.001)
    monitor.update([dict(rows[0],force_n=3.)],0.,.001)
    assert monitor.pair_peaks['food|spoon']==3.
    assert monitor.pair_impulses['food|spoon']==pytest.approx(.002)


def passing_reports():
    hashes={'input':'hash'}
    report=dict(model_version='single_bean_native_v1',snapshot_schema_version=3,status='passed',hashes_unchanged=True,
                input_hashes=hashes,final_input_hashes=hashes,
                stages={**{n:'passed' for n in ('M1-A','M1-B','M1-C','M1-D')}, 'M3':'not_verified','M4':'not_verified'},
                cases={n:dict(status='passed') for n in d.REQUIRED})
    return hashes,{r:copy.deepcopy(report) for r in ('panda','ur5e')}


@pytest.mark.parametrize('damage',['partial','viewer','old_model','old_beans_model','multi_bean','drift','robot','candidate'])
def test_invalid_runs_cannot_publish_freeze(monkeypatch,tmp_path,damage):
    hashes,reports=passing_reports()
    monkeypatch.setattr(d,'input_hashes',lambda:hashes)
    monkeypatch.setattr(d,'load_json',lambda _:dict(beans=dict(contact_parameter_status='frozen', count=1)))
    if damage=='partial':
        reports['panda']['cases']['hold']['status']='not_verified'
    elif damage=='viewer':
        reports['ur5e']['cases']['viewer']['status']='failed'
    elif damage=='old_model':
        reports['panda']['model_version']='new_tableware_v2'
    elif damage=='old_beans_model':
        reports['panda']['model_version']='beans_native_v1'
    elif damage=='multi_bean':
        monkeypatch.setattr(d,'load_json',lambda _:dict(beans=dict(contact_parameter_status='frozen', count=15)))
    elif damage=='drift':
        reports['ur5e']['final_input_hashes']={}
    elif damage=='robot':
        reports['ur5e']['status']='failed'
    else:
        monkeypatch.setattr(d,'load_json',lambda _:dict(beans=dict(contact_parameter_status='candidate')))
    path=tmp_path/'freeze_manifest.json'
    with pytest.raises(AssertionError):
        d.publish_freeze(reports,{},path,hashes)
    assert not path.exists()


def test_partial_entry_keeps_unselected_not_verified(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'input_hashes',lambda:{})
    monkeypatch.setattr(d,'source_manifest',lambda:dict(status='passed'))
    report=d.validate('panda',tmp_path,['manifest'])
    assert report['status']=='incomplete'
    assert all(report['cases'][n]['status']=='not_verified' for n in d.REQUIRED if n!='manifest')
    assert report['stages']['M3']==report['stages']['M4']=='not_verified'


@pytest.mark.parametrize('dt',[.001,.0005])
def test_move_pose_refreshes_commands_every_twenty_ms(dt):
    from types import SimpleNamespace
    from feedingrobot.scripts.validate_m1c import move_pose
    times=[]
    task=SimpleNamespace(dt=dt,tick=0,robot_config={'linear_speed_limit':.03},
                         index=SimpleNamespace(base=0),
                         data=SimpleNamespace(time=0.,site_xmat=np.eye(3).reshape(1,9)),
                         adapter=SimpleNamespace(set_twist=lambda value,now,end:times.append(now)))
    task.snapshot=lambda:dict(tcp_position=np.zeros(3),tcp_rotation=np.eye(3))
    def step():
        task.data.time+=dt
        task.tick+=1
    task.step_physics=step
    cfg=load_json('configs/acceptance.json'); cfg['reachability_timeout_s']=.06
    with pytest.raises(AssertionError,match='Pose timeout'):
        move_pose(task,SimpleNamespace(phase='control'),[.1,0,0],np.eye(3),cfg)
    np.testing.assert_allclose(times,[0.,.02,.04],atol=1e-12)


@pytest.mark.parametrize('case,expected',[
    ('convergence',set(d.MOTION_CASES)),('viewer',{'sweep_seed_0'}),('performance',set(d.CONTROL_CASES))])
def test_local_case_runs_its_dependencies_without_claiming_unselected_cases(monkeypatch,tmp_path,case,expected):
    observed=[]
    def baseline(robot,folder,selected):
        observed.append(set(selected))
        return dict(stages={'M1-C':'passed'},cases={n:dict(status='passed' if n in selected else 'not_verified',
            performance={},load_wall_s=0.,model_load_wall_s=0.) for n in d.CONTROL_CASES})
    monkeypatch.setattr(d,'input_hashes',lambda:{})
    monkeypatch.setattr(d.c,'validate',baseline)
    monkeypatch.setattr(d,'convergence',lambda *args:dict(status='passed'))
    monkeypatch.setattr(d,'viewer',lambda *args:dict(status='passed'))
    result=d.validate('panda',tmp_path,[case])
    assert observed==[expected]
    assert result['status']=='incomplete'
    assert result['cases'][case]['status']=='passed'
    assert all(result['cases'][n]['status']=='not_verified' for n in d.REQUIRED if n!=case)


def test_final_robot_failure_restores_candidate_without_publication(monkeypatch,tmp_path):
    calls=[];states=[]
    monkeypatch.setattr(d.sys,'argv',['validate_m1','--robot','all','--output',str(tmp_path)])
    monkeypatch.setattr(d,'input_hashes',lambda:{})
    monkeypatch.setattr(d,'load_json',lambda _:dict(beans=dict(contact_parameter_status='frozen', count=1)))
    def validate(robot,*args):
        calls.append(robot)
        return dict(status='passed' if robot=='panda' else 'failed')
    monkeypatch.setattr(d,'validate',validate)
    monkeypatch.setattr(d,'parameter_status',states.append)
    with pytest.raises(SystemExit) as error:
        d.main()
    assert error.value.code==1 and calls==['panda','ur5e'] and states==['candidate']
    assert not (tmp_path/'freeze_manifest.json').exists()


@pytest.mark.parametrize('drift',[False,True])
def test_freeze_receipt_hashes_evidence_and_rechecks_inputs(monkeypatch,tmp_path,drift):
    import hashlib
    hashes,reports=passing_reports();folders={}
    for robot,report in reports.items():
        report['cases']['assembly']['solver']={'timestep':.001}
        report['cases']['contacts']['contact_parameters']={'solref':[.002,1.]}
        folder=tmp_path/robot;folder.mkdir();folders[robot]=folder
        d.write_json(folder/'m1d_report.json',report)
    reads=[]
    def read_hashes():
        reads.append(1)
        return {} if drift and len(reads)>1 else hashes
    monkeypatch.setattr(d,'input_hashes',read_hashes)
    monkeypatch.setattr(d,'load_json',lambda _:dict(beans=dict(contact_parameter_status='frozen', count=1)))
    monkeypatch.setattr(d.importlib.metadata,'version',lambda _:'test')
    path=tmp_path/'freeze_manifest.json'
    if drift:
        with pytest.raises(AssertionError,match='receipt'):
            d.publish_freeze(reports,folders,path,hashes)
        assert not path.exists()
    else:
        manifest=d.publish_freeze(reports,folders,path,hashes)
        assert path.exists() and len(reads)==2
        for folder in folders.values():
            file=folder/'m1d_report.json'
            assert manifest['evidence_sha256'][str(file)]==hashlib.sha256(file.read_bytes()).hexdigest()
        assert manifest['stages']['M3']==manifest['stages']['M4']=='not_verified'


@pytest.mark.parametrize('dt',[.001,.0005])
def test_phase_transition_keeps_episode_command_grid(dt):
    from types import SimpleNamespace
    from feedingrobot.scripts.validate_m1c import move_pose
    times=[]
    task=SimpleNamespace(dt=dt,tick=round(.007/dt),robot_config={'linear_speed_limit':.03},
        index=SimpleNamespace(base=0),data=SimpleNamespace(time=.007,site_xmat=np.eye(3).reshape(1,9)),
        adapter=SimpleNamespace(set_twist=lambda value,now,end:times.append(now)))
    task.snapshot=lambda:dict(tcp_position=np.zeros(3),tcp_rotation=np.eye(3))
    def step():
        task.data.time+=dt
        task.tick+=1
    task.step_physics=step
    cfg=load_json('configs/acceptance.json');cfg['reachability_timeout_s']=.06
    with pytest.raises(AssertionError,match='Pose timeout'):
        move_pose(task,SimpleNamespace(phase='control'),[.1,0,0],np.eye(3),cfg)
    np.testing.assert_allclose(times,[.02,.04,.06],atol=1e-12)
