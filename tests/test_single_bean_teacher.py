"""M4 migration preserves the frozen pickup geometry and native Bean physics."""

import numpy as np
import pytest

from feedingrobot.data.recipes import recipe
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts.bean_path import pickup_path
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.scripts.validate_m1c import sweep_path
from feedingrobot.sim.model import load_json


@pytest.mark.parametrize('robot', ['panda', 'ur5e'])
def test_pickup_geometry_matches_frozen_m3_without_mutating_physics(robot):
    env = FeedingGymEnv(robot)
    try:
        env.reset(seed=0)
        before = env.task.get_state()['physics'].copy()
        geometry = teacher_geometry(env.task)
        expected = sweep_path(env.task, load_json('configs/acceptance.json'))
        actual = pickup_path(geometry)
        assert len(actual) == len(expected) == 12
        for a, b in zip(actual, expected):
            assert a[0] == b[0] and a[3] == b[3]
            np.testing.assert_allclose(a[1], b[1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(a[2], b[2], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(before, env.task.get_state()['physics'])
        assert not geometry['scoop_points'].flags.writeable
        assert not {'seed', 'scenario_state', 'food_mass_kg', 'food_friction'}.intersection(geometry)
    finally:
        env.close()


@pytest.mark.parametrize('recover', [False, True])
def test_single_bean_recipe_preserves_frozen_materials_and_layout(recover):
    config = load_json('configs/collect.json')
    _, scenario, _, _ = recipe(config, 'acceptance', 0, recover=recover)
    assert not {'food_mass_kg', 'food_friction', 'food_offset_m'}.intersection(scenario)
    assert scenario['recover'] == recover
    env = FeedingGymEnv()
    try:
        env.reset(seed=0, options={'scenario': scenario})
        assert env.task.failure_reason is None
    finally:
        env.close()


def test_receiver_observation_and_snapshot_schema_are_explicit():
    import copy
    env = FeedingGymEnv()
    try:
        vector, _ = env.reset(seed=0)
        obs = env.task.provider.observe()['policy_obs']
        receiver = env.task.model.site('mouth_receiver').id
        assert env.schema['version'] == 3 and vector.shape == (108,)
        np.testing.assert_array_equal(obs['receiver_relative_world'],
                                      env.task.data.site_xpos[receiver]-obs['tcp_position'])
        np.testing.assert_array_equal(obs['receiver_rotation'], env.task.data.site_xmat[receiver].reshape(3,3))
        state = env.get_state()
        assert state['schema_version'] == 3 and state['task']['signature']['schema_version'] == 4
        old = copy.deepcopy(state)
        old['schema_version'] = 2
        with pytest.raises(ValueError):
            env.set_state(old)
        old = copy.deepcopy(state)
        old['observation_schema']['version'] = 2
        with pytest.raises(ValueError):
            env.set_state(old)
    finally:
        env.close()


def test_release_progresses_across_retract_and_clears_before_retreat():
    import mink
    from feedingrobot.experts import Teacher
    env = FeedingGymEnv()
    try:
        env.reset(seed=0)
        teacher = Teacher(env.task.robot_config, load_json('configs/collect.json'))
        teacher.reset({}, geometry=teacher_geometry(env.task))
        teacher.part = len(teacher.path)
        teacher.pickup_lift_complete = teacher.wait_level_complete = True
        obs = env.task.provider.observe()['policy_obs']
        mouth = obs['tcp_position']+obs['mouth_relative_world']
        receiver = obs['tcp_position']+obs['receiver_relative_world']
        bean = obs['tcp_position']+obs['bean_relative_world'][0]
        def move(position, rotation):
            obs.update(tcp_position=position.copy(), tcp_rotation=rotation.copy(),
                       mouth_relative_world=mouth-position, receiver_relative_world=receiver-position,
                       bean_relative_world=np.array([bean-position]))
        move(mouth+obs['mouth_rotation'] @ [.004,0.,-.006], obs['mouth_rotation'])
        obs['stage'] = 'TRANSFER'
        teacher.act(obs)
        assert teacher.stage == 'release_lower'
        move(teacher.target_position, teacher.target_rotation)
        obs['time'] = .05
        teacher.act(obs)
        assert teacher.release_part == 1
        obs['time'] = .1
        teacher.act(obs)
        assert teacher.stage == 'release_roll'
        angle = np.linalg.norm(mink.SO3.from_matrix(teacher.target_rotation @ obs['mouth_rotation'].T).log())
        assert 0 < angle < .1
        obs.update(time=teacher.roll_start+1./teacher.parameters['release_rate_s_inv']+1e-9,stage='RETRACT')
        teacher.act(obs)
        assert teacher.stage == 'release_roll' and teacher.release_part == 1
        move(teacher.target_position,teacher.target_rotation)
        teacher.act(obs)
        assert teacher.release_part == 1
        obs['time'] += teacher.parameters['release_settle_s']
        teacher.act(obs)
        assert teacher.release_part == 2
        roll_target = teacher.target_position.copy()
        teacher.act(obs)
        assert teacher.stage == 'release_clear'
        np.testing.assert_allclose(teacher.target_position-roll_target,
                                   obs['mouth_rotation'] @ [-.003,-.003,.001],atol=1e-12)
        move(teacher.target_position,teacher.target_rotation)
        teacher.act(obs)
        assert teacher.release_part == 3
        teacher.act(obs)
        assert teacher.stage == 'retract'
        relative = obs['mouth_rotation'].T @ (teacher.target_position-mouth)
        tool = teacher.geometry['tool_points'] @ (obs['mouth_rotation'].T @ teacher.target_rotation).T
        np.testing.assert_allclose(np.max(tool[:,0])+relative[0],
                                   -teacher.geometry['task_config']['clearance_margin_m'],atol=1e-12)
        np.testing.assert_allclose(relative[1],teacher.retreat_offset[1],atol=1e-12)
        normal = obs['receiver_rotation'][:,2]
        lowest = np.min((teacher.geometry['scoop_points'] @ teacher.target_rotation.T) @ normal)
        np.testing.assert_allclose(lowest+normal @ (teacher.target_position-receiver),
                                   teacher.geometry['task_config']['clearance_margin_m'],atol=1e-12)
        endpoint = teacher.target_position.copy()
        obs['time'] += .5
        teacher.act(obs)
        np.testing.assert_allclose(teacher.target_position,endpoint,atol=1e-12)
        receiver += np.array([0.,0.,.003])
        obs['receiver_relative_world'] = receiver-obs['tcp_position']
        teacher.act(obs)
        np.testing.assert_allclose(teacher.target_position-endpoint,[0.,0.,.003],atol=1e-12)
    finally:
        env.close()


def test_common_initial_state_convergence_restores_boundary_state(tmp_path):
    import pickle
    from feedingrobot.data.rollout import run_episode
    config = load_json('configs/collect.json')
    env = FeedingGymEnv()
    try:
        env.reset(seed=0,options={'scenario':config['scene']})
        state = env.task.get_state()
    finally:
        env.close()
    result = run_episode('panda',0,config,tmp_path/'fine',timestep=.0005,
                         initial_state=state,max_episode_s=.01)
    restored = pickle.loads((tmp_path/'fine/initial_state.pkl').read_bytes())
    np.testing.assert_array_equal(restored['physics'],state['physics'])
    for field in ('qacc','sensordata'):
        np.testing.assert_array_equal(restored['boundary'][field],state['boundary'][field])
    assert result['dt'] == .0005 and result['truncated']


def test_recovery_selection_requires_complete_action_inside_finished_segment():
    from feedingrobot.data.episodes import recovery_action_mask
    segment = dict(recovery_valid=True,start_s=.101,end_s=.102)
    assert not recovery_action_mask([segment],[7],[100],[150],[True],.001).any()
    segment['end_s'] = .25
    np.testing.assert_array_equal(recovery_action_mask([segment],[7,7,7],[100,150,200],
                                  [150,200,250],[True,False,True],.001),[False,False,True])


def test_recovery_departure_trigger_preserves_default_m3_behavior():
    env = FeedingGymEnv()
    try:
        env.reset(seed=0, options={'scenario': {'recover': True, 'recover_departure_m': .008}})
        task = env.task
        task.logic.phase = 'APPROACH'
        mouth = task.model.site('mouth_entry').id
        wait = task.data.site_xpos[mouth]-task.data.site_xmat[mouth].reshape(3,3)[:,0]*task.task_config['wait_offset_m']
        task.data.site_xpos[task.index.tcp] = wait
        task._write_drivers()
        assert task.scenario_state['closure_start'] is None
        task.data.site_xpos[task.index.tcp] = wait + [.009, 0., 0.]
        task._write_drivers()
        assert task.scenario_state['closure_start'] == 0.
        task.scenario_state['closure_start'] = None
        task.scenario_state['parameters'].pop('recover_departure_m')
        task.data.site_xpos[task.index.tcp] = wait
        task._write_drivers()
        assert task.scenario_state['closure_start'] == 0.
    finally:
        env.close()


def test_episode_loader_rejects_old_version_and_changed_field_contract(tmp_path):
    import json
    from feedingrobot.data.rollout import run_episode
    from feedingrobot.data.episodes import load_episode
    run_episode('panda',0,load_json('configs/collect.json'),tmp_path/'episode',max_episode_s=.01)
    path=tmp_path/'episode/manifest.json'
    original=path.read_text()
    manifest=json.loads(original)
    manifest['observation_schema']['version']=2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='observation schema'):
        load_episode(path.parent)
    manifest=json.loads(original)
    manifest['observation_schema']['fields'][-1][2]='invalid units'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='observation schema'):
        load_episode(path.parent)


def test_m4_contact_time_revision_preserves_other_frozen_tolerances():
    m3 = load_json('configs/acceptance_m3.json')
    m4 = load_json('configs/acceptance_m4.json')
    assert m3['contact_event_time_tolerance_s'] == 1.
    assert m4['contact_event_time_tolerance_s'] == 1.1
    excluded = {'scope', 'contact_event_time_tolerance_s'}
    assert {k: v for k, v in m3.items() if k not in excluded} == {
        k: v for k, v in m4.items() if k not in excluded}


def test_convergence_runs_both_variants_after_first_numeric_failure(tmp_path, monkeypatch):
    import pickle
    from feedingrobot.scripts import validate_m4 as m4, validate_m3 as m3
    config = load_json('configs/collect.json')
    seed = m4.recipe(config,'acceptance',0,recover=False)[0]
    path = tmp_path/'episodes'/f'normal_{seed}'
    path.mkdir(parents=True)
    (path/'initial_state.pkl').write_bytes(pickle.dumps({'complete_initial_state':True}))
    manifest = dict(dt=.001,solver_iterations=100,solver_tolerance=1e-8,
                    comparison={},events=[],success=True,failure_reason=None,phase='SUCCESS',
                    time_s=55.,contact_peak_n=0.,contact_impulse_ns=0.,
                    contact_group_peaks_n={},contact_group_impulses_ns={},wrist_peak_n=0.,
                    accepted_recovery=False)
    monkeypatch.setattr(m4,'load_episode',lambda _: (manifest,{}))
    monkeypatch.setattr(m4,'physical_success',lambda _: True)
    executed=[]
    def saved(*args,**kwargs):
        assert kwargs['initial_state']=={'complete_initial_state':True}
        executed.append(args[2].name)
        return manifest
    monkeypatch.setattr(m4,'saved_or_run',saved)
    numeric=iter((False,True))
    monkeypatch.setattr(m3,'compare_runs',lambda *args: {'checks':{'tcp_path':next(numeric)}})
    results=m4.convergence_trial('panda',config,tmp_path,dict(index=0,recover=False))
    assert executed==[f'{seed}_half_dt',f'{seed}_refined_solver']
    assert [r['comparison']['passed'] for r in results]==[False,True]


@pytest.mark.parametrize("picked_up,frame", [(False,"world"),(True,"mouth")])
def test_scoop_entry_and_oral_entry_record_their_actual_frames(tmp_path,monkeypatch,picked_up,frame):
    from feedingrobot.experts import Teacher
    from feedingrobot.data.rollout import run_episode
    def entry_action(teacher,observation):
        teacher.stage = 'entry'
        teacher.pickup_lift_complete = picked_up
        teacher.proposal = np.zeros(6)
        return teacher.proposal.copy()
    monkeypatch.setattr(Teacher,'act',entry_action)
    result=run_episode('panda',0,load_json('configs/collect.json'),tmp_path/frame,max_episode_s=.04)
    samples=result['comparison']['tcp_samples']
    assert samples and all(s['frame']==frame for s in samples)
    assert result['comparison']['tcp_comparison_frame']==frame
    if not picked_up:
        np.testing.assert_allclose(samples[0]['comparison_position'],samples[0]['position'],atol=1e-12)
    else:
        assert np.linalg.norm(np.array(samples[0]['comparison_position'])-samples[0]['position'])>.1


def test_release_roll_tracks_known_rate_from_current_and_past_observations():
    import mink
    from feedingrobot.experts import Teacher
    env = FeedingGymEnv()
    try:
        env.reset(seed=0)
        teacher = Teacher(env.task.robot_config, load_json('configs/collect.json'))
        teacher.reset({}, geometry=teacher_geometry(env.task))
        teacher.part = len(teacher.path)
        teacher.pickup_lift_complete = teacher.wait_level_complete = True
        teacher.release_part = 1
        teacher.roll_start = teacher.transfer_start = 0.
        obs = env.task.provider.observe()['policy_obs']
        mouth = obs['tcp_position']+obs['mouth_relative_world']
        receiver = obs['tcp_position']+obs['receiver_relative_world']
        bean = obs['tcp_position']+obs['bean_relative_world'][0]
        obs.update(stage='TRANSFER', time=1.)
        teacher.act(obs)
        position = teacher.target_position.copy()
        obs.update(time=1.05, tcp_position=position,
                   mouth_relative_world=mouth-position, receiver_relative_world=receiver-position,
                   bean_relative_world=np.array([bean-position]))
        axis = np.asarray(teacher.parameters['release_rotation_rad'])
        rate = teacher.parameters['release_rate_s_inv']
        obs['tcp_rotation'] = obs['mouth_rotation'] @ mink.SO3.exp(axis*rate*obs['time']).as_matrix()
        command = teacher.act(obs)
        np.testing.assert_allclose(command[3:],teacher.base_rotation.T @ (obs['mouth_rotation'] @ axis*rate),atol=1e-9)
        assert teacher.last_roll_time == obs['time']
    finally:
        env.close()
