"""Handover physics, ownership, causal windows, quotas and zero-update audit."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import mink
import numpy as np
import pytest
import torch

from correction_v2 import controller as c,corpus,rollout,run
from feedingrobot.data.episodes import load_episode,write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.data import ActionWindows,features,field_slices
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.policies.evaluation import ActionChunk
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT


def fake_teacher():
    return SimpleNamespace(path=[('above',np.array([0.,0.,.14]),np.eye(3),.03),
        ('pre_entry',np.array([0.,0.,.025]),np.eye(3),.03),
        ('entry',np.zeros(3),np.eye(3),.005)],geometry={'bowl_position':np.zeros(3)},
        base_rotation=np.eye(3),parameters=dict(position_gain=2.,orientation_gain=2.,
            transport_speed_m_s=.03,linear_speed_m_s=.005),
        robot_config=dict(linear_speed_limit=.05,angular_speed_limit=.5),
        act=lambda obs:np.array([0.,0.,-.0035,0.,0.,0.]))


def obs(z=.062,y=-.0015,degrees=3.,vz=0.,omega=0.):
    return dict(tcp_position=np.array([0.,y,z]),tcp_rotation=mink.SO3.exp(np.array([0.,-np.deg2rad(degrees),0.])).as_matrix(),
        tcp_twist_world=np.array([0.,0.,vz,0.,omega,0.]),stage='ACQUIRE')


def test_cancel_chunk_preserves_velocity_and_rejects_future_progress():
    chunk=ActionChunk(0,'ACQUIRE',np.ones((16,6)));velocity=np.arange(6,dtype=float)
    saved=velocity.copy();chunk=c.cancel_chunk(chunk)
    assert chunk is None;np.testing.assert_array_equal(velocity,saved)
    p=c.Progress();teacher=fake_teacher()
    assert not p.candidate('p2_moving',teacher,obs(z=.007,vz=-.004))
    p.update(teacher,obs(z=.14,y=0.));assert p.above and not p.pre_entry
    assert not p.candidate('p2_moving',teacher,obs(z=.007,vz=-.004))
    p.update(teacher,obs(z=.025,y=0.));assert p.pre_entry
    assert p.candidate('p2_moving',teacher,obs(z=.007,vz=-.004))


def test_alignment_holds_height_then_restores_actual_waypoint():
    teacher=fake_teacher();progress=c.Progress();progress.above=True
    expert=c.AlignmentTeacher(teacher,obs(),100,'p1_stopped',progress)
    command,owner=expert.act(obs(),100)
    assert owner=='alignment_teacher' and command[1]>0 and command[4]>0 and command[2]==0
    assert teacher.part==1
    for tick in (150,200,250,300):
        _,owner=expert.act(obs(y=0.,degrees=0.),tick)
        assert owner=='alignment_teacher'
    command,owner=expert.act(obs(y=0.,degrees=0.),350)
    assert owner=='path_teacher' and expert.completed_tick==350 and command[2]<0
    assert expert.qualified()


def test_alignment_stability_and_height_gates():
    progress=c.Progress();progress.above=True;progress.pre_entry=True
    expert=c.AlignmentTeacher(fake_teacher(),obs(z=.007,vz=-.004),0,'p2_moving',progress)
    assert expert.path_part==2
    expert.act(obs(z=.007,y=0.,degrees=0.),50)
    expert.act(obs(z=.007,y=0.,degrees=0.,vz=-.003),200)
    assert expert.stable is None and expert.completed_tick is None
    with pytest.raises(ValueError,match='height_drift'): expert.act(obs(z=.006,y=0.,degrees=0.),250)
    progress.pre_entry=False
    with pytest.raises(ValueError,match='progress'):
        c.AlignmentTeacher(fake_teacher(),obs(),0,'p2_stopped',progress)


def test_stopped_requires_both_measured_and_adapter_velocities():
    assert c.stopped(obs(),np.zeros(6))
    assert not c.stopped(obs(vz=-.003),np.zeros(6))
    assert not c.stopped(obs(),np.array([0,0,.003,0,0,0]))


def label_example():
    m=dict(success=True,pickup=True,delivery=True,accepted_normal=True,qualified_correction=True,
        split='train',teacher_version=c.VERSION,source_unchanged=True,truncated=False,abort_reason=None,
        max_episode_s=60.,simulated_s=58.,scenario={},category='p1_stopped',prefix_source='dp',corrected_tick=250,
        geometry_checks=[dict(passed=True)],handover=dict(release_tick=150,release_residual=c.residual(fake_teacher(),obs())))
    a=dict(action_mask=np.array([False,False,False,True,True]),
        action_owner=np.array(['dp','pulse','external_brake','alignment_teacher','path_teacher']),
        action_ticks=np.arange(5)*50,action_end_ticks=np.arange(5)*50+50)
    for name in ('physics','observations','action_observations','actions','proposals'): a[name]=np.zeros((5,6))
    return m,a


@pytest.mark.parametrize('index',(0,1,2))
def test_non_teacher_labels_never_enter(index):
    m,a=label_example();assert corpus.check_labels(m,a)['valid_labels']==2
    a['action_mask'][index]=True
    with pytest.raises(ValueError,match='Non-teacher'): corpus.check_labels(m,a)


@pytest.mark.parametrize('field,value',(('split','validation'),('truncated',True),('success',False),('simulated_s',60.1)))
def test_validation_failed_or_timeout_cannot_enter(field,value):
    m,a=label_example();m[field]=value
    with pytest.raises(ValueError,match='complete'): corpus.check_labels(m,a)


def test_release_gate_and_nonfinite_data_rejected():
    m,a=label_example();m['handover']['release_residual']['angle_deg']=1.
    with pytest.raises(ValueError,match='category gates'): corpus.check_labels(m,a)
    m,a=label_example();a['observations'][0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'): corpus.check_labels(m,a)


def test_file_tree_rejects_extra_or_changed_file(tmp_path):
    p=tmp_path/'a';p.write_text('a');expected=corpus.file_hashes(tmp_path)
    corpus.verify_files(tmp_path,expected);p.write_text('b')
    with pytest.raises(ValueError,match='SHA256'): corpus.verify_files(tmp_path,expected)
    p.write_text('a');(tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError,match='coverage'): corpus.verify_files(tmp_path,expected)


def test_split_seed_isolation_includes_v1_and_history(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    v1=dict(accepted=[dict(seed=740001)],rejected=[dict(seed=740005)])
    for seeds in ([1],[740001],[740005]):
        with pytest.raises(ValueError,match='overlaps'): run.seed_check(seeds,[dict(seed=1)],v1)
    directory=tmp_path/'outputs/single_bean/v1/m5/dit/descent_calibration_old';directory.mkdir(parents=True)
    write_json(directory/'report.json',dict(pairs=[dict(seed=760001)]))
    with pytest.raises(ValueError,match='overlaps'): run.seed_check([760001],[],v1)


def test_three_pools_keep_post_alignment_entry_and_sweep(tmp_path):
    paths=[]
    for name in ('correction','aligned'):
        path=tmp_path/name;path.mkdir();paths.append(path)
    np.save(paths[0]/'action_owner.npy',np.array(['dp','alignment_teacher','path_teacher','path_teacher']))
    np.save(paths[0]/'action_stages.npy',np.array(['dp','alignment','entry','sweep']))
    np.save(paths[1]/'action_owner.npy',np.array(['path_teacher','path_teacher']))
    np.save(paths[1]/'action_stages.npy',np.array(['entry','sweep']))
    windows=SimpleNamespace(episodes=[(paths[0],dict(category='p1_stopped',handover=dict(release_tick=50),corrected_tick=100)),
        (paths[1],dict(category='aligned'))],windows=[(0,1,2,1),(0,2,3,1),(0,3,4,1),(1,0,1,1),(1,1,2,1)],
        arrays=lambda e:dict(action_ticks=np.arange(4)*50))
    pools,_=corpus.pool_indices(windows)
    assert pools==dict(alignment={0:[0]},transition={0:[1,2]},aligned={1:[3,4]})
    np.save(paths[0]/'action_stages.npy',np.array(['dp','alignment','entry','entry']))
    with pytest.raises(ValueError,match='entry and sweep'): corpus.pool_indices(windows)


def test_exact_mixing_fraction_and_equal_pool_selection():
    class Data:
        def __init__(self,base): self.base=base
        def sample_indices(self,rng,count): return list(range(count))
        def __getitem__(self,i): return dict(pool=torch.tensor(-1 if self.base else i))
    mixed=object.__new__(corpus.MixedV2);mixed.base=Data(True);mixed.extra=Data(False)
    mixed.pools=dict(alignment={0:[0]},transition={1:[1]},aligned={2:[2]})
    rng=np.random.default_rng(13);batches=[mixed.batch(rng,64)['pool'].numpy() for _ in range(100)]
    assert all(np.count_nonzero(batch==-1)==48 for batch in batches)
    all_pools=np.concatenate(batches);counts=[np.count_nonzero(all_pools==i) for i in range(3)]
    assert min(counts)>450 and max(counts)<650


@pytest.fixture
def physical_config():
    setup(read_json(ROOT/'configs/dp_dit.json'),'cpu')
    manifest=read_json(sorted((ROOT/'datasets/single_bean/v1/m4/panda/train').glob('normal_*/manifest.json'))[0])
    return manifest['teacher_config'],dict(manifest['teacher_config']['scene'],recover=False)


def test_baseline_records_exact_replay_and_original_prefix(tmp_path,physical_config):
    from feedingrobot.data.rollout import run_episode
    config,scenario=physical_config;directory=tmp_path/'baseline'
    m=rollout.episode(790001,'p1_stopped',config,scenario,directory,source='baseline',max_s=.3)
    assert replay_episode(directory)['status']=='passed'
    original=tmp_path/'original';run_episode('panda',790001,config,original,scenario=scenario,max_episode_s=.3,split='calibration')
    _,a=load_episode(directory);_,b=load_episode(original)
    np.testing.assert_allclose(a['actions'],b['actions'],rtol=0,atol=1e-10)
    np.testing.assert_array_equal(a['action_observations'],b['action_observations'])
    assert not a['action_mask'].any() and m['truncated']


def test_whole_spoon_clearance_and_screen_do_not_change_physics(physical_config):
    from feedingrobot.envs import FeedingGymEnv
    from feedingrobot.experts import Teacher
    from feedingrobot.experts.geometry import teacher_geometry
    config,scenario=physical_config;env=FeedingGymEnv('panda')
    try:
        env.reset(seed=790002,options={'scenario':scenario});task=env.task
        teacher=Teacher(task.robot_config,config);teacher.reset({},geometry=teacher_geometry(task))
        before=task.get_state()['physics'].copy();positions=task.data.geom_xpos.copy()
        result=c.swept_clearance(task,teacher,task.provider.observe()['policy_obs'])
        assert set(result)=={'minimum_m','reserve_m','net_m','required_m','passed'}
        np.testing.assert_array_equal(task.get_state()['physics'],before)
        np.testing.assert_array_equal(task.data.geom_xpos,positions)
    finally: env.close()


def test_causal_features_ignore_future_and_keep_non_teacher_history(physical_config):
    from feedingrobot.envs import FeedingGymEnv
    _,scenario=physical_config;env=FeedingGymEnv('panda')
    try:
        env.reset(seed=790003,options={'scenario':scenario});value=env.observe_policy()
        ticks=np.array([0,20,50,100]);values=np.tile(value,(4,1));fields=field_slices(env.schema)
        first=features(ticks,values,np.ones(4,bool),50,fields)
        values[-1]=1e6;second=features(ticks,values,np.ones(4,bool),50,fields)
        for key in first: np.testing.assert_array_equal(first[key],second[key])
        assert first['state_mask'].all() and first['history_mask'][-1]
    finally: env.close()


def test_zero_update_audit_loads_only_100k_ema(tmp_path,monkeypatch):
    config=deepcopy(read_json(ROOT/'configs/dp_dit.json'))
    config['model'].update(hidden_size=16,heads=2,depth=1,mlp_hidden=32,horizon=2)
    model=ActionDiT(config);ema={k:torch.ones_like(v)*.01 for k,v in model.state_dict().items()}
    trained=dict(ema=ema,model={k:torch.zeros_like(v) for k,v in ema.items()})
    binding=dict(parent_checkpoint_sha256='parent',normalization={})
    monkeypatch.setattr(run,'ancestry',lambda args:(dict(config=config,normalization={}),trained,{}, {},[],{},binding))
    calibration=tmp_path/'calibration.json';calibration.write_text('{}')
    report=dict(calibration_report=str(calibration),calibration_report_sha256=sha256(calibration),dataset='ignored')
    monkeypatch.setattr(run,'audit_corpus',lambda *args:report)
    monkeypatch.setattr(run,'calibration_check',lambda *args:{})
    item=dict(states=torch.zeros(2,122),history=torch.zeros(10,28),phase=torch.tensor(1),interaction=torch.zeros(4),
        state_mask=torch.ones(2,dtype=torch.bool),history_mask=torch.ones(10,dtype=torch.bool),
        actions=torch.zeros(2,6),action_mask=torch.ones(2,dtype=torch.bool))
    class Mixed:
        pools=dict(alignment={0:[0]},transition={0:[0]},aligned={1:[0]})
        extra=[item]
        def __init__(self,*args): pass
        def batch(self,rng,count):
            from torch.utils.data import default_collate
            assert count==64
            return default_collate([item]*count)
        def summary(self): return dict(correction_fraction=.25)
    monkeypatch.setattr(run,'MixedV2',Mixed)
    def forbidden(*args,**kwargs): raise AssertionError('Optimizer/backward must not run')
    monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    monkeypatch.setattr(torch.Tensor,'backward',forbidden)
    monkeypatch.setattr(run,'start_output',lambda *args:(tmp_path,dict(optimizer_created=False,optimizer_updates=0)))
    monkeypatch.setattr(run,'finish',lambda output,provenance,binding,**kwargs:dict(provenance,**kwargs))
    (tmp_path/'data.json').write_text('{}')
    args=SimpleNamespace(data_report=str(tmp_path/'data.json'))
    result=run.audit(args)
    assert result['step']==100000 and result['model_exact'] and result['ema_exact']
    assert result['optimizer_updates']==0 and not result['optimizer_created'] and result['batch_size']==64


def test_unpassed_calibration_stops_before_cuda_or_output(tmp_path,monkeypatch):
    args=SimpleNamespace(calibration_report='failed',output=str(tmp_path/'output'))
    monkeypatch.setattr(run,'ancestry',lambda args:({}, {},{}, {},[],{},{'parent_checkpoint_sha256':'p'}))
    monkeypatch.setattr(run,'verify_evidence',lambda *args:dict(mode='descent_calibration_v2',status='failed'))
    with pytest.raises(ValueError,match='Unpassed'): run.collect(args)
    assert not Path(args.output).exists()


def test_legal_windows_stop_at_invalid_prefix_and_phase_boundary(tmp_path):
    from feedingrobot.envs.feeding import observation_schema
    from feedingrobot.data.episodes import annotate
    schema=observation_schema('panda',7);width=sum(s for _,s,_ in schema['fields'])
    directory=tmp_path/'data';path=directory/'train'/'episode';path.mkdir(parents=True)
    write_json(directory/'normalization.json',dict(observation_schema=schema))
    events=[dict(name='phase',phase='ACQUIRE',time=0.),dict(name='phase',phase='TRANSPORT',time=.2),dict(name='success',time=.3)]
    write_json(path/'manifest.json',dict(robot_id='panda',status='complete',dt=.001,observation_schema=schema,
        segments=annotate(events),events=events,split='train',group_id='new',seed=770001,scenario={},
        accepted_normal=True,success=True,recovery_action_rows=0,accepted_recovery=False,observation_rows=6))
    values=dict(action_mask=np.array([False,True,True,False,True,True]),action_phases=np.array([1,1,1,1,2,2]),
        action_ticks=np.arange(6)*50,action_end_ticks=np.arange(6)*50+50,
        actions=np.zeros((6,6)),action_observations=np.zeros((6,width)),observations=np.zeros((6,width)),
        observation_ticks=np.arange(6)*50,observation_valid=np.ones(6,dtype=bool))
    for name,value in values.items(): np.save(path/f'{name}.npy',value)
    windows=ActionWindows(directory,'train',16)
    assert windows.windows==[(0,1,3,1),(0,2,3,1),(0,4,6,2),(0,5,6,2)]
    assert windows[0]['action_mask'].tolist()==[True,True]+[False]*14


@pytest.fixture
def complete_corpus(tmp_path,monkeypatch):
    root=tmp_path/'collection';data=root/'data';original=tmp_path/'original'
    original.mkdir();(data/'train').mkdir(parents=True)
    for p in (original/'normalization.json',data/'normalization.json'): p.write_text('{}')
    accepted=[];category_list=[category for category in c.CATEGORIES for _ in range(c.QUOTAS[category])]
    # Category and seed records here isolate report-level checks; ownership is
    # exercised independently above with real masks and physical recordings.
    for i,category in enumerate(category_list):
        path=data/'train'/f'episode_{i}';path.mkdir()
        write_json(path/'manifest.json',dict(seed=770001+i,group_id=f'v2_{i}',category=category))
        accepted.append(dict(seed=770001+i,category=category,path=f'train/episode_{i}',
            replay=dict(status='passed',max_physics_reference_error=0.,max_observation_error=0.)))
    binding=dict(parent='100k')
    report=dict(mode='correction_dataset_v2',status='ready_experimental',binding=binding,source_unchanged=True,
        normalization={},dataset=str(data),accepted=accepted,counts=c.QUOTAS,
        dataset_sha256=corpus.file_hashes(data),evidence_sha256=corpus.file_hashes(root))
    report_path=root/'report.json';write_json(report_path,report)
    monkeypatch.setattr(corpus,'load_episode',lambda p:(read_json(p/'manifest.json'),{}))
    monkeypatch.setattr(corpus,'check_labels',lambda *args:{})
    return report_path,report,binding,original


def test_corpus_requires_exact_quotas_ancestry_and_replay(complete_corpus):
    path,report,binding,original=complete_corpus
    assert corpus.audit_corpus(path,binding,{},original)['counts']==c.QUOTAS
    with pytest.raises(ValueError,match='ancestry'): corpus.audit_corpus(path,{}, {},original)
    report['accepted'][0]['replay']['max_observation_error']=1e-5;write_json(path,report)
    with pytest.raises(ValueError,match='tolerance'): corpus.audit_corpus(path,binding,{},original)
    report['accepted'][0]['replay']['max_observation_error']=0.;report['counts']=dict(c.QUOTAS,p1_stopped=1);write_json(path,report)
    with pytest.raises(ValueError,match='quotas'): corpus.audit_corpus(path,binding,{},original)


def test_corpus_rejects_seed_from_any_original_split(complete_corpus):
    path,report,binding,original=complete_corpus
    directory=original/'validation'/'episode';directory.mkdir(parents=True)
    write_json(directory/'manifest.json',dict(seed=770001,group_id='original_validation'))
    with pytest.raises(ValueError,match='leakage'): corpus.audit_corpus(path,binding,{},original)


def test_collection_evidence_cannot_be_added_or_changed(complete_corpus):
    path,report,binding,original=complete_corpus
    (path.parent/'unlisted.json').write_text('{}')
    with pytest.raises(ValueError,match='evidence'): corpus.audit_corpus(path,binding,{},original)


@pytest.mark.parametrize('bf16',(True,False))
def test_host_precision_failure_does_not_create_collection(tmp_path,monkeypatch,bf16):
    args=SimpleNamespace(calibration_report='passed',output=str(tmp_path/'output'))
    monkeypatch.setattr(run,'ancestry',lambda args:(dict(config={}), {},{}, {},[],{},{'parent_checkpoint_sha256':'p'}))
    monkeypatch.setattr(run,'calibration_check',lambda *args:{})
    monkeypatch.setattr(run,'seed_check',lambda *args:None)
    if bf16:
        def unavailable(*args): raise RuntimeError('CUDA unavailable; no CPU fallback')
        monkeypatch.setattr(run,'setup',unavailable)
        message='CUDA unavailable'
    else:
        monkeypatch.setattr(run,'setup',lambda *args:(torch.device('cuda'),{}))
        monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
        message='BF16'
    with pytest.raises(RuntimeError,match=message): run.collect(args)
    assert not Path(args.output).exists()


def test_real_loop_cancels_dp_queue_without_resetting_velocity(tmp_path,monkeypatch,physical_config):
    from feedingrobot.envs import FeedingGymEnv
    config,scenario=physical_config;captured={};samples=[];cancellations=[]
    def make_env(*args,**kwargs):
        captured['env']=FeedingGymEnv(*args,**kwargs)
        return captured['env']
    def predict(*args):
        samples.append(captured['env'].task.tick)
        actions=torch.zeros(1,16,6);actions[:,:,0]=.02
        return actions
    def cancel(chunk):
        task=captured['env'].task
        if task.tick==100: cancellations.append((task.adapter.velocity.copy(),task.data.qpos.copy()))
        return None
    class Expert:
        completed_tick=None
        release_residual={}
        path_part=1
        def __init__(self,*args):
            task=captured['env'].task
            assert np.linalg.norm(task.adapter.velocity[:3])>.001
            for velocity,q in cancellations:
                np.testing.assert_array_equal(task.adapter.velocity,velocity)
                np.testing.assert_array_equal(task.data.qpos,q)
        def act(self,obs,tick): return np.zeros(6),'alignment_teacher'
        def qualified(self): return False
    monkeypatch.setattr(rollout,'FeedingGymEnv',make_env)
    monkeypatch.setattr(rollout,'sample_actions',predict)
    monkeypatch.setattr(rollout,'cancel_chunk',cancel)
    monkeypatch.setattr(rollout,'AlignmentTeacher',Expert)
    monkeypatch.setattr(c.Progress,'candidate',lambda self,category,teacher,obs,calibration=False:obs['time']>=.1)
    monkeypatch.setattr(rollout,'swept_clearance',lambda *args:dict(passed=True))
    monkeypatch.setattr(rollout,'clearance',lambda *args:.1)
    # Normalization is not required by the stub predictor; features remain causal.
    row=rollout.episode(790004,'p1_moving',config,scenario,tmp_path/'dp',source='dp',model=object(),
        dp_config=read_json(ROOT/'configs/dp_dit.json'),normalization=None,device=torch.device('cpu'),max_s=.3)
    assert samples==[50] and len(cancellations)==2 and row['handover']['release_tick']==100
    _,a=load_episode(tmp_path/'dp')
    assert a['action_owner'][0]=='dp' and not a['action_mask'][0]
    assert np.all(a['action_owner'][1:]=='alignment_teacher')
    assert replay_episode(tmp_path/'dp')['status']=='passed'
