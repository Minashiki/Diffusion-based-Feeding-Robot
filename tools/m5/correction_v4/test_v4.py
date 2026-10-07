"""Stable boundary, target ownership, endpoint and source-binding regressions."""

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import correction_v4
import numpy as np
import pytest
import torch

from correction_v4 import controller as c,corpus,run
from correction_v2.test_v2 import fake_teacher,obs
from correction_v3.run import v3_hashes
from feedingrobot.data.episodes import write_json
from feedingrobot.policies.audit import read_json
from feedingrobot.sim.model import ROOT


def expert():
    progress=c.Progress();progress.above=True
    return c.AlignmentTeacher(fake_teacher(),obs(),0,'p1_stopped',progress)


def test_every_physical_boundary_resets_stability_between_decisions():
    teacher=expert();teacher.observe(obs(),0)
    for tick in range(1,451):
        good=obs(y=0.,degrees=0.,vz=-.003 if tick==201 else 0.)
        teacher.observe(good,tick)
        if tick%50==0:
            _,owner=teacher.act(good,tick)
            assert owner==('path_teacher' if tick==450 else 'alignment_teacher')
    assert teacher.stable==202 and teacher.completed_tick==450
    assert sum(row[-1] for row in teacher.records)==1


def test_200ms_minimum_and_missing_boundary_are_enforced():
    teacher=expert();teacher.observe(obs(y=0.,degrees=0.),0)
    for tick in range(1,201):
        good=obs(y=0.,degrees=0.);teacher.observe(good,tick)
        if tick%50==0:
            _,owner=teacher.act(good,tick)
            assert owner==('path_teacher' if tick==200 else 'alignment_teacher')
    teacher=expert();teacher.observe(obs(),0)
    with pytest.raises(ValueError,match='physical stability'):teacher.observe(obs(),2)
    with pytest.raises(ValueError,match='current physical'):teacher.act(obs(),50)


@pytest.mark.parametrize('cell',c.CELLS)
def test_signed_direction_and_velocity(cell):
    teacher=fake_teacher();state=obs(z=.062,degrees=2.5)
    state['tcp_position'][:2]=np.asarray(cell['quadrant'])*.0015/np.sqrt(2)
    state['tcp_twist_world'][:2]=c.direction_velocity(cell)
    assert c.release_direction(cell,teacher,state)
    state['tcp_twist_world'][:2]*=-1
    assert not c.release_direction(cell,teacher,state)


def test_perturbation_feedback_preserves_signed_target_without_changing_state():
    cell=c.CELLS[3];teacher=fake_teacher();state=obs()
    target=np.asarray(cell['quadrant'])*.0015/np.sqrt(2)
    state['tcp_position'][:2]=target+np.array([-.0003,.0003])
    before=state['tcp_position'].copy();feedback=c.perturbation_feedback(cell,teacher,state)
    assert feedback[0]>0 and feedback[1]<0
    np.testing.assert_array_equal(before,state['tcp_position'])


def fake_windows(monkeypatch,tmp_path):
    stages=['above']+['alignment']*6+list(corpus.STAGES)+['pickup_hold','transport','entry']
    owners=['prefix_teacher']+['alignment_teacher']*6+['path_teacher']*(len(stages)-7)
    phase=np.ones(len(stages),int);phase[-3:-1]=2;phase[-1]=4
    ticks=np.arange(len(stages))*50
    path=tmp_path/'episode';path.mkdir()
    for name,value in dict(action_stages=stages,action_owner=owners,action_phases=phase,
            action_mask=np.ones(len(stages),bool)).items(): np.save(path/f'{name}.npy',np.asarray(value))
    metadata=dict(category='p1_stopped',corrected_tick=350,handover=dict(release_tick=50))
    def initialize(self,*args,**kwargs):
        self.episodes=[(path,metadata)];self.windows=[]
        for start in range(len(stages)):
            end=start+1
            while end<len(stages) and end-start<16 and phase[end]==phase[start]:end+=1
            self.windows.append((0,start,end,int(phase[start])))
        self.arrays=lambda e:dict(action_ticks=ticks)
    monkeypatch.setattr(corpus.ActionWindows,'__init__',initialize)
    return corpus.PickupWindows('unused','train'),stages


def test_expanded_sampler_clips_transport_and_excludes_prefix_and_approach(monkeypatch,tmp_path):
    windows,stages=fake_windows(monkeypatch,tmp_path)
    chosen=[i for rows in windows.pools.values() for phases in rows.values() for ids in phases.values() for i in ids]
    targets={row for i in chosen for row in range(windows.windows[i][1],windows.windows[i][2])}
    assert 0 not in targets and len(stages)-2 not in targets and len(stages)-1 not in targets
    assert {stages[row] for row in targets}==set(corpus.STAGES)|{'alignment'}
    assert windows.clipped==1
    hold=windows.pools['transition'][0]['pickup_hold']
    assert {windows.windows[i][3] for i in hold}=={1,2}
    assert set(windows.pools['alignment'][0])=={'correcting','stable_wait'}


def test_real_old_corpus_baseline_and_loss_coverage():
    report=read_json(ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_dataset_001/report.json')
    windows=corpus.PickupWindows(report['dataset'],'train',16,report['normalization']);summary=windows.summary()
    assert [summary['pools'][p]['windows'] for p in windows.pools]==[119,5419,2980]
    assert summary['starts_by_stage']['correcting']==87 and summary['starts_by_stage']['stable_wait']==32
    assert summary['clipped_windows']==84
    assert summary['unique_actions_by_stage']['pickup_hold']==244
    assert summary['loss_positions_by_stage']['pickup_hold']==2902
    assert all(summary['unique_actions_by_stage'][s]>0 for s in corpus.STAGES)
    wait=sum(v for k,v in summary['stage_probabilities']['alignment'].items() if k.endswith(':stable_wait'))
    assert wait==pytest.approx(1/24)


def test_exact_mixture_stage_balance_and_rng_repeatability():
    class Base:
        def sample_indices(self,rng,count):return np.zeros(count,dtype=int)
        def __getitem__(self,index):return dict(code=torch.tensor(-1))
    class Extra:
        pools={p:{0:{'short':[i*2],'long':[i*2+1]*100}} for i,p in enumerate(('alignment','transition','aligned'))}
        def __getitem__(self,index):return dict(code=torch.tensor(index))
    mixed=corpus.MixedV4.__new__(corpus.MixedV4);mixed.base=Base();mixed.extra=Extra()
    first=mixed.batch(np.random.default_rng(7),64)['code']
    torch.testing.assert_close(first,mixed.batch(np.random.default_rng(7),64)['code'])
    rng=np.random.default_rng(3);batches=[mixed.batch(rng,64)['code'].numpy() for _ in range(200)]
    assert all(np.count_nonzero(b==-1)==48 for b in batches)
    counts=np.bincount(np.concatenate(batches)[np.concatenate(batches)>=0],minlength=6)
    assert np.max(np.abs(counts-3200/6))<80


def test_historical_seed_overlap_and_unchanged_v3_tools(monkeypatch,tmp_path):
    expected=read_json(ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_dataset_001/report.json')['binding']['v3_tools']
    assert v3_hashes()==expected
    monkeypatch.setattr(run,'ROOT',tmp_path)
    path=tmp_path/'outputs/single_bean/v1/m5/dit/failed/attempts/episode';path.mkdir(parents=True)
    write_json(path/'manifest.json',dict(seed=800001))
    with pytest.raises(ValueError,match='overlaps'):run.seed_check([800001],[],dict(accepted=[],rejected=[]))


def test_changed_or_unpassed_report_cannot_enter_collection(tmp_path):
    binding={'test':1};path=tmp_path/'report.json'
    write_json(path,dict(binding=binding,status='failed',source_unchanged=True))
    with pytest.raises(ValueError,match='Unpassed'):run.verify_report(path,binding,'passed')
    write_json(path,dict(binding=binding,status='passed',source_unchanged=True,evidence_sha256={}))
    (tmp_path/'extra').write_text('changed')
    with pytest.raises(ValueError,match='Changed'):run.verify_report(path,binding,'passed')


def test_critical_teacher_coverage_requires_independent_episodes_per_category_and_sign():
    rows=[dict(cell=cell,coverage={'x_critical_action_ticks':[50] if cell['velocity']=='toward' else []})
        for cell in c.CELLS]
    summary=run.aggregate_coverage(rows)
    assert summary['passed'] and set(summary['critical_independent_episodes'].values())=={2}
    rows[0]['coverage']['x_critical_action_ticks']=[]
    assert not run.aggregate_coverage(rows)['passed']


def test_physics_workers_have_one_torch_and_interop_thread(monkeypatch):
    calls=[]
    monkeypatch.setattr(run.torch,'set_num_threads',lambda n:calls.append(('torch',n)))
    monkeypatch.setattr(run.torch,'get_num_interop_threads',lambda:8)
    monkeypatch.setattr(run.torch,'set_num_interop_threads',lambda n:calls.append(('interop',n)))
    run.worker_setup();assert calls==[('torch',1),('interop',1)]


def test_spawn_physics_worker_budget():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn'),initializer=run.worker_setup) as pool:
        rows=[future.result() for future in [pool.submit(run.worker_setup) for _ in range(5)]]
    assert all(r['torch_threads']==r['interop_threads']==1 and len(r['cpu_affinity'])<=6 for r in rows)


def test_pickup_capture_preserves_logic_and_records_exact_velocity(monkeypatch,tmp_path):
    from correction_v4 import diagnostics as d
    logic=SimpleNamespace(timers={'pickup':.4})
    calls=[]
    def update(e,dt,time,**kwargs):
        calls.append(e);logic.timers['pickup']=0.
    logic.update=update
    task=SimpleNamespace(logic=logic,model=None,data=None,index=SimpleNamespace(bean_bodies=[0]),tick=50,
        bean_acceptance=dict(linear_speed_m_s=.001,angular_speed_rad_s=.1))
    def velocity(m,data,kind,body,result,local):result[:]=[.2,0,0,.0005,0,0]
    monkeypatch.setattr(d.mujoco,'mj_objectVelocity',velocity)
    recorder=d.PickupRecorder();e=dict(supported=True,off_bowl=True,pickup_eligible=False,bowl_clearance_m=.01)
    with recorder.capture(task):logic.update(e,.001,.05)
    assert calls==[e] and recorder.rows[0][-1]==8
    assert logic.update is update
    assert recorder.save(tmp_path)['reset_causes']['angular_speed']==1


@pytest.mark.parametrize('index,seed',[(0,801001),(31,801002)])
def test_real_controlled_handover_and_exact_replay(index,seed,tmp_path):
    from correction_v4.rollout import episode
    from correction_v2.run import scenario_for
    from feedingrobot.data.episodes import load_episode
    from feedingrobot.data.replay import replay_episode
    old=read_json(ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_calibration_001/episodes/780001_calibration/manifest.json')
    config=old['teacher_config'];cell=c.CELLS[index];directory=tmp_path/'case'
    m=episode(seed,cell,config,scenario_for(seed,config),directory,max_s=12.)
    assert m['qualified_correction'] and not m['abort_reason'],m['handover']
    assert m['alignment_max_drift_m']<.0007
    _,a=load_episode(directory)
    assert not a['action_mask'][~np.isin(a['action_owner'],c.TEACHER_OWNERS)].any()
    detail=run.coverage(m,a,directory)
    assert detail['wait_action_bins']==[1,1,1,1] and detail['x_critical_action_ticks']
    masked=dict(a,action_mask=a['action_mask'].copy())
    masked['action_mask'][np.isin(a['action_ticks'],detail['x_critical_action_ticks'])]=False
    assert run.coverage(m,masked,directory)['x_critical_action_ticks']==[]
    assert replay_episode(directory)['status']=='passed'


def short_matrix_case(index,root):
    from correction_v4.rollout import episode
    from correction_v2.run import scenario_for
    from feedingrobot.data.episodes import load_episode
    from feedingrobot.data.replay import replay_episode
    config=read_json(ROOT/'outputs/single_bean/v1/m5/dit/correction_v3_calibration_001/episodes/780001_calibration/manifest.json')['teacher_config']
    cell=c.CELLS[index];seed=803001+index;directory=Path(root)/str(seed)
    m=episode(seed,cell,config,scenario_for(seed,config),directory,max_s=12.)
    assert m['qualified_correction'] and not m['abort_reason'],(cell,m['abort_reason'],m['handover'])
    _,a=load_episode(directory);detail=run.coverage(m,a,directory)
    assert replay_episode(directory)['status']=='passed'
    return dict(cell=cell,coverage=detail,release=m['handover']['release_residual'])


def test_all_32_real_direction_velocity_handovers_and_replays(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn'),initializer=run.worker_setup) as pool:
        rows=list(pool.map(short_matrix_case,range(32),[str(tmp_path)]*32))
    assert run.aggregate_coverage(rows)['passed'],run.aggregate_coverage(rows)
