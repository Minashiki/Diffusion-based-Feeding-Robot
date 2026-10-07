"""Evaluator contracts and short fixed-action physics; no production policy runs."""

import importlib
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]))
entry=importlib.import_module('correction_v4.evaluation.run')
b=entry.b
policy=importlib.import_module('policy_rollout')

import numpy as np
import pytest
import torch

from feedingrobot.data.episodes import load_episode,write_json
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.sim.model import ROOT

OUT=ROOT/'outputs/single_bean/v1/m5/dit'


def test_105k_prefix_keeps_later_training_rows(tmp_path):
    path=tmp_path/'metrics.jsonl'
    rows=[dict(step=i,loss=.01,gradient_norm=.1) for i in range(100001,110001)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));before=path.read_bytes()
    report=b.records(path,105000)
    assert report['prefix_rows']==5000 and report['log_last_step']==110000
    assert path.read_bytes()==before
    rows[6000]['step']+=1;path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='continuous'): b.records(path,105000)
    rows[6000]['step']-=1;rows[-1]['loss']=float('nan')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='finite'): b.records(path,105000)


def test_checkpoint_checks_saved_provenance_snapshot_and_finite_ema(tmp_path):
    snapshot=tmp_path/'tool_snapshot';snapshot.mkdir();(snapshot/'common.py').write_text('original')
    binding=dict(collection={'legacy':{'source_hashes':{'source':'hash'}}},entry_tools={'common.py':sha256(snapshot/'common.py')})
    provenance=dict(schema_version=4,fork_kind=b.training.FORK,diagnostic=True,binding=binding,config={},normalization={},source_hashes={'source':'hash'})
    write_json(tmp_path/'provenance.json',provenance)
    (tmp_path/'metrics.jsonl').write_text(json.dumps(dict(step=100001,loss=.1,gradient_norm=.2))+'\n')
    value=dict(provenance,step=100001,optimizer_updates=1,model={'w':torch.tensor(1.)},ema={'w':torch.tensor(2.)})
    path=tmp_path/'step.pt';torch.save(value,path)
    _,audit=b.checkpoint(path,binding,{},{});assert audit['step']==100001
    value['ema']['w']=torch.tensor(float('nan'));torch.save(value,path)
    with pytest.raises(FloatingPointError): b.checkpoint(path,binding,{},{})
    value['ema']['w']=torch.tensor(2.);value['optimizer_updates']=2;torch.save(value,path)
    with pytest.raises(ValueError,match='update'): b.checkpoint(path,binding,{},{})
    value['optimizer_updates']=1;torch.save(value,path);(snapshot/'common.py').write_text('changed')
    with pytest.raises(ValueError,match='snapshot'): b.checkpoint(path,binding,{},{})


@pytest.mark.parametrize('field,replacement,accepted',(
    (('sampler','config','_use_default_values'),['beta_end','beta_start'],True),
    (('sampler','config','_use_default_values'),['beta_start'],False),
    (('sampler','config','_use_default_values'),['beta_start','beta_end','steps_offset'],False),
    (('sampler','config','_use_default_values'),['beta_start','beta_end','beta_end'],False),
    (('sampler','config','_use_default_values'),None,False),
    (('sampler','config','beta_schedule'),'linear',False),
    (('sampler','config','timestep_spacing'),'trailing',False),
    (('sampler','timesteps'),[0,10,20],False),
    (('precision',),'fp32',False),
    (('hardware','gpu'),'different',False),
    (('config',),{'changed':True},False),
    (('binding',),{'changed':True},False),
    (('source_hashes',),{'source':'changed'},False),
))
def test_checkpoint_accepts_only_reordered_default_names(tmp_path,field,replacement,accepted):
    snapshot=tmp_path/'tool_snapshot';snapshot.mkdir();(snapshot/'common.py').write_text('original')
    binding=dict(collection={'legacy':{'source_hashes':{'source':'hash'}}},entry_tools={'common.py':sha256(snapshot/'common.py')})
    provenance=dict(schema_version=4,fork_kind=b.training.FORK,diagnostic=True,binding=binding,config={},normalization={},
        source_hashes={'source':'hash'},precision='bf16_autocast',hardware={'gpu':'same'},
        sampler=dict(config={'_use_default_values':['beta_start','beta_end'],
            'beta_schedule':'squaredcos_cap_v2','timestep_spacing':'leading'},timesteps=[20,10,0]))
    write_json(tmp_path/'provenance.json',provenance)
    (tmp_path/'metrics.jsonl').write_text(json.dumps(dict(step=100001,loss=.1,gradient_norm=.2))+'\n')
    value=dict(deepcopy(provenance),step=100001,optimizer_updates=1,model={'w':torch.tensor(1.)},ema={'w':torch.tensor(2.)})
    target=value
    for key in field[:-1]: target=target[key]
    if replacement is None: target.pop(field[-1])
    else: target[field[-1]]=replacement
    path=tmp_path/'step.pt';torch.save(value,path)
    before={p:sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    if accepted:
        loaded,_=b.checkpoint(path,binding,{},{})
        assert loaded['sampler']==value['sampler']
    else:
        with pytest.raises(ValueError): b.checkpoint(path,binding,{},{})
    assert {p:sha256(p) for p in before}==before


@pytest.mark.parametrize('threads',(6,7,8))
def test_cuda_budget_does_not_mutate_training_binding(monkeypatch,threads):
    calls=[];config=dict(training=dict(workers=2,cpu_budget=6,torch_threads=4))
    monkeypatch.setattr(b,'CPUS',list(range(8)))
    monkeypatch.setattr(b,'affinity',lambda cpus:calls.append(cpus))
    def setup(runtime,device):
        assert device=='cuda';assert runtime['training']==dict(workers=0,cpu_budget=threads,torch_threads=threads)
        return torch.device('cuda'),{}
    monkeypatch.setattr(b,'setup',setup);monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    _,hardware=b.compute(config,threads)
    assert calls==[list(range(threads))] and hardware['replay_workers']==threads-1
    assert config['training']==dict(workers=2,cpu_budget=6,torch_threads=4)
    with pytest.raises(ValueError): b.compute(config,9)


def test_new_directory_preserves_collection_and_training_hashes():
    saved=read_json(OUT/'correction_v4_run_001/provenance.json')['binding']
    from correction_v4.run import tools_hashes
    assert b.training.entry_hashes()==saved['entry_tools']
    assert tools_hashes()==saved['collection']['v4_tools']


def obs(speed=.001,height=0.):
    return dict(tcp_position=np.array([0.,0.,height]),r=dict(angle_deg=.2,lateral_m=.0002,linear_speed=speed,angular_speed=.001))


def alignment(monkeypatch):
    monkeypatch.setattr(policy,'residual',lambda reference,value:value['r'])
    return policy.Alignment(SimpleNamespace(path=[None,None,(None,np.zeros(3),None)]),0.)


def test_1ms_instability_between_action_ticks_resets_200ms(monkeypatch,tmp_path):
    tracker=alignment(monkeypatch)
    for tick in range(339):
        tracker.observe(obs(.003 if tick==137 else .001),tick)
        if tick==200: assert tracker.completed is None
        if tick==337: assert tracker.completed is None
    assert tracker.completed==338
    result=tracker.save(tmp_path)
    assert result['resets']==1 and result['longest_stable_ms']==200
    records=np.load(tmp_path/'alignment_boundaries.npy')
    assert records[137,-1]==1 and records[138,8]==138


@pytest.mark.parametrize('field,value',(('angle_deg',np.rad2deg(.01)),('lateral_m',.0007),('linear_speed',.002),('angular_speed',.05)))
def test_qualification_threshold_equality_is_not_good(monkeypatch,field,value):
    tracker=alignment(monkeypatch);tracker.observe(obs(),0)
    bad=obs();bad['r'][field]=value;tracker.observe(bad,1)
    assert tracker.stable is None and tracker.rows[-1][-1]


def test_height_and_missing_physics_boundary(monkeypatch):
    tracker=alignment(monkeypatch);tracker.observe(obs(),0);tracker.observe(obs(height=.0007),1)
    assert tracker.stable is None
    with pytest.raises(ValueError,match='1ms'): tracker.observe(obs(),3)


def test_fixed_small_and_full_case_coverage():
    config=read_json(ROOT/'configs/dp_dit.json')
    data=read_json(OUT/'correction_v4_dataset_001/report.json');data['old_report']=str(OUT/'correction_v3_dataset_001/report.json')
    calibration=read_json(Path(data['calibration_report']))
    small=entry.select_cases(config,data,calibration,'small');full=entry.select_cases(config,data,calibration,'validation')
    assert len(small)==16 and len(full)==66
    selected=[calibration['pairs'][i]['cell'] for i in entry.SMALL_CELLS]
    for category in ('p1_stopped','p1_moving','p2_stopped','p2_moving'):
        cells=[c for c in selected if c['category']==category]
        assert len(cells)==2 and {c['velocity'] for c in cells}=={'toward','away'}
        assert {c['quadrant'][0] for c in cells}=={-1,1}
    assert len(entry.select_cases(config,data,calibration,'test'))==30
    calibration['pairs'][0]['cell']['velocity']='away'
    with pytest.raises(ValueError,match='coverage'): entry.select_cases(config,data,calibration,'small')


def test_test_requires_unchanged_full_validation_and_selected_checkpoint(tmp_path):
    bound=tmp_path/'checkpoint.pt';bound.write_bytes(b'weight')
    output=tmp_path/'validation';output.mkdir();(output/'evidence.json').write_text('{}')
    report=dict(mode='validation',status='validation_passed',binding={},evaluation_tools={},protocol={},
        source_unchanged=True,formal_test_run=False,full_validation_run=True,
        input_sha256={str(bound):sha256(bound)},arms=[dict(name='v4_105000',sha256='selected',step=105000,validation_passed=True)],
        evidence_sha256=b.file_hashes(output))
    path=output/'report.json';write_json(path,report)
    assert b.validation_evidence(path,'selected',{},{},{})['selected_arm']=='v4_105000'
    with pytest.raises(ValueError,match='Selected'): b.validation_evidence(path,'other',{},{},{})
    report['full_validation_run']=False;write_json(path,report)
    with pytest.raises(ValueError,match='full'): b.validation_evidence(path,'selected',{},{},{})
    report['full_validation_run']=True;write_json(path,report);bound.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'): b.validation_evidence(path,'selected',{},{},{})


def rows():
    result=[]
    for i in range(62):
        correction=i<32;recover=i>=47
        result.append(dict(seed=i,source_episode=str(i),cohort='v4_calibration' if correction else 'validation',
            correction_case=correction,recover=recover,success=True,physical_success=True,pickup=True,delivery=True,
            transport_completed=True,recovery_completed=recover,entered=['SELECT','ACQUIRE','TRANSFER','RETRACT']+(['RECOVER'] if recover else []),
            model_alignment={'success':True},external_braking={'interventions':[]},replay={'status':'passed'},contact_peak_n=0.,wrist_peak_n=0.))
        result[-1]['failure_reason']=None
    return result


def test_validation_does_not_hide_recovery_bypass_or_safety_intervention():
    config=dict(acceptance=dict(full_success=.8,stage_success=.9))
    cases=rows();assert entry.arm_summary(cases,config)['validation_passed']
    cases[-1]['entered'].remove('RECOVER')
    assert not entry.arm_summary(cases,config)['validation_passed']
    cases=rows();cases[0]['external_braking']['interventions']=[{'reason':'clearance'}]
    assert not entry.arm_summary(cases,config)['validation_passed']
    cases=rows();cases[0]['replay']['status']='failed'
    assert not entry.arm_summary(cases,config)['validation_passed']


def test_leading_sampler_matches_frozen_sampler():
    from diagnose_sampling import spacing_override
    from feedingrobot.policies.dit import sample_actions
    config=read_json(ROOT/'configs/dp_dit.json');config['model'].update(depth=1,hidden_size=32,heads=4,mlp_hidden=64)
    torch.manual_seed(11);model=ActionDiT(config).eval()
    batch=dict(states=torch.randn(1,2,model.encoder.state[0].in_features),history=torch.randn(1,10,28),
        phase=torch.ones(1,dtype=torch.long),interaction=torch.zeros(1,4),
        state_mask=torch.ones(1,2,dtype=torch.bool),history_mask=torch.ones(1,10,dtype=torch.bool))
    norm=dict(action_mean=[0.]*6,action_std=[1.]*6)
    with spacing_override('leading'):
        expected=sample_actions(model,config,batch,norm,torch.Generator().manual_seed(3))
    actual=policy.sample_actions(model,config,batch,norm,torch.Generator().manual_seed(3))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.fixture
def physical(request):
    seed=getattr(request,'param',800015)
    source=OUT/f'correction_v4_calibration_003/episodes/{seed}_candidate'
    m=read_json(source/'manifest.json')
    return source,m,read_json(ROOT/'configs/dp_dit.json'),read_json(source.parents[1]/'report.json')['normalization']


@pytest.mark.parametrize('physical',(800015,800023,800031,800039),indirect=True)
def test_fixed_action_physics_causality_chunk_indices_and_parallel_replay(monkeypatch,tmp_path,physical):
    source,m,config,norm=physical;plans=[]
    def sample(model,config,batch,norm,generator):
        plans.append((batch['states'].clone(),generator.initial_seed()))
        actions=torch.zeros(1,16,6);actions[0,:,1]=torch.arange(16)*.0001
        return actions
    monkeypatch.setattr(policy,'sample_actions',sample)
    monkeypatch.setattr(policy.Teacher,'act',lambda *a:pytest.fail('Teacher must not execute'))
    output=tmp_path/'case'
    result=policy.rollout(None,config,norm,source,output,torch.device('cpu'),correction=True,max_ticks=m['handover']['release_tick']+300)
    trace=read_json(output/'trace.json');_,a=load_episode(output)
    assert [t['chunk_action_index'] for t in trace]==[0,1,2,3,0,1]
    assert [seed for _,seed in plans]==[m['handover']['release_tick']//50,(m['handover']['release_tick']+200)//50]
    assert plans[0][0][0,:,-1].all() and len(plans)==2
    assert not a['action_mask'].any() and set(a['action_owner'])=={'model'}
    boundaries=np.load(output/'alignment_boundaries.npy')
    np.testing.assert_array_equal(boundaries[:,0],np.arange(m['handover']['release_tick'],m['handover']['release_tick']+301))
    jobs=[(output,result)];entry.replays(jobs,b.CPUS[:6],tmp_path)
    assert result['replay']['status']=='passed' and result['replay']['qualification_equal']
    assert result['replay']['worker']['torch_threads']==1 and not result['replay']['worker']['cuda_initialized']


def test_safety_braking_is_separate_and_replayable(monkeypatch,tmp_path,physical):
    source,m,config,norm=physical
    monkeypatch.setattr(policy,'sample_actions',lambda *a:torch.tensor([0.,0.,.02,0.,0.,0.]).repeat(1,16,1))
    result=policy.rollout(None,config,norm,source,tmp_path/'unsafe',torch.device('cpu'),correction=True,max_ticks=m['handover']['release_tick']+1000)
    assert result['failure_reason']=='alignment_height_drift' and not result['model_alignment']['success']
    assert result['external_braking']['interventions'] and result['model_alignment']['maximum_height_drift_m']>=.0007
    assert entry.replay_case(tmp_path/'unsafe')['status']=='passed'


def test_invalid_proposal_cannot_become_a_label_or_physical_failure(monkeypatch,tmp_path,physical):
    source,m,config,norm=physical
    monkeypatch.setattr(policy,'sample_actions',lambda *a:torch.full((1,16,6),float('nan')))
    result=policy.rollout(None,config,norm,source,tmp_path/'invalid',torch.device('cpu'),correction=True,max_ticks=m['handover']['release_tick']+100)
    assert result['failure_reason']=='invalid_command' and not result['success']
    manifest,arrays=load_episode(tmp_path/'invalid')
    assert manifest['failure_reason'] is None and not arrays['action_mask'].any()
    assert entry.replay_case(tmp_path/'invalid')['status']=='passed'


def test_offline_cases_cover_direction_wait_ages_and_complete_hold():
    config=read_json(ROOT/'configs/dp_dit.json')
    norm=read_json(OUT/'correction_v4_run_001/provenance.json')['normalization']
    data=read_json(OUT/'correction_v4_dataset_001/report.json')
    cases,items=entry.offline.windows(config,norm,data)
    extra=[c for c in cases if c['source']=='v4_train_diagnostic']
    assert len({c['cell_key'] for c in extra if 'cell_key' in c})==32
    assert {c['wait_bin'] for c in extra if 'wait_bin' in c}=={'0-50','50-100','100-150','150-200'}
    holds=[i for c,i in zip(cases,items) if c['stage']=='pickup_hold']
    assert len(holds)==88 and any(i['phase'].item()==2 for i in holds)
    assert all(i['action_mask'].any() for i in items)


@pytest.fixture
def synthetic_run(monkeypatch,tmp_path):
    config=dict(dataset=str(tmp_path/'original'),model={'horizon':16},acceptance={'full_success':.8,'stage_success':.9})
    names=('base_checkpoint','parent_checkpoint','v1_report','old_data_report','data_report','audit_report')
    values={}
    for name in names:
        path=tmp_path/name;path.write_bytes(b'unchanged');values[name]=str(path)
    checkpoint=tmp_path/'step_105000.pt';checkpoint.write_bytes(b'unchanged weights')
    write_json(tmp_path/'provenance.json',dict(hardware=dict(device='cuda',gpu='same',torch='same',cuda_runtime='same')))
    (tmp_path/'metrics.jsonl').write_text('unchanged log')
    binding={'collection':{'legacy':{'source_hashes':{}}}}
    data=dict(dataset=str(tmp_path/'extra'),calibration_report=values['data_report'])
    state={'weight':torch.ones(1,1)}
    evidence=lambda args:({'config':config,'normalization':{},'parent':{'m4':'frozen'}},{'ema':state},{},data,{},binding)
    monkeypatch.setattr(b.training,'evidence',evidence);monkeypatch.setattr(b.training,'check_unchanged',lambda _:None)
    monkeypatch.setattr(b,'checkpoint',lambda *a:({'ema':state,'step':105000},dict(path=str(checkpoint),sha256=sha256(checkpoint),step=105000,weights='ema')))
    monkeypatch.setattr(b,'compute',lambda *a:(torch.device('cpu'),dict(device='cuda',gpu='same',torch='same',cuda_runtime='same',cpu_affinity=b.CPUS[:6])))
    monkeypatch.setattr(entry,'sampler_settings',lambda *a:{'timestep_spacing':'leading'})
    monkeypatch.setattr(entry,'ActionDiT',lambda _:torch.nn.Linear(1,1,bias=False))
    all_rows=rows();sources={}
    for row in all_rows:
        directory=tmp_path/'sources'/str(row['seed']);directory.mkdir(parents=True)
        write_json(directory/'manifest.json',dict(seed=row['seed'],files_sha256={}))
        row['source_episode']=str(directory);sources[str(directory)]=row
    monkeypatch.setattr(entry,'select_cases',lambda config,data,calibration,mode:[dict(name=str(r['seed']),source=r['source_episode'],
        correction=r['correction_case'],cohort=r['cohort']) for r in all_rows if mode!='test' or not r['correction_case']])
    monkeypatch.setattr(entry,'load_episode',lambda path:(read_json(Path(path)/'manifest.json'),{}))
    def rollout(model,config,norm,source,output,device,**kwargs):
        output.mkdir(parents=True);(output/'initial_state.pkl').write_bytes(b'paired state')
        return json.loads(json.dumps(sources[str(source)]))
    monkeypatch.setattr(entry,'rollout',rollout)
    def replay(jobs,cpus,output):
        for directory,row in jobs: row['replay']={'status':'passed'};write_json(directory/'result.json',row)
    monkeypatch.setattr(entry,'replays',replay)
    args=SimpleNamespace(**values,checkpoint=[str(checkpoint)],output=str(tmp_path/'validation'),mode='validation',
        validation_report=None,cpu_threads=6,noise_seed=0,freeze=False)
    return args,checkpoint


def test_orchestration_preserves_weights_and_freezes_only_after_gated_test(synthetic_run,tmp_path,monkeypatch):
    args,checkpoint=synthetic_run;before=sha256(checkpoint)
    validation=entry.run(args)
    assert validation['status']=='validation_passed' and validation['m5_status']=='incomplete'
    assert not (Path(args.output)/'freeze_manifest.json').exists()
    args.mode='test';args.validation_report=str(Path(args.output)/'report.json');args.output=str(tmp_path/'test')
    report=entry.run(args)
    assert report['status']=='passed' and report['m5_status']=='incomplete'
    assert not (Path(args.output)/'freeze_manifest.json').exists()
    args.freeze=True;args.output=str(tmp_path/'test_release');report=entry.run(args)
    frozen=read_json(Path(args.output)/'freeze_manifest.json')
    assert report['status']=='passed' and report['m5_status']=='complete'
    assert frozen['checkpoint_diagnostic'] and frozen['checkpoint_sha256']==before
    assert frozen['report_sha256']==sha256(Path(args.output)/'report.json')
    assert sha256(checkpoint)==before and not list(Path(args.output).rglob('*.pt'))
    def failed_replay(jobs,cpus,output):
        for directory,row in jobs: row['replay']={'status':'failed'};write_json(directory/'result.json',row)
    monkeypatch.setattr(entry,'replays',failed_replay);args.output=str(tmp_path/'failed_test')
    failed=entry.run(args)
    assert failed['status']=='failed' and failed['m5_status']=='incomplete'
    assert not (Path(args.output)/'freeze_manifest.json').exists()


def test_audit_constructs_neither_model_nor_optimizer(synthetic_run,monkeypatch,tmp_path):
    args,_=synthetic_run;args.mode='audit';args.output=str(tmp_path/'audit')
    monkeypatch.setattr(entry,'ActionDiT',lambda *a:pytest.fail('Audit must not construct a model'))
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**kw:pytest.fail('Evaluator must not construct an optimizer'))
    report=entry.run(args)
    assert report['status']=='audit_passed' and not report['model_executed'] and not report['physics_executed']


def test_small_replay_failure_is_not_successful_completion(synthetic_run,monkeypatch):
    args,_=synthetic_run;args.mode='small'
    def failed_replay(jobs,cpus,output):
        for directory,row in jobs: row['replay']={'status':'failed'};write_json(directory/'result.json',row)
    monkeypatch.setattr(entry,'replays',failed_replay)
    report=entry.run(args)
    assert report['status']=='diagnostic_failed' and report['m5_status']=='incomplete'


def test_different_paired_initial_state_stops_before_release(synthetic_run,monkeypatch):
    args,_=synthetic_run;original=entry.rollout
    def changed_state(model,config,norm,source,output,device,**kwargs):
        row=original(model,config,norm,source,output,device,**kwargs)
        if output.parent.name.startswith('v4_'): (output/'initial_state.pkl').write_bytes(b'different state')
        return row
    monkeypatch.setattr(entry,'rollout',changed_state)
    with pytest.raises(AssertionError,match='initial physics'): entry.run(args)
    report=read_json(Path(args.output)/'report.json')
    assert report['status']=='error' and report['m5_status']=='incomplete'
