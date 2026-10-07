"""Private v4 audit, paired EMA diagnostics, validation and gated test release."""

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os
from pathlib import Path
import shutil

import bindings as b
import numpy as np
import torch

import offline
from policy_rollout import rollout
from correction_v4.controller import CALIBRATION_SEEDS,CELLS
from correction_v4.diagnostics import replay_with_pickup
from diagnose_sampling import sampler_settings
from feedingrobot.data.episodes import load_episode,write_json
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import ActionDiT,schedulers
from feedingrobot.policies.evaluation import summarize
from feedingrobot.policies.training import validation_loss
from feedingrobot.sim.model import ROOT

SMALL_CELLS=(0,7,10,13,19,20,25,30)


def select_cases(config,data,calibration,mode):
    cases=[]
    if mode!='test':
        if ([p['seed'] for p in calibration['pairs']]!=list(CALIBRATION_SEEDS)
                or [p['cell'] for p in calibration['pairs']]!=list(CELLS)
                or any(p['status']!='passed' for p in calibration['pairs'])):
            raise ValueError('Independent v4 calibration coverage differs')
        root=Path(data['calibration_report']).resolve().parent/'episodes'
        ids=SMALL_CELLS if mode=='small' else range(32)
        cases.extend(dict(name=f'v4_{CALIBRATION_SEEDS[i]}',source=str(root/f'{CALIBRATION_SEEDS[i]}_candidate'),
            correction=True,cohort='v4_calibration') for i in ids)
        old=read_json(Path(data['old_report']))
        history=Path(old['calibration_report']).resolve().parent/'episodes'
        cases.extend(dict(name=f'history_{seed}',source=str(history/f'{seed}_calibration'),
            correction=True,cohort='historical_diagnostic') for seed in (780001,780003,780005,780007))
    split='test' if mode=='test' else 'validation'
    paths=sorted((ROOT/config['dataset']/split).glob('*/manifest.json'))
    normal=[p for p in paths if not read_json(p)['scenario'].get('recover',False)]
    recovery=[p for p in paths if read_json(p)['scenario'].get('recover',False)]
    if len(normal)!=15 or len(recovery)!=15: raise ValueError('Original evaluation requires exactly 15+15 frozen cases')
    chosen=normal[:3]+recovery[:1] if mode=='small' else normal+recovery
    cases.extend(dict(name=f'{split}_{p.parent.name}',source=str(p.parent),correction=False,cohort=split) for p in chosen)
    return cases


def replay_setup(cpus):
    b.affinity(cpus);torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)


def replay_case(directory):
    directory=Path(directory);output=directory/'replay_diagnostics';output.mkdir(exist_ok=False)
    try:
        evidence=replay_with_pickup(directory,output)
        live=np.load(directory/'pickup_boundaries.npy');replayed=np.load(output/'pickup_boundaries.npy')
        np.testing.assert_allclose(replayed,live,rtol=0,atol=1e-10)
        return dict(status='passed',**evidence,qualification_equal=True,
            worker=dict(torch_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),
                cuda_initialized=torch.cuda.is_initialized()))
    except Exception as error:
        return dict(status='failed',error=f'{type(error).__name__}: {error}')


def replays(jobs,cpus,output):
    # CUDA rollouts have finished. Parent CPU work and each physics worker use one thread.
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=len(cpus)-1,mp_context=multiprocessing.get_context('spawn'),
            initializer=replay_setup,initargs=(cpus,)) as pool:
        futures={pool.submit(replay_case,directory):(directory,row) for directory,row in jobs}
        for future in as_completed(futures):
            directory,row=futures[future];row['replay']=future.result();write_json(directory/'result.json',row)
            print(f'replay {directory.parent.name}/{directory.name}: {row["replay"]["status"]}',flush=True)
            write_json(output/'replay_progress.json',[dict(episode=str(p),status=r['replay']['status'])
                for p,r in jobs if 'replay' in r])


def arm_summary(rows,config,parent_rows=None):
    ordinary=[r for r in rows if not r['correction_case']]
    correction=[r for r in rows if r['cohort']=='v4_calibration']
    regressions=[]
    if parent_rows is not None:
        for old,new in zip(parent_rows,rows):
            if old['source_episode']!=new['source_episode']: raise ValueError('Paired case order differs')
            for flag in ('pickup','transport_completed','delivery','success','recovery_completed'):
                if old[flag] and not new[flag]: regressions.append(dict(seed=new['seed'],cohort=new['cohort'],metric=flag))
            if old['correction_case'] and old['model_alignment']['success'] and not new['model_alignment']['success']:
                regressions.append(dict(seed=new['seed'],cohort=new['cohort'],metric='alignment'))
            if not old['external_braking']['interventions'] and new['external_braking']['interventions']:
                regressions.append(dict(seed=new['seed'],cohort=new['cohort'],metric='safety_intervention'))
    original=summarize(ordinary,config)
    local=dict(attempts=len(correction),aligned=sum(r['model_alignment']['success'] for r in correction),
        pickups=sum(r['pickup'] for r in correction),complete_success=sum(r['success'] for r in correction),
        safety_interventions=sum(len(r['external_braking']['interventions']) for r in correction))
    passed=(len(correction)==32 and local['aligned']==32 and local['complete_success']==32
        and local['safety_interventions']==0 and original['status']=='passed' and not regressions
        and all(r['replay']['status']=='passed' for r in rows))
    history=[r for r in rows if r['cohort']=='historical_diagnostic']
    return dict(original=original,correction=local,historical=dict(attempts=len(history),
        aligned=sum(r['model_alignment']['success'] for r in history),pickups=sum(r['pickup'] for r in history),
        complete_success=sum(r['success'] for r in history)),regressions=regressions,validation_passed=passed)


def freeze(output,provenance,arm):
    write_json(output/'freeze_manifest.json',dict(schema_version=4,status='frozen',policy_version='DP_v1',
        backbone='dit_adaln_cross_attention',robot_id='panda',checkpoint=arm['path'],checkpoint_sha256=arm['sha256'],
        checkpoint_diagnostic=True,release_kind='audited_v4_fork',training_step=arm['step'],weights='ema',
        config=provenance['config'],normalization=provenance['normalization'],sampler=provenance['protocol']['sampler'],
        parent=provenance['parent'],binding=provenance['binding'],source_hashes=provenance['source_hashes'],
        evaluation_tools=provenance['evaluation_tools'],validation=provenance['validation'],
        report_sha256=sha256(output/'report.json'),evidence_sha256=b.file_hashes(output)))


def run(args):
    output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError('Evaluation output must not already exist')
    if args.mode=='test' and (len(args.checkpoint)!=1 or not args.validation_report):
        raise ValueError('Test requires one selected v4 checkpoint and --validation-report')
    if args.mode!='test' and args.validation_report: raise ValueError('--validation-report is only used for test')
    if args.freeze and args.mode!='test': raise ValueError('--freeze is only used with gated test')
    if args.cpu_threads not in (6,7,8) or len(b.CPUS)<args.cpu_threads:
        raise ValueError('Requested CPU budget exceeds the available 6..8 logical CPUs')
    print('Auditing frozen ancestry, supervision and zero-update evidence; no old physics reruns...',flush=True)
    base,parent,parent_audit,data,calibration,binding=b.training.evidence(args)
    config=base['config'];normalization=base['normalization'];tools=b.tool_hashes()
    paths=[Path(p).resolve() for p in args.checkpoint]
    if len(set(paths))!=len(paths): raise ValueError('Duplicate checkpoint')
    arms=[];states={}
    if args.mode!='test':
        arms.append(dict(name='parent_100000',path=str(Path(args.parent_checkpoint).resolve()),
            sha256=sha256(args.parent_checkpoint),step=100000,weights='ema'))
        states[arms[-1]['name']]=parent['ema']
    for path in paths:
        value,audit=b.checkpoint(path,binding,config,normalization)
        name=f'v4_{value["step"]}'
        if name in states: raise ValueError('Duplicate training step')
        arms.append(dict(audit,name=name));states[name]=value['ema']
        del value
    input_paths=[args.base_checkpoint,args.parent_checkpoint,args.v1_report,args.old_data_report,args.data_report,
        args.audit_report,data['calibration_report']]+[str(p) for p in paths]
    inputs={str(Path(p).resolve()):sha256(p) for p in input_paths}
    for path in paths:
        for name in ('provenance.json','metrics.jsonl'):
            p=path.parent/name;inputs[str(p)]=sha256(p)
    protocol=dict(sampler=sampler_settings(config,'leading'),noise_seed=args.noise_seed,
        noise_pairing='seed*1000003 + physical_tick//50; equal initial noise at equal ticks',
        physics_ms=1,history_ms=20,action_ms=50,replan_ms=200,max_episode_s=60.,alignment_boundary_ms=1,
        assisted_alignment=False,teacher_in_execution=False,visualization='disabled')
    validation=b.validation_evidence(args.validation_report,arms[0]['sha256'],binding,tools,protocol) if args.mode=='test' else None
    if validation: inputs[validation['path']]=validation['sha256']
    device,hardware=b.compute(config,args.cpu_threads)
    for path in paths:
        saved=read_json(path.parent/'provenance.json')['hardware']
        if any(hardware[k]!=saved[k] for k in ('device','gpu','torch','cuda_runtime')):
            raise ValueError('Private comparisons require the original training CUDA runtime and GPU')
    provenance=dict(schema_version=4,mode=args.mode,diagnostic=args.mode!='test',checkpoint_diagnostic=True,
        config=config,normalization=normalization,binding=binding,parent=base['parent'],warmstart_parent=parent_audit,
        source_hashes=binding['collection']['legacy']['source_hashes'],evaluation_tools=tools,
        hardware=hardware,protocol=protocol,validation=validation,input_sha256=inputs,arms=arms,publication_requested=args.freeze,
        optimizer_created=False,optimizer_updates=0,formal_test_run=False,full_validation_run=False,
        m5_status='incomplete',dp_v1='not_frozen',model_executed=False,physics_executed=False,
        limitation='Calibration snapshots follow teacher/external preparation; autonomous high-speed braking is not established')
    output.mkdir(parents=True,exist_ok=False);snapshot=output/'tool_snapshot';snapshot.mkdir()
    for name in tools: shutil.copyfile(b.HERE/name,snapshot/name)
    write_json(output/'provenance.json',provenance)
    results={};jobs=[]
    def finish(status,**extra):
        report=dict(provenance,status=status,results=results,**extra,evidence_sha256=b.file_hashes(output))
        report['evidence_sha256'].pop('report.json',None)
        write_json(output/'report.json',report);return report
    try:
        if args.mode!='audit':
            model=ActionDiT(config).to(device).eval().requires_grad_(False)
            provenance['model_executed']=True
            if args.mode=='offline':
                cases,items=offline.windows(config,normalization,data)
                dataset=ActionWindows(ROOT/config['dataset'],'validation',config['model']['horizon'],normalization)
                scheduler,_=schedulers(config)
                for arm in arms:
                    model.load_state_dict(states[arm['name']])
                    results[arm['name']]=offline.evaluate(model,config,normalization,device,output/arm['name'],cases,items)
                    arm['validation_loss']=validation_loss(model,dataset,scheduler,config,device)
                    if not np.isfinite(arm['validation_loss']): raise FloatingPointError('Nonfinite EMA validation loss')
                    write_json(output/arm['name']/'result.json',results[arm['name']])
                for seed in ('0','1','2'):
                    if any(results[a['name']]['by_seed'][seed]['initial_noise_sha256']!=results[arms[0]['name']]['by_seed'][seed]['initial_noise_sha256'] for a in arms):
                        raise AssertionError('Offline diffusion noise pairing differs')
            else:
                cases=select_cases(config,dict(data,old_report=args.old_data_report),calibration,args.mode)
                sources={c['source']:load_episode(c['source']) for c in cases}
                train_seeds={read_json(p)['seed'] for p in Path(data['dataset']).glob('train/*/manifest.json')}
                if train_seeds.intersection(m['seed'] for m,_ in sources.values()): raise ValueError('Evaluation seed overlaps v4 train')
                provenance.update(cases=cases,physics_executed=True)
                for source,(m,_) in sources.items():
                    inputs[str(Path(source)/'manifest.json')]=sha256(Path(source)/'manifest.json')
                    inputs.update({str(Path(source)/name):digest for name,digest in m['files_sha256'].items()})
                write_json(output/'provenance.json',provenance)
                for arm in arms:
                    model.load_state_dict(states[arm['name']]);results[arm['name']]=[]
                    for case_index,case in enumerate(cases):
                        directory=output/arm['name']/case['name']
                        row=rollout(model,config,normalization,case['source'],directory,device,
                            correction=case['correction'],noise_seed=args.noise_seed,source_data=sources[case['source']])
                        row['initial_state_sha256']=sha256(directory/'initial_state.pkl')
                        if arm['step']>100000 and 'parent_100000' in results:
                            if row['initial_state_sha256']!=results['parent_100000'][case_index]['initial_state_sha256']:
                                raise AssertionError('Paired initial physics state differs')
                        row['cohort']=case['cohort'];write_json(directory/'result.json',row)
                        results[arm['name']].append(row);jobs.append((directory,row))
                        print(f'{arm["name"]} {case["name"]}: aligned={row["model_alignment"] and row["model_alignment"]["success"]} pickup={row["pickup"]} success={row["success"]} reason={row["failure_reason"]}',flush=True)
                    finish('evaluation_running')
                del model;torch.cuda.empty_cache()
                replays(jobs,hardware['cpu_affinity'],output)
                provenance.update(full_validation_run=args.mode=='validation',formal_test_run=args.mode=='test')
                for arm in arms:
                    arm.update(arm_summary(results[arm['name']],config,results.get('parent_100000') if arm['step']>100000 else None))
        b.check_files(inputs);b.training.check_unchanged(binding)
        if b.tool_hashes()!=tools: raise ValueError('Evaluation tools changed during execution')
        if b.file_hashes(snapshot)!=tools: raise ValueError('Evaluation source snapshot changed')
        if validation and b.validation_evidence(args.validation_report,arms[0]['sha256'],binding,tools,protocol)!=validation:
            raise ValueError('Validation gate changed during test')
        # Recheck the full M4/correction ancestry, without running old acceptance or replay.
        _,_,_,_,_,final_binding=b.training.evidence(args)
        if final_binding!=binding: raise ValueError('Evaluation ancestry changed')
        if args.mode=='validation':
            passed=any(a['validation_passed'] for a in arms if a['step']>100000)
            return finish('validation_passed' if passed else 'validation_failed',source_unchanged=True)
        if args.mode=='test':
            passed=arms[0]['original']['status']=='passed' and all(r['replay']['status']=='passed' for r in results[arms[0]['name']])
            if passed and args.freeze: provenance.update(m5_status='complete',dp_v1='frozen')
            report=finish('passed' if passed else 'failed',source_unchanged=True)
            if passed and args.freeze: freeze(output,provenance,arms[0])
            return report
        if args.mode=='small' and any(row['replay']['status']!='passed' for rows in results.values() for row in rows):
            return finish('diagnostic_failed',source_unchanged=True)
        return finish('audit_passed' if args.mode=='audit' else 'diagnostic_completed',source_unchanged=True)
    except Exception as error:
        provenance.update(m5_status='incomplete',dp_v1='not_frozen')
        finish('error',error=f'{type(error).__name__}: {error}')
        raise


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode',choices=('audit','offline','small','validation','test'))
    for name in ('base-checkpoint','parent-checkpoint','v1-report','old-data-report','data-report','audit-report','output'):
        result.add_argument('--'+name,required=True)
    result.add_argument('--checkpoint',nargs='+',required=True)
    result.add_argument('--cpu-threads',type=int,choices=(6,7,8),default=6)
    result.add_argument('--noise-seed',type=int,default=0)
    result.add_argument('--validation-report')
    result.add_argument('--freeze',action='store_true',help='Publish DP_v1 only after matching full validation and test pass')
    return result


if __name__=='__main__':
    args=parser().parse_args();report=run(args)
    print(f'{report["status"]}: {Path(args.output).resolve()/"report.json"}',flush=True)
    raise SystemExit(1 if report['status'] in ('error','failed','validation_failed','diagnostic_failed') else 0)
