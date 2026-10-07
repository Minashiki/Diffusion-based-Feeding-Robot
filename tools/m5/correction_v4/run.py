"""Reindex, paired calibration, collection and zero-update audit; never train."""

import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import correction_v4
import numpy as np
import torch

from correction_v4.controller import CELLS,CALIBRATION_SEEDS,VERSION
from correction_v4.corpus import MixedV4,PickupWindows
from correction_v4.diagnostics import replay_with_pickup
from correction_v4.rollout import episode
from correction_v3.corpus import audit_corpus,file_hashes,verify_files
from correction_v3.run import ancestry,unchanged,calibration_check
from correction_v2.run import paired,scenario_for,success
from feedingrobot.data.episodes import load_episode,write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.dit import ActionDiT,noise_loss,schedulers
from feedingrobot.policies.runtime import setup,to_device
from feedingrobot.sim.model import ROOT


def tools_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def cell_key(cell):
    return f"{cell['category']}:{cell['quadrant'][0]}:{cell['quadrant'][1]}:{cell['velocity']}"


def seed_check(seeds,metadata,v1):
    used={m['seed'] for m in metadata}|{m['seed'] for m in v1['accepted']+v1['rejected']}
    def extract(value):
        if isinstance(value,dict):
            if 'seed' in value: used.add(value['seed'])
            for item in value.values(): extract(item)
        elif isinstance(value,list):
            for item in value: extract(item)
    for pattern in ('**/report.json','**/manifest.json'):
        for path in (ROOT/'outputs/single_bean/v1/m5/dit').glob(pattern): extract(read_json(path))
    if used.intersection(seeds): raise ValueError('Seed overlaps historical evidence')


def inputs(args):
    base,parent,parent_audit,v1,metadata,config,legacy=ancestry(args)
    old=audit_corpus(args.data_report,legacy,base['normalization'],ROOT/base['config']['dataset'])
    calibration_check(old['calibration_report'],legacy)
    if sha256(old['calibration_report'])!=old['calibration_report_sha256']: raise ValueError('Old calibration changed')
    runtime=deepcopy(base['config']);runtime['training'].update(cpu_budget=6,torch_threads=6,workers=0)
    _,hardware=setup(runtime,'cpu')
    binding=dict(legacy=legacy,v4_tools=tools_hashes(),old_report_sha256=sha256(args.data_report),
        old_dataset_sha256=old['dataset_sha256'],normalization=base['normalization'],cpu_budget=6)
    return base,parent,v1,metadata,config,old,binding,hardware


def start(args,binding,hardware):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'tool_snapshot';snapshot.mkdir()
    for name in binding['v4_tools']: shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    provenance=dict(schema_version=4,mode=args.mode,teacher_version=VERSION,binding=binding,
        old_report=str(Path(args.data_report).resolve()),hardware=hardware,optimizer_created=False,
        optimizer_updates=0,model_policy_executed=False,model_executed=False,formal_test_run=False,full_validation_run=False,
        m5_status='incomplete',dp_v1='not_frozen',normalization=binding['normalization'])
    write_json(output/'provenance.json',provenance)
    return output,provenance


def finish(output,provenance,**result):
    binding=provenance['binding'];unchanged(binding['legacy'])
    if tools_hashes()!=binding['v4_tools']: raise ValueError('v4 tools changed during execution')
    verify_files(Path(read_json(provenance['old_report'])['dataset']),binding['old_dataset_sha256'])
    if sha256(provenance['old_report'])!=binding['old_report_sha256']: raise ValueError('Old report changed')
    report=dict(provenance,**result,source_unchanged=True,evidence_sha256=file_hashes(output))
    write_json(output/'report.json',report)
    return report


def coverage(m,a,directory):
    boundaries=np.load(Path(directory)/'alignment_boundaries.npy',allow_pickle=False)
    if not len(boundaries): raise ValueError('Missing physical alignment evidence')
    h=m['handover'];wait=h['stable_start_tick'];complete=m['corrected_tick']
    final=boundaries[(boundaries[:,0]>=wait)&(boundaries[:,0]<=complete)]
    qualified=(np.linalg.norm(final[:,1:3],axis=1)<.0007)&(final[:,3]<np.rad2deg(.01))&(final[:,4]<.002)&(final[:,5]<.05)
    if (complete-wait<200 or not np.all(final[:,6]) or not np.all(qualified)
            or not np.all(final[:,7]==wait) or not np.array_equal(final[:,0],np.arange(wait,complete+1))):
        raise ValueError('Not continuously stable for 200ms')
    if np.any(a['action_mask']&(a['action_owner']=='path_teacher')&(a['action_ticks']<wait+200)):
        raise ValueError('Path teacher started before stable qualification')
    mask=a['action_mask']&(a['action_owner']=='alignment_teacher')
    age=a['action_ticks'][mask]-wait
    bins=[int(np.count_nonzero((age>=i)&(age<i+50))) for i in (0,50,100,150)]
    if not all(bins): raise ValueError('Missing stable-wait action age bucket')
    sign=m['cell']['quadrant'][0]
    critical=(boundaries[:,1]*sign>=.00045)&(boundaries[:,1]*sign<=.00065)
    critical_actions=np.intersect1d(a['action_ticks'][mask],boundaries[critical,0])
    if not critical.any(): raise ValueError('Missing physical signed X critical crossing')
    return dict(stable_start_tick=wait,stable_duration_ms=complete-wait,wait_action_bins=bins,
        x_critical_boundaries=int(critical.sum()),x_critical_action_ticks=critical_actions.astype(int).tolist(),
        stability_resets=int(boundaries[:,8].sum()))


def aggregate_coverage(rows):
    counts={f'{cell["category"]}:x{cell["quadrant"][0]:+d}':0 for cell in CELLS}
    for row in rows:
        if row.get('coverage') and row['coverage']['x_critical_action_ticks']:
            cell=row['cell'];counts[f'{cell["category"]}:x{cell["quadrant"][0]:+d}']+=1
    return dict(critical_independent_episodes=counts,minimum_required=2,
        passed=all(count>=2 for count in counts.values()))


def worker_setup():
    # Five physics workers plus the parent fit the six-CPU budget. No policy
    # model is constructed in a worker; all native numerical pools are one.
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    return dict(torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)))


def paired_case(seed,cell,config,root,*,split='calibration'):
    root=Path(root);scenario=scenario_for(seed,config)
    baseline_dir=root/f'{seed}_baseline'
    baseline=episode(seed,cell,config,scenario,baseline_dir,baseline=True)
    baseline_replay=replay_episode(baseline_dir)
    directory=root/f'{seed}_candidate'
    case=episode(seed,cell,config,scenario,directory,split=split) if success(baseline) else None
    replay=replay_episode(directory) if case else None;detail=None;reason=None
    try:
        if not case or not paired(baseline,case): raise ValueError('Baseline or paired initial state failed')
        m,a=load_episode(directory);detail=check_case(m,a,directory,train=split=='train')
    except ValueError as error: reason=str(error)
    passed=bool(detail and replay['status']=='passed' and baseline_replay['status']=='passed')
    return dict(seed=seed,cell=cell,status='passed' if passed else 'failed',reason=reason,
        baseline={k:v for k,v in baseline.items() if k!='files_sha256'},baseline_replay=baseline_replay,
        case={k:v for k,v in case.items() if k!='files_sha256'} if case else None,replay=replay,coverage=detail,
        worker=dict(torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            cpu_affinity=sorted(os.sched_getaffinity(0))))


def collect_cell(index,cell,config,root):
    rejected=[]
    for seed in range(800101+2*index,800103+2*index):
        row=paired_case(seed,cell,config,root,split='train')
        if row['status']=='passed': return row,rejected
        rejected.append(row)
    return None,rejected


def check_case(m,a,directory,*,train=False):
    if not (success(m) and m['qualified_correction'] and m['accepted_normal'] and not m['abort_reason']
            and not m['truncated'] and m['source_unchanged'] and m['teacher_version']==VERSION
            and m['prefix_source']=='controlled_perturbation' and m['max_episode_s']==60.
            and m['simulated_s']<=60. and not m['scenario'].get('recover',False)
            and not m['model_policy_executed'] and (not train or m['split']=='train')):
        raise ValueError('Requires complete controlled-perturbation teacher episode')
    if (m['cell'] not in CELLS or m['category']!=m['cell']['category']
            or m['group_id']!=f"panda:descent_correction_v4:{m['seed']}"):
        raise ValueError('Cell/group identity differs')
    mask=np.asarray(a['action_mask']);ticks=np.asarray(a['action_ticks']);owners=np.asarray(a['action_owner'])
    if np.any(mask&~np.isin(owners,('alignment_teacher','path_teacher'))): raise ValueError('Non-teacher label')
    if np.any(mask&(ticks<m['handover']['release_tick'])): raise ValueError('Prefix labels leaked')
    if np.any(a['action_end_ticks'][mask]!=ticks[mask]+50): raise ValueError('Incomplete label')
    for name in ('physics','observations','action_observations','actions','proposals'):
        if not np.isfinite(a[name]).all(): raise ValueError('Nonfinite label evidence')
    h=m['handover'];r=h['release_residual'];cell=m['cell']
    error=np.asarray(h['release_error']);twist=np.asarray(h['release_twist'])
    radial=float(error@twist[:2]/np.linalg.norm(error));sign=-1 if cell['velocity']=='toward' else 1
    stopped=(r['linear_speed']<.002 and r['angular_speed']<.02
        and np.linalg.norm(h['release_velocity'][:3])<.002 and np.linalg.norm(h['release_velocity'][3:])<.02)
    moving=(.002<r['downward_speed']<=.005 and r['linear_speed']<=.005
        and np.linalg.norm(h['release_velocity'][:3])<=.0045 and np.linalg.norm(h['release_velocity'][3:])<.05)
    if not (2<=r['angle_deg']<=4 and .001<=r['lateral_m']<=.002 and r['angular_speed']<.05
            and np.all(error*np.asarray(cell['quadrant'])>0) and .0005<=sign*radial<=.0015
            and (stopped if m['category'].endswith('stopped') else moving)
            and m['alignment_max_drift_m']<.0007 and not m['safety_checks']
            and m['geometry_checks'] and all(g['passed'] for g in m['geometry_checks'])):
        raise ValueError('Unqualified release or physical safety')
    if m['category'].startswith('p1') and not (h['pulse_end_tick']<h['trigger_tick']<h['release_tick']
            and .045<=h['trigger_residual']['downward_speed']<=.051):
        raise ValueError('P1 requires actual high-speed braking')
    return coverage(m,a,directory)


def verify_report(path,binding,status):
    path=Path(path).resolve();r=read_json(path)
    if r.get('binding')!=binding or r.get('status')!=status or not r.get('source_unchanged'):
        raise ValueError('Unpassed or mismatched v4 report')
    actual=file_hashes(path.parent);actual.pop(path.name,None)
    if actual!=r['evidence_sha256']: raise ValueError('Changed v4 evidence')
    return r


def reindex(args):
    base,_,_,_,_,old,binding,hardware=inputs(args);output,p=start(args,binding,hardware)
    windows=PickupWindows(old['dataset'],'train',base['config']['model']['horizon'],base['normalization'])
    summary=windows.summary()
    if ([summary['pools'][pool]['windows'] for pool in ('alignment','transition','aligned')]!=[119,5419,2980]
            or summary['clipped_windows']!=84 or summary['unique_actions_by_stage']['pickup_hold']!=244):
        raise ValueError('Old corpus reindex baseline differs')
    return finish(output,p,status='reindex_passed',data=summary,model_executed=False)


def calibrate(args):
    base,_,v1,metadata,config,old,binding,hardware=inputs(args)
    verify_report(args.reindex_report,binding,'reindex_passed')
    seed_check(CALIBRATION_SEEDS,metadata,v1)
    output,p=start(args,binding,hardware);pairs=[]
    try:
        with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn'),initializer=worker_setup) as pool:
            futures={pool.submit(paired_case,seed,cell,config,output/'episodes'):i
                for i,(seed,cell) in enumerate(zip(CALIBRATION_SEEDS,CELLS))}
            for future in as_completed(futures):
                row=future.result();pairs.append(row);pairs.sort(key=lambda r:r['seed'])
                print(f'calibration {len(pairs)}/32 {cell_key(row["cell"])} seed={row["seed"]} '
                    f'passed={row["status"]=="passed"} reason={row["reason"]}',flush=True)
                write_json(output/'progress.json',dict(pairs=pairs))
                if row['status']!='passed':
                    for pending in futures: pending.cancel()
                    pool.shutdown(wait=True,cancel_futures=True)
                    break
    except Exception as error:
        finish(output,p,status='error',pairs=pairs,error=f'{type(error).__name__}: {error}');raise
    batch=aggregate_coverage(pairs)
    return finish(output,p,status='passed' if len(pairs)==32 and all(r['status']=='passed' for r in pairs) and batch['passed'] else 'failed',
        pairs=pairs,batch_coverage=batch,physics_workers=5,worker_torch_threads=1,
        reindex_report=str(Path(args.reindex_report).resolve()),reindex_sha256=sha256(args.reindex_report))


def collect(args):
    base,_,v1,metadata,config,old,binding,hardware=inputs(args)
    cal=verify_report(args.calibration_report,binding,'passed')
    if (len(cal['pairs'])!=32 or [r['cell'] for r in cal['pairs']]!=list(CELLS)
            or [r['seed'] for r in cal['pairs']]!=list(CALIBRATION_SEEDS)
            or not aggregate_coverage(cal['pairs'])['passed']): raise ValueError('Calibration coverage differs')
    verify_report(cal['reindex_report'],binding,'reindex_passed')
    if sha256(cal['reindex_report'])!=cal['reindex_sha256']: raise ValueError('Reindex evidence changed')
    for row in cal['pairs']:
        if row['status']!='passed': raise ValueError('Unpassed calibration cell')
        directory=Path(args.calibration_report).resolve().parent/'episodes'/f"{row['seed']}_candidate"
        m,a=load_episode(directory);check_case(m,a,directory)
    seed_check(range(800101,800165),metadata,v1)
    output,p=start(args,binding,hardware);data=output/'data';(data/'train').mkdir(parents=True)
    shutil.copyfile(Path(old['dataset'])/'normalization.json',data/'normalization.json')
    for row in old['accepted']:
        shutil.copytree(Path(old['dataset'])/row['path'],data/row['path'])
    accepted=[];rejected=[];counts={cell_key(c):0 for c in CELLS}
    try:
        with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn'),initializer=worker_setup) as pool:
            futures={pool.submit(collect_cell,i,cell,config,output/'attempts'):i for i,cell in enumerate(CELLS)}
            for future in as_completed(futures):
                row,failures=future.result();rejected.extend(failures)
                for failure in failures:
                    print(f'rejected {cell_key(failure["cell"])} seed={failure["seed"]} reason={failure["reason"]}',flush=True)
                if row:
                    seed=row['seed'];cell=row['cell'];directory=output/'attempts'/f'{seed}_candidate'
                    destination=data/'train'/f'correction_{seed}';shutil.move(str(directory),str(destination))
                    row['replay']['episode']=str(destination)
                    accepted.append(dict(seed=seed,cell=cell,path=str(destination.relative_to(data)),
                        replay=row['replay'],baseline_replay=row['baseline_replay'],coverage=row['coverage'],worker=row['worker']))
                    accepted.sort(key=lambda r:r['seed']);counts[cell_key(cell)]=1
                    print(f'accepted {len(accepted)}/32 {cell_key(cell)} seed={seed}',flush=True)
                write_json(output/'progress.json',dict(accepted=accepted,rejected=rejected,counts=counts))
    except Exception as error:
        finish(output,p,status='error',accepted=accepted,rejected=rejected,counts=counts,error=f'{type(error).__name__}: {error}');raise
    windows=PickupWindows(data,'train',base['config']['model']['horizon'],base['normalization'])
    batch=aggregate_coverage(accepted)
    status='quota_incomplete' if len(accepted)!=32 else 'ready_experimental' if batch['passed'] else 'coverage_incomplete'
    return finish(output,p,status=status,dataset=str(data),batch_coverage=batch,physics_workers=5,worker_torch_threads=1,
        accepted=accepted,rejected=rejected,counts=counts,old_accepted=old['accepted'],data=windows.summary(),
        dataset_sha256=file_hashes(data),calibration_report=str(Path(args.calibration_report).resolve()),
        calibration_sha256=sha256(args.calibration_report))


def audit(args):
    base,parent,_,_,_,old,binding,hardware=inputs(args)
    report=verify_report(args.collection_report,binding,'ready_experimental')
    cal=verify_report(report['calibration_report'],binding,'passed')
    if sha256(report['calibration_report'])!=report['calibration_sha256'] or len(cal['pairs'])!=32:
        raise ValueError('Calibration changed')
    data=Path(report['dataset']);verify_files(data,report['dataset_sha256'])
    expected={r['path'] for r in old['accepted']+report['accepted']}
    actual={str(p.parent.relative_to(data)) for p in data.glob('*/*/manifest.json')}
    if (expected!=actual or len(actual)!=44 or report['counts']!={cell_key(c):1 for c in CELLS}
            or not aggregate_coverage(report['accepted'])['passed']):
        raise ValueError('Merged corpus quota or episode inventory differs')
    seeds=set();groups=set()
    for row in old['accepted']:
        if file_hashes(data/row['path'])!=file_hashes(Path(old['dataset'])/row['path']): raise ValueError('Old episode copy changed')
    for path in data.glob('*/*/manifest.json'):
        m=read_json(path)
        if m['seed'] in seeds or m['group_id'] in groups: raise ValueError('Duplicate seed/group')
        seeds.add(m['seed']);groups.add(m['group_id'])
    for row in report['accepted']:
        m,a=load_episode(data/row['path']);check_case(m,a,data/row['path'],train=True)
        index=CELLS.index(m['cell'])
        if (row['seed']!=m['seed'] or row['cell']!=m['cell']
                or m['seed'] not in range(800101+2*index,800103+2*index)):
            raise ValueError('Collection cell/seed binding differs')
        if row['replay']['status']!='passed' or row['baseline_replay']['status']!='passed': raise ValueError('Unpassed replay')
    if len({cell_key(r['cell']) for r in report['accepted']})!=32: raise ValueError('Repeated collection cell')
    if sha256(data/'normalization.json')!=sha256(Path(old['dataset'])/'normalization.json'): raise ValueError('Normalization changed')
    mixed=MixedV4(base['config'],base['normalization'],data)
    model=ActionDiT(base['config']).eval().requires_grad_(False);model.load_state_dict(parent['ema']);ema=deepcopy(model)
    torch.manual_seed(base['config']['training']['seed']);rng=np.random.default_rng(base['config']['training']['seed'])
    scheduler,_=schedulers(base['config'])
    with torch.no_grad(): loss=noise_loss(model,scheduler,to_device(mixed.batch(rng,64),torch.device('cpu')))
    exact=all(torch.equal(v,parent['ema'][k]) for k,v in model.state_dict().items())
    ema_exact=all(torch.equal(v,parent['ema'][k]) for k,v in ema.state_dict().items())
    if not torch.isfinite(loss) or not exact or not ema_exact: raise ValueError('Failed zero-update audit')
    verify_files(data,report['dataset_sha256']);verify_report(args.collection_report,binding,'ready_experimental')
    output,p=start(args,binding,hardware)
    return finish(output,p,status='audit_passed',step=100000,precision='fp32',forward_loss=float(loss),
        model_exact=exact,ema_exact=ema_exact,model_executed=True,data=mixed.extra.summary(),episodes=44,
        collection_sha256=sha256(args.collection_report),optimizer_state_loaded=False,rng_state_loaded=False)


def pickup_audit(args):
    _,_,_,_,_,_,binding,hardware=inputs(args);output,p=start(args,binding,hardware)
    hashes=file_hashes(args.episode)
    try: result=replay_with_pickup(args.episode,output)
    except Exception as error:
        finish(output,p,status='error',episode=str(Path(args.episode).resolve()),error=str(error));raise
    verify_files(args.episode,hashes)
    return finish(output,p,status='pickup_replay_passed',episode=str(Path(args.episode).resolve()),
        episode_sha256=hashes,**result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('reindex','calibrate','collect','audit','pickup-audit'))
    for name in ('base-checkpoint','checkpoint','v1-report','data-report','output'): parser.add_argument('--'+name,required=True)
    for name in ('reindex-report','calibration-report','collection-report','episode'): parser.add_argument('--'+name)
    args=parser.parse_args()
    required={'calibrate':'reindex_report','collect':'calibration_report','audit':'collection_report','pickup-audit':'episode'}
    if args.mode in required and not getattr(args,required[args.mode]): parser.error(f'{required[args.mode]} is required')
    result=globals()[args.mode.replace('-','_')](args)
    print(f"{result['status']}: {args.output}",flush=True)
    if result['status'] in ('failed','error','quota_incomplete','coverage_incomplete'): raise SystemExit(1)


if __name__=='__main__': main()
