"""Paired parent/v3 closed loops with a required live MuJoCo observer by default."""

import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import time

from common import check_fork,check_unchanged,compute,evidence,input_arguments
import numpy as np
import torch

from correction_v3.controller import BRAKE_HORIZON_S,clearance,residual,safety_gate,stopped
from correction_v3.corpus import file_hashes
from diagnose_sampling import spacing_override
from feedingrobot.control.adapter import clip_norm
from feedingrobot.data.episodes import EpisodeWriter,input_hashes,load_episode,write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.policies.data import features,field_slices
from feedingrobot.policies.dit import ActionDiT,sample_actions
from feedingrobot.policies.evaluation import ActionChunk,summarize
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.runtime import precision
from feedingrobot.sim.events import PHASES
from feedingrobot.sim.model import ROOT
from train import check_records


def alignment_measure(reference,obs,height,tick,stable):
    r=residual(reference,obs);drift=abs(float(obs['tcp_position'][2])-height)
    good=(r['angle_deg']<np.rad2deg(.01) and r['lateral_m']<.0007
        and r['linear_speed']<.002 and r['angular_speed']<.05 and drift<.0007)
    stable=(tick if stable is None else stable) if good else None
    return r,drift,stable,bool(stable is not None and tick-stable>=200)


def select_cases(config,calibration,full):
    # calibration is independent of the 12 train episodes; only low-speed snapshots are restored.
    calibration_root=Path(calibration['calibration_directory'])
    pairs=calibration['pairs'] if full else calibration['pairs'][::2]
    corrections=[(calibration_root/'episodes'/f'{p["seed"]}_calibration',True) for p in pairs]
    validation=[]
    for p in sorted((ROOT/config['dataset']/'validation').glob('*/manifest.json')):
        m=read_json(p)
        validation.append((p.parent,bool(m['scenario'].get('recover',False))))
    if not full:
        validation=([row for row in validation if not row[1]][:3]+[row for row in validation if row[1]][:1])
    return corrections+[(p,False) for p,_ in validation]


@torch.no_grad()
def rollout(model,config,normalization,source,output,device,*,correction,viewer=True,noise_seed=0,max_ticks=None):
    m,a=load_episode(source);env=FeedingGymEnv('panda');task=env.task
    display=None;writer=None;trace=[];stable=None;completed=None;maximum_drift=0.;safety=[];abort=None
    brake_tick=None;wrist_peak=0.;owners=[];alignment_end=None;brake_commands=0
    try:
        env.reset(seed=m['seed'],options={'scenario':m['scenario']})
        snapshot=source/('handover_state.pkl' if correction else 'initial_state.pkl')
        task.set_state(pickle.loads(snapshot.read_bytes()))
        if task.state_signature()!=m['signature']: raise ValueError('Saved model signature differs')
        initial=task.get_state();initial_tick=task.tick
        if correction and initial_tick!=m['handover']['release_tick']: raise ValueError('Low-speed snapshot tick differs')
        reference=Teacher(task.robot_config,m['teacher_config'])
        reference.reset({},geometry=teacher_geometry(task))
        height=float(task.provider.observe()['policy_obs']['tcp_position'][2])
        initial_residual=residual(reference,task.provider.observe()['policy_obs']) if correction else None
        writer=EpisodeWriter(output,env.schema,task.dt,env.max_episode_s)
        (output/'initial_state.pkl').write_bytes(pickle.dumps(initial,protocol=5))
        fields=field_slices(env.schema)
        ticks=[];observations=[]
        if correction:
            # Preserve causal history across handover, including exact 20Hz action observations.
            prior_ticks=np.r_[a['observation_ticks'],a['action_ticks']]
            prior_values=np.concatenate([a['observations'],a['action_observations']])
            indices=np.argsort(prior_ticks,kind='stable')
            indices=indices[(prior_ticks[indices]<task.tick)&(prior_ticks[indices]>=task.tick-400)]
            ticks=prior_ticks[indices].tolist();observations=list(prior_values[indices])
        writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy())
        if viewer:
            from feedingrobot.sim.observer_viewer import ObserverViewer
            display=ObserverViewer(task.model,task.data)
            if display.start()['status']!='running': raise RuntimeError('MuJoCo viewer unavailable; use a desktop display or explicitly --headless')
        wall=time.monotonic();chunk=None;last_plan=-200
        generator=torch.Generator(device=device)
        limit=round(env.max_episode_s/task.dt) if max_ticks is None else min(max_ticks,round(env.max_episode_s/task.dt))
        while not task.terminated and task.tick<limit:
            phase=task.logic.phase
            if task.tick%20==0 or task.tick%50==0:
                ticks.append(task.tick);observations.append(env.observe_policy())
                ticks=ticks[-32:];observations=observations[-32:]
            if task.tick%50==0:
                obs=task.provider.observe()['policy_obs'];r=None
                if correction and completed is None and brake_tick is None:
                    r,drift,stable,done=alignment_measure(reference,obs,height,task.tick,stable)
                    alignment_end=r
                    maximum_drift=max(maximum_drift,drift)
                    if done: completed=task.tick
                    elif phase!='ACQUIRE':
                        abort='left_acquire_before_alignment';safety.append(dict(tick=task.tick,reason=abort))
                    else:
                        gate=safety_gate(task,obs)
                        if gate['stop_required']:
                            abort='alignment_safety_clearance';safety.append(dict(tick=task.tick,reason=abort,**gate))
                    if abort:
                        brake_tick=task.tick;chunk=None
                if brake_tick is not None:
                    command=np.zeros(6);owner='external_brake'
                    if stopped(obs,task.adapter.velocity): break
                    if (task.tick-brake_tick)*task.dt>BRAKE_HORIZON_S:
                        abort+=':brake_timeout';break
                elif phase!='SELECT':
                    if chunk is None or task.tick-last_plan>=200 or chunk.phase!=phase:
                        x=features(np.asarray(ticks),np.asarray(observations),np.ones(len(ticks),bool),task.tick,fields,normalization)
                        batch={k:torch.as_tensor(v,device=device)[None] for k,v in x.items()}
                        # Equal initial diffusion noise at equal physical ticks in both arms.
                        generator.manual_seed(noise_seed*1000003+task.tick//50)
                        with precision(device): actions=sample_actions(model,config,batch,normalization,generator)
                        chunk=ActionChunk(task.tick,phase,actions[0].float().cpu().numpy());last_plan=task.tick
                    action=chunk.current(task.tick,phase)
                    if action is None or not np.isfinite(action).all():
                        task._terminate('invalid_command');break
                    command=np.r_[clip_norm(action[:3],task.robot_config['linear_speed_limit']),
                        clip_norm(action[3:],task.robot_config['angular_speed_limit'])]
                    owner='model'
                else:
                    command=None
                if command is not None:
                    task.adapter.set_twist(command,task.data.time,(task.tick+50)*task.dt)
                    writer.command(task.tick,command,(task.tick+50)*task.dt)
                    proposal=action if owner=='model' else command
                    writer.action(task.tick,PHASES.index(phase),proposal,command,env.observe_policy(),task.tick+50)
                    writer.actions[-1]['valid']=False;owners.append(owner)
                    brake_commands+=int(owner=='external_brake')
                    trace.append(dict(tick=task.tick,phase=phase,owner=owner,command=command.tolist(),
                        predicted=proposal.tolist(),residual=r,height_drift_m=abs(float(obs['tcp_position'][2])-height) if correction else None))
            physical=task.step_physics();wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
            if task.logic.phase!=phase and not task.terminated:
                task.adapter.stop(hold_reference=True);writer.command(task.tick,hold_reference=True,after_physics=True)
                writer.interrupt(task.tick);chunk=None
            writer.record_physics(task,physical)
            if correction and completed is None and brake_tick is None:
                drift=abs(float(physical['tcp_position'][2])-height);maximum_drift=max(maximum_drift,drift)
                if not task.terminated and task.tick<limit and (drift>=.0007 or clearance(task)<=.0005):
                    abort='alignment_height_drift' if drift>=.0007 else 'alignment_clearance'
                    brake_tick=task.tick;chunk=None;safety.append(dict(tick=task.tick,reason=abort,height_drift_m=drift))
                    # Physical-boundary safety brake preserves measured/reference velocity.
                    task.adapter.set_twist(np.zeros(6),task.data.time,(task.tick+50)*task.dt)
                    writer.command(task.tick,np.zeros(6),(task.tick+50)*task.dt)
                    brake_commands+=1
                    writer.interrupt(task.tick)
            if task.tick%20==0 or task.terminated:
                valid=task.failure_reason!='nonfinite_state'
                writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy() if valid else observations[-1],valid)
                if display:
                    # Pace the observer run so the user can watch the physical motion.
                    time.sleep(max(0.,(task.tick-initial_tick)*task.dt-(time.monotonic()-wall)))
                    display.update(phase=task.logic.phase,result=f'{owner if trace else "running"}; aligned={completed is not None}')
                    if display.report['status']!='running': raise RuntimeError('MuJoCo viewer closed during validation')
        valid=task.failure_reason!='nonfinite_state'
        if writer.obs_ticks[-1]!=task.tick:
            writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy() if valid else observations[-1],valid)
        truncated=not task.terminated
        if truncated: task.logic.emit('time_limit',task.data.time)
        task.adapter.stop(hold_reference=True);writer.interrupt(task.tick);writer.command(task.tick,hold_reference=True)
        events=deepcopy(task.logic.events);new_events=[e for e in events if e['time']>=initial_tick*task.dt]
        names={e['name'] for e in new_events};entered={'SELECT'}|{e['phase'] for e in events if e['name']=='phase'}
        recovered=any(e['name']=='phase' and e.get('previous')=='RECOVER' and e['phase']=='WAIT_READY' for e in new_events)
        visualization=display.close() if display else dict(status='disabled')
        if viewer and (not visualization.get('displayed_frames') or visualization.get('reason')!='episode_finished'):
            raise RuntimeError('MuJoCo did not visibly complete the episode')
        result=dict(robot_id='panda',seed=m['seed'],scenario=m['scenario'],split='validation_diagnostic',
            source_episode=str(source),signature=m['signature'],input_hashes=input_hashes(),initial_tick=initial_tick,
            teacher_in_execution=False,training_export=False,correction_case=correction,category=m.get('category'),
            success=bool(task.logic.success and (recovered or not m['scenario'].get('recover',False)) and not abort),
            physical_success=bool(task.logic.success),pickup='pickup' in names,delivery='delivery' in names,
            transport_completed=any(e['name']=='phase' and e.get('previous')=='TRANSPORT' and e['phase']=='WAIT_READY' for e in new_events),
            recover=bool(m['scenario'].get('recover',False)),recovery_completed=recovered,entered=sorted(entered),
            events=events,truncated=truncated,failure_reason=task.failure_reason or abort or ('time_limit' if truncated else None),
            simulated_s=task.tick*task.dt,max_episode_s=env.max_episode_s,wall_s=time.monotonic()-wall,
            solver_iterations=int(task.model.opt.iterations),solver_tolerance=float(task.model.opt.tolerance),
            contact_peak_n=task.monitor.peak_n,wrist_peak_n=wrist_peak,
            model_alignment=dict(initial=initial_residual,completed_tick=completed,success=completed is not None,
                final=alignment_end,
                maximum_height_drift_m=maximum_drift if correction else None),
            external_braking=dict(prefix_commands=int(np.sum((a['action_owner']=='external_brake')&(a['action_ticks']<initial_tick))) if correction else 0,
                prefix_braking_is_model=False,interventions=safety,commands=brake_commands,
                autonomous_high_speed_braking_proven=False),visualization=visualization)
        np.save(output/'action_owner.npy',np.asarray(owners,dtype='U20'))
        write_json(output/'trace.json',trace)
        # Replay checks engine truth; task-level diagnostic failures remain separate in result.json.
        writer.finish(dict(result,success=bool(task.logic.success),failure_reason=task.failure_reason,
            diagnostic_failure_reason=result['failure_reason']))
        return result
    finally:
        if display: display.close()
        env.close()


def run(args):
    base,parent,_,data,calibration,binding=evidence(args)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=base['config'];normalization=base['normalization']
    check_fork(checkpoint,binding,config,normalization)
    check_records(Path(args.checkpoint).resolve().parent/'metrics.jsonl',checkpoint['step'])
    device,hardware=compute(config,args.cpu_threads)
    calibration=dict(calibration,calibration_directory=str(Path(data['calibration_report']).resolve().parent))
    cases=select_cases(config,calibration,args.mode=='validation')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'tool_snapshot';snapshot.mkdir()
    for name in binding['entry_tools']: shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    provenance=dict(schema_version=3,diagnostic=True,mode=args.mode,binding=binding,hardware=hardware,
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256(args.checkpoint),
        training_step=checkpoint['step'],weights='ema',noise_seed=args.noise_seed,paired_initial_noise=True,
        sampler='leading',viewer_required=not args.headless,full_validation_run=args.mode=='validation',
        formal_test_run=False,m5_status='incomplete',dp_v1='not_frozen',teacher_in_execution=False,
        limitation='Correction snapshots follow external braking; high-speed autonomous braking is not tested or claimed.',
        cases=[dict(source=str(p),correction=c) for p,c in cases])
    write_json(output/'provenance.json',provenance)
    model=ActionDiT(config).to(device).eval().requires_grad_(False)
    results={'parent':[],'v3':[]}
    with spacing_override('leading'):
        for arm,weights in (('parent',parent['ema']),('v3',checkpoint['ema'])):
            model.load_state_dict(weights)
            for source,correction in cases:
                directory=output/arm/source.name
                row=rollout(model,config,normalization,source,directory,device,correction=correction,
                    viewer=not args.headless,noise_seed=args.noise_seed)
                row['replay']=replay_episode(directory)
                write_json(directory/'result.json',row);results[arm].append(row)
                print(f'{arm} seed={row["seed"]} aligned={row["model_alignment"]["success"]} pickup={row["pickup"]} transport={row["transport_completed"]} delivery={row["delivery"]} success={row["success"]}',flush=True)
    regressions=[]
    for old,new in zip(results['parent'],results['v3']):
        for flag in ('pickup','transport_completed','delivery','success','recovery_completed'):
            if old[flag] and not new[flag]: regressions.append(dict(seed=new['seed'],metric=flag))
    summaries={arm:summarize([r for r in rows if not r['correction_case']],config) for arm,rows in results.items()}
    correction_summary={arm:dict(attempts=sum(r['correction_case'] for r in rows),
        aligned=sum(r['correction_case'] and r['model_alignment']['success'] for r in rows),
        complete_success=sum(r['correction_case'] and r['success'] for r in rows),
        safety_interventions=sum(len(r['external_braking']['interventions']) for r in rows)) for arm,rows in results.items()}
    candidate=correction_summary['v3']
    alignment_passed=(candidate['aligned']==candidate['attempts'] and candidate['complete_success']==candidate['attempts']
        and candidate['safety_interventions']==0)
    full_passed=args.mode=='validation' and alignment_passed and not regressions and summaries['v3']['status']=='passed'
    evidence(args);check_unchanged(binding)
    report=dict(provenance,status=('validation_passed' if full_passed else 'validation_failed') if args.mode=='validation' else 'diagnostic_completed',
        source_unchanged=True,validation=summaries,validation_acceptance_passed=full_passed,
        regressions=regressions,regression_free=not regressions,
        correction=correction_summary,
        evidence_sha256=file_hashes(output))
    write_json(output/'report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('small','validation'));input_arguments(parser)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--noise-seed',type=int,default=0)
    parser.add_argument('--headless',action='store_true',help='Explicitly disable the live MuJoCo observer')
    args=parser.parse_args();report=run(args)
    print(Path(args.output).resolve()/'report.json',flush=True)
    raise SystemExit(1 if report['status']=='validation_failed' else 0)
