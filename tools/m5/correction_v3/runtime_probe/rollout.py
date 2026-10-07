"""Local diagnostic rollout derived from the unchanged v3 evaluation entry."""

from copy import deepcopy
from pathlib import Path
import pickle
import time

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import common  # Preserve the original import-time native thread limits.

import numpy as np
import torch

from correction_v3.controller import BRAKE_HORIZON_S,clearance,residual,safety_gate,stopped
from diffusers import DDIMScheduler
from feedingrobot.control.adapter import clip_norm
from feedingrobot.data.episodes import EpisodeWriter,input_hashes,load_episode,write_json
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.policies.data import features,field_slices
from feedingrobot.policies.dit import schedulers
from feedingrobot.policies.evaluation import ActionChunk
from feedingrobot.policies.runtime import precision
from feedingrobot.sim.events import PHASES
from evaluate import alignment_measure


VARIANTS=('baseline200','replan50','holdgate200')


def period(variant):
    if variant not in VARIANTS: raise ValueError('Unknown probe variant')
    return 50 if variant=='replan50' else 200


def start_viewer(display,output):
    report=display.start();deadline=time.monotonic()+1.
    # The spawned viewer sets ready before its queue feeder publishes "running".
    while report['status']=='starting' and time.monotonic()<deadline:
        time.sleep(.01);display._messages();report=display.report.copy()
    if report['status']!='running':
        write_json(output/'viewer_error.json',report)
        raise RuntimeError(f'MuJoCo viewer unavailable: {report}; use a desktop display or explicitly --headless')


def hold_gate(command,base_rotation,active):
    applied=command.copy();world=base_rotation@command[:3]
    changed=bool(active and world[2]<0.)
    if changed:
        world[2]=0.;applied[:3]=base_rotation.T@world
    return applied,changed


def leading_scheduler(config):
    _,original=schedulers(config)
    return DDIMScheduler.from_config(original.config,timestep_spacing='leading')


@torch.no_grad()
def sample_actions(model,config,batch,normalization,generator=None):
    scheduler=leading_scheduler(config)
    scheduler.set_timesteps(config['diffusion']['inference_steps'],device=batch['states'].device)
    condition=model.encoder(batch)
    actions=torch.randn((len(batch['states']),model.horizon,6),device=batch['states'].device,generator=generator)
    for step in scheduler.timesteps:
        noise=model.denoiser(actions,step.expand(len(actions)),condition)
        actions=scheduler.step(noise,step,actions,eta=0).prev_sample
    mean=torch.as_tensor(normalization['action_mean'],device=actions.device)
    scale=torch.as_tensor(normalization['action_std'],device=actions.device)
    return actions*scale+mean


def distribution(values):
    if not len(values): return dict(count=0,p50=None,p95=None,maximum=None)
    return dict(count=len(values),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),maximum=float(np.max(values)))


def motion_changes(trace):
    result={}
    for field in ('predicted','pre_gate_command','command','measured_twist_world'):
        buckets={'ordinary':[],'boundary':[]}
        for old,new in zip(trace,trace[1:]):
            if new['tick']-old['tick']!=50: continue
            boundary=(old['phase']!=new['phase'] or old['owner']!=new['owner']
                or old['gate_active']!=new['gate_active'] or new['owner']=='external_brake')
            delta=np.asarray(new[field])-old[field]
            buckets['boundary' if boundary else 'ordinary'].append([np.linalg.norm(delta[:3]),np.linalg.norm(delta[3:])])
        result[field]={name:dict(linear=distribution([d[0] for d in rows]),angular=distribution([d[1] for d in rows])) for name,rows in buckets.items()}
    result['gate_modification']=dict(linear=distribution([np.linalg.norm(np.asarray(t['command'])[:3]-t['pre_gate_command'][:3]) for t in trace]),
        angular=distribution([np.linalg.norm(np.asarray(t['command'])[3:]-t['pre_gate_command'][3:]) for t in trace]))
    return result


@torch.no_grad()
def rollout(model,config,normalization,source,output,device,*,correction,viewer=True,noise_seed=0,max_ticks=None,variant='baseline200'):
    replan_ms=period(variant);inferences=[];gate_count=0;first_good=None;longest_stable=0;first_down=None
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
            start_viewer(display,output)
        wall=time.monotonic();chunk=None;last_plan=-200
        generator=torch.Generator(device=device)
        limit=round(env.max_episode_s/task.dt) if max_ticks is None else min(max_ticks,round(env.max_episode_s/task.dt))
        while not task.terminated and task.tick<limit:
            phase=task.logic.phase
            if task.tick%20==0 or task.tick%50==0:
                ticks.append(task.tick);observations.append(env.observe_policy())
                ticks=ticks[-32:];observations=observations[-32:]
            if task.tick%50==0:
                obs=task.provider.observe()['policy_obs'];r=None;changed=False;gate_active=False;pre_gate_command=None
                if correction and completed is None and brake_tick is None:
                    r,drift,stable,done=alignment_measure(reference,obs,height,task.tick,stable)
                    alignment_end=r
                    if stable is not None:
                        if first_good is None: first_good=task.tick
                        longest_stable=max(longest_stable,task.tick-stable)
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
                    if chunk is None or task.tick-last_plan>=replan_ms or chunk.phase!=phase:
                        x=features(np.asarray(ticks),np.asarray(observations),np.ones(len(ticks),bool),task.tick,fields,normalization)
                        batch={k:torch.as_tensor(v,device=device)[None] for k,v in x.items()}
                        # Equal initial diffusion noise at equal physical ticks in both arms.
                        generator.manual_seed(noise_seed*1000003+task.tick//50)
                        inference_start=time.monotonic()
                        with precision(device): actions=sample_actions(model,config,batch,normalization,generator)
                        predicted_actions=actions[0].float().cpu().numpy()
                        inferences.append(dict(tick=task.tick,seconds=time.monotonic()-inference_start))
                        chunk=ActionChunk(task.tick,phase,predicted_actions);last_plan=task.tick
                    action=chunk.current(task.tick,phase)
                    if action is None or not np.isfinite(action).all():
                        task._terminate('invalid_command');break
                    command=np.r_[clip_norm(action[:3],task.robot_config['linear_speed_limit']),
                        clip_norm(action[3:],task.robot_config['angular_speed_limit'])]
                    pre_gate_command=command.copy()
                    gate_active=bool(variant=='holdgate200' and correction and completed is None)
                    command,changed=hold_gate(command,task.data.site_xmat[task.index.base].reshape(3,3),gate_active)
                    gate_count+=int(changed)
                    owner='model_hold_gate' if changed else 'model'
                    if first_down is None and (task.data.site_xmat[task.index.base].reshape(3,3)@command[:3])[2]<0.:
                        first_down=task.tick
                else:
                    command=None
                if command is not None:
                    task.adapter.set_twist(command,task.data.time,(task.tick+50)*task.dt)
                    writer.command(task.tick,command,(task.tick+50)*task.dt)
                    proposal=action if owner in ('model','model_hold_gate') else command
                    if pre_gate_command is None: pre_gate_command=command.copy()
                    writer.action(task.tick,PHASES.index(phase),proposal,command,env.observe_policy(),task.tick+50)
                    writer.actions[-1]['valid']=False;owners.append(owner)
                    brake_commands+=int(owner=='external_brake')
                    trace.append(dict(tick=task.tick,phase=phase,owner=owner,command=command.tolist(),
                        predicted=proposal.tolist(),pre_gate_command=pre_gate_command.tolist(),
                        gate_active=gate_active,gate_modified=changed,stable_since=stable,stable_ms=0 if stable is None else task.tick-stable,
                        chunk_issued_tick=None if chunk is None else chunk.issued_tick,
                        chunk_action_index=None if chunk is None else (task.tick-chunk.issued_tick)//50,
                        measured_twist_world=np.asarray(obs['tcp_twist_world']).tolist(),
                        world_vz_before=float((task.data.site_xmat[task.index.base].reshape(3,3)@pre_gate_command[:3])[2]),
                        world_vz_after=float((task.data.site_xmat[task.index.base].reshape(3,3)@command[:3])[2]),
                        residual=r,height_drift_m=abs(float(obs['tcp_position'][2])-height) if correction else None))
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
            variant=variant,replan_ms=replan_ms,
            hold_gate=dict(applications=gate_count,assisted_alignment_pass=completed is not None and gate_count>0),
            alignment_diagnostics=dict(conditions_met=completed is not None,first_good_tick=first_good,longest_stable_ms=longest_stable,first_downward_tick=first_down),
            inference=dict(first_s=inferences[0]['seconds'] if inferences else None,subsequent=distribution([t['seconds'] for t in inferences[1:]]),
                calls=len(inferences),over_period_fraction=float(np.mean([t['seconds']>replan_ms/1000 for t in inferences])) if inferences else None,
                over_50ms_fraction=float(np.mean([t['seconds']>.05 for t in inferences])) if inferences else None,records=inferences),
            motion_changes=motion_changes(trace),
            model_alignment=dict(initial=initial_residual,completed_tick=completed,success=completed is not None and gate_count==0,
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
