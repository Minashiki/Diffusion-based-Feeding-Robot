"""Unassisted 200ms DiT plans, 1ms qualification and replayable actual commands."""

from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from diffusers import DDIMScheduler

from correction_v3.controller import BRAKE_HORIZON_S,clearance,residual,safety_gate,stopped
from correction_v4.diagnostics import PickupRecorder
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


class Alignment:
    def __init__(self,reference,height):
        self.reference=reference;self.height=height;self.stable=None;self.completed=None
        self.rows=[];self.first_good=None;self.longest=0;self.maximum_drift=0.;self.final=None

    def observe(self,obs,tick):
        if self.completed is not None: return
        if self.rows and tick!=self.rows[-1][0]+1: raise ValueError('Missing 1ms alignment boundary')
        r=residual(self.reference,obs);drift=abs(float(obs['tcp_position'][2])-self.height)
        good=(r['angle_deg']<np.rad2deg(.01) and r['lateral_m']<.0007
            and r['linear_speed']<.002 and r['angular_speed']<.05 and drift<.0007)
        reset=self.stable is not None and not good
        self.stable=(tick if self.stable is None else self.stable) if good else None
        age=0 if self.stable is None else tick-self.stable
        if good and self.first_good is None: self.first_good=tick
        self.longest=max(self.longest,age);self.maximum_drift=max(self.maximum_drift,drift);self.final=r
        error=np.asarray(obs['tcp_position'])[:2]-self.reference.path[2][1][:2]
        self.rows.append([tick,*error,r['angle_deg'],r['linear_speed'],r['angular_speed'],
            drift,good,-1 if self.stable is None else self.stable,age,reset])
        if age>=200: self.completed=tick

    def save(self,output):
        np.save(output/'alignment_boundaries.npy',np.asarray(self.rows,dtype=float).reshape(-1,11))
        return dict(success=self.completed is not None,completed_tick=self.completed,first_good_tick=self.first_good,
            longest_stable_ms=self.longest,maximum_height_drift_m=self.maximum_drift,final=self.final,
            resets=sum(int(row[-1]) for row in self.rows),boundary_ms=1,
            fields=['tick','error_x_m','error_y_m','angle_deg','linear_speed','angular_speed',
                'height_drift_m','good','stable_since','age_ms','reset'])


@torch.no_grad()
def sample_actions(model,config,batch,normalization,generator):
    _,original=schedulers(config)
    sampler=DDIMScheduler.from_config(original.config,timestep_spacing='leading')
    sampler.set_timesteps(config['diffusion']['inference_steps'],device=batch['states'].device)
    condition=model.encoder(batch)
    actions=torch.randn((len(batch['states']),model.horizon,6),device=batch['states'].device,generator=generator)
    for step in sampler.timesteps:
        prediction=model.denoiser(actions,step.expand(len(actions)),condition)
        actions=sampler.step(prediction,step,actions,eta=0).prev_sample
    return actions*torch.as_tensor(normalization['action_std'],device=actions.device)+torch.as_tensor(
        normalization['action_mean'],device=actions.device)


@torch.no_grad()
def rollout(model,config,normalization,source,output,device,*,correction,noise_seed=0,max_ticks=None,source_data=None):
    source=Path(source);output=Path(output);m,a=load_episode(source) if source_data is None else source_data
    env=FeedingGymEnv('panda');task=env.task;recorder=PickupRecorder()
    trace=[];owners=[];interventions=[];brake=None;abort=None;chunk=None;last_plan=-200;wrist_peak=0.
    started=time.monotonic();alignment=None;times=[]
    try:
        env.reset(seed=m['seed'],options={'scenario':m['scenario']})
        snapshot=source/('handover_state.pkl' if correction else 'initial_state.pkl')
        task.set_state(pickle.loads(snapshot.read_bytes()))
        if task.state_signature()!=m['signature'] or task.dt!=.001 or env.max_episode_s!=60.:
            raise ValueError('Evaluation signature, physics clock or time limit differs')
        initial_tick=task.tick
        if correction and initial_tick!=m['handover']['release_tick']: raise ValueError('Handover tick differs')
        if correction:
            reference=Teacher(task.robot_config,m['teacher_config'])
            reference.reset({},geometry=teacher_geometry(task))
            alignment=Alignment(reference,float(task.provider.observe()['policy_obs']['tcp_position'][2]))
            alignment.observe(task.provider.observe()['policy_obs'],task.tick)
        writer=EpisodeWriter(output,env.schema,task.dt,env.max_episode_s)
        (output/'initial_state.pkl').write_bytes(pickle.dumps(task.get_state(),protocol=5))
        fields=field_slices(env.schema);ticks=[];observations=[]
        if correction:
            prior=np.r_[a['observation_ticks'],a['action_ticks']]
            values=np.concatenate([a['observations'],a['action_observations']])
            valid=np.r_[a['observation_valid'],np.ones(len(a['action_ticks']),bool)]
            order=np.argsort(prior,kind='stable')
            order=order[valid[order]&(prior[order]<task.tick)&(prior[order]>=task.tick-400)]
            ticks=prior[order].tolist();observations=list(values[order])
        writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy())
        generator=torch.Generator(device=device)
        limit=60000 if max_ticks is None else min(max_ticks,60000)
        with recorder.capture(task):
            while not task.terminated and task.tick<limit:
                phase=task.logic.phase;obs=task.provider.observe()['policy_obs']
                if alignment and alignment.completed is None and brake is None:
                    if task.logic.phase!='ACQUIRE': abort='left_acquire_before_alignment'
                    elif alignment.completed is None:
                        distance=clearance(task)
                        if alignment.maximum_drift>=.0007: abort='alignment_height_drift'
                        elif distance<=.0005: abort='alignment_clearance'
                        elif task.tick%50==0 and safety_gate(task,obs)['stop_required']: abort='alignment_safety_clearance'
                    if abort:
                        brake=task.tick;chunk=None;writer.interrupt(task.tick)
                        interventions.append(dict(tick=task.tick,reason=abort))
                        if not stopped(obs,task.adapter.velocity):
                            task.adapter.set_twist(np.zeros(6),task.data.time,(task.tick+50)*task.dt)
                            writer.command(task.tick,np.zeros(6),(task.tick+50)*task.dt)
                if task.tick%20==0 or task.tick%50==0:
                    ticks.append(task.tick);observations.append(env.observe_policy())
                    ticks=ticks[-32:];observations=observations[-32:]
                if task.tick%50==0:
                    command=None;owner='model';action=None
                    if brake is not None:
                        if stopped(obs,task.adapter.velocity): break
                        if (task.tick-brake)*task.dt>BRAKE_HORIZON_S: abort+=':brake_timeout';break
                        command=np.zeros(6);action=command;owner='external_brake'
                    elif phase!='SELECT':
                        if chunk is None or task.tick-last_plan>=200 or chunk.phase!=phase:
                            x=features(np.asarray(ticks),np.asarray(observations),np.ones(len(ticks),bool),task.tick,fields,normalization)
                            batch={k:torch.as_tensor(v,device=device)[None] for k,v in x.items()}
                            generator.manual_seed(noise_seed*1000003+task.tick//50)
                            before=time.monotonic()
                            with precision(device): actions=sample_actions(model,config,batch,normalization,generator)
                            actions=actions[0].float().cpu().numpy()
                            times.append(dict(tick=task.tick,seconds=time.monotonic()-before))
                            chunk=ActionChunk(task.tick,phase,actions);last_plan=task.tick
                        action=chunk.current(task.tick,phase)
                        if action is None or not np.isfinite(action).all(): abort='invalid_command';break
                        command=np.r_[clip_norm(action[:3],task.robot_config['linear_speed_limit']),
                            clip_norm(action[3:],task.robot_config['angular_speed_limit'])]
                    if command is not None:
                        task.adapter.set_twist(command,task.data.time,(task.tick+50)*task.dt)
                        writer.command(task.tick,command,(task.tick+50)*task.dt)
                        writer.action(task.tick,PHASES.index(phase),action,command,env.observe_policy(),task.tick+50)
                        writer.actions[-1]['valid']=False;owners.append(owner)
                        trace.append(dict(tick=task.tick,phase=phase,owner=owner,predicted=action.tolist(),command=command.tolist(),
                            chunk_issued_tick=None if chunk is None else chunk.issued_tick,
                            chunk_action_index=None if chunk is None else (task.tick-chunk.issued_tick)//50,
                            world_command=(task.data.site_xmat[task.index.base].reshape(3,3)@command[:3]).tolist(),
                            shaped_twist_base=task.adapter.velocity.tolist(),measured_twist_world=np.asarray(obs['tcp_twist_world']).tolist()))
                physical=task.step_physics();wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
                if (alignment and alignment.completed is None and brake is None and task.failure_reason!='nonfinite_state'
                        and task.tick>alignment.rows[-1][0]):
                    alignment.observe(physical,task.tick)
                if task.logic.phase!=phase and not task.terminated:
                    task.adapter.stop(hold_reference=True);writer.command(task.tick,hold_reference=True,after_physics=True)
                    writer.interrupt(task.tick);chunk=None
                writer.record_physics(task,physical)
                if task.tick%20==0 or task.terminated:
                    valid=task.failure_reason!='nonfinite_state'
                    writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy() if valid else observations[-1],valid)
        valid=task.failure_reason!='nonfinite_state'
        if writer.obs_ticks[-1]!=task.tick:
            writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy() if valid else observations[-1],valid)
        truncated=not task.terminated
        if truncated: task.logic.emit('time_limit',task.data.time)
        task.adapter.stop(hold_reference=True);writer.interrupt(task.tick);writer.command(task.tick,hold_reference=True)
        events=deepcopy(task.logic.events);new_events=[e for e in events if e['time']>=initial_tick*.001]
        names={e['name'] for e in new_events};entered={'SELECT'}|{e['phase'] for e in new_events if e['name']=='phase'}
        recovered=any(e['name']=='phase' and e.get('previous')=='RECOVER' and e['phase']=='WAIT_READY' for e in new_events)
        alignment_result=alignment.save(output) if alignment else None
        pickup=recorder.save(output)
        success=bool(task.logic.success and (recovered or not m['scenario'].get('recover',False)) and not abort)
        result=dict(robot_id='panda',seed=m['seed'],scenario=m['scenario'],signature=m['signature'],
            split='test' if m['split']=='test' and not correction else 'validation_diagnostic',
            source_episode=str(source),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
            input_hashes=input_hashes(),initial_tick=initial_tick,events=events,success=success,physical_success=bool(task.logic.success),
            pickup='pickup' in names,delivery='delivery' in names,recover=bool(m['scenario'].get('recover',False)),
            recovery_completed=recovered,transport_completed=any(e['name']=='phase' and e.get('previous')=='TRANSPORT'
                and e['phase']=='WAIT_READY' for e in new_events),entered=sorted(entered),truncated=truncated,
            failure_reason=task.failure_reason or abort or ('time_limit' if truncated else None),
            model_alignment=alignment_result,pickup_diagnostics=pickup,correction_case=correction,category=m.get('category'),cell=m.get('cell'),
            external_braking=dict(interventions=interventions,commands=sum(o=='external_brake' for o in owners),
                prefix_braking_is_model=False,autonomous_high_speed_braking_proven=False),
            teacher_in_execution=False,training_export=False,simulated_s=task.tick*.001,max_episode_s=env.max_episode_s,
            solver_iterations=int(task.model.opt.iterations),solver_tolerance=float(task.model.opt.tolerance),
            contact_peak_n=task.monitor.peak_n,wrist_peak_n=wrist_peak,contact_impulse_ns=task.monitor.impulse_ns,
            inference=times,wall_s=time.monotonic()-started,visualization=dict(status='disabled'))
        np.save(output/'action_owner.npy',np.asarray(owners,dtype='U20'));write_json(output/'trace.json',trace)
        writer.finish(dict(result,success=bool(task.logic.success),failure_reason=task.failure_reason,
            diagnostic_failure_reason=result['failure_reason']))
        return result
    finally: env.close()
