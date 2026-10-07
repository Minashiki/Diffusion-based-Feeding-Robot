"""Record real prefixes, staged teacher ownership and exact physics commands."""

from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .controller import (AlignmentTeacher,Progress,TEACHER_OWNERS,VERSION,cancel_chunk,
    BRAKE_HORIZON_S,REQUIRED_CLEARANCE,alignment_residual,braking_reserve,
    clearance,moving_release,residual,safety_gate,stopped,swept_clearance)
from feedingrobot.control.adapter import clip_norm
from feedingrobot.data.episodes import EpisodeWriter,annotate,input_hashes,load_episode,write_json
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.policies.data import features,field_slices
from feedingrobot.policies.dit import sample_actions
from feedingrobot.policies.evaluation import ActionChunk
from feedingrobot.policies.runtime import precision
from feedingrobot.sim.events import PHASES


def episode(seed,category,config,scenario,directory,*,source,model=None,dp_config=None,
            normalization=None,device=None,max_s=None,resume_from=None,prefix_from=None):
    """Saved DP states/commands are diagnostic only; collection uses CUDA DP."""
    env=FeedingGymEnv('panda',max_episode_s=max_s)
    task=env.task;writer=None;owners=[];stages=[];trace=[];plans=[]
    state=dict(mode='waiting',source=source,category=category)
    alignment=None;chunk=None;last_plan=-200;progress=Progress();wrist_peak=0.;alignment_max_drift=None
    observations=[];ticks=[];abort_reason=None;geometry_checks=[];safety_checks=[]
    prefix_commands={}
    started=time.perf_counter();source_hashes=input_hashes()
    try:
        env.reset(seed=seed,options={'scenario':scenario})
        geometry=teacher_geometry(task);teacher=Teacher(task.robot_config,config)
        teacher.reset({},geometry=geometry)
        if source=='diagnostic':
            old,_=load_episode(resume_from or prefix_from)
            if old['input_hashes']!=source_hashes or old['seed']!=seed or old['scenario']!=scenario or old['category']!=category:
                raise ValueError('Saved diagnostic inputs differ')
            if resume_from:
                task.set_state(pickle.loads((Path(resume_from)/'handover_state.pkl').read_bytes()))
                progress.above=old['handover']['progress']['above']
                progress.pre_entry=old['handover']['progress']['pre_entry']
            else:
                import json
                prefix_commands={c['tick']:c['twist'] for c in json.loads((Path(prefix_from)/'commands.json').read_text()) if c['kind']=='twist'}
        initial=task.get_state();signature=task.state_signature()
        writer=EpisodeWriter(directory,env.schema,task.dt,env.max_episode_s)
        directory=Path(directory)
        (directory/'initial_state.pkl').write_bytes(pickle.dumps(initial,protocol=5))
        writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy())
        fields=field_slices(env.schema)
        generator=torch.Generator(device=device).manual_seed(dp_config['training']['seed']) if source=='dp' else None
        while not task.terminated and task.tick<round(env.max_episode_s/task.dt):
            phase=task.logic.phase;policy=env.observe_policy()
            if task.tick%20==0 or task.tick%50==0:
                observations.append(policy);ticks.append(task.tick)
                if len(ticks)>32: observations.pop(0);ticks.pop(0)
            if task.tick%50==0:
                obs=task.provider.observe()['policy_obs'];progress.update(teacher,obs)
                command=None;proposal=None;owner=None
                if source in ('baseline','aligned'):
                    command=teacher.act(obs);owner='path_teacher' if source=='aligned' else 'prefix_teacher'
                elif (source=='calibration' and category.startswith('p1') and state['mode']=='waiting'
                        and obs['stage']=='ACQUIRE' and progress.above and teacher.part==1):
                    state.update(mode='preparing',preparation_tick=task.tick)
                    chunk=cancel_chunk(chunk)
                elif state['mode'] in ('waiting','descent') and progress.candidate(category,teacher,obs,
                        source=='calibration' and category.startswith('p2')):
                    gate=swept_clearance(task,teacher,obs);geometry_checks.append(dict(tick=task.tick,**gate))
                    if not gate['passed']:
                        state.update(mode='safety_braking',safety_tick=task.tick,safety_reason='unsafe_handover_clearance',
                            safety_residual=residual(teacher,obs),safety_height=float(obs['tcp_position'][2]))
                    else:
                        prepare=source=='calibration' and category.startswith('p2')
                        state.update(mode='preparing' if prepare else 'braking' if category.endswith('stopped') or not moving_release(obs,task.adapter.velocity) else 'handover',
                            trigger_tick=task.tick,trigger_residual=residual(teacher,obs),progress=vars(progress).copy(),
                            brake_start_tick=task.tick,brake_start_height=float(obs['tcp_position'][2]),
                            brake_limit_m=braking_reserve(task,obs),brake_max_drop_m=0.,preparation_tick=task.tick)
                    chunk=cancel_chunk(chunk)
                if source in ('dp','diagnostic') and state['mode']=='waiting':
                    safety=safety_gate(task,obs)
                    if safety['stop_required']:
                        safety_checks.append(dict(tick=task.tick,**safety))
                        state.update(mode='safety_braking',safety_tick=task.tick,
                            safety_residual=residual(teacher,obs),safety_height=float(obs['tcp_position'][2]))
                        chunk=cancel_chunk(chunk)
                if state['mode']=='safety_braking':
                    command=np.zeros(6);owner='external_brake'
                    state['safety_minimum_clearance_m']=min(state.get('safety_minimum_clearance_m',float('inf')),clearance(task))
                    if stopped(obs,task.adapter.velocity):
                        state.update(safety_stop_tick=task.tick,safety_stop_height=float(obs['tcp_position'][2]))
                        abort_reason=state.get('safety_reason','unsafe_unqualified_prefix');break
                    if (task.tick-state['safety_tick'])*task.dt>BRAKE_HORIZON_S:
                        abort_reason='safety_brake_timeout';break
                if state['mode']=='braking':
                    command=np.zeros(6);owner='external_brake'
                    state['brake_max_drop_m']=max(state['brake_max_drop_m'],state['brake_start_height']-float(obs['tcp_position'][2]))
                    if clearance(task)<=REQUIRED_CLEARANCE or state['brake_max_drop_m']>state['brake_limit_m']:
                        abort_reason='unsafe_braking';break
                    ready=stopped(obs,task.adapter.velocity) if category.endswith('stopped') else moving_release(obs,task.adapter.velocity)
                    if ready and (task.tick-state['brake_start_tick']>=250 or category.endswith('moving')):
                        state['mode']='handover'
                    elif (task.tick-state['brake_start_tick'])*task.dt>BRAKE_HORIZON_S:
                        abort_reason='brake_release_not_covered';break
                if source=='calibration' and state['mode']=='preparing':
                    command=np.zeros(6);owner='external_brake'
                    if task.tick-state['preparation_tick']>=250 and stopped(obs,task.adapter.velocity):
                        state.update(mode='pulse',pulse_start_tick=task.tick,origin_y=float(obs['tcp_position'][1]))
                if source=='calibration' and state['mode']=='pulse':
                    r=residual(teacher,obs)
                    angular_done=r['angle_deg']>=2.5
                    lateral_done=state['origin_y']-obs['tcp_position'][1]>=.0015
                    command=np.r_[teacher.base_rotation.T@np.array([0.,0. if lateral_done else -.0025,0.]),
                        teacher.base_rotation.T@np.array([0.,0. if angular_done else -.05,0.])]
                    owner='pulse'
                    if angular_done and lateral_done:
                        command=np.zeros(6);state.update(mode='settling',pulse_end_tick=task.tick)
                if source=='calibration' and state['mode']=='settling':
                    command=np.zeros(6);owner='external_brake'
                    if task.tick-state['pulse_end_tick']>=250 and stopped(obs,task.adapter.velocity):
                        if category.startswith('p1'):
                            state['mode']='descent'
                        elif category.endswith('moving'):
                            state.update(mode='momentum',momentum_tick=task.tick)
                        else: state['mode']='handover'
                if source=='calibration' and state['mode']=='momentum':
                    command=np.r_[teacher.base_rotation.T@np.array([0.,0.,-.0045]),np.zeros(3)]
                    owner='momentum_pulse'
                    if task.tick-state['momentum_tick']>=250 and -obs['tcp_twist_world'][2]>=.004:
                        state['mode']='handover'
                if source=='calibration' and state['mode']=='descent':
                    command=np.r_[teacher.base_rotation.T@np.array([0.,0.,-.05]),np.zeros(3)]
                    owner='momentum_pulse'
                if state['mode']=='handover':
                    ready=stopped(obs,task.adapter.velocity) if category.endswith('stopped') else moving_release(obs,task.adapter.velocity)
                    if not ready or not alignment_residual(teacher,obs) or not progress.corridor(category,teacher,obs):
                        abort_reason='release_category_not_covered';break
                    gate=swept_clearance(task,teacher,obs);geometry_checks.append(dict(tick=task.tick,**gate))
                    if not gate['passed']:
                        abort_reason='unsafe_release_clearance';break
                    chunk=cancel_chunk(chunk)
                    alignment=AlignmentTeacher(teacher,obs,task.tick,category,progress)
                    state.update(mode='teacher',release_tick=task.tick,release_residual=alignment.release_residual,
                        release_velocity=task.adapter.velocity.tolist(),release_height=float(obs['tcp_position'][2]),path_part=alignment.path_part)
                    (directory/'handover_state.pkl').write_bytes(pickle.dumps(task.get_state(),protocol=5))
                if source not in ('baseline','aligned') and state['mode']=='teacher':
                    if alignment.completed_tick is None and clearance(task)<=.0005:
                        abort_reason='alignment_clearance';break
                    try: command,owner=alignment.act(obs,task.tick)
                    except ValueError as error:
                        abort_reason=str(error);break
                    state['corrected_tick']=alignment.completed_tick
                if source not in ('baseline','aligned') and state['mode']=='waiting':
                    if source=='calibration':
                        command=teacher.act(obs);owner='prefix_teacher'
                    elif source=='diagnostic':
                        if task.tick in prefix_commands:
                            command=np.asarray(prefix_commands[task.tick]);owner='dp'
                        elif phase!='SELECT':
                            abort_reason='diagnostic_prefix_exhausted';break
                    elif phase!='SELECT':
                        if chunk is None or task.tick-last_plan>=200:
                            x=features(np.asarray(ticks),np.asarray(observations),np.ones(len(ticks),bool),
                                task.tick,fields,normalization)
                            batch={k:torch.as_tensor(v,device=device)[None] for k,v in x.items()}
                            with precision(device):
                                actions=sample_actions(model,dp_config,batch,normalization,generator)[0].float().cpu().numpy()
                            if not np.isfinite(actions).all():
                                abort_reason='nonfinite_prediction';break
                            plans.append(dict(tick=task.tick,condition=x,actions=actions))
                            chunk=ActionChunk(task.tick,phase,actions);last_plan=task.tick
                        proposal=chunk.current(task.tick,phase)
                        if proposal is None:
                            abort_reason='invalid_chunk';break
                        command=np.r_[clip_norm(proposal[:3],task.robot_config['linear_speed_limit']),
                            clip_norm(proposal[3:],task.robot_config['angular_speed_limit'])]
                        owner='dp'
                if command is not None:
                    if not np.isfinite(command).all():
                        abort_reason='nonfinite_command';break
                    if owner in ('path_teacher','prefix_teacher') and teacher.stop_requested:
                        task.adapter.stop(hold_reference=True);writer.command(task.tick,hold_reference=True)
                    if proposal is None: proposal=teacher.proposal.copy() if owner in TEACHER_OWNERS else command.copy()
                    task.adapter.set_twist(command,task.data.time,(task.tick+50)*task.dt)
                    writer.command(task.tick,command,(task.tick+50)*task.dt)
                    # Correction labels use the pre-command observation, as do DP features.
                    # Baseline is a comparison-only copy of original teacher logs.
                    action_observation=env.observe_policy() if source=='baseline' else policy
                    writer.action(task.tick,PHASES.index(phase),proposal,command,action_observation,task.tick+50)
                    writer.actions[-1]['valid']=owner in TEACHER_OWNERS
                    owners.append(owner);stages.append('alignment' if owner=='alignment_teacher' else teacher.stage if owner in ('path_teacher','prefix_teacher') else owner)
                    trace.append(dict(tick=task.tick,owner=owner,teacher_part=teacher.part,teacher_stage=stages[-1],
                        mode=state['mode'],position=obs['tcp_position'].tolist(),twist=obs['tcp_twist_world'].tolist(),
                        reference_velocity=task.adapter.velocity.tolist(),residual=residual(teacher,obs)))
            physical=task.step_physics();wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
            if task.logic.phase!=phase and not task.terminated:
                task.adapter.stop(hold_reference=True);writer.command(task.tick,hold_reference=True,after_physics=True)
                writer.interrupt(task.tick);chunk=cancel_chunk(chunk)
            writer.record_physics(task,physical)
            if alignment and alignment.completed_tick is None:
                drift=abs(physical['tcp_position'][2]-alignment.height)
                alignment_max_drift=max(alignment_max_drift or 0.,float(drift))
                if drift>=.0007:
                    abort_reason='alignment_height_drift';break
            if task.tick%20==0 or task.terminated:
                valid=task.failure_reason!='nonfinite_state'
                writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy() if valid else policy,valid)
        # Terminal observation is explicit; same-tick duplicates have equal values.
        valid=task.failure_reason!='nonfinite_state'
        if writer.obs_ticks[-1]!=task.tick:
            writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy() if valid else policy,valid)
        truncated=not task.terminated
        if truncated: task.logic.emit('time_limit',task.data.time)
        task.adapter.stop(hold_reference=True);writer.interrupt(task.tick);writer.command(task.tick,hold_reference=True)
        events=deepcopy(task.logic.events);names={e['name'] for e in events}
        qualified=bool(alignment and alignment.qualified()) if source not in ('baseline','aligned') else source=='aligned'
        # All labels remain diagnostic until the collector accepts and replays.
        metadata=dict(robot_id='panda',seed=seed,split='train' if source in ('dp','aligned') else 'calibration',
            group_id=f'panda:descent_correction_v3:{seed}',category=category,prefix_source=source,
            teacher_version=VERSION,teacher_config=config,teacher_parameters={},scenario=scenario,
            signature=signature,initial_signature=signature,initial_physics_sha256=hashlib.sha256(initial['physics'].tobytes()).hexdigest(),
            teacher_geometry_sha256=geometry['sha256'],input_hashes=source_hashes,
            success=bool(task.logic.success),pickup='pickup' in names,delivery='delivery' in names,
            failure_reason=task.failure_reason,abort_reason=abort_reason,truncated=truncated,
            qualified_correction=qualified,accepted_normal=bool(qualified and task.logic.success and not abort_reason),
            accepted_recovery=False,recovery_action_rows=0,m5_extension=True,training_export=False,
            segments=annotate(events),events=events,simulated_s=task.tick*task.dt,max_episode_s=env.max_episode_s,
            solver_iterations=int(task.model.opt.iterations),solver_tolerance=float(task.model.opt.tolerance),
            contact_peak_n=task.monitor.peak_n,wrist_peak_n=wrist_peak,wall_s=time.perf_counter()-started,
            handover=state,corrected_tick=alignment.completed_tick if alignment else None,alignment_max_drift_m=alignment_max_drift,
            geometry_checks=geometry_checks,safety_checks=safety_checks,source_unchanged=input_hashes()==source_hashes,
            diagnostic_resume=str(Path(resume_from).resolve()) if resume_from else None,
            diagnostic_prefix=str(Path(prefix_from).resolve()) if prefix_from else None,
            model_policy_executed=source=='dp',initial_tick=initial['task']['tick'])
        np.save(directory/'action_owner.npy',np.asarray(owners,dtype='U20'))
        np.save(directory/'action_stages.npy',np.asarray(stages,dtype='U24'))
        write_json(directory/'trace.json',trace);write_json(directory/'plans.json',plans)
        writer.finish(metadata)
        return metadata
    finally:
        env.close()
