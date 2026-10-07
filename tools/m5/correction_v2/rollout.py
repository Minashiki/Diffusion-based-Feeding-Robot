"""Record real prefixes, staged teacher ownership and exact physics commands."""

from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .controller import (AlignmentTeacher,Progress,TEACHER_OWNERS,VERSION,cancel_chunk,
    clearance,residual,stopped,swept_clearance)
from feedingrobot.control.adapter import clip_norm
from feedingrobot.data.episodes import EpisodeWriter,annotate,input_hashes,write_json
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.policies.data import features,field_slices
from feedingrobot.policies.dit import sample_actions
from feedingrobot.policies.evaluation import ActionChunk
from feedingrobot.policies.runtime import precision
from feedingrobot.sim.events import PHASES


def episode(seed,category,config,scenario,directory,*,source,model=None,dp_config=None,
            normalization=None,device=None,max_s=None):
    """source is baseline, calibration, dp, or aligned; only dp uses a model."""
    env=FeedingGymEnv('panda',max_episode_s=max_s)
    task=env.task;writer=None;owners=[];stages=[];trace=[];plans=[]
    state=dict(mode='waiting',source=source,category=category)
    alignment=None;chunk=None;last_plan=-200;progress=Progress();wrist_peak=0.
    observations=[];ticks=[];abort_reason=None;geometry_checks=[]
    started=time.perf_counter();source_hashes=input_hashes()
    try:
        env.reset(seed=seed,options={'scenario':scenario})
        geometry=teacher_geometry(task);teacher=Teacher(task.robot_config,config)
        teacher.reset({},geometry=geometry)
        initial=task.get_state();signature=task.state_signature()
        writer=EpisodeWriter(directory,env.schema,task.dt,env.max_episode_s)
        directory=Path(directory)
        (directory/'initial_state.pkl').write_bytes(pickle.dumps(initial,protocol=5))
        writer.record_observation(0,PHASES.index(task.logic.phase),env.observe_policy())
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
                elif state['mode']=='waiting' and progress.candidate(category,teacher,obs,source=='calibration'):
                    gate=swept_clearance(task,teacher,obs);geometry_checks.append(dict(tick=task.tick,**gate))
                    if not gate['passed']:
                        abort_reason='unsafe_handover_clearance';break
                    state.update(mode='braking' if source=='calibration' or category.endswith('stopped') else 'handover',
                        trigger_tick=task.tick,trigger_residual=residual(teacher,obs),progress=vars(progress).copy())
                    chunk=cancel_chunk(chunk)
                if source not in ('baseline','aligned') and state['mode']=='braking':
                    command=np.zeros(6);owner='external_brake'
                    if task.tick-state['trigger_tick']>=250 and stopped(obs,task.adapter.velocity):
                        if source=='calibration':
                            state.update(mode='pulse',pulse_start_tick=task.tick,origin_y=float(obs['tcp_position'][1]))
                        else: state['mode']='handover'
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
                        if category.endswith('moving'):
                            state.update(mode='momentum',momentum_tick=task.tick)
                        else: state['mode']='handover'
                if source=='calibration' and state['mode']=='momentum':
                    command=np.r_[teacher.base_rotation.T@np.array([0.,0.,-.0045]),np.zeros(3)]
                    owner='momentum_pulse'
                    if task.tick-state['momentum_tick']>=100 and -obs['tcp_twist_world'][2]>.002:
                        state['mode']='handover'
                if state['mode']=='handover':
                    gate=swept_clearance(task,teacher,obs);geometry_checks.append(dict(tick=task.tick,**gate))
                    if not gate['passed']:
                        abort_reason='unsafe_release_clearance';break
                    chunk=cancel_chunk(chunk)
                    alignment=AlignmentTeacher(teacher,obs,task.tick,category,progress)
                    state.update(mode='teacher',release_tick=task.tick,release_residual=alignment.release_residual,
                        release_velocity=task.adapter.velocity.tolist(),path_part=alignment.path_part)
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
                    # v2 labels use the pre-command observation, as did DP features.
                    # Baseline is a comparison-only copy of original teacher logs.
                    action_observation=env.observe_policy() if source=='baseline' else policy
                    writer.action(task.tick,PHASES.index(phase),proposal,command,action_observation,task.tick+50)
                    writer.actions[-1]['valid']=owner in TEACHER_OWNERS
                    owners.append(owner);stages.append('alignment' if owner=='alignment_teacher' else teacher.stage if owner in ('path_teacher','prefix_teacher') else owner)
                    trace.append(dict(tick=task.tick,owner=owner,teacher_part=teacher.part,teacher_stage=stages[-1],
                        position=obs['tcp_position'].tolist(),twist=obs['tcp_twist_world'].tolist(),residual=residual(teacher,obs)))
            physical=task.step_physics();wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
            if task.logic.phase!=phase and not task.terminated:
                task.adapter.stop(hold_reference=True);writer.command(task.tick,hold_reference=True,after_physics=True)
                writer.interrupt(task.tick);chunk=cancel_chunk(chunk)
            writer.record_physics(task,physical)
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
            group_id=f'panda:descent_correction_v2:{seed}',category=category,prefix_source=source,
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
            handover=state,corrected_tick=alignment.completed_tick if alignment else None,
            geometry_checks=geometry_checks,source_unchanged=input_hashes()==source_hashes)
        np.save(directory/'action_owner.npy',np.asarray(owners,dtype='U20'))
        np.save(directory/'action_stages.npy',np.asarray(stages,dtype='U24'))
        write_json(directory/'trace.json',trace);write_json(directory/'plans.json',plans)
        writer.finish(metadata)
        return metadata
    finally:
        env.close()
