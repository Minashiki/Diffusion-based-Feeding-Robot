"""CPU teacher prefixes with measured direction and momentum handovers."""

from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import time

import numpy as np

from .controller import (AlignmentTeacher,Progress,VERSION,TEACHER_OWNERS,residual,
    alignment_residual,moving_release,stopped,swept_clearance,clearance,braking_reserve,
    BRAKE_HORIZON_S,REQUIRED_CLEARANCE,direction_velocity,release_direction,perturbation_feedback)
from .diagnostics import PickupRecorder
from feedingrobot.data.episodes import EpisodeWriter,annotate,input_hashes,write_json
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.sim.events import PHASES
from feedingrobot.control.adapter import clip_norm


def episode(seed,cell,config,scenario,directory,*,baseline=False,split='calibration',max_s=None):
    env=FeedingGymEnv('panda',max_episode_s=max_s);task=env.task;directory=Path(directory)
    category=cell['category'];progress=Progress();alignment=None;writer=None
    state=dict(mode='waiting');owners=[];stages=[];trace=[];geometry_checks=[]
    recorder=PickupRecorder();abort=None;maximum_drift=0.;wrist_peak=0.
    started=time.perf_counter();source_hashes=input_hashes()
    try:
        env.reset(seed=seed,options={'scenario':scenario})
        geometry=teacher_geometry(task);teacher=Teacher(task.robot_config,config);teacher.reset({},geometry=geometry)
        initial=task.get_state();signature=task.state_signature()
        writer=EpisodeWriter(directory,env.schema,task.dt,env.max_episode_s)
        (directory/'initial_state.pkl').write_bytes(pickle.dumps(initial,protocol=5))
        writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy())
        while not task.terminated and task.tick<round(env.max_episode_s/task.dt):
            phase=task.logic.phase
            if task.tick%50==0:
                policy=env.observe_policy();obs=task.provider.observe()['policy_obs'];progress.update(teacher,obs)
                command=None;owner=None
                if baseline:
                    command=teacher.act(obs);owner='prefix_teacher'
                else:
                    if state['mode']=='waiting':
                        prepare=(category.startswith('p1') and obs['stage']=='ACQUIRE' and progress.above and teacher.part==1)
                        prepare|=(category.startswith('p2') and progress.candidate(category,teacher,obs,True))
                        if prepare:
                            gate=swept_clearance(task,teacher,obs);geometry_checks.append(dict(tick=task.tick,**gate))
                            if not gate['passed']: abort='unsafe_preparation';break
                            state.update(mode='preparing',preparation_tick=task.tick,progress=vars(progress).copy())
                        else: command=teacher.act(obs);owner='prefix_teacher'
                    if state['mode']=='preparing':
                        command=np.zeros(6);owner='external_brake'
                        if task.tick-state['preparation_tick']>=250 and stopped(obs,task.adapter.velocity):
                            state.update(mode='pulse',pulse_start_tick=task.tick)
                    if state['mode']=='pulse':
                        error=np.asarray(obs['tcp_position'])[:2]-teacher.path[2][1][:2]
                        target=np.asarray(cell['quadrant'])*.0015/np.sqrt(2)
                        lateral=target-error
                        done=np.abs(lateral)<.00008
                        angular_done=residual(teacher,obs)['angle_deg']>=2.5
                        linear=np.r_[np.where(done,0,np.sign(lateral)*.0018),0.]
                        angular=np.array([0.,0. if angular_done else -.05,0.])
                        command=np.r_[teacher.base_rotation.T@linear,teacher.base_rotation.T@angular];owner='direction_pulse'
                        if done.all() and angular_done:
                            command=np.zeros(6);state.update(mode='settling',pulse_end_tick=task.tick)
                    if state['mode']=='settling':
                        command=np.zeros(6);owner='external_brake'
                        if task.tick-state['pulse_end_tick']>=250 and stopped(obs,task.adapter.velocity):
                            state.update(mode='descent' if category.startswith('p1') else 'velocity',velocity_tick=task.tick)
                    if state['mode']=='descent':
                        linear=clip_norm(np.r_[perturbation_feedback(cell,teacher,obs),-.05],task.robot_config['linear_speed_limit'])
                        command=np.r_[teacher.base_rotation.T@linear,np.zeros(3)];owner='momentum_pulse'
                        if progress.candidate(category,teacher,obs):
                            gate=swept_clearance(task,teacher,obs);geometry_checks.append(dict(tick=task.tick,**gate))
                            if not gate['passed']: abort='unsafe_braking';break
                            state.update(mode='braking',trigger_tick=task.tick,trigger_residual=residual(teacher,obs),
                                brake_start_tick=task.tick,brake_start_height=float(obs['tcp_position'][2]),
                                brake_limit_m=braking_reserve(task,obs),brake_max_drop_m=0.)
                    if state['mode']=='braking':
                        command=np.r_[teacher.base_rotation.T@np.r_[perturbation_feedback(cell,teacher,obs),0.],np.zeros(3)]
                        owner='external_brake'
                        state['brake_max_drop_m']=max(state['brake_max_drop_m'],state['brake_start_height']-obs['tcp_position'][2])
                        ready=stopped(obs,task.adapter.velocity) if category.endswith('stopped') else moving_release(obs,task.adapter.velocity)
                        if ready:
                            state.update(mode='velocity',velocity_tick=task.tick)
                        elif task.tick-state['brake_start_tick']>round(BRAKE_HORIZON_S/task.dt):
                            abort='brake_release_not_covered';break
                    if state['mode']=='velocity':
                        linear=np.r_[direction_velocity(cell),-.004 if category.endswith('moving') else 0.]
                        command=np.r_[teacher.base_rotation.T@linear,np.zeros(3)];owner='velocity_pulse'
                        ready=stopped(obs,task.adapter.velocity) if category.endswith('stopped') else moving_release(obs,task.adapter.velocity)
                        if (task.tick-state['velocity_tick']>=150 and ready and release_direction(cell,teacher,obs)):
                            if not alignment_residual(teacher,obs) or not progress.corridor(category,teacher,obs):
                                abort='release_category_not_covered';break
                            gate=swept_clearance(task,teacher,obs);geometry_checks.append(dict(tick=task.tick,**gate))
                            if not gate['passed']: abort='unsafe_release_clearance';break
                            alignment=AlignmentTeacher(teacher,obs,task.tick,category,progress);alignment.observe(obs,task.tick)
                            state.update(mode='teacher',release_tick=task.tick,release_residual=alignment.release_residual,
                                release_velocity=task.adapter.velocity.tolist(),release_height=alignment.height,
                                release_error=(np.asarray(obs['tcp_position'])[:2]-teacher.path[2][1][:2]).tolist(),
                                release_twist=obs['tcp_twist_world'].tolist())
                            (directory/'handover_state.pkl').write_bytes(pickle.dumps(task.get_state(),protocol=5))
                        elif task.tick-state['velocity_tick']>=500:
                            abort='direction_velocity_not_covered';break
                    if state['mode']=='teacher':
                        command,owner=alignment.act(obs,task.tick)
                        state['corrected_tick']=alignment.completed_tick
                        if alignment.completed_tick is not None: state['stable_start_tick']=alignment.stable
                if command is not None:
                    if not np.isfinite(command).all(): raise ValueError('Nonfinite command')
                    if owner in ('prefix_teacher','path_teacher') and teacher.stop_requested:
                        task.adapter.stop(hold_reference=True);writer.command(task.tick,hold_reference=True)
                    proposal=teacher.proposal.copy() if owner in TEACHER_OWNERS else command.copy()
                    task.adapter.set_twist(command,task.data.time,(task.tick+50)*task.dt)
                    writer.command(task.tick,command,(task.tick+50)*task.dt)
                    writer.action(task.tick,PHASES.index(phase),proposal,command,
                        env.observe_policy() if baseline else policy,task.tick+50)
                    writer.actions[-1]['valid']=not baseline and owner in TEACHER_OWNERS
                    owners.append(owner);stages.append('alignment' if owner=='alignment_teacher' else
                        teacher.stage if owner in ('path_teacher','prefix_teacher') else owner)
                    trace.append(dict(tick=task.tick,owner=owner,mode=state['mode'],teacher_stage=stages[-1],
                        position=obs['tcp_position'].tolist(),residual=residual(teacher,obs),
                        stable_since=alignment.stable if alignment else None))
            with recorder.capture(task): physical=task.step_physics()
            wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
            if task.logic.phase!=phase and not task.terminated:
                task.adapter.stop(hold_reference=True);writer.command(task.tick,hold_reference=True,after_physics=True)
                writer.interrupt(task.tick)
            writer.record_physics(task,physical)
            if alignment and alignment.completed_tick is None:
                drift=abs(physical['tcp_position'][2]-alignment.height);maximum_drift=max(maximum_drift,float(drift))
                alignment.observe(physical,task.tick)
                if drift>=.0007: abort='alignment_height_drift';break
                if clearance(task)<=.0005: abort='alignment_clearance';break
            elif not baseline and state['mode'] not in ('waiting','teacher'):
                distance=clearance(task)
                speed=max(np.linalg.norm(physical['tcp_twist_world'][:3]),np.linalg.norm(task.adapter.velocity[:3]))
                if distance<=REQUIRED_CLEARANCE or speed>task.robot_config['linear_speed_limit']+.001:
                    abort='unsafe_perturbation';break
                if state['mode']=='braking':
                    drop=state['brake_start_height']-physical['tcp_position'][2]
                    state['brake_max_drop_m']=max(state['brake_max_drop_m'],float(drop))
                    if drop>state['brake_limit_m']: abort='unsafe_braking';break
            if task.tick%20==0 or task.terminated:
                writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy())
        if writer.obs_ticks[-1]!=task.tick:
            writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy())
        truncated=not task.terminated
        if truncated: task.logic.emit('time_limit',task.data.time)
        task.adapter.stop(hold_reference=True);writer.interrupt(task.tick);writer.command(task.tick,hold_reference=True)
        events=deepcopy(task.logic.events);names={e['name'] for e in events}
        qualified=bool(alignment and alignment.qualified())
        metadata=dict(robot_id='panda',seed=seed,split=split,group_id=f'panda:descent_correction_v4:{seed}',
            category=category,cell=cell,prefix_source='baseline' if baseline else 'controlled_perturbation',
            teacher_version=VERSION,teacher_config=config,teacher_parameters={},scenario=scenario,signature=signature,
            initial_signature=signature,initial_physics_sha256=hashlib.sha256(initial['physics'].tobytes()).hexdigest(),
            teacher_geometry_sha256=geometry['sha256'],input_hashes=source_hashes,
            success=bool(task.logic.success),pickup='pickup' in names,delivery='delivery' in names,
            failure_reason=task.failure_reason,abort_reason=abort,truncated=truncated,qualified_correction=qualified,
            accepted_normal=bool(qualified and task.logic.success and not abort),accepted_recovery=False,recovery_action_rows=0,
            m5_extension=True,training_export=False,segments=annotate(events),events=events,simulated_s=task.tick*task.dt,
            max_episode_s=env.max_episode_s,solver_iterations=int(task.model.opt.iterations),
            solver_tolerance=float(task.model.opt.tolerance),contact_peak_n=task.monitor.peak_n,wrist_peak_n=wrist_peak,
            wall_s=time.perf_counter()-started,handover=state,corrected_tick=alignment.completed_tick if alignment else None,
            alignment_max_drift_m=maximum_drift if alignment else None,geometry_checks=geometry_checks,
            safety_checks=[],source_unchanged=input_hashes()==source_hashes,model_policy_executed=False,initial_tick=initial['task']['tick'])
        np.save(directory/'action_owner.npy',np.asarray(owners,dtype='U24'))
        np.save(directory/'action_stages.npy',np.asarray(stages,dtype='U24'))
        np.save(directory/'alignment_boundaries.npy',np.asarray(alignment.records if alignment else [],dtype=float).reshape(-1,9))
        write_json(directory/'trace.json',trace);recorder.save(directory);writer.finish(metadata)
        return metadata
    finally: env.close()
