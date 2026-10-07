"""Independent-seed teacher rotation-pulse calibration; never exports training data."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import shutil
import time

import mink
import numpy as np
import torch

from acquire_diagnosis import verify_evidence
from diagnose_sampling import check_checkpoint
from feedingrobot.data.episodes import EpisodeWriter, annotate, input_hashes, write_json
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.experts.teacher import rotation_error
from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT
from feedingrobot.sim.events import PHASES


CASES=((730001,3.),(730002,5.),(730003,8.))


def pulse_command(state,teacher,observation,velocity,base_rotation,tick):
    target_p,target_r=teacher.path[0][1:3]
    rotation=np.asarray(observation['tcp_rotation'])
    error=float(np.linalg.norm(rotation_error(target_r,rotation)))
    angular_speed=float(np.linalg.norm(observation['tcp_twist_world'][3:]))
    if state['mode']=='waiting':
        if teacher.part!=0 or not teacher.reached(target_p,target_r,np.asarray(observation['tcp_position']),rotation):
            return None
        state.update(mode='pulse',pulse_start_tick=tick)
    if state['mode']=='pulse':
        # Predict the short stopping arc under the unchanged 2 rad/s^2 adapter limit.
        stopping_arc=float(np.linalg.norm(velocity[3:]))**2/(2*state['angular_acceleration_limit'])
        if error>=np.deg2rad(state['requested_deg']+.2)-stopping_arc:
            state.update(mode='settling',pulse_end_tick=tick)
        else:
            return np.r_[np.zeros(3),-.05*(base_rotation.T@target_r[:,1])]
    if state['mode']=='settling':
        if tick-state['pulse_end_tick']>=250 and angular_speed<.02 and np.linalg.norm(velocity[3:])<.02:
            state.update(mode='released',release_tick=tick,achieved_deg=float(np.rad2deg(error)),
                release_angular_speed=angular_speed,residual_slow_match=3.<=np.rad2deg(error)<=10. and angular_speed<.05)
            return None
        return np.zeros(6)
    return None


def run_case(seed,requested_deg,perturbed,config,scenario,output,max_ticks=None,episode_directory=None):
    env=FeedingGymEnv('panda')
    task=env.task
    trace=[]
    perturbation=dict(mode='waiting' if perturbed else 'disabled',requested_deg=requested_deg,
        angular_acceleration_limit=task.robot_config['angular_acceleration_limit'])
    wrist_peak=0.;corrected_tick=None;stable_start=None
    start=time.perf_counter()
    writer=None
    ownership=[]
    try:
        env.reset(seed=seed,options={'scenario':scenario})
        geometry=teacher_geometry(task)
        teacher=Teacher(task.robot_config,config)
        teacher.reset({},geometry=geometry)
        base=mink.SO3(np.array(task.robot_config['base_quaternion'])).as_matrix()
        prefix='pulse' if perturbed else 'baseline'
        initial_state=task.get_state()
        (output/f'{seed}_{prefix}_initial.pkl').write_bytes(pickle.dumps(initial_state,protocol=5))
        initial_physics_sha256=hashlib.sha256(initial_state['physics'].tobytes()).hexdigest()
        initial_signature=task.state_signature()
        if episode_directory is not None:
            writer=EpisodeWriter(episode_directory,env.schema,task.dt,env.max_episode_s)
            (Path(episode_directory)/'initial_state.pkl').write_bytes(pickle.dumps(initial_state,protocol=5))
            writer.record_observation(0,PHASES.index(task.logic.phase),env.observe_policy())
        target=teacher.path[0][2]
        limit=round(env.max_episode_s/task.dt) if max_ticks is None else min(max_ticks,round(env.max_episode_s/task.dt))
        while not task.terminated and task.tick<limit:
            phase=task.logic.phase
            if task.tick%50==0:
                observation=task.provider.observe()['policy_obs']
                previous_mode=perturbation['mode']
                command=pulse_command(perturbation,teacher,observation,task.adapter.velocity,base,task.tick) if perturbed else None
                if previous_mode=='settling' and perturbation['mode']=='released':
                    perturbation['release_phase']=phase
                    (output/f'{seed}_pulse_release.pkl').write_bytes(pickle.dumps(dict(task=task.get_state(),teacher=deepcopy(teacher.__dict__)),protocol=5))
                    if writer is not None:
                        (Path(episode_directory)/'correction_state.pkl').write_bytes(pickle.dumps(task.get_state(),protocol=5))
                teacher_owned=command is None
                if teacher_owned:
                    command=teacher.act(observation)
                    if teacher.stop_requested:
                        task.adapter.stop(hold_reference=True)
                        if writer is not None: writer.command(task.tick,hold_reference=True)
                if not np.isfinite(command).all():
                    task._terminate('invalid_command');break
                error=np.linalg.norm(rotation_error(target,np.asarray(observation['tcp_rotation'])))
                angular_speed=np.linalg.norm(observation['tcp_twist_world'][3:])
                if perturbed and perturbation['mode']=='released' and corrected_tick is None:
                    if error<.01 and angular_speed<.05:
                        stable_start=task.tick if stable_start is None else stable_start
                        if task.tick-stable_start>=200: corrected_tick=task.tick
                    else: stable_start=None
                task.adapter.set_twist(command,task.data.time,(task.tick+50)*task.dt)
                if writer is not None:
                    writer.command(task.tick,command,(task.tick+50)*task.dt)
                    writer.action(task.tick,PHASES.index(phase),teacher.proposal if teacher_owned else command,
                        command,env.observe_policy(),task.tick+50)
                    writer.actions[-1]['valid']=bool(teacher_owned)
                    ownership.append(bool(teacher_owned))
                trace.append(dict(tick=task.tick,phase=phase,teacher_stage=teacher.stage,teacher_owned=teacher_owned,
                    pulse_mode=perturbation['mode'],observation=env.observe_policy().tolist(),
                    early_target_error_deg=float(np.rad2deg(error)),angular_speed=float(angular_speed),command=command.tolist()))
            state=task.step_physics()
            wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
            if task.logic.phase!=phase and not task.terminated:
                task.adapter.stop(hold_reference=True)
                if writer is not None:
                    writer.command(task.tick,hold_reference=True,after_physics=True)
                    writer.interrupt(task.tick)
            if writer is not None:
                writer.record_physics(task,state)
                if task.tick%20==0 or task.terminated:
                    writer.record_observation(task.tick,PHASES.index(task.logic.phase),env.observe_policy(),task.failure_reason!='nonfinite_state')
            if trace and task.tick%50==0:
                trace[-1].update(shaped=task.adapter.velocity.tolist(),measured=task.snapshot()['tcp_twist_world'].tolist())
        if not task.terminated:
            task.adapter.stop(hold_reference=True)
            if writer is not None: task.logic.emit('time_limit',task.data.time)
        events=deepcopy(task.logic.events)
        names={e['name'] for e in events}
        qualified=bool(perturbed and perturbation.get('residual_slow_match') and perturbation.get('release_phase')=='ACQUIRE'
            and abs(perturbation['achieved_deg']-requested_deg)<=.75 and corrected_tick is not None)
        result=dict(seed=seed,split='calibration',perturbed=perturbed,teacher_config=config,scenario=scenario,
            initial_signature=initial_signature,initial_physics_sha256=initial_physics_sha256,geometry_sha256=geometry['sha256'],dt=task.dt,max_episode_s=env.max_episode_s,
            requested_residual_deg=requested_deg,perturbation=perturbation,qualified_correction=qualified,
            corrected_tick=corrected_tick,correction_duration_s=(corrected_tick-perturbation['release_tick'])*task.dt if corrected_tick is not None else None,
            pickup='pickup' in names,delivery='delivery' in names,success=bool(task.logic.success),
            failure_reason=task.failure_reason or (None if task.logic.success else 'time_limit' if task.tick>=round(env.max_episode_s/task.dt) else 'diagnostic_prefix'),
            events=events,entered=sorted({'SELECT'}|{e['phase'] for e in events if e['name']=='phase'}),
            simulated_s=task.tick*task.dt,wall_s=time.perf_counter()-start,contact_peak_n=task.monitor.peak_n,wrist_peak_n=wrist_peak,
            failure_contacts=[{k:c[k] for k in ('group1','group2','force_n','distance')} for c in sorted(task.contacts+task.applied_contacts,key=lambda c:c['force_n'],reverse=True)[:10]],
            trace=trace,training_export=False,model_policy_executed=False)
        if writer is not None:
            writer.interrupt(task.tick)
            writer.command(task.tick,hold_reference=True)
            np.save(Path(episode_directory)/'teacher_owned.npy',np.asarray(ownership,dtype=bool))
            metadata={k:v for k,v in result.items() if k!='trace'}
            metadata.update(robot_id='panda',split='train',group_id=f'panda:rotation_correction:{seed}',signature=initial_signature,
                input_hashes=input_hashes(),segments=annotate(events),accepted_normal=bool(qualified and task.logic.success),
                accepted_recovery=False,recovery_action_rows=0,truncated=not task.terminated,
                failure_reason=task.failure_reason,rejection_reason=None if qualified and task.logic.success else result['failure_reason'] or 'correction_gate',
                solver_iterations=int(task.model.opt.iterations),solver_tolerance=float(task.model.opt.tolerance),
                m5_extension=True,training_export='experimental_extension_only',collector_sha256=sha256(__file__))
            writer.finish(metadata)
            result['recorded_episode']=str(episode_directory)
        return result
    finally: env.close()


def may_expand(baseline,pulse):
    return (baseline['success'] and pulse['qualified_correction'] and pulse['pickup']
            and pulse['failure_reason'] not in ('invalid_command','nonfinite_state'))


def run(args):
    checkpoint_path,output=Path(args.checkpoint).resolve(),Path(args.output).resolve()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(checkpoint);digest=sha256(checkpoint_path)
    previous=verify_evidence(args.previous_report,digest)
    if previous['mode']!='rotation_feedback' or not previous['all_samples_finite']:
        raise ValueError('Requires finite rotation feedback evidence')
    device,hardware=setup(checkpoint['config'],'cpu')
    dataset=ROOT/checkpoint['config']['dataset']
    manifests=[read_json(p) for p in dataset.glob('*/*/manifest.json')]
    used={m['seed'] for m in manifests}
    if any(seed in used for seed,degrees in CASES): raise ValueError('Calibration seed overlaps frozen data')
    config=next(m['teacher_config'] for m in manifests if m['split']=='train' and not m['scenario'].get('recover',False))
    output.mkdir(parents=True,exist_ok=False);snapshot=output/'tool_snapshot';snapshot.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file(): shutil.copyfile(file,snapshot/file.name)
    provenance=dict(schema_version=1,mode='teacher_rotation_calibration',diagnostic=True,checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest,training_step=50000,weights='ema',checkpoint_use='Ancestry audit only; no model inference or optimizer.',
        parent=parent,source_check='passed',original_source_hashes=checkpoint['source_hashes'],hardware=hardware,precision='native simulation FP64',
        previous_report=str(Path(args.previous_report).resolve()),previous_report_sha256=sha256(args.previous_report),
        tool_sha256={p.name:sha256(p) for p in snapshot.iterdir()},calibration_cases=[dict(seed=s,requested_residual_deg=d) for s,d in CASES],
        controller='Original teacher; perturbation is zero linear plus -0.05 rad/s about early target y-axis, then zero twist for settling.',
        correction_criterion='Real residual 3..10deg, within 0.75deg of request, speed<0.02 at release in ACQUIRE; error<0.01rad and speed<0.05 for 200ms after release.',
        teacher_parameters={},teacher_config=config,scope='up to three paired independent calibration scenes',
        optimizer_updates=0,formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete',acceptance_status='not_evaluated',
        original_data_unchanged=True,training_export=False,model_policy_executed=False,seed_overlap=False)
    write_json(output/'provenance.json',provenance)
    pairs=[]
    try:
        for seed,degrees in CASES:
            rng=np.random.default_rng(seed)
            scenario=dict(config['scene'],recover=False,head_amp_m=float(rng.uniform(.005,.01)),head_freq_hz=float(rng.uniform(.1,.2)),head_phase_rad=float(rng.uniform(-np.pi,np.pi)))
            baseline=run_case(seed,degrees,False,config,scenario,output)
            write_json(output/f'{seed}_baseline.json',baseline)
            print(f'baseline seed={seed}: pickup={baseline["pickup"]} success={baseline["success"]} reason={baseline["failure_reason"]}',flush=True)
            if not baseline['success']:
                pairs.append(dict(seed=seed,baseline={k:baseline[k] for k in ('success','pickup','failure_reason','simulated_s')},pulse_skipped=True));break
            pulse=run_case(seed,degrees,True,config,scenario,output)
            write_json(output/f'{seed}_pulse.json',pulse)
            if (baseline['initial_signature']!=pulse['initial_signature'] or baseline['geometry_sha256']!=pulse['geometry_sha256']
                    or baseline['initial_physics_sha256']!=pulse['initial_physics_sha256']):
                raise ValueError('Pair environment or geometry differs')
            fields=('success','pickup','delivery','failure_reason','simulated_s','contact_peak_n','wrist_peak_n','wall_s')
            pairs.append(dict(seed=seed,requested_residual_deg=degrees,baseline={k:baseline[k] for k in fields},
                pulse={k:pulse[k] for k in fields+('qualified_correction','perturbation','correction_duration_s')},pair_initial_signature_verified=True,pair_initial_physics_verified=True))
            print(f'pulse seed={seed}: residual={pulse["perturbation"].get("achieved_deg")} correction={pulse["qualified_correction"]} pickup={pulse["pickup"]} success={pulse["success"]} reason={pulse["failure_reason"]}',flush=True)
            if not may_expand(baseline,pulse): break
    except Exception as error:
        write_json(output/'report.json',dict(provenance,status='error',pairs=pairs,error=f'{type(error).__name__}: {error}'));raise
    check_checkpoint(checkpoint)
    report=dict(provenance,status='completed',pairs=pairs,source_unchanged=input_hashes()==checkpoint['source_hashes'])
    report['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='report.json'}
    write_json(output/'report.json',report)
    if not report['source_unchanged']: raise ValueError('Original runtime inputs changed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--previous-report',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args)
    print('report:',Path(args.output).resolve()/'report.json',flush=True)
