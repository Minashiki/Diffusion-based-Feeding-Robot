"""One independent teacher pair: safe descent lateral/rotation perturbation calibration."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import mink
import numpy as np
import torch

from acquire_diagnosis import verify_evidence
from diagnose_sampling import check_checkpoint
from train_correction import tool_hashes
from feedingrobot.data.episodes import input_hashes,write_json
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts import Teacher
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.experts.teacher import rotation_error
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.sim.model import ROOT
from feedingrobot.policies.runtime import setup


SEED=750001


def control(state,teacher,obs,adapter_velocity,base,tick):
    if state['mode']=='waiting':
        if teacher.part!=1 or obs['tcp_position'][2]>state['trigger_z']: return None
        state.update(mode='braking',trigger_tick=tick,teacher_part=teacher.part,
            rotation_target=teacher.path[1][2].tolist(),position_target=teacher.path[1][1].tolist())
    if state['mode']=='braking':
        if tick-state['trigger_tick']>=250 and np.linalg.norm(obs['tcp_twist_world'][:3])<.002 and np.linalg.norm(adapter_velocity[:3])<.002:
            state.update(mode='pulse',pulse_start_tick=tick,origin_y=float(obs['tcp_position'][1]),pulse_position=np.asarray(obs['tcp_position']).tolist())
        else: return np.zeros(6)
    target=np.asarray(state['rotation_target'])
    error=np.linalg.norm(rotation_error(target,obs['tcp_rotation']))
    if state['mode']=='pulse':
        angular_done=error>=np.deg2rad(state['requested_deg'])
        lateral_done=state['origin_y']-obs['tcp_position'][1]>=state['requested_lateral_m']
        command=np.r_[base.T@np.array([0.,0. if lateral_done else -.0025,0.]),
                      base.T@np.array([0.,0. if angular_done else -.05,0.])]
        if angular_done and lateral_done: state.update(mode='settling',pulse_end_tick=tick)
        return command
    if state['mode']=='settling':
        twist=np.asarray(obs['tcp_twist_world'])
        if (tick-state['pulse_end_tick']>=250 and np.linalg.norm(twist[:3])<.002
                and np.linalg.norm(twist[3:])<.02 and np.linalg.norm(adapter_velocity[:3])<.002
                and np.linalg.norm(adapter_velocity[3:])<.02):
            state.update(mode='released',release_tick=tick,release_part=teacher.part,
                achieved_deg=float(np.rad2deg(error)),achieved_lateral_m=float(state['origin_y']-obs['tcp_position'][1]),
                release_angular_speed=float(np.linalg.norm(twist[3:])),release_linear_speed=float(np.linalg.norm(twist[:3])),
                release_position=np.asarray(obs['tcp_position']).tolist())
            return None
        return np.zeros(6)
    return None


def run_case(config,scenario,perturbed,output):
    env=FeedingGymEnv('panda');task=env.task;trace=[];corrected_tick=None;stable=None;wrist_peak=0.
    state=dict(mode='waiting' if perturbed else 'disabled',requested_deg=2.5,requested_lateral_m=.0015,trigger_z=.07)
    start=time.perf_counter();prefix='perturbed' if perturbed else 'baseline'
    try:
        env.reset(seed=SEED,options={'scenario':scenario});geometry=teacher_geometry(task)
        teacher=Teacher(task.robot_config,config);teacher.reset({},geometry=geometry)
        base=mink.SO3(np.asarray(task.robot_config['base_quaternion'])).as_matrix()
        initial=task.get_state();initial_physics=hashlib.sha256(initial['physics'].tobytes()).hexdigest()
        (output/f'{prefix}_initial.pkl').write_bytes(pickle.dumps(initial,protocol=5))
        while not task.terminated and task.tick<round(env.max_episode_s/task.dt):
            phase=task.logic.phase
            if task.tick%50==0:
                obs=task.provider.observe()['policy_obs'];previous_mode=state['mode']
                command=control(state,teacher,obs,task.adapter.velocity,base,task.tick) if perturbed else None
                owned=command is None
                if previous_mode!='released' and state['mode']=='released':
                    (output/'release.pkl').write_bytes(pickle.dumps(dict(task=task.get_state(),teacher=deepcopy(teacher.__dict__)),protocol=5))
                if owned:
                    command=teacher.act(obs)
                    if teacher.stop_requested: task.adapter.stop(hold_reference=True)
                if not np.isfinite(command).all(): task._terminate('invalid_command');break
                if perturbed and state['mode']=='released' and corrected_tick is None:
                    error=np.linalg.norm(rotation_error(np.asarray(state['rotation_target']),obs['tcp_rotation']))
                    lateral=abs(obs['tcp_position'][1]-state['position_target'][1])
                    if error<.01 and lateral<.0007 and np.linalg.norm(obs['tcp_twist_world'][3:])<.05:
                        stable=task.tick if stable is None else stable
                        if task.tick-stable>=200: corrected_tick=task.tick
                    else: stable=None
                task.adapter.set_twist(command,task.data.time,(task.tick+50)*task.dt)
                trace.append(dict(tick=task.tick,phase=phase,teacher_part=teacher.part,teacher_stage=teacher.stage,
                    perturbation_mode=state['mode'],teacher_owned=owned,command=command.tolist(),observation=env.observe_policy().tolist()))
            task.step_physics();wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
            if task.logic.phase!=phase and not task.terminated: task.adapter.stop(hold_reference=True)
            if trace and task.tick%50==0: trace[-1].update(shaped=task.adapter.velocity.tolist(),measured=task.snapshot()['tcp_twist_world'].tolist())
        if not task.terminated: task.adapter.stop(hold_reference=True)
        events=deepcopy(task.logic.events);names={e['name'] for e in events}
        qualified=bool(perturbed and state['mode']=='released' and state['teacher_part']==state['release_part']==1
            and abs(state['achieved_deg']-state['requested_deg'])<=.75
            and abs(state['achieved_lateral_m']-state['requested_lateral_m'])<=.0005
            and state['release_angular_speed']<.02 and state['release_linear_speed']<.002 and corrected_tick is not None)
        result=dict(seed=SEED,split='independent_calibration',perturbed=perturbed,success=bool(task.logic.success),
            pickup='pickup' in names,delivery='delivery' in names,failure_reason=task.failure_reason or (None if task.logic.success else 'time_limit'),
            simulated_s=task.tick*task.dt,wall_s=time.perf_counter()-start,qualified_correction=qualified,
            perturbation=state,corrected_tick=corrected_tick,correction_duration_s=(corrected_tick-state['release_tick'])*task.dt if corrected_tick else None,
            contact_peak_n=task.monitor.peak_n,wrist_peak_n=wrist_peak,initial_physics_sha256=initial_physics,
            geometry_sha256=geometry['sha256'],initial_signature=task.state_signature(),events=events,trace=trace,
            model_policy_executed=False,training_export=False,teacher_config=config,scenario=scenario)
        write_json(output/f'{prefix}.json',result);return result
    finally: env.close()


def run(args):
    base_path=Path(args.base_checkpoint).resolve();base=torch.load(base_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(base);_,hardware=setup(base['config'],'cpu')
    report_path=Path(args.replay_report).resolve();digest=sha256(args.checkpoint)
    evidence=verify_evidence(report_path,digest)
    if (evidence['binding']['base_checkpoint_sha256']!=sha256(base_path)
            or evidence['binding']['tool_hashes']!=tool_hashes() or evidence['binding']['source_hashes']!=input_hashes()
            or len(evidence['results'])!=3 or any(r['status']!='passed' for r in evidence['results'])):
        raise ValueError('Requires audited matching CUDA failure replays')
    original=ROOT/base['config']['dataset'];metadata=[read_json(p) for p in original.glob('*/*/manifest.json')]
    if any(m['seed']==SEED for m in metadata): raise ValueError('Calibration seed overlaps frozen splits')
    data=read_json(ROOT/'outputs/single_bean/v1/m5/dit/correction_dataset_v1_001/report.json')
    if any(m['seed']==SEED for m in data['accepted']+data['rejected']): raise ValueError('Calibration seed overlaps correction data')
    config=next(m['teacher_config'] for m in metadata if m['split']=='train' and not m['scenario'].get('recover',False))
    rng=np.random.default_rng(SEED)
    scenario=dict(config['scene'],recover=False,head_amp_m=float(rng.uniform(.005,.01)),head_freq_hz=float(rng.uniform(.1,.2)),head_phase_rad=float(rng.uniform(-np.pi,np.pi)))
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/Path(__file__).name)
    provenance=dict(diagnostic=True,mode='descent_coupled_calibration',checkpoint_sha256=digest,training_step=100000,weights='ema',
        parent=parent,binding=evidence['binding'],hardware=hardware,physics_precision='fp64',optimizer_updates=0,training_export=False,
        model_policy_executed=False,formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete',
        original_source_hashes=base['source_hashes'],input_report=str(report_path),input_report_sha256=sha256(report_path),
        correction_data_report_sha256=sha256(ROOT/'outputs/single_bean/v1/m5/dit/correction_dataset_v1_001/report.json'))
    write_json(output/'provenance.json',provenance)
    baseline=run_case(config,scenario,False,output)
    print('baseline',baseline['success'],baseline['simulated_s'],baseline['failure_reason'],flush=True)
    pulse=run_case(config,scenario,True,output) if baseline['success'] else None
    if pulse: print('perturbed',pulse['success'],pulse['qualified_correction'],pulse['simulated_s'],pulse['failure_reason'],flush=True)
    paired=bool(pulse and pulse['initial_physics_sha256']==baseline['initial_physics_sha256'] and pulse['geometry_sha256']==baseline['geometry_sha256'])
    unchanged=input_hashes()==base['source_hashes'] and tool_hashes()==evidence['binding']['tool_hashes']
    if not unchanged: raise ValueError('Source/bound tools changed')
    report=dict(provenance,baseline={k:v for k,v in baseline.items() if k!='trace'},
        perturbed={k:v for k,v in pulse.items() if k!='trace'} if pulse else None,paired_initial_state=paired,
        status='passed' if paired and pulse['success'] and pulse['qualified_correction'] else 'failed',source_unchanged=unchanged)
    report['evidence_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base-checkpoint','checkpoint','replay-report','output'): parser.add_argument('--'+key,required=True)
    run(parser.parse_args())
