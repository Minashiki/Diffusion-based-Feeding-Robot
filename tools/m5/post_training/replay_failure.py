"""Replay audited CUDA commands, locate contacts, and retain diagnostic-only states."""

import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from acquire_diagnosis import verify_evidence,angle_degrees
from correction_data import audit_data
from diagnose_sampling import check_checkpoint
from rotation_feedback import early_target,rotation_descriptor
from train_correction import tool_hashes
from feedingrobot.data.episodes import input_hashes,load_episode,write_json
from feedingrobot.envs import FeedingGymEnv
from feedingrobot.experts.geometry import teacher_geometry
from feedingrobot.experts.bean_path import pickup_path
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.data import features,field_slices
from feedingrobot.policies.runtime import setup


SNAPSHOT_TICKS=(2050,3050,4050,6050,8050)


def replay_case(case,conditions,norm,output):
    metadata,teacher_arrays=load_episode(case['source_episode'])
    env=FeedingGymEnv('panda');task=env.task
    traces={r['tick']:r for r in case['trace']};plans={r['tick']:r for r in conditions}
    rows,contacts,snapshots=[],[],[];observation_ticks,observations=[],[]
    feature_error=shaped_error=measured_error=0.;wrist_peak=0.;command_count=0
    start=time.perf_counter()
    try:
        env.reset(seed=metadata['seed'],options={'scenario':metadata['scenario']})
        task.set_state(pickle.loads((Path(case['source_episode'])/'initial_state.pkl').read_bytes()))
        if task.state_signature()!=metadata['signature']: raise ValueError('Initial model signature differs')
        geometry=teacher_geometry(task);fields=field_slices(env.schema)
        nominal_parameters=dict(metadata['teacher_config']['teacher'],entry_pitch_offset_rad=0.)
        nominal_path=pickup_path(geometry,nominal_parameters)
        target=early_target(metadata)
        nominal_target=early_target(dict(metadata,teacher_parameters=nominal_parameters))
        reference_ticks=np.asarray(teacher_arrays['action_ticks'])
        eligible=(reference_ticks<=15000)&(teacher_arrays['action_phases']==1)
        path_ticks=reference_ticks[eligible];path_obs=np.asarray(teacher_arrays['action_observations'])[eligible]
        starts=metadata['comparison']['action_starts']
        def stage(tick):
            return next(r['stage'] for r in reversed(starts) if r['time']<=tick*task.dt+1e-8)
        limit=round(case['simulated_s']/task.dt)
        while not task.terminated and task.tick<limit:
            phase=task.logic.phase
            if task.tick%20==0 or task.tick%50==0:
                observation_ticks.append(task.tick);observations.append(env.observe_policy())
                if len(observations)>32: observation_ticks.pop(0);observations.pop(0)
            if task.tick in plans:
                x=features(np.asarray(observation_ticks),np.asarray(observations),np.ones(len(observations),bool),task.tick,fields,norm)
                plan=plans[task.tick]
                for key,value in x.items():
                    expected=plan['phase_index'] if key=='phase' else plan[key]
                    feature_error=max(feature_error,float(np.max(np.abs(np.asarray(value,dtype=float)-np.asarray(expected,dtype=float)))))
            if task.tick in SNAPSHOT_TICKS:
                name=f'{case["seed"]}_tick_{task.tick}.pkl'
                (output/name).write_bytes(pickle.dumps(task.get_state(),protocol=5))
                snapshots.append(dict(tick=task.tick,path=name,split='validation_diagnostic_only',training_export=False))
            if task.tick%50==0 and phase!='SELECT':
                row=traces.get(task.tick)
                if row is None or row['phase']!=phase: raise ValueError('Recorded command/phase mismatch')
                task.adapter.set_twist(np.asarray(row['command']),task.data.time,(task.tick+50)*task.dt)
                command_count+=1
                obs=env.observe_policy();position=obs[fields['tcp_position']]
                nearest=int(np.linalg.norm(path_obs[:,fields['tcp_position']]-position,axis=1).argmin())
                timed=int(np.searchsorted(reference_ticks,task.tick))
                matched=teacher_arrays['action_observations'][timed]
                rows.append(dict(tick=task.tick,position=position.tolist(),**rotation_descriptor(obs,target,fields),
                    nominal_target_error_deg=rotation_descriptor(obs,nominal_target,fields)['target_error_deg'],
                    time_position_gap_mm=float(np.linalg.norm(position-matched[fields['tcp_position']])*1000),
                    time_reference_stage=stage(task.tick),nearest_teacher_tick=int(path_ticks[nearest]),
                    nearest_teacher_stage=stage(path_ticks[nearest]),nearest_path_gap_mm=float(np.linalg.norm(position-path_obs[nearest,fields['tcp_position']])*1000),
                    nearest_path_angle_deg=angle_degrees(obs[fields['tcp_rotation']],path_obs[nearest,fields['tcp_rotation']]),
                    nominal_above_gap_mm=float(np.linalg.norm(position-nominal_path[0][1])*1000)))
            task.step_physics();wrist_peak=max(wrist_peak,task.substep_wrist_peak_n)
            if task.logic.phase!=phase and not task.terminated: task.adapter.stop(hold_reference=True)
            if task.substep_contact_peak_n>.1:
                contacts.append(dict(tick=task.tick,peak_n=task.substep_contact_peak_n,
                    pairs=deepcopy(task.contacts+task.applied_contacts)))
            if task.tick%50==0:
                row=traces.get(task.tick-50)
                if row and 'shaped' in row:
                    shaped_error=max(shaped_error,float(np.max(np.abs(task.adapter.velocity-row['shaped']))))
                    measured_error=max(measured_error,float(np.max(np.abs(task.snapshot()['tcp_twist_world']-row['measured']))))
        events=deepcopy(task.logic.events)
        checks=dict(command_count=command_count,condition_max_error=feature_error,shaped_max_error=shaped_error,
            measured_max_error=measured_error,events_equal=events==case['events'],
            termination_equal=task.tick==limit and task.failure_reason==case['failure_reason'] and task.logic.success==case['physical_success'],
            contact_peak_error=abs(task.monitor.peak_n-case['contact_peak_n']),wrist_peak_error=abs(wrist_peak-case['wrist_peak_n']))
        passed=(checks['events_equal'] and checks['termination_equal'] and command_count==len(traces)
            and feature_error<=1e-6 and shaped_error<=1e-10 and measured_error<=1e-10
            and checks['contact_peak_error']<=1e-10 and checks['wrist_peak_error']<=1e-10)
        result=dict(seed=case['seed'],source_episode=case['source_episode'],status='passed' if passed else 'mismatch',
            replay_checks=checks,simulated_s=task.tick*task.dt,wall_s=time.perf_counter()-start,
            failure_reason=task.failure_reason,rows=rows,contacts=contacts,snapshots=snapshots,
            model_inference=False,teacher_executed=False,training_export=False,split='validation',
            reference_note='Teacher path and episode target are references, not unique task targets. Nominal 65deg target is reported separately.')
        write_json(output/f'{case["seed"]}.json',result)
        if not passed: raise ValueError(f'CUDA command replay differs: {checks}')
        return result
    finally: env.close()


def run(args):
    path=Path(args.checkpoint).resolve();digest=sha256(path)
    report_path=Path(args.pickup_report).resolve();report=verify_evidence(report_path,digest)
    base_path=Path(args.base_checkpoint).resolve()
    base=torch.load(base_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(base);audit_data(base,base_path,args.data_report)
    if (report['binding']['base_checkpoint_sha256']!=sha256(base_path)
            or report['binding']['data_report_sha256']!=sha256(args.data_report)
            or report['binding']['source_hashes']!=input_hashes()
            or report['binding']['tool_hashes']!=tool_hashes()):
        raise ValueError('CUDA report ancestry/source/tool binding differs')
    _,hardware=setup(base['config'],'cpu');output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/Path(__file__).name)
    conditions=read_json(report_path.parent/'conditions.json');results=[]
    for episode in report['result']['completed_scenes']:
        name=Path(episode).name;case=read_json(report_path.parent/f'{name}.json')
        result=replay_case(case,conditions[name],base['normalization'],output)
        results.append(dict(seed=result['seed'],status=result['status'],replay_checks=result['replay_checks']))
        print('replayed',name,result['replay_checks'],flush=True)
    unchanged=input_hashes()==base['source_hashes'] and tool_hashes()==report['binding']['tool_hashes']
    if not unchanged: raise ValueError('Source or bound tools changed')
    final=dict(diagnostic=True,mode='cuda_command_failure_replay',checkpoint_sha256=digest,training_step=report['training_step'],weights='ema',
        hardware=hardware,physics_precision='fp64',policy_sampling='none; exact CUDA-issued commands',
        source_unchanged=unchanged,parent=parent,original_source_hashes=base['source_hashes'],binding=report['binding'],
        input_report=str(report_path),input_report_sha256=sha256(report_path),results=results,
        training_export=False,snapshots_split='validation_diagnostic_only',optimizer_updates=0,
        full_validation_run=False,formal_test_run=False,m5_status='incomplete',dp_v1='not_frozen')
    final['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'report.json',final)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base-checkpoint','checkpoint','pickup-report','data-report','output'): parser.add_argument('--'+key,required=True)
    run(parser.parse_args())
