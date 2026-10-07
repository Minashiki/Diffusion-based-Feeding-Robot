"""Offline early rotation feedback and exhaustive early-train condition coverage."""

import argparse
from pathlib import Path
import shutil

import mink
import numpy as np
import torch
from torch.utils.data import default_collate

from acquire_diagnosis import sample_from_noise, verify_evidence
from diagnose_sampling import check_checkpoint, spacing_override
from followup import vector
from step_sensitivity import inference_config, noise_for, settings
from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.experts.teacher import rotation_error
from feedingrobot.control.adapter import clip_norm
from feedingrobot.policies import dit
from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.runtime import precision, setup, to_device
from feedingrobot.sim.model import ROOT, load_json


KEYS=('states','history','phase','interaction','state_mask','history_mask')


def early_target(metadata):
    acquisition=load_json('configs/acceptance.json')['beans_native']['m1c']
    yaw=np.deg2rad(acquisition['entry_yaw_deg'])
    pitch=np.deg2rad(acquisition['entry_pitch_deg'])+metadata['teacher_parameters']['entry_pitch_offset_rad']
    return (mink.SO3.exp(np.array([0.,0.,yaw]))@mink.SO3.exp(np.array([0.,pitch,0.]))).as_matrix()


def rotation_descriptor(observation,target,fields):
    error=rotation_error(target,np.asarray(observation)[fields['tcp_rotation']].reshape(3,3))
    angular=np.asarray(observation)[fields['tcp_twist_world']][3:]
    return dict(target_error_deg=float(np.rad2deg(np.linalg.norm(error))),
        target_error_world=error.tolist(),measured_angular_speed=float(np.linalg.norm(angular)))


def residual_slow(row):
    return 3.<=row['target_error_deg']<=10. and row['measured_angular_speed']<.05


def correction_projection(angular,reference):
    denominator=np.dot(reference,reference)
    return float(np.dot(angular,reference)/denominator) if denominator>1e-12 else None


def plan_item(plan):
    return {k:torch.as_tensor(plan['condition'][k],dtype=torch.bool if k.endswith('_mask') else torch.long if k=='phase' else torch.float32) for k in KEYS}


def nearest_conditions(matrix,queries):
    # Block the search to respect the original CPU budget without a giant 3-D tensor.
    values=[]
    for query in queries:
        best_distance,best_index=np.inf,None
        for start in range(0,len(matrix),512):
            distance=np.mean((matrix[start:start+512]-query)**2,axis=1)
            index=int(distance.argmin())
            if distance[index]<best_distance:
                best_distance,best_index=float(distance[index]),start+index
        values.append(dict(index=best_index,normalized_condition_rms_distance=float(np.sqrt(best_distance))))
    return values


def coverage(train,validation,live,episode):
    records,matrix=[],[]
    robot=load_json('configs/robots/panda.json')
    base=mink.SO3(np.array(robot['base_quaternion'])).as_matrix()
    maximum_label_difference=0.
    for i,(e,start,end,phase) in enumerate(train.windows):
        arrays=train.arrays(e)
        tick=int(arrays['action_ticks'][start])
        if phase!=1 or tick>5000: continue
        path,metadata=train.episodes[e]
        target=early_target(metadata)
        descriptor=rotation_descriptor(arrays['action_observations'][start],target,train.fields)
        reference=clip_norm(base.T@(metadata['teacher_config']['teacher']['orientation_gain']*np.array(descriptor['target_error_world'])),robot['angular_speed_limit'])
        maximum_label_difference=max(maximum_label_difference,float(np.max(np.abs(reference-arrays['actions'][start,3:]))))
        records.append(dict(index=i,episode=str(path),tick=tick,kind='recovery' if metadata['scenario'].get('recover',False) else 'normal',
            **descriptor,teacher_angular_speed=float(np.linalg.norm(arrays['actions'][start,3:]))))
        matrix.append(vector({k:v.numpy() for k,v in train[i].items()}).astype(np.float32))
    matrix=np.stack(matrix)
    if maximum_label_difference>1e-6:
        raise ValueError('Early angular reference differs from train labels')
    reference=[]
    validation_values=[]
    for i,(e,start,end,phase) in enumerate(validation.windows):
        arrays=validation.arrays(e);tick=int(arrays['action_ticks'][start])
        if phase==1 and tick in range(50,5000,200) and not validation.episodes[e][1]['scenario'].get('recover',False):
            validation_values.append(vector({k:v.numpy() for k,v in validation[i].items()}).astype(np.float32))
    validation_distances=nearest_conditions(matrix,validation_values)
    for plan,nearest in zip(live,nearest_conditions(matrix,[vector(p['condition']).astype(np.float32) for p in live])):
        nearest['nearest_train']=records[nearest.pop('index')]
        reference.append(dict(tick=plan['tick'],**nearest))
    slow=[r for r in records if residual_slow(r)]
    return dict(train_selection='Every legal ACQUIRE train window at tick <=5000; no subsampling inside that interval.',
        train_windows=len(records),train_episodes=len({r['episode'] for r in records}),
        train_angular_reference_maximum_label_difference=maximum_label_difference,
        residual_slow_definition='Target error 3..10 degrees AND measured angular speed <0.05 rad/s; exploratory descriptor, not calibrated OOD.',
        train_residual_slow_windows=len(slow),train_residual_slow_by_kind={kind:sum(r['kind']==kind for r in slow) for kind in ('normal','recovery')},
        train_residual_slow_examples=slow[:20],validation_reference_windows=len(validation_values),
        validation_distance_p50=float(np.median([r['normalized_condition_rms_distance'] for r in validation_distances])),
        validation_distance_p95=float(np.quantile([r['normalized_condition_rms_distance'] for r in validation_distances],.95)),
        live_nearest=reference,episode=episode,
        limitation='Exhaustive only within early train ACQUIRE; feature RMS is not an encoder metric or a calibrated coverage threshold.')


@torch.no_grad()
def analyze(model,config,norm,device,validation,train,rollout,output):
    episode=Path(rollout['source_episode']);metadata=read_json(episode/'manifest.json')
    target=early_target(metadata)
    robot=load_json('configs/robots/panda.json')
    base=mink.SO3(np.array(robot['base_quaternion'])).as_matrix()
    gain=metadata['teacher_config']['teacher']['orientation_gain']
    fields=validation.fields
    e=next(i for i,(path,m) in enumerate(validation.episodes) if path==episode)
    arrays=validation.arrays(e)
    early=(arrays['action_ticks']<5000)&(arrays['action_ticks']>0)
    recomputed=np.array([clip_norm(base.T@(gain*rotation_error(target,r[fields['tcp_rotation']].reshape(3,3))),robot['angular_speed_limit']) for r in arrays['action_observations'][early]])
    difference=float(np.max(np.abs(recomputed-arrays['actions'][early,3:])))
    if difference>1e-6: raise ValueError('Early angular reference does not reproduce teacher labels')
    trace={r['tick']:r for r in rollout['trace']}
    rows=[]
    for tick,r in trace.items():
        if not 2000<=tick<5000: continue
        descriptor=rotation_descriptor(r['observation'],target,fields)
        reference=clip_norm(base.T@(gain*np.array(descriptor['target_error_world'])),robot['angular_speed_limit'])
        rows.append(dict(tick=tick,**descriptor,teacher_local_angular_reference=reference.tolist(),
            reference_speed=float(np.linalg.norm(reference)),
            angular={key:np.asarray(r[key])[3:].tolist() for key in ('predicted','command','shaped','measured') if key in r},
            correction_ratio=correction_projection(np.array(r['command'][3:]),reference)))
    live=[p for p in rollout['plans'] if 2000<=p['tick']<5000]
    windows={(int(validation.arrays(j)['action_ticks'][start])):i for i,(j,start,end,phase) in enumerate(validation.windows) if j==e and phase==1}
    items,cases,reference=[],[],[]
    for plan in live:
        teacher=validation[windows[plan['tick']]]
        r=trace[plan['tick']]
        error=rotation_error(target,np.asarray(r['observation'])[fields['tcp_rotation']].reshape(3,3))
        for domain in ('teacher','live'):
            items.append({k:teacher[k] for k in KEYS} if domain=='teacher' else plan_item(plan))
            cases.append(dict(tick=plan['tick'],domain=domain))
            reference.append((teacher['actions'][0].numpy()*np.array(norm['action_std'])+np.array(norm['action_mean']))[3:] if domain=='teacher'
                else clip_norm(base.T@(gain*error),robot['angular_speed_limit']))
    samples=[]
    with spacing_override('leading'),precision(device):
        for seed in (0,1,2):
            predictions=[]
            for start in range(0,len(items),8):
                batch=to_device(default_collate(items[start:start+8]),device)
                noise=noise_for(cases[start:start+8],seed,model.horizon,device)
                predictions.append(sample_from_noise(model,inference_config(config,20),batch,norm,noise).float().cpu().numpy())
            samples.append(np.concatenate(predictions))
            print(f'rotation feedback seed={seed}: {len(items)} teacher/live windows',flush=True)
    samples=np.stack(samples);reference=np.array(reference)
    finite=bool(np.isfinite(samples).all())
    np.savez_compressed(output/'samples.npz',prediction=samples,current_angular_reference=reference)
    evaluations=[]
    if finite:
        for seed,prediction in enumerate(samples):
            for i,case in enumerate(cases):
                angular=clip_norm(prediction[i,0,3:],robot['angular_speed_limit'])
                evaluations.append(dict(seed=seed,**case,predicted_angular=angular.tolist(),
                    reference_speed=float(np.linalg.norm(reference[i])),correction_ratio=correction_projection(angular,reference[i])))
    live_descriptors=[r for r in rows if any(p['tick']==r['tick'] for p in live)]
    result=dict(status='completed' if finite else 'nonfinite_samples',all_samples_finite=finite,
        teacher_reference_validation=dict(maximum_absolute_difference=difference,checked_actions=int(early.sum()),
            target_rotation=target.tolist(),orientation_gain=gain),trace_alignment=rows,
        trace_alignment_note='Observation/error at command start; shaped and measured are saved at the 50ms interval end.',
        model_cases=cases,model_evaluations=evaluations,
        live_residual_slow_windows=sum(residual_slow(r) for r in live_descriptors),
        coverage=coverage(train,validation,live,str(episode)),
        limitation='Local early teacher angular law on live pose is an offline control reference, not a proven safe complete off-demonstration teacher action; never executed.',
        physics_rollouts=0,teacher_in_execution=False)
    return result


def run(args):
    checkpoint_path,output=Path(args.checkpoint).resolve(),Path(args.output).resolve()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(checkpoint);digest=sha256(checkpoint_path)
    previous=verify_evidence(args.previous_report,digest)
    if previous['mode']!='protected_twenty_step_probe' or len(previous['scenes'])!=1 or previous['scenes'][0]['pickup']:
        raise ValueError('Requires the failed single-scene 20-step diagnostic')
    episode=Path(previous['scenes'][0]['source_episode'])
    rollout=read_json(Path(args.previous_report).resolve().parent/f'{episode.name}_leading_20.json')
    config,norm=checkpoint['config'],checkpoint['normalization']
    device,hardware=setup(config,args.device)
    model=dit.ActionDiT(config).to(device);model.load_state_dict(checkpoint['ema']);model.eval().requires_grad_(False)
    validation=ActionWindows(ROOT/config['dataset'],'validation',model.horizon,norm)
    train=ActionWindows(ROOT/config['dataset'],'train',model.horizon,norm)
    output.mkdir(parents=True,exist_ok=False);snapshot=output/'tool_snapshot';snapshot.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file(): shutil.copyfile(file,snapshot/file.name)
    with spacing_override('leading'): sampler=settings(config,20)
    provenance=dict(schema_version=1,mode='rotation_feedback',diagnostic=True,checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest,training_step=50000,weights='ema',parent=parent,source_check='passed',
        original_source_hashes=checkpoint['source_hashes'],training_config=config,normalization=norm,hardware=hardware,
        precision='bf16_autocast' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32',sampler=sampler,
        previous_report=str(Path(args.previous_report).resolve()),previous_report_sha256=sha256(args.previous_report),
        tool_sha256={p.name:sha256(p) for p in snapshot.iterdir()},optimizer_updates=0,formal_test_run=False,
        dp_v1='not_frozen',m5_status='incomplete',scope='offline single failed validation episode plus early train coverage')
    write_json(output/'provenance.json',provenance)
    try: result=analyze(model,config,norm,device,validation,train,rollout,output)
    except Exception as error:
        write_json(output/'report.json',dict(provenance,status='error',error=f'{type(error).__name__}: {error}'));raise
    report=dict(provenance,**result,source_unchanged=input_hashes()==checkpoint['source_hashes'])
    report['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='report.json'}
    write_json(output/'report.json',report)
    if not report['source_unchanged']: raise ValueError('Original runtime inputs changed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--previous-report',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=run(args)
    print('report:',Path(args.output).resolve()/'report.json',flush=True)
    raise SystemExit(0 if report['status']=='completed' else 1)
