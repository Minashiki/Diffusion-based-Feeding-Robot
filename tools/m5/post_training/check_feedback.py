"""Compare learned correction on recorded train windows and failed live conditions."""

import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import mink
import numpy as np
import torch
from torch.utils.data import default_collate

from acquire_diagnosis import condition_item,sample_from_noise,verify_evidence
from correction_data import MixedCorrection
from diagnose_sampling import spacing_override
from rotation_feedback import early_target,rotation_descriptor,correction_projection,nearest_conditions
from followup import vector
from train_correction import tool_hashes
from feedingrobot.control.adapter import clip_norm
from feedingrobot.data.episodes import input_hashes,write_json
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.policies.runtime import setup,to_device
from feedingrobot.sim.model import load_json


KEYS=('states','history','phase','interaction','state_mask','history_mask')


def summary(rows):
    return dict(windows=len(rows),target_error_deg_p50=float(np.median([r['target_error_deg'] for r in rows])),
        measured_angular_speed_p50=float(np.median([r['measured_angular_speed'] for r in rows])),
        reference_speed_p50=float(np.median([r['reference_speed'] for r in rows])))


def run(args):
    path=Path(args.checkpoint).resolve();digest=sha256(path)
    offline=verify_evidence(args.offline_report,digest);rollout=verify_evidence(args.pickup_report,digest)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if checkpoint['binding']!=offline['binding'] or checkpoint['binding']!=rollout['binding']:
        raise ValueError('Evidence/checkpoint binding mismatch')
    if input_hashes()!=checkpoint['source_hashes'] or tool_hashes()!=checkpoint['binding']['tool_hashes']:
        raise ValueError('Source or bound tool version changed')
    config,norm=checkpoint['config'],checkpoint['normalization']
    device,hardware=setup(config,'cpu');output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    data_report=read_json(args.data_report)
    if sha256(args.data_report)!=checkpoint['binding']['data_report_sha256']:
        raise ValueError('Correction data evidence differs')
    mixed=MixedCorrection(config,norm,data_report);fields=mixed.extra.fields
    robot=load_json('configs/robots/panda.json');base=mink.SO3(np.asarray(robot['base_quaternion'])).as_matrix()
    items,records=[],[]
    max_label_difference=0.
    for e,pool in sorted(mixed.pools.items()):
        metadata=mixed.extra.episodes[e][1]
        target=early_target(dict(metadata,teacher_parameters=metadata['teacher_config']['teacher']))
        for i in pool:
            _,start,end,phase=mixed.extra.windows[i];arrays=mixed.extra.arrays(e)
            obs=arrays['action_observations'][start];descriptor=rotation_descriptor(obs,target,fields)
            reference=clip_norm(base.T@(metadata['teacher_config']['teacher']['orientation_gain']*np.asarray(descriptor['target_error_world'])),robot['angular_speed_limit'])
            max_label_difference=max(max_label_difference,float(np.max(np.abs(reference-arrays['actions'][start,3:]))))
            item=mixed.extra[i];items.append({k:item[k] for k in KEYS})
            records.append(dict(kind='correction_train',episode=mixed.extra.episodes[e][0].name,
                tick=int(arrays['action_ticks'][start]),**descriptor,reference=reference.tolist(),
                reference_speed=float(np.linalg.norm(reference)),position=obs[fields['tcp_position']].tolist()))
    if max_label_difference>1e-6: raise ValueError('Correction teacher angular law differs from labels')
    correction_count=len(items)
    conditions=read_json(Path(args.pickup_report).parent/'conditions.json')
    for episode,plans in conditions.items():
        case=read_json(Path(args.pickup_report).parent/f'{episode}.json')
        metadata=read_json(Path(case['source_episode'])/'manifest.json');target=early_target(metadata)
        traces={r['tick']:r for r in case['trace']}
        for plan in plans:
            if plan['phase']!='ACQUIRE' or not 2000<=plan['tick']<=5000: continue
            obs=np.asarray(plan['states'])[-1,:len(norm['observation_mean'])]*np.asarray(norm['observation_std'])+np.asarray(norm['observation_mean'])
            descriptor=rotation_descriptor(obs,target,fields)
            reference=clip_norm(base.T@(metadata['teacher_config']['teacher']['orientation_gain']*np.asarray(descriptor['target_error_world'])),robot['angular_speed_limit'])
            item=condition_item(plan);items.append(item)
            records.append(dict(kind='live',episode=episode,tick=plan['tick'],**descriptor,
                reference=reference.tolist(),reference_speed=float(np.linalg.norm(reference)),
                position=obs[fields['tcp_position']].tolist(),command=traces[plan['tick']]['command'][3:],
                actual_projection=correction_projection(np.asarray(traces[plan['tick']]['command'][3:]),reference)))
    matrices=np.stack([vector({k:v.numpy() for k,v in item.items()}) for item in items])
    nearest=nearest_conditions(matrices[:correction_count],matrices[correction_count:])
    for row,match in zip(records[correction_count:],nearest):
        reference=records[match['index']]
        row['nearest_correction']=dict(match,episode=reference['episode'],tick=reference['tick'],
            target_error_deg=reference['target_error_deg'],position_gap_mm=float(np.linalg.norm(np.asarray(row['position'])-reference['position'])*1000))
    predictions={}
    for label,weight_path in (('base_50000',Path(args.base_checkpoint)),('trained_100000',path)):
        weight_path=weight_path.resolve()
        if label=='base_50000' and sha256(weight_path)!=checkpoint['binding']['base_checkpoint_sha256']:
            raise ValueError('Original EMA checkpoint differs')
        weights=torch.load(weight_path,map_location='cpu',weights_only=False,mmap=True)
        model=ActionDiT(config).to(device);model.load_state_dict(weights['ema']);model.eval().requires_grad_(False)
        predictions[label]={}
        with torch.no_grad(),spacing_override('leading'):
            for seed in (0,1,2):
                rows=[]
                for start in range(0,len(items),8):
                    batch=to_device(default_collate(items[start:start+8]),device)
                    noise=torch.randn((len(batch['states']),model.horizon,6),generator=torch.Generator().manual_seed(seed),device=device)
                    rows.extend(sample_from_noise(model,config,batch,norm,noise).cpu().numpy()[:,0,3:])
                angular=np.asarray(rows)
                if not np.isfinite(angular).all(): raise FloatingPointError('Nonfinite feedback sample')
                groups={}
                for kind in ('correction_train','live'):
                    indices=[i for i,r in enumerate(records) if r['kind']==kind and r['reference_speed']>=.05]
                    estimates=angular[indices];references=np.asarray([records[i]['reference'] for i in indices])
                    groups[kind]=dict(windows=len(indices),angular_vector_rmse=float(np.sqrt(np.mean(np.sum((estimates-references)**2,axis=1)))),
                        predicted_speed_p50=float(np.median(np.linalg.norm(estimates,axis=1))),
                        projection_ratio_p50=float(np.median([correction_projection(a,b) for a,b in zip(estimates,references)])))
                predictions[label][str(seed)]=groups
                print(label,seed,groups,flush=True)
        del model,weights
    script=output/Path(__file__).name;shutil.copyfile(__file__,script)
    report=dict(diagnostic=True,mode='trained_feedback',checkpoint_sha256=digest,training_step=checkpoint['step'],weights='ema',
        hardware=hardware,precision='fp32',source_unchanged=input_hashes()==checkpoint['source_hashes'],
        original_source_hashes=checkpoint['source_hashes'],binding=checkpoint['binding'],
        inputs={str(Path(p).resolve()):sha256(p) for p in (args.offline_report,args.pickup_report,args.data_report)},
        model_comparison=predictions,records=records,correction_train=summary(records[:correction_count]),
        live=summary(records[correction_count:]),teacher_label_maximum_difference=max_label_difference,
        noise_seeds=[0,1,2],paired_initial_noise=True,inference_action_mask='all valid',teacher_reference_executed=False,
        limitation='Local early teacher angular reference is a diagnostic, not a safe complete off-demonstration controller. Normalized feature RMS is not a calibrated OOD threshold.',
        formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete',evidence_sha256={script.name:sha256(script)})
    write_json(output/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','base-checkpoint','offline-report','pickup-report','data-report','output'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
