"""Sample true held-out entry/sweep windows and CUDA failure conditions offline."""

import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import default_collate

from acquire_diagnosis import condition_item,sample_from_noise,verify_evidence
from diagnose_sampling import spacing_override,action_metrics
from train_correction import tool_hashes
from feedingrobot.data.episodes import input_hashes,write_json
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.policies.runtime import setup,to_device
from feedingrobot.sim.model import ROOT


KEYS=('states','history','phase','interaction','state_mask','history_mask')


def run(args):
    path=Path(args.checkpoint).resolve();digest=sha256(path)
    evidence=verify_evidence(args.pickup_report,digest)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if checkpoint['binding']!=evidence['binding'] or input_hashes()!=checkpoint['source_hashes'] or tool_hashes()!=checkpoint['binding']['tool_hashes']:
        raise ValueError('Trained checkpoint/source/tool binding differs')
    config,norm=checkpoint['config'],checkpoint['normalization'];device,hardware=setup(config,'cpu')
    dataset=ActionWindows(ROOT/config['dataset'],'validation',config['model']['horizon'],norm)
    ticks=set(range(9050,11451,200))
    names={Path(p).name for p in evidence['result']['completed_scenes']}
    items,records,labels,masks=[],[],[],[]
    for i,(e,start,end,phase) in enumerate(dataset.windows):
        directory,metadata=dataset.episodes[e];tick=int(dataset.arrays(e)['action_ticks'][start])
        if directory.name not in names or phase!=1 or tick not in ticks: continue
        item=dataset[i];items.append({k:item[k] for k in KEYS})
        labels.append(item['actions'].numpy()*np.asarray(norm['action_std'])+np.asarray(norm['action_mean']))
        masks.append(item['action_mask'].numpy())
        stage=next(r['stage'] for r in reversed(metadata['comparison']['action_starts']) if r['time']<=tick*.001+1e-8)
        records.append(dict(kind='teacher',episode=directory.name,tick=tick,teacher_stage=stage))
    teacher_count=len(items)
    conditions=read_json(Path(args.pickup_report).parent/'conditions.json')
    for name,plans in conditions.items():
        for plan in plans:
            if plan['tick'] not in ticks: continue
            items.append(condition_item(plan));records.append(dict(kind='live',episode=name,tick=plan['tick']))
    model=ActionDiT(config).to(device);model.load_state_dict(checkpoint['ema']);model.eval().requires_grad_(False)
    predictions,metrics={},{}
    with torch.no_grad(),spacing_override('leading'):
        for seed in (0,1,2):
            parts=[]
            for start in range(0,len(items),8):
                batch=to_device(default_collate(items[start:start+8]),device)
                noise=torch.randn((len(batch['states']),model.horizon,6),generator=torch.Generator().manual_seed(seed),device=device)
                parts.append(sample_from_noise(model,config,batch,norm,noise).cpu().numpy())
            prediction=np.concatenate(parts)
            if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite transition predictions')
            predictions[str(seed)]=prediction
            groups={}
            for stage in sorted({r['teacher_stage'] for r in records[:teacher_count]}):
                indices=[i for i,r in enumerate(records[:teacher_count]) if r['teacher_stage']==stage]
                mask=np.asarray(masks)[indices].copy();mask[:,4:]=False
                truth=np.asarray(labels)[indices];estimate=prediction[indices]
                legal=mask
                ref=truth[legal,:3];pred=estimate[legal,:3]
                active=np.linalg.norm(ref,axis=1)>=.002
                groups[stage]=dict(metrics=action_metrics(estimate,truth,mask,(.05,.5)),
                    reference_horizontal_speed_p50=float(np.median(np.linalg.norm(ref[:,:2],axis=1))),
                    predicted_horizontal_speed_p50=float(np.median(np.linalg.norm(pred[:,:2],axis=1))),
                    reference_z_p50=float(np.median(ref[:,2])),predicted_z_p50=float(np.median(pred[:,2])),
                    linear_projection_p50=float(np.median(np.sum(ref[active]*pred[active],axis=1)/np.sum(ref[active]**2,axis=1))))
            live=prediction[teacher_count:,:4,:3].reshape(-1,3)
            groups['live']=dict(windows=len(items)-teacher_count,predicted_horizontal_speed_p50=float(np.median(np.linalg.norm(live[:,:2],axis=1))),
                predicted_z_p50=float(np.median(live[:,2])),predicted_angular_speed_p50=float(np.median(np.linalg.norm(prediction[teacher_count:,:4,3:],axis=2))))
            metrics[str(seed)]=groups;print(seed,groups,flush=True)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'samples.npz',truth=np.asarray(labels),mask=np.asarray(masks),**predictions)
    shutil.copyfile(__file__,output/Path(__file__).name)
    report=dict(diagnostic=True,mode='entry_transition_offline',checkpoint_sha256=digest,training_step=checkpoint['step'],weights='ema',
        source_unchanged=input_hashes()==checkpoint['source_hashes'],binding=checkpoint['binding'],hardware=hardware,precision='fp32',
        input_report=str(Path(args.pickup_report).resolve()),input_report_sha256=sha256(args.pickup_report),records=records,
        teacher_windows=teacher_count,live_windows=len(items)-teacher_count,noise_seeds=[0,1,2],metrics=metrics,
        inference_action_mask='all valid; label mask used only for teacher error statistics',
        limitation='Live predictions have no ground-truth labels. Teacher entry/sweep targets depend on recorded teacher parameters.',
        optimizer_updates=0,formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete')
    report['evidence_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','pickup-report','output'): parser.add_argument('--'+key,required=True)
    run(parser.parse_args())
