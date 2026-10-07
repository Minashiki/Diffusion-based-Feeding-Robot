"""Fixed held-out original windows and explicitly labelled v4 train diagnostics."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from correction_v4.corpus import PickupWindows
from diagnose_sampling import action_metrics,selected_windows
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.runtime import precision,to_device
from feedingrobot.sim.model import ROOT
from policy_rollout import sample_actions


def windows(config,normalization,data):
    original=ActionWindows(ROOT/config['dataset'],'validation',config['model']['horizon'],normalization)
    extra=PickupWindows(data['dataset'],'train',config['model']['horizon'],normalization)
    cases=[];items=[]
    for case in selected_windows(original):
        cases.append(dict(case,source='original_validation',stage=case['phase']))
        items.append(original[case['index']])
    for pool,episodes in extra.pools.items():
        for e,stages in episodes.items():
            path,m=extra.episodes[e];ticks=extra.arrays(e)['action_ticks']
            for stage,indices in stages.items():
                chosen=[indices[0]]
                if stage=='stable_wait':
                    wait=m['handover'].get('stable_start_tick',m['corrected_tick']-200)
                    chosen=[]
                    for age in (0,50,100,150):
                        bucket=[i for i in indices if age<=ticks[extra.windows[i][1]]-wait<age+50]
                        if bucket: chosen.append(bucket[0])
                elif stage=='correcting' and (path/'alignment_boundaries.npy').exists():
                    boundaries=np.load(path/'alignment_boundaries.npy',mmap_mode='r')
                    critical=set(boundaries[(np.abs(boundaries[:,1])>=.00045)&(np.abs(boundaries[:,1])<=.00065),0])
                    matching=[i for i in indices if ticks[extra.windows[i][1]] in critical]
                    if matching: chosen.append(matching[0])
                elif stage=='pickup_hold': chosen.append(indices[-1])
                for i in dict.fromkeys(chosen):
                    _,start,_,_=extra.windows[i]
                    case=dict(source='v4_train_diagnostic',episode=str(path),index=i,pool=pool,stage=stage,
                        category=m['category'],cell=m.get('cell'),tick=int(ticks[start]))
                    if m.get('cell'):
                        from correction_v4.run import cell_key
                        case['cell_key']=cell_key(m['cell'])
                    if stage=='stable_wait':
                        age=int(ticks[start]-wait);case.update(wait_age_ms=age,wait_bin=f'{age//50*50}-{age//50*50+50}')
                    cases.append(case)
                    items.append(extra[i])
    return cases,items


@torch.no_grad()
def evaluate(model,config,normalization,device,output,cases,items):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    labels=default_collate(items);mask=labels['action_mask'].numpy()
    truth=labels['actions'].numpy()*np.asarray(normalization['action_std'])+np.asarray(normalization['action_mean'])
    archives=dict(truth=truth,action_mask=mask);results={}
    # Use the actual frozen robot protection thresholds.
    from feedingrobot.sim.model import load_json
    robot=load_json('configs/robots/panda.json');limits=(robot['linear_speed_limit'],robot['angular_speed_limit'])
    for seed in (0,1,2):
        predictions=[];hashes=[]
        for start in range(0,len(items),8):
            batch=to_device(default_collate(items[start:start+8]),device)
            generator=torch.Generator(device=device).manual_seed(seed)
            noise=torch.randn((len(batch['states']),model.horizon,6),device=device,generator=generator)
            hashes.extend(hashlib.sha256(n.cpu().numpy().tobytes()).hexdigest() for n in noise)
            generator.manual_seed(seed)
            with precision(device): prediction=sample_actions(model,config,batch,normalization,generator)
            predictions.append(prediction.float().cpu().numpy())
        prediction=np.concatenate(predictions)
        if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite offline DDIM actions')
        archives[f'prediction_{seed}']=prediction;groups={}
        group_names={f"source:{c['source']}" for c in cases}|{f"stage:{c['stage']}" for c in cases}
        for field in ('category','cell_key','wait_bin'):
            group_names|={f'{field}:{c[field]}' for c in cases if field in c}
        for group in sorted(group_names):
            field,value=group.split(':',1);ids=[i for i,c in enumerate(cases) if c.get(field)==value]
            prefix=mask[ids].copy();prefix[:,4:]=False
            groups[group]=dict(windows=len(ids),legal_prefix=action_metrics(prediction[ids],truth[ids],prefix,limits),
                legal_horizon=action_metrics(prediction[ids],truth[ids],mask[ids],limits))
        results[str(seed)]=dict(initial_noise_sha256=hashes,groups=groups)
    np.savez_compressed(output/'samples.npz',**archives)
    return dict(status='completed',windows=len(cases),noise_seeds=[0,1,2],by_seed=results,cases=cases,
        inference_action_mask='all valid; teacher masks are used only for error metrics',
        limitation='v4 train windows diagnose supervision fit; they are not independent policy acceptance')
