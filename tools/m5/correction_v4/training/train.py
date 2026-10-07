"""v4 100k EMA warmstart or same-fork resume; CUDA training only."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from common import START,FORK,check_fork,check_unchanged,compute,evidence,input_arguments
import numpy as np
import torch

from correction_v4.corpus import MixedV4
from correction_v3.corpus import file_hashes
from diagnose_sampling import sampler_settings
from feedingrobot.data.episodes import write_json
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import ActionDiT,noise_loss,schedulers
from feedingrobot.policies.runtime import precision,restore_rng,rng_state,save_checkpoint,to_device
from feedingrobot.policies.training import update_ema,validation_loss
from feedingrobot.sim.model import ROOT


def check_records(path,step):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    if ([r['step'] for r in rows]!=list(range(START+1,step+1))
            or not all(math.isfinite(r[k]) for r in rows for k in ('loss','gradient_norm') )
            or not all(math.isfinite(r.get('validation_loss',0.)) for r in rows)):
        raise ValueError('Metrics must be finite and continuous through the checkpoint step')
    return dict(rows=len(rows),first_step=rows[0]['step'],last_step=rows[-1]['step'])


def run(args):
    if args.cpu_threads!=6: raise ValueError('--cpu-threads must be six')
    base,parent,parent_audit,data,_,binding=evidence(args)
    config=base['config'];settings=config['training'];normalization=base['normalization']
    output=Path(args.output).resolve();resumed=None
    if args.resume:
        if args.mode!='train': raise ValueError('Preflight cannot load optimizer/RNG state')
        if Path(args.resume).resolve().parent!=output: raise ValueError('Resume must stay in its v4 run directory')
        resumed=torch.load(args.resume,map_location='cpu',weights_only=False)
        check_fork(resumed,binding,config,normalization)
        check_records(output/'metrics.jsonl',resumed['step'])
    start=resumed['step'] if resumed else START
    if args.mode=='train' and (args.updates is None or args.updates<=start):
        raise ValueError('--updates is a cumulative target and must exceed the current step')
    mixed=MixedV4(config,normalization,data['dataset'])
    summary=mixed.extra.summary()
    provenance=dict(schema_version=4,diagnostic=True,fork_kind=FORK,binding=binding,
        config=config,normalization=normalization,parent=parent_audit,
        base_checkpoint=str(Path(args.base_checkpoint).resolve()),
        parent_checkpoint=str(Path(args.parent_checkpoint).resolve()),
        data_report=str(Path(args.data_report).resolve()),audit_report=str(Path(args.audit_report).resolve()),
        source_hashes=binding['collection']['legacy']['source_hashes'],data=summary,
        initialization='100000 parent EMA; fresh optimizer/RNG unless resuming this v4 fork',
        sampler=sampler_settings(config,'leading'),effective_workers=0,automatic_closed_loop=False,
        formal_test_run=False,full_validation_run=False,m5_status='incomplete',dp_v1='not_frozen',
        supervision='v4 direction/stable-wait and complete pickup_hold; no controlled-perturbation or braking labels')
    if args.mode=='preflight':
        check_unchanged(binding);output.mkdir(parents=True,exist_ok=False)
        result=dict(provenance,status='preflight_passed',step=START,optimizer_created=False,optimizer_updates=0,
            model_executed=False,source_unchanged=True)
        write_json(output/'report.json',result);return result
    device,hardware=compute(config,args.cpu_threads)
    if not resumed: output.mkdir(parents=True,exist_ok=False)
    provenance.update(hardware=hardware,precision='bf16_autocast' if torch.cuda.is_bf16_supported() else 'fp32')
    if not resumed:
        snapshot=output/'tool_snapshot';snapshot.mkdir()
        for name in binding['entry_tools']: shutil.copyfile(Path(__file__).parent/name,snapshot/name)
        write_json(output/'provenance.json',provenance)
    model=ActionDiT(config).to(device);model.load_state_dict(parent['ema'])
    ema=deepcopy(model).eval().requires_grad_(False)
    del parent,base
    # Fresh RNG follows construction so the parent optimizer/RNG are never inherited.
    torch.manual_seed(settings['seed']);np.random.seed(settings['seed']);torch.cuda.manual_seed_all(settings['seed'])
    rng=np.random.default_rng(settings['seed']);scheduler,_=schedulers(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    if resumed:
        model.load_state_dict(resumed['model']);ema.load_state_dict(resumed['ema'])
        optimizer.load_state_dict(resumed['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value): state[key]=value.to(device)
        restore_rng(resumed['rng'],rng)
        del resumed
    validation=ActionWindows(ROOT/config['dataset'],'validation',config['model']['horizon'],normalization)
    step=start
    def payload():
        check_unchanged(binding)
        return dict(provenance,step=step,model=model.state_dict(),ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),rng=rng_state(rng),optimizer_updates=step-START)
    write_json(output/'status.json',dict(status='training_experimental',step=step,target=args.updates))
    with (output/'metrics.jsonl').open('a') as metrics:
        while step<args.updates:
            batch=to_device(mixed.batch(rng,settings['batch_size']),device)
            model.train();optimizer.zero_grad(set_to_none=True)
            with precision(device): loss=noise_loss(model,scheduler,batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite v4 diffusion loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();update_ema(ema,model,settings['ema_decay']);step+=1
            record=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm))
            if step%settings['validation_every']==0:
                record['validation_loss']=validation_loss(ema,validation,scheduler,config,device)
                if not math.isfinite(record['validation_loss']): raise FloatingPointError('Nonfinite validation loss')
            metrics.write(json.dumps(record)+'\n')
            if step%100==0 or step==args.updates:
                metrics.flush();print(f'update={step}/{args.updates} loss={record["loss"]:.6f}',flush=True)
            if step%settings['checkpoint_every']==0:
                metrics.flush();save_checkpoint(output/f'step_{step}.pt',payload());save_checkpoint(output/'last.pt',payload())
        # Reaudit data and all bound evidence before accepting the final checkpoint.
        evidence(args)
        save_checkpoint(output/f'step_{step}.pt',payload());save_checkpoint(output/'last.pt',payload())
    result=dict(status='trained_experimental_not_released',step=step,optimizer_updates=step-START,
        updates_this_run=step-start,m5_status='incomplete',formal_test_run=False,dp_v1='not_frozen',
        source_unchanged=True,evidence_sha256=file_hashes(output/'tool_snapshot'))
    write_json(output/'status.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','train'));input_arguments(parser)
    parser.add_argument('--updates',type=int);parser.add_argument('--resume')
    args=parser.parse_args();result=run(args)
    print(f'{result["status"]}: {Path(args.output).resolve()}',flush=True)
