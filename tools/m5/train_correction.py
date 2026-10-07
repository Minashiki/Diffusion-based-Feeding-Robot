"""Experimental EMA warm start with audited correction data; never formal resume."""

import argparse
import copy
import json
from pathlib import Path
import shutil

import diffusers
import numpy as np
import torch

from correction_data import audit_data,MixedCorrection
from diagnose_sampling import check_checkpoint,sampler_settings
from feedingrobot.data.episodes import input_hashes,write_json
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import ActionDiT,noise_loss,schedulers
from feedingrobot.policies.runtime import precision,restore_rng,rng_state,save_checkpoint,setup,to_device
from feedingrobot.policies.training import update_ema,validation_loss
from feedingrobot.sim.model import ROOT


FORK='m5_rotation_correction_v1'


def tool_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def fork_check(checkpoint,binding,output,resume):
    if (checkpoint.get('schema_version')!=2 or checkpoint.get('diagnostic') is not True
            or checkpoint.get('fork_kind')!=FORK or checkpoint.get('binding')!=binding):
        raise ValueError('Experimental checkpoint ancestry/data/tool version mismatch')
    if Path(resume).resolve().parent!=Path(output).resolve():
        raise ValueError('Resume must stay in its experimental run directory')
    if checkpoint['step']<50000: raise ValueError('Invalid cumulative update count')


def initialize(model,base):
    model.load_state_dict(base['ema'])


def run(args):
    base_path=Path(args.checkpoint).resolve();report_path=Path(args.data_report).resolve()
    base=torch.load(base_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(base);data=audit_data(base,base_path,report_path)
    config=base['config'];settings=config['training'];source=input_hashes()
    device,hardware=setup(config,args.device)
    binding=dict(base_checkpoint_sha256=sha256(base_path),data_report_sha256=sha256(report_path),
        source_hashes=source,tool_hashes=tool_hashes(),normalization=base['normalization'],
        fork_kind=FORK,sampling_fraction=.25,workers=0,seed=settings['seed'])
    output=Path(args.output).resolve()
    resumed=None
    if args.resume:
        if args.mode!='train': raise ValueError('Audit does not resume optimizer state')
        resumed=torch.load(args.resume,map_location='cpu',weights_only=False)
        fork_check(resumed,binding,output,args.resume)
        if (resumed['config']!=config or resumed['normalization']!=base['normalization']
                or resumed['source_hashes']!=source):
            raise ValueError('Experimental checkpoint configuration/source/normalization differs')
    start=resumed['step'] if resumed else 50000
    if args.mode=='train' and (args.updates is None or args.updates<=start):
        raise ValueError('--updates is required and must exceed the current cumulative step')
    if not resumed: output.mkdir(parents=True,exist_ok=False)
    mixed=MixedCorrection(config,base['normalization'],data)
    model=ActionDiT(config).to(device);initialize(model,base)
    # Reset randomness after model construction: new branch, not original optimizer/RNG resume.
    torch.manual_seed(settings['seed']);np.random.seed(settings['seed'])
    if device.type=='cuda': torch.cuda.manual_seed_all(settings['seed'])
    rng=np.random.default_rng(settings['seed'])
    scheduler,_=schedulers(config)
    provenance=dict(schema_version=2,diagnostic=True,fork_kind=FORK,binding=binding,
        base_checkpoint=str(base_path),base_checkpoint_sha256=sha256(base_path),
        data_report=str(report_path),original_source_hashes=source,diffusers_version=diffusers.__version__,
        precision='bf16_autocast' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32',
        config=config,parent=parent,hardware=hardware,normalization=base['normalization'],
        sampler=sampler_settings(config,'leading'),initialization='formal 50000 EMA; fresh optimizer and RNG',
        data=mixed.summary(),effective_workers=0,automatic_closed_loop=False,
        formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete')
    if not resumed:
        snapshot=output/'tool_snapshot';snapshot.mkdir()
        for name in binding['tool_hashes']: shutil.copyfile(Path(__file__).parent/name,snapshot/name)
        write_json(output/'provenance.json',provenance)
    if args.mode=='audit':
        model.eval()
        with torch.no_grad(),precision(device):
            batch=to_device(mixed.batch(rng,4),device)
            loss=noise_loss(model,scheduler,batch)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite audit forward')
        exact=all(torch.equal(v.detach().cpu(),base['ema'][k]) for k,v in model.state_dict().items())
        if not exact: raise ValueError('EMA initialization differs')
        if input_hashes()!=source or tool_hashes()!=binding['tool_hashes']: raise ValueError('Runtime source changed')
        result=dict(provenance,status='audit_passed',step=50000,optimizer_updates=0,
            optimizer_created=False,ema_exact=exact,forward_loss=float(loss),source_unchanged=True)
        result['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
        write_json(output/'report.json',result);return result
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    step=start
    if resumed:
        model.load_state_dict(resumed['model']);ema.load_state_dict(resumed['ema'])
        optimizer.load_state_dict(resumed['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value): state[key]=value.to(device)
        restore_rng(resumed['rng'],rng)
    validation=ActionWindows(ROOT/config['dataset'],'validation',config['model']['horizon'],base['normalization'])
    def payload():
        if input_hashes()!=source or tool_hashes()!=binding['tool_hashes']: raise ValueError('Runtime source changed')
        return dict(provenance,step=step,source_hashes=source,model=model.state_dict(),ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),rng=rng_state(rng),best_score=-1.)
    write_json(output/'status.json',dict(status='training_experimental',step=step,target=args.updates))
    with (output/'metrics.jsonl').open('a') as metrics:
        while step<args.updates:
            batch=to_device(mixed.batch(rng,settings['batch_size']),device)
            model.train();optimizer.zero_grad(set_to_none=True)
            with precision(device): loss=noise_loss(model,scheduler,batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite diffusion loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();update_ema(ema,model,settings['ema_decay']);step+=1
            record=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm))
            if step%settings['validation_every']==0:
                record['validation_loss']=validation_loss(ema,validation,scheduler,config,device)
                if not np.isfinite(record['validation_loss']): raise FloatingPointError('Nonfinite validation loss')
            metrics.write(json.dumps(record)+'\n')
            if step%100==0 or step==args.updates:
                metrics.flush();print(f'update={step}/{args.updates} loss={record["loss"]:.6f}',flush=True)
            if step%settings['checkpoint_every']==0:
                save_checkpoint(output/f'step_{step}.pt',payload());save_checkpoint(output/'last.pt',payload())
        save_checkpoint(output/f'step_{step}.pt',payload());save_checkpoint(output/'last.pt',payload())
    result=dict(status='trained_experimental_not_released',step=step,optimizer_updates=step-50000,
        m5_status='incomplete',test_status='not_run',dp_v1='not_frozen')
    write_json(output/'status.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('audit','train'))
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--data-report',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--updates',type=int);parser.add_argument('--resume')
    args=parser.parse_args();print(run(args),flush=True)
