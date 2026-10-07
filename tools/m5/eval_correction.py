"""Validation-only diagnostics for experimental correction-fork EMA checkpoints."""

import argparse
from pathlib import Path
import shutil

import torch

from correction_data import audit_data
from diagnose_sampling import check_checkpoint,rollouts,sampler_settings
from train_correction import fork_check,tool_hashes
from feedingrobot.data.episodes import input_hashes,write_json
from feedingrobot.policies.audit import sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT,load_json


def run(args):
    base_path=Path(args.base_checkpoint).resolve();path=Path(args.checkpoint).resolve()
    base=torch.load(base_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(base);audit_data(base,base_path,args.data_report)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    binding=dict(base_checkpoint_sha256=sha256(base_path),data_report_sha256=sha256(args.data_report),
        source_hashes=input_hashes(),tool_hashes=tool_hashes(),normalization=base['normalization'],
        fork_kind='m5_rotation_correction_v1',sampling_fraction=.25,workers=0,seed=base['config']['training']['seed'])
    fork_check(checkpoint,binding,path.parent,path)
    if (checkpoint['config']!=base['config'] or checkpoint['normalization']!=base['normalization']
            or checkpoint['source_hashes']!=binding['source_hashes']):
        raise ValueError('Experimental model configuration/normalization differs')
    device,hardware=setup(base['config'],args.device)
    model=ActionDiT(base['config']).to(device)
    model.load_state_dict(checkpoint['ema']);model.eval().requires_grad_(False)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite EMA: physics skipped')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'tool_snapshot';snapshot.mkdir()
    for name in tool_hashes(): shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    dataset=ActionWindows(ROOT/base['config']['dataset'],'validation',model.horizon,base['normalization'])
    robot=load_json('configs/robots/panda.json')
    provenance=dict(diagnostic=True,mode=args.mode,checkpoint=str(path),checkpoint_sha256=sha256(path),
        base_checkpoint_sha256=sha256(base_path),training_step=checkpoint['step'],weights='ema',parent=parent,
        binding=binding,hardware=hardware,precision='bf16_autocast' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32',
        sampler=sampler_settings(base['config'],'leading'),split='validation',formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete')
    write_json(output/'provenance.json',provenance)
    # Same original protective execution as the 50k leading probe. No historical
    # trailing comparison is attached to the experimental checkpoint.
    result=rollouts(model,base['config'],base['normalization'],device,output,dataset,args.mode,
        (robot['linear_speed_limit'],robot['angular_speed_limit']),path)
    report=dict(provenance,result=result,source_unchanged=input_hashes()==binding['source_hashes'])
    if not report['source_unchanged'] or tool_hashes()!=binding['tool_hashes']:
        raise ValueError('Runtime source changed')
    report['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'report.json',report)
    print(output/'report.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('pickup','validation'))
    parser.add_argument('--base-checkpoint',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-report',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
