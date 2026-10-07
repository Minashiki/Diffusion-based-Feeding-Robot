"""Audit a trained correction fork and compare samplers on validation windows."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch

from correction_data import audit_data
from diagnose_sampling import check_checkpoint,offline,sampler_settings,selected_windows
from train_correction import fork_check,tool_hashes
from feedingrobot.data.episodes import input_hashes,write_json
from feedingrobot.policies.audit import sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import ActionDiT
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT,load_json


def check_records(path,step):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(50001,step+1)):
        raise ValueError('Training records are not continuous from the 50k warm start')
    if not all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
               and math.isfinite(r.get('validation_loss',0.)) for r in rows):
        raise ValueError('Nonfinite training record')
    return dict(rows=len(rows),first_step=rows[0]['step'],last_step=rows[-1]['step'],continuous=True,
        finite=True,last_record=rows[-1],validation=[r for r in rows if 'validation_loss' in r])


def run(args):
    base_path=Path(args.base_checkpoint).resolve();path=Path(args.checkpoint).resolve()
    data_report=Path(args.data_report).resolve();output=Path(args.output).resolve()
    base=torch.load(base_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(base);data=audit_data(base,base_path,data_report)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    source=input_hashes();original_tools=tool_hashes()
    binding=dict(base_checkpoint_sha256=sha256(base_path),data_report_sha256=sha256(data_report),
        source_hashes=source,tool_hashes=original_tools,normalization=base['normalization'],
        fork_kind='m5_rotation_correction_v1',sampling_fraction=.25,workers=0,seed=base['config']['training']['seed'])
    fork_check(checkpoint,binding,path.parent,path)
    if (checkpoint['config']!=base['config'] or checkpoint['normalization']!=base['normalization']
            or checkpoint['source_hashes']!=source):
        raise ValueError('Checkpoint configuration/normalization/source differs')
    training=check_records(path.parent/'metrics.jsonl',checkpoint['step'])
    device,hardware=setup(base['config'],args.device)
    for kind in ('model','ema'):
        if not all(torch.isfinite(v).all() for v in checkpoint[kind].values()):
            raise FloatingPointError(f'Nonfinite {kind} weights; physical validation blocked')
    model=ActionDiT(base['config']).to(device)
    model.load_state_dict(checkpoint['ema']);model.eval().requires_grad_(False)
    dataset=ActionWindows(ROOT/base['config']['dataset'],'validation',model.horizon,base['normalization'])
    cases=selected_windows(dataset);robot=load_json('configs/robots/panda.json')
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'tool_snapshot';snapshot.mkdir()
    for name in original_tools: shutil.copyfile(Path(__file__).resolve().parents[1]/name,snapshot/name)
    shutil.copyfile(__file__,snapshot/Path(__file__).name)
    provenance=dict(schema_version=1,mode='trained_fork_offline',diagnostic=True,checkpoint=str(path),
        checkpoint_sha256=sha256(path),training_step=checkpoint['step'],weights='ema',parent=parent,binding=binding,
        original_source_hashes=source,original_tools=original_tools,training_records=training,hardware=hardware,
        tool_sha256={p.name:sha256(p) for p in snapshot.iterdir()},normalization=base['normalization'],
        precision='bf16_autocast' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32',
        samplers={s:sampler_settings(base['config'],s) for s in ('trailing','leading')},
        optimizer_updates_this_run=0,formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete',
        data_report_sha256=sha256(data_report),correction_episodes=len(data['accepted']))
    write_json(output/'provenance.json',provenance)
    result=offline(model,base['config'],base['normalization'],device,output,dataset,cases,
        (robot['linear_speed_limit'],robot['angular_speed_limit']))
    unchanged=input_hashes()==source and tool_hashes()==original_tools
    if not unchanged: raise ValueError('Runtime source or bound tools changed')
    report=dict(provenance,result=result,source_unchanged=unchanged)
    report['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'report.json',report)
    print(output/'report.json',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-report',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    run(parser.parse_args())
