"""Separate training bindings; preserve the completed v4 collection audit."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import correction_v4
import diffusers
import torch

from correction_v4.run import inputs,verify_report,verify_files,tools_hashes
from correction_v3.run import unchanged
from feedingrobot.policies.audit import sha256
from feedingrobot.policies.runtime import setup

START=100000
FORK='m5_descent_alignment_v4'


def entry_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def evidence(args):
    inherited=SimpleNamespace(base_checkpoint=args.base_checkpoint,checkpoint=args.parent_checkpoint,
        v1_report=args.v1_report,data_report=args.old_data_report)
    base,parent,_,_,_,_,collection,_=inputs(inherited)
    data=verify_report(args.data_report,collection,'ready_experimental')
    audit=verify_report(args.audit_report,collection,'audit_passed')
    if (audit.get('step')!=START or audit.get('episodes')!=44
            or not audit.get('model_exact') or not audit.get('ema_exact')
            or audit.get('optimizer_created') is not False or audit.get('optimizer_updates')!=0
            or audit.get('collection_sha256')!=sha256(args.data_report)):
        raise ValueError('Requires the matching passed v4 zero-update audit')
    verify_files(data['dataset'],data['dataset_sha256'])
    calibration=verify_report(data['calibration_report'],collection,'passed')
    if len(calibration['pairs'])!=32 or sha256(data['calibration_report'])!=data['calibration_sha256']:
        raise ValueError('Calibration binding differs')
    binding=dict(fork_kind=FORK,collection=collection,entry_tools=entry_hashes(),
        report_sha256={str(Path(p).resolve()):sha256(p) for p in (args.data_report,args.audit_report,args.old_data_report)},
        dataset=data['dataset'],dataset_sha256=data['dataset_sha256'],workers=0,cpu_budget=6,
        torch_threads=6,torch_version=torch.__version__,diffusers_version=diffusers.__version__)
    return base,parent,dict(step=START,ema_sha256=collection['legacy']['parent_checkpoint_sha256']),data,calibration,binding


def compute(config,threads):
    if threads!=6: raise ValueError('v4 training uses exactly six CPU threads')
    runtime=deepcopy(config);runtime['training'].update(cpu_budget=6,torch_threads=6,workers=0)
    return setup(runtime,'cuda')


def check_unchanged(binding):
    unchanged(binding['collection']['legacy'])
    if tools_hashes()!=binding['collection']['v4_tools'] or entry_hashes()!=binding['entry_tools']:
        raise ValueError('Bound collection or training tools changed')
    for path,digest in binding['report_sha256'].items():
        if sha256(path)!=digest: raise ValueError('Bound report changed')
    verify_files(binding['dataset'],binding['dataset_sha256'])


def check_fork(checkpoint,binding,config,normalization):
    if (checkpoint.get('schema_version')!=4 or checkpoint.get('fork_kind')!=FORK
            or checkpoint.get('binding')!=binding or checkpoint.get('config')!=config
            or checkpoint.get('normalization')!=normalization or checkpoint.get('step',0)<=START):
        raise ValueError('v4 resume ancestry/data/tool/runtime mismatch')
    for kind in ('model','ema'):
        if not all(torch.isfinite(v).all() for v in checkpoint[kind].values()):
            raise FloatingPointError(f'Nonfinite v4 {kind} weights')


def input_arguments(parser):
    for name in ('base-checkpoint','parent-checkpoint','v1-report','old-data-report','data-report','audit-report','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--cpu-threads',type=int,choices=(6,),default=6)
