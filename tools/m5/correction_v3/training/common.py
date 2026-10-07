"""Separate v3 training bindings, leaving collection evidence unchanged."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

# Bound native import-time pools too; compute() narrows the runtime budget to 6 by default.
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
if hasattr(os,'sched_getaffinity'):
    selected=sorted(os.sched_getaffinity(0))[:8]
    for thread in Path('/proc/self/task').iterdir():
        try: os.sched_setaffinity(int(thread.name),selected)
        except ProcessLookupError: pass

import diffusers
import torch
torch.set_num_threads(6)

from acquire_diagnosis import verify_evidence
from correction_v3.corpus import audit_corpus
from correction_v3.run import ancestry,calibration_check,unchanged
from feedingrobot.policies.audit import sha256
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT


FORK='m5_descent_alignment_v3'
START=100000


def entry_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def evidence(args):
    inherited=SimpleNamespace(base_checkpoint=args.base_checkpoint,checkpoint=args.parent_checkpoint,v1_report=args.v1_report)
    base,parent,parent_audit,_,_,_,collection_binding=ancestry(inherited)
    data=audit_corpus(args.data_report,collection_binding,base['normalization'],ROOT/base['config']['dataset'])
    calibration=calibration_check(data['calibration_report'],collection_binding)
    if sha256(data['calibration_report'])!=data['calibration_report_sha256']:
        raise ValueError('Calibration report changed')
    audit=verify_evidence(args.audit_report,collection_binding['parent_checkpoint_sha256'])
    if (audit.get('mode')!='correction_warmstart_audit_v3' or audit.get('status')!='audit_passed'
            or audit.get('binding')!=collection_binding or audit.get('step')!=START
            or audit.get('optimizer_created') is not False or audit.get('optimizer_updates')!=0
            or audit.get('model_exact') is not True or audit.get('ema_exact') is not True
            or audit.get('data_report_sha256')!=sha256(args.data_report)):
        raise ValueError('Requires the matching passed zero-update v3 warmstart audit')
    binding=dict(fork_kind=FORK,collection=collection_binding,data_report_sha256=sha256(args.data_report),
        audit_report_sha256=sha256(args.audit_report),entry_tools=entry_hashes(),sampling_fraction=.25,
        workers=0,cpu_budget=args.cpu_threads,torch_threads=args.cpu_threads,
        torch_version=torch.__version__,diffusers_version=diffusers.__version__)
    return base,parent,parent_audit,data,calibration,binding


def compute(config,threads,device='cuda'):
    if not 6<=threads<=8: raise ValueError('--cpu-threads must be within 6..8 logical CPUs')
    runtime=deepcopy(config)
    runtime['training'].update(cpu_budget=threads,torch_threads=threads,workers=0)
    return setup(runtime,device)


def check_fork(checkpoint,binding,config,normalization):
    if (checkpoint.get('schema_version')!=3 or checkpoint.get('diagnostic') is not True
            or checkpoint.get('fork_kind')!=FORK or checkpoint.get('binding')!=binding
            or checkpoint.get('config')!=config or checkpoint.get('normalization')!=normalization
            or checkpoint.get('source_hashes')!=binding['collection']['source_hashes']
            or checkpoint.get('step',0)<=START):
        raise ValueError('v3 checkpoint ancestry/data/tool/runtime mismatch')
    for kind in ('model','ema'):
        if not all(torch.isfinite(v).all() for v in checkpoint[kind].values()):
            raise FloatingPointError(f'Nonfinite v3 {kind} weights')


def check_unchanged(binding):
    unchanged(binding['collection'])
    if entry_hashes()!=binding['entry_tools']: raise ValueError('v3 training/validation tools changed')


def input_arguments(parser):
    for name in ('base-checkpoint','parent-checkpoint','v1-report','data-report','audit-report','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--cpu-threads',type=int,default=6)
