"""Read-only v4 ancestry, checkpoint-prefix checks and bounded evaluation CPUs."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

# Capture the host allowance before the frozen v4 package narrows imports to six.
CPUS=sorted(os.sched_getaffinity(0))[:8]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'


def affinity(cpus):
    for thread in Path('/proc/self/task').iterdir():
        try: os.sched_setaffinity(int(thread.name),cpus)
        except ProcessLookupError: pass


affinity(CPUS)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import torch
torch.set_num_threads(6)
if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)

from correction_v4.training import common as training
from correction_v3.corpus import file_hashes
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.runtime import setup

HERE=Path(__file__).resolve().parent


def tool_hashes():
    return {p.name:sha256(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}


def compute(config,threads):
    if threads not in (6,7,8) or len(CPUS)<threads:
        raise ValueError('Evaluation requires 6..8 available logical CPUs')
    affinity(CPUS[:threads])
    runtime=deepcopy(config)
    runtime['training'].update(cpu_budget=threads,torch_threads=threads,workers=0)
    device,hardware=setup(runtime,'cuda')
    hardware.update(cpu_budget=threads,dataloader_workers=0,physics_owners=1,
        replay_workers=threads-1,precision='bf16_autocast' if torch.cuda.is_bf16_supported() else 'fp32')
    return device,hardware


def records(path,step):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    if (not rows or [r['step'] for r in rows]!=list(range(training.START+1,rows[-1]['step']+1))
            or not all(math.isfinite(r[k]) for r in rows for k in ('loss','gradient_norm'))
            or not all(math.isfinite(r.get('validation_loss',0.)) for r in rows)
            or not training.START<step<=rows[-1]['step']):
        raise ValueError('Training log must be finite and continuous through the selected checkpoint')
    prefix=rows[:step-training.START]
    return dict(first_step=prefix[0]['step'],checkpoint_step=prefix[-1]['step'],
        prefix_rows=len(prefix),log_last_step=rows[-1]['step'],log_sha256=sha256(path),
        validation=[r for r in prefix if 'validation_loss' in r])


def checkpoint(path,binding,config,normalization):
    path=Path(path).resolve()
    value=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    training.check_fork(value,binding,config,normalization)
    if (value.get('diagnostic') is not True or value.get('optimizer_updates')!=value['step']-training.START
            or value.get('source_hashes')!=binding['collection']['legacy']['source_hashes']):
        raise ValueError('v4 checkpoint supervision/update/source provenance differs')
    log=records(path.parent/'metrics.jsonl',value['step'])
    provenance=read_json(path.parent/'provenance.json')
    actual={k:deepcopy(value.get(k)) for k in provenance}
    for metadata in (actual,provenance):
        defaults=(metadata.get('sampler') or {}).get('config',{}).get('_use_default_values')
        # Diffusers builds this name list from a set; order can change on resume.
        if isinstance(defaults,list): defaults.sort()
    if actual!=provenance:
        raise ValueError('Checkpoint does not match its saved run provenance')
    snapshots={str(Path('tool_snapshot')/name):digest for name,digest in binding['entry_tools'].items()}
    if file_hashes(path.parent/'tool_snapshot')!=binding['entry_tools']:
        raise ValueError('Training source snapshot changed')
    return value,dict(path=str(path),sha256=sha256(path),step=value['step'],weights='ema',
        training_log=log,provenance_sha256=sha256(path.parent/'provenance.json'),training_snapshot=snapshots)


def check_files(files):
    for path,digest in files.items():
        if sha256(path)!=digest: raise ValueError(f'Bound evaluation input changed: {path}')


def validation_evidence(path,checkpoint_sha256,binding,tools,protocol):
    path=Path(path).resolve();report=read_json(path)
    if (report.get('mode')!='validation' or report.get('status')!='validation_passed'
            or report.get('binding')!=binding or report.get('evaluation_tools')!=tools
            or report.get('protocol')!=protocol or report.get('source_unchanged') is not True
            or report.get('formal_test_run') is not False or report.get('full_validation_run') is not True):
        raise ValueError('Test requires matching, passed, full v4 validation evidence')
    actual=file_hashes(path.parent);actual.pop(path.name,None)
    if actual!=report.get('evidence_sha256'): raise ValueError('Validation evidence changed')
    if not report.get('input_sha256'): raise ValueError('Validation input bindings are missing')
    check_files(report['input_sha256'])
    selected=[arm for arm in report['arms'] if arm['sha256']==checkpoint_sha256
        and arm.get('validation_passed') is True and arm['step']>training.START]
    if len(selected)!=1: raise ValueError('Selected checkpoint did not pass full validation')
    return dict(path=str(path),sha256=sha256(path),selected_arm=selected[0]['name'])
