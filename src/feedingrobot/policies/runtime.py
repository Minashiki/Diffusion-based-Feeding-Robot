"""Explicit compute budgets, device checks and resumable run storage."""

import contextlib
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


def setup(config, device='cuda'):
    budget = config['training']['cpu_budget']
    if not 1 <= budget <= 8:
        raise ValueError('CPU budget must be within 1..8 logical CPUs')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        selected = allowed[:budget]
        # Include native library threads created during imports, not only the main thread.
        for thread in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(thread.name), selected)
            except ProcessLookupError:
                pass
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    torch.set_num_threads(min(config['training']['torch_threads'], budget))
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this session; run in the feedingrobot host environment. No CPU fallback.')
    seed = config['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    info = dict(python=sys.executable, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                device=str(device), cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
    return device, info


def precision(device):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if device.type == 'cuda' and torch.cuda.is_bf16_supported() else contextlib.nullcontext())


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def rng_state(rng):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], sampler=rng.bit_generator.state)


def restore_rng(state, rng):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    rng.bit_generator.state = state['sampler']
