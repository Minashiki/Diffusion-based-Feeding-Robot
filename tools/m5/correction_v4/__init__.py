"""Independent data repair; no optimizer or policy training."""

import os
from pathlib import Path

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
if hasattr(os,'sched_getaffinity'):
    selected=sorted(os.sched_getaffinity(0))[:6]
    for thread in Path('/proc/self/task').iterdir():
        try: os.sched_setaffinity(int(thread.name),selected)
        except ProcessLookupError: pass

