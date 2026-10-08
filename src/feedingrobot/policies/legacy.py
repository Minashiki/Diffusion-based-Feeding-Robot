"""Run historical MinaDiTv1 entries against their verified source snapshot."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.sim.model import ROOT

RUNTIME = ROOT / 'outputs/single_bean/v1/m5/dit/MinaDiTv1/runtime_001'


def verify_runtime(expected=None):
    manifest = read_json(RUNTIME / 'legacy_manifest.json')
    if expected is not None and manifest['source_hashes'] != expected:
        raise ValueError('Historical checkpoint does not match the preserved MinaDiTv1 runtime')
    for relative, digest in manifest['source_hashes'].items():
        if sha256(RUNTIME / relative) != digest:
            raise ValueError(f'Historical MinaDiTv1 source changed: {relative}')
    return manifest


def run(entry, arguments, *, expected=None):
    verify_runtime(expected)
    env = dict(os.environ, PYTHONPATH=str(RUNTIME / 'src'))
    code = """
import json, runpy, sys
from pathlib import Path
from feedingrobot.sim.model import ROOT
from feedingrobot.data.episodes import input_hashes
assert input_hashes() == json.loads((ROOT/'legacy_manifest.json').read_text())['source_hashes'], 'Legacy runtime input set changed'
entry = sys.argv[1]
sys.argv = sys.argv[1:]
if entry.endswith('.py'):
    path = ROOT / entry
    sys.path.insert(0, str(path.parent))
    runpy.run_path(str(path), run_name='__main__')
else:
    runpy.run_module(entry, run_name='__main__')
"""
    print(f'MinaDiTv1 historical runtime: {RUNTIME}', flush=True)
    return subprocess.run([sys.executable, '-c', code, entry, *map(str, arguments)], cwd=RUNTIME, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entry', help='Original module name or runtime-relative Python script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return run(args.entry, args.arguments)


if __name__ == '__main__':
    raise SystemExit(main())
