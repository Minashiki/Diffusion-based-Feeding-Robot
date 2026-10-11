"""Verify the historical source snapshot bound to reusable correction data."""

from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.sim.model import ROOT

RUNTIME = ROOT / 'outputs/single_bean/v1/m5/dit/MinaDiTv1/runtime_001'


def verify_runtime(expected=None):
    manifest = read_json(RUNTIME / 'legacy_manifest.json')
    if expected is not None and manifest['source_hashes'] != expected:
        raise ValueError('Correction data does not match the preserved source snapshot')
    for relative, digest in manifest['source_hashes'].items():
        if sha256(RUNTIME / relative) != digest:
            raise ValueError(f'Correction source snapshot changed: {relative}')
    return manifest
