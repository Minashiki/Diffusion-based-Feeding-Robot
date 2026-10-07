"""Strict train-only ownership audit and three-pool correction sampling."""

from pathlib import Path

import numpy as np

from correction_v2.corpus import file_hashes,verify_files,pool_indices,MixedV2 as MixedV3

from .controller import CATEGORIES,QUOTAS,TEACHER_OWNERS,VERSION,MOVING_MAX,REFERENCE_MAX
from feedingrobot.data.episodes import load_episode
from feedingrobot.policies.audit import read_json,sha256


def check_labels(m,a):
    if not (m['success'] and m['pickup'] and m['delivery'] and m['accepted_normal'] and m['qualified_correction']
            and m['split']=='train' and m['teacher_version']==VERSION and m['source_unchanged']
            and not m['truncated'] and not m['abort_reason'] and m['max_episode_s']==60.
            and m['simulated_s']<=60. and not m['scenario'].get('recover',False)):
        raise ValueError('Only complete qualified train-only v3 episodes may enter')
    mask=np.asarray(a['action_mask']);owners=np.asarray(a['action_owner']);ticks=np.asarray(a['action_ticks'])
    if mask.shape!=owners.shape or np.any(mask & ~np.isin(owners,TEACHER_OWNERS)):
        raise ValueError('Non-teacher label')
    if m['category']=='aligned':
        if m['prefix_source']!='aligned' or not np.all(owners=='path_teacher'):
            raise ValueError('Aligned control cannot contain a perturbed prefix')
    else:
        if m['prefix_source']!='dp' or m['corrected_tick'] is None or m['handover'].get('release_tick') is None:
            raise ValueError('Correction needs an actual DP prefix and completed alignment')
        if m['alignment_max_drift_m'] is None or m['alignment_max_drift_m']>=.0007:
            raise ValueError('Unverified alignment height')
        if (np.any(mask[ticks<m['handover']['release_tick']]) or not np.any(owners=='dp')
                or not np.any(mask & (owners=='alignment_teacher'))):
            raise ValueError('Missing DP/alignment ownership or prefix labels leaked')
        r=m['handover']['release_residual']
        if not (2.<=r['angle_deg']<=4. and .001<=r['lateral_m']<=.002 and r['angular_speed']<.05
                and (r['linear_speed']<.002 if m['category'].endswith('stopped') else .002<r['downward_speed']<=MOVING_MAX and r['linear_speed']<=MOVING_MAX)):
            raise ValueError('Release outside category gates')
        if m['safety_checks'] or (m['category'].endswith('moving')
                and np.linalg.norm(m['handover']['release_velocity'][:3])>REFERENCE_MAX):
            raise ValueError('Safety intervention or unverified moving reference')
        if not m['geometry_checks'] or any(not g['passed'] for g in m['geometry_checks']):
            raise ValueError('Unverified handover geometry')
    for name in ('physics','observations','action_observations','actions','proposals'):
        if not np.isfinite(a[name]).all(): raise ValueError(f'Nonfinite {name}')
    if np.any(a['action_end_ticks'][mask]!=ticks[mask]+50): raise ValueError('Incomplete teacher label')
    return dict(valid_labels=int(mask.sum()),invalid_labels=int((~mask).sum()),non_teacher_labels_excluded=True)


def audit_corpus(report_path,binding,normalization,original):
    report_path=Path(report_path).resolve();r=read_json(report_path)
    if (r.get('mode')!='correction_dataset_v3' or r.get('status')!='ready_experimental'
            or r.get('binding')!=binding or not r.get('source_unchanged') or r.get('normalization')!=normalization):
        raise ValueError('Unready corpus or v3 ancestry/normalization mismatch')
    evidence={str(p.relative_to(report_path.parent)):sha256(p) for p in report_path.parent.rglob('*')
        if p.is_file() and p!=report_path and '__pycache__' not in p.parts}
    if evidence!=r['evidence_sha256']: raise ValueError('Changed collection evidence')
    root=Path(r['dataset']);verify_files(root,r['dataset_sha256'])
    if sha256(root/'normalization.json')!=sha256(Path(original)/'normalization.json'):
        raise ValueError('Normalization changed')
    expected={row['path'] for row in r['accepted']}
    actual={str(p.parent.relative_to(root)) for p in root.glob('*/*/manifest.json')}
    if actual!=expected or len(expected)!=12: raise ValueError('Episode coverage changed')
    counts=dict.fromkeys(CATEGORIES,0)
    seeds={read_json(p)['seed'] for p in Path(original).glob('*/*/manifest.json')}
    groups={read_json(p)['group_id'] for p in Path(original).glob('*/*/manifest.json')}
    for row in r['accepted']:
        m,a=load_episode(root/row['path']);check_labels(m,a)
        if m['seed'] in seeds or m['group_id'] in groups or m['seed'] not in range(780101,780125):
            raise ValueError('Train split/seed leakage')
        if row['seed']!=m['seed'] or row['category']!=m['category'] or row['replay']['status']!='passed':
            raise ValueError('Episode binding/replay mismatch')
        if row['replay']['max_physics_reference_error']>1e-10 or row['replay']['max_observation_error']>1e-7:
            raise ValueError('Replay tolerance exceeded')
        counts[m['category']]+=1;seeds.add(m['seed']);groups.add(m['group_id'])
    if counts!=QUOTAS or r['counts']!=counts: raise ValueError('Category quotas differ')
    return r
