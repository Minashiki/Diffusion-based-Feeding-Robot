"""Strict train-only ownership audit and three-pool correction sampling."""

from pathlib import Path

import numpy as np
from torch.utils.data import default_collate

from .controller import CATEGORIES,QUOTAS,TEACHER_OWNERS,VERSION
from feedingrobot.data.episodes import load_episode
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.sim.model import ROOT


def file_hashes(root):
    root=Path(root)
    return {str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def verify_files(root,expected):
    if file_hashes(root)!=expected: raise ValueError('File coverage or SHA256 changed')


def check_labels(m,a):
    if not (m['success'] and m['pickup'] and m['delivery'] and m['accepted_normal'] and m['qualified_correction']
            and m['split']=='train' and m['teacher_version']==VERSION and m['source_unchanged']
            and not m['truncated'] and not m['abort_reason'] and m['max_episode_s']==60.
            and m['simulated_s']<=60. and not m['scenario'].get('recover',False)):
        raise ValueError('Only complete qualified train-only v2 episodes may enter')
    mask=np.asarray(a['action_mask']);owners=np.asarray(a['action_owner']);ticks=np.asarray(a['action_ticks'])
    if mask.shape!=owners.shape or np.any(mask & ~np.isin(owners,TEACHER_OWNERS)):
        raise ValueError('Non-teacher label')
    if m['category']=='aligned':
        if m['prefix_source']!='aligned' or not np.all(owners=='path_teacher'):
            raise ValueError('Aligned control cannot contain a perturbed prefix')
    else:
        if m['prefix_source']!='dp' or m['corrected_tick'] is None or m['handover'].get('release_tick') is None:
            raise ValueError('Correction needs an actual DP prefix and completed alignment')
        if (np.any(mask[ticks<m['handover']['release_tick']]) or not np.any(owners=='dp')
                or not np.any(mask & (owners=='alignment_teacher'))):
            raise ValueError('Missing DP/alignment ownership or prefix labels leaked')
        r=m['handover']['release_residual']
        if not (2.<=r['angle_deg']<=4. and .001<=r['lateral_m']<=.002 and r['angular_speed']<.05
                and (r['linear_speed']<.002 if m['category'].endswith('stopped') else r['downward_speed']>.002)):
            raise ValueError('Release outside category gates')
        if not m['geometry_checks'] or any(not g['passed'] for g in m['geometry_checks']):
            raise ValueError('Unverified handover geometry')
    for name in ('physics','observations','action_observations','actions','proposals'):
        if not np.isfinite(a[name]).all(): raise ValueError(f'Nonfinite {name}')
    if np.any(a['action_end_ticks'][mask]!=ticks[mask]+50): raise ValueError('Incomplete teacher label')
    return dict(valid_labels=int(mask.sum()),invalid_labels=int((~mask).sum()),non_teacher_labels_excluded=True)


def pool_indices(windows):
    pools={name:{} for name in ('alignment','transition','aligned')}
    stages_seen={}
    intervals={}
    for e,(path,m) in enumerate(windows.episodes):
        owners=np.load(path/'action_owner.npy',allow_pickle=False)
        stages=np.load(path/'action_stages.npy',allow_pickle=False)
        stages_seen[e]=set();intervals[e]={}
        a=windows.arrays(e)
        for i,(episode,start,end,phase) in enumerate(windows.windows):
            if episode!=e or phase!=1: continue
            tick=int(a['action_ticks'][start]);owner=owners[start];stage=stages[start]
            pool=None
            if m['category']=='aligned' and owner=='path_teacher' and stage in ('pre_entry','entry','sweep'):
                pool='aligned'
            elif m['category']!='aligned':
                if owner=='alignment_teacher' and m['handover']['release_tick']<=tick<m['corrected_tick']:
                    pool='alignment'
                elif owner=='path_teacher' and tick>=m['corrected_tick'] and stage in ('pre_entry','entry','sweep'):
                    pool='transition';stages_seen[e].add(str(stage))
            if pool:
                pools[pool].setdefault(e,[]).append(i)
                intervals[e].setdefault(pool,[]).append(tick)
        required=('aligned',) if m['category']=='aligned' else ('alignment','transition')
        if any(e not in pools[p] for p in required): raise ValueError('Missing legal v2 window pool')
        if m['category']!='aligned' and not {'entry','sweep'}<=stages_seen[e]:
            raise ValueError('P3 must contain real entry and sweep teacher windows')
    return pools,{str(e):{p:[min(t),max(t)] for p,t in row.items()} for e,row in intervals.items()}


class MixedV2:
    def __init__(self,config,normalization,report):
        self.base=ActionWindows(ROOT/config['dataset'],'train',config['model']['horizon'],normalization)
        self.extra=ActionWindows(report['dataset'],'train',config['model']['horizon'],normalization)
        self.pools,self.intervals=pool_indices(self.extra)

    def batch(self,rng,count):
        if count<4 or count%4: raise ValueError('Batch size must be divisible by four')
        indices=self.base.sample_indices(rng,count*3//4)
        items=[self.base[i] for i in indices]
        names=tuple(self.pools)
        for _ in range(count//4):
            pool=self.pools[names[int(rng.integers(3))]]
            episodes=sorted(pool);e=episodes[int(rng.integers(len(episodes)))];windows=pool[e]
            items.append(self.extra[windows[int(rng.integers(len(windows)))]] )
        return default_collate([items[i] for i in rng.permutation(count)])

    def summary(self):
        return dict(original_windows=len(self.base),extension_windows=len(self.extra),correction_fraction=.25,
            pools={p:dict(episodes=len(rows),windows=sum(map(len,rows.values()))) for p,rows in self.pools.items()},
            pool_intervals=self.intervals,
            sampling='75% original phase/episode-balanced; 25% v2 equal pool, then episode/window-balanced')


def audit_corpus(report_path,binding,normalization,original):
    report_path=Path(report_path).resolve();r=read_json(report_path)
    if (r.get('mode')!='correction_dataset_v2' or r.get('status')!='ready_experimental'
            or r.get('binding')!=binding or not r.get('source_unchanged') or r.get('normalization')!=normalization):
        raise ValueError('Unready corpus or v2 ancestry/normalization mismatch')
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
        if m['seed'] in seeds or m['group_id'] in groups or m['seed'] not in range(770001,770025):
            raise ValueError('Train split/seed leakage')
        if row['seed']!=m['seed'] or row['category']!=m['category'] or row['replay']['status']!='passed':
            raise ValueError('Episode binding/replay mismatch')
        if row['replay']['max_physics_reference_error']>1e-10 or row['replay']['max_observation_error']>1e-7:
            raise ValueError('Replay tolerance exceeded')
        counts[m['category']]+=1;seeds.add(m['seed']);groups.add(m['group_id'])
    if counts!=QUOTAS or r['counts']!=counts: raise ValueError('Category quotas differ')
    return r
