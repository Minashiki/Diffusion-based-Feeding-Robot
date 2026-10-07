"""Versioned experimental correction corpus and phase-balanced mixed train sampler."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import default_collate

from acquire_diagnosis import verify_evidence
from diagnose_sampling import check_checkpoint
from teacher_rotation_calibration import run_case
from feedingrobot.data.episodes import input_hashes, load_episode, write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json, sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT


TRAIN_CASES=((740001,3.),(740002,5.),(740003,3.),(740004,5.),(740005,3.),(740006,5.))


def label_check(metadata,arrays):
    if not (metadata['success'] and metadata['accepted_normal'] and metadata['qualified_correction']
            and metadata['split']=='train' and metadata['m5_extension'] and not metadata['scenario'].get('recover',False)):
        raise ValueError('Only complete successful correction episodes may train')
    mask,owned=arrays['action_mask'],arrays['teacher_owned']
    if mask.shape!=owned.shape or np.any(mask & ~owned):
        raise ValueError('Perturbation/settling actions cannot be training labels')
    ticks=arrays['action_ticks']
    start=metadata['perturbation']['pulse_start_tick'];end=metadata['perturbation']['release_tick']
    if not (~mask[(ticks>=start)&(ticks<end)]).all():
        raise ValueError('Pulse interval labels must be invalid')
    for name in ('physics','observations','action_observations','actions','proposals'):
        if not np.isfinite(arrays[name]).all(): raise ValueError(f'Nonfinite correction data: {name}')
    return dict(valid_labels=int(mask.sum()),invalid_labels=int((~mask).sum()),perturbation_labels_excluded=True)


def audit_data(checkpoint,checkpoint_path,report_path):
    check_checkpoint(checkpoint)
    report_path=Path(report_path).resolve()
    report=verify_evidence(report_path,sha256(checkpoint_path))
    if (report['mode']!='correction_dataset' or report['status']!='ready_experimental'
            or report['parent']['dataset_binding']!=checkpoint['parent']['dataset_binding']
            or report['normalization']!=checkpoint['normalization']):
        raise ValueError('Correction data ancestry or normalization differs')
    if sha256(report['checkpoint'])!=report['checkpoint_sha256']:
        raise ValueError('Base checkpoint changed')
    root=Path(report['dataset'])
    expected_episodes={episode['path'] for episode in report['accepted']}
    actual_episodes={str(p.parent.relative_to(root)) for p in root.glob('*/*/manifest.json')}
    if actual_episodes!=expected_episodes or len(expected_episodes)!=4:
        raise ValueError('Correction episode coverage changed')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=set(report['dataset_sha256']): raise ValueError('Correction dataset file coverage changed')
    for relative,expected in report['dataset_sha256'].items():
        if sha256(root/relative)!=expected: raise ValueError(f'Correction dataset changed: {relative}')
    original=ROOT/checkpoint['config']['dataset']
    if sha256(root/'normalization.json')!=sha256(original/'normalization.json'):
        raise ValueError('Observation normalization changed')
    original_metadata=[read_json(p) for p in original.glob('*/*/manifest.json')]
    used_seeds={m['seed'] for m in original_metadata};used_groups={m['group_id'] for m in original_metadata}
    for episode in report['accepted']:
        m,a=load_episode(root/episode['path'])
        if m['seed'] in used_seeds or m['group_id'] in used_groups: raise ValueError('Correction split leakage')
        if episode['replay']['status']!='passed': raise ValueError('Unverified correction replay')
        label_check(m,a);used_seeds.add(m['seed']);used_groups.add(m['group_id'])
    return report


class MixedCorrection:
    def __init__(self,config,normalization,report):
        self.base=ActionWindows(ROOT/config['dataset'],'train',config['model']['horizon'],normalization)
        self.extra=ActionWindows(report['dataset'],'train',config['model']['horizon'],normalization)
        self.pools={}
        for i,(e,start,end,phase) in enumerate(self.extra.windows):
            m=self.extra.episodes[e][1];tick=int(self.extra.arrays(e)['action_ticks'][start])
            if phase==1 and m['perturbation']['release_tick']<=tick<=m['corrected_tick']:
                self.pools.setdefault(e,[]).append(i)
        if len(self.pools)!=len(self.extra.episodes): raise ValueError('Missing legal correction windows')

    def batch(self,rng,count):
        if count<4 or count%4: raise ValueError('Mixed batch must be divisible by four')
        base=self.base.sample_indices(rng,count*3//4)
        episodes=sorted(self.pools);extra=[]
        for _ in range(count//4):
            e=episodes[int(rng.integers(len(episodes)))];indices=self.pools[e]
            extra.append(indices[int(rng.integers(len(indices)))])
        items=[self.base[i] for i in base]+[self.extra[i] for i in extra]
        order=rng.permutation(count)
        return default_collate([items[i] for i in order])

    def summary(self):
        return dict(original_windows=len(self.base),extension_windows=len(self.extra),
            correction_windows=sum(map(len,self.pools.values())),correction_episodes=len(self.pools),
            correction_fraction=.25,other_fraction=.75,
            sampling='75% original phase/episode-balanced; 25% correction episode/window-balanced; shuffled each batch')


def collect(args):
    checkpoint_path,output=Path(args.checkpoint).resolve(),Path(args.output).resolve()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(checkpoint);digest=sha256(checkpoint_path)
    previous=verify_evidence(args.previous_report,digest)
    if previous['mode']!='teacher_rotation_calibration' or sum(p['pulse']['success'] and p['pulse']['qualified_correction'] for p in previous['pairs'])<2:
        raise ValueError('Requires two complete successful teacher correction calibrations')
    device,hardware=setup(checkpoint['config'],'cpu')
    original=ROOT/checkpoint['config']['dataset']
    metadata=[read_json(p) for p in original.glob('*/*/manifest.json')]
    used={m['seed'] for m in metadata}|{p['seed'] for p in previous['pairs']}
    if any(s in used for s,d in TRAIN_CASES): raise ValueError('New train seed overlaps earlier evidence')
    config=next(m['teacher_config'] for m in metadata if m['split']=='train' and not m['scenario'].get('recover',False))
    output.mkdir(parents=True,exist_ok=False)
    data=output/'data';(data/'train').mkdir(parents=True);attempts=output/'attempts';attempts.mkdir()
    shutil.copyfile(original/'normalization.json',data/'normalization.json')
    snapshot=output/'tool_snapshot';snapshot.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file(): shutil.copyfile(file,snapshot/file.name)
    provenance=dict(schema_version=1,mode='correction_dataset',diagnostic=True,checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest,training_step=50000,weights='ema',parent=parent,original_source_hashes=checkpoint['source_hashes'],
        normalization=checkpoint['normalization'],hardware=hardware,dataset=str(data),source_check='passed',
        previous_report=str(Path(args.previous_report).resolve()),previous_report_sha256=sha256(args.previous_report),
        tool_sha256={p.name:sha256(p) for p in snapshot.iterdir()},optimizer_updates=0,formal_test_run=False,
        dp_v1='not_frozen',m5_status='incomplete',scope='four new train-only corrections, two near 3deg and two near 5deg; maximum six attempts',
        normalization_refitted=False,formal_m4_dataset=False,base_data_modified=False)
    write_json(output/'provenance.json',provenance)
    accepted,rejected=[],[];counts={3.:0,5.:0}
    try:
        for seed,degrees in TRAIN_CASES:
            if counts[degrees]>=2: continue
            rng=np.random.default_rng(seed)
            scenario=dict(config['scene'],recover=False,head_amp_m=float(rng.uniform(.005,.01)),head_freq_hz=float(rng.uniform(.1,.2)),head_phase_rad=float(rng.uniform(-np.pi,np.pi)))
            directory=attempts/f'episode_{seed}'
            row=run_case(seed,degrees,True,config,scenario,attempts,episode_directory=directory)
            write_json(attempts/f'{seed}.json',row)
            if not row['success'] or not row['qualified_correction']:
                rejected.append(dict(seed=seed,reason=row['failure_reason'] or 'correction_gate',path=str(directory)))
                print(f'rejected seed={seed} reason={rejected[-1]["reason"]}',flush=True);continue
            m,a=load_episode(directory);labels=label_check(m,a)
            replay=replay_episode(directory)
            destination=data/'train'/f'correction_{seed}'
            shutil.move(str(directory),str(destination));replay['episode']=str(destination)
            accepted.append(dict(seed=seed,path=str(destination.relative_to(data)),requested_deg=degrees,
                achieved_deg=row['perturbation']['achieved_deg'],correction_duration_s=row['correction_duration_s'],
                replay=replay,**labels));counts[degrees]+=1
            print(f'accepted/replayed seed={seed} residual={row["perturbation"]["achieved_deg"]:.3f} labels={labels["valid_labels"]}',flush=True)
            if len(accepted)==4: break
    except Exception as error:
        write_json(output/'report.json',dict(provenance,status='error',accepted=accepted,rejected=rejected,error=f'{type(error).__name__}: {error}'));raise
    check_checkpoint(checkpoint)
    report=dict(provenance,status='ready_experimental' if len(accepted)==4 else 'quota_incomplete',accepted=accepted,rejected=rejected,
        source_unchanged=input_hashes()==checkpoint['source_hashes'],dataset_sha256={str(p.relative_to(data)):sha256(p) for p in data.rglob('*') if p.is_file()})
    report['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='report.json'}
    write_json(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--previous-report',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    report=collect(args);print('report:',Path(args.output).resolve()/'report.json',flush=True)
    raise SystemExit(0 if report['status']=='ready_experimental' else 1)
