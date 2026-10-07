"""Exploratory protected 20-step rollout after the original clipping gate failed."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

from acquire_diagnosis import pose_reference, verify_evidence
from diagnose_sampling import action_metrics, check_checkpoint, rollout_metrics, spacing_override
from prefix_rollout import run_prefix_policy
from step_sensitivity import inference_config, settings
from feedingrobot.control.adapter import clip_norm
from feedingrobot.data.episodes import input_hashes, write_json
from feedingrobot.policies import dit
from feedingrobot.policies.audit import sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.runtime import setup
from feedingrobot.sim.model import ROOT, load_json


def protected_review(predictions, truth, mask, limits):
    projected = predictions.copy()
    overshoot = {}
    for name, columns, limit in (('linear',slice(0,3),limits[0]),('angular',slice(3,6),limits[1])):
        vectors = predictions[...,columns]
        speed = np.linalg.norm(vectors[mask],axis=-1)
        projected[...,columns] = np.array([clip_norm(v,limit) for v in vectors.reshape(-1,3)]).reshape(vectors.shape)
        overshoot[name] = dict(p99=float(np.quantile(speed,.99)),maximum=float(speed.max()),
            fraction_above_105_percent=float(np.mean(speed>limit*1.05)),limit=limit)
    return dict(raw=action_metrics(predictions,truth,mask,limits),speed_projected=action_metrics(projected,truth,mask,limits),
        overshoot=overshoot,limitation='Speed projection only; excludes acceleration shaping, reference guards and physical response.')


def supports_exploratory_rollout(reviews):
    for seed in ('0','1','2'):
        baseline,candidate = reviews['10'][seed],reviews['20'][seed]
        if not baseline['raw']['finite'] or not candidate['raw']['finite']:
            return False
        for component in ('linear','angular'):
            if candidate['speed_projected'][component]['vector_rmse'] >= baseline['speed_projected'][component]['vector_rmse']:
                return False
    return True


@torch.no_grad()
def run(args):
    checkpoint_path,output = Path(args.checkpoint).resolve(),Path(args.output).resolve()
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
    parent = check_checkpoint(checkpoint)
    digest = sha256(checkpoint_path)
    offline = verify_evidence(args.offline_report,digest)
    baseline = verify_evidence(args.baseline_report,digest)
    if (offline['mode']!='step_sensitivity' or not offline['all_samples_finite']
            or baseline['mode']!='cadence' or baseline['status']!='completed'):
        raise ValueError('Requires completed finite step sensitivity and matched cadence evidence')
    config,norm = checkpoint['config'],checkpoint['normalization']
    device,hardware = setup(config,args.device)
    robot = load_json('configs/robots/panda.json')
    limits = robot['linear_speed_limit'],robot['angular_speed_limit']
    archives = np.load(Path(args.offline_report).resolve().parent/'samples.npz')
    reviews = {str(count):{str(seed):protected_review(archives[f'leading_{count}_seed_{seed}'],archives['truth'],
        archives['legal_prefix_mask'],limits) for seed in (0,1,2)} for count in (10,20)}
    eligible = supports_exploratory_rollout(reviews)
    output.mkdir(parents=True,exist_ok=False)
    snapshot = output/'tool_snapshot'
    snapshot.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file(): shutil.copyfile(file,snapshot/file.name)
    with spacing_override('leading'):
        sampler = settings(config,20)
    provenance = dict(schema_version=1,mode='protected_twenty_step_probe',diagnostic=True,checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest,training_step=50000,weights='ema',parent=parent,source_check='passed',
        original_source_hashes=checkpoint['source_hashes'],training_config=config,normalization=norm,hardware=hardware,
        precision='bf16_autocast' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32',sampler=sampler,
        previous_report=str(Path(args.offline_report).resolve()),previous_report_sha256=sha256(args.offline_report),
        baseline_report=str(Path(args.baseline_report).resolve()),baseline_report_sha256=sha256(args.baseline_report),
        tool_sha256={p.name:sha256(p) for p in snapshot.iterdir()},review=reviews,
        exploratory_eligible=eligible,original_gate_status='failed; unchanged',selection='Exploratory 20-step choice after observing lower projected errors and smaller overshoot/cost than 50 steps; not independent selection evidence.',
        optimizer_updates=0,formal_test_run=False,dp_v1='not_frozen',m5_status='incomplete',acceptance_status='not_evaluated',
        scope='first_normal_validation_scene; expand only after real pickup',execute_steps=4,replan_ticks=200,
        noise_pairing='base seed 0; initial seed=tick//50 at common planning ticks; matches cadence 4-step reference',
        teacher_in_execution=False,realtime_status='not_verified')
    write_json(output/'provenance.json',provenance)
    rows=[]
    try:
        if eligible:
            model = dit.ActionDiT(config).to(device)
            model.load_state_dict(checkpoint['ema'])
            model.eval().requires_grad_(False)
            dataset = ActionWindows(ROOT/config['dataset'],'validation',model.horizon,norm)
            episodes = [(i,p) for i,(p,m) in enumerate(dataset.episodes) if not m['scenario'].get('recover',False)][:3]
            offline_episodes = {c['episode'] for c in offline['cases']}
            if {str(p) for i,p in episodes} != offline_episodes:
                raise ValueError('Scene list differs from the offline evidence')
            revised = inference_config(config,20)
            with spacing_override('leading'):
                for i,episode in episodes:
                    row=run_prefix_policy(model,revised,norm,episode,device,execute_steps=4)
                    row['predicted_velocity']=rollout_metrics(row,limits)
                    teacher=dict(dataset.arrays(i),action_phases=np.load(episode/'action_phases.npy',mmap_mode='r'))
                    row['pose_reference']=pose_reference([r['observation'] for r in row['trace']],[r['tick'] for r in row['trace']],teacher,dataset.fields)
                    reference=Path(args.baseline_report).resolve().parent/f'{episode.name}_prefix_4.json'
                    from feedingrobot.policies.audit import read_json
                    original=read_json(reference)
                    shared_seeds={p['tick']:p['noise_seed'] for p in original['plans']}
                    common=[p for p in row['plans'] if p['tick'] in shared_seeds]
                    paired=bool(common) and all(p['noise_seed']==shared_seeds[p['tick']] for p in common)
                    row['reference_comparison']=dict(baseline=str(reference),baseline_sha256=sha256(reference),
                        common_noise_seed_pairing_verified=paired,common_planning_ticks=len(common),
                        baseline_pickup=original['pickup'],baseline_failure_reason=original['failure_reason'],
                        baseline_simulated_s=original['simulated_s'],baseline_contact_peak_n=original['contact_peak_n'])
                    write_json(output/f'{episode.name}_leading_20.json',row)
                    rows.append({k:row[k] for k in ('source_episode','pickup','success','failure_reason','entered','simulated_s','wall_s','contact_peak_n','wrist_peak_n','predicted_velocity','reference_comparison')})
                    print(f'leading20 pickup={row["pickup"]} success={row["success"]} reason={row["failure_reason"]} seconds={row["simulated_s"]}',flush=True)
                    if not row['pickup'] or row['failure_reason'] in ('invalid_command','nonfinite_state'):
                        break
    except Exception as error:
        write_json(output/'report.json',dict(provenance,status='error',error=f'{type(error).__name__}: {error}'))
        raise
    report=dict(provenance,status='completed' if eligible else 'review_rejected',scenes=rows,
        pickup_successes=sum(r['pickup'] for r in rows),source_unchanged=input_hashes()==checkpoint['source_hashes'])
    report['evidence_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name!='report.json'}
    write_json(output/'report.json',report)
    if not report['source_unchanged']: raise ValueError('Original runtime inputs changed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--offline-report',required=True)
    parser.add_argument('--baseline-report',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=run(args)
    print('report:',Path(args.output).resolve()/'report.json',flush=True)
    raise SystemExit(0 if report['status']=='completed' else 1)
