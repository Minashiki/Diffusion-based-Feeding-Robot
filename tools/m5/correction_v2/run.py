"""Calibrate, collect on host CUDA/BF16, or audit 100k EMA with zero updates."""

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from correction_v2.controller import ATTEMPTS,CATEGORIES,QUOTAS,VERSION
from correction_v2.corpus import MixedV2,audit_corpus,check_labels,file_hashes,pool_indices
from correction_v2.rollout import episode
from acquire_diagnosis import verify_evidence
from correction_data import audit_data
from diagnose_sampling import check_checkpoint,sampler_settings,spacing_override
from train_correction import FORK,fork_check,tool_hashes
from feedingrobot.data.episodes import input_hashes,load_episode,write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.data import ActionWindows
from feedingrobot.policies.dit import ActionDiT,noise_loss,schedulers
from feedingrobot.policies.runtime import setup,to_device
from feedingrobot.sim.model import ROOT


def v2_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def ancestry(args):
    paths={key:Path(getattr(args,key)).resolve() for key in ('base_checkpoint','checkpoint','v1_report')}
    base=torch.load(paths['base_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    parent=check_checkpoint(base);v1=audit_data(base,paths['base_checkpoint'],paths['v1_report'])
    trained=torch.load(paths['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    source=input_hashes();old_tools=tool_hashes()
    old_binding=dict(base_checkpoint_sha256=sha256(paths['base_checkpoint']),data_report_sha256=sha256(paths['v1_report']),
        source_hashes=source,tool_hashes=old_tools,normalization=base['normalization'],fork_kind=FORK,
        sampling_fraction=.25,workers=0,seed=base['config']['training']['seed'])
    fork_check(trained,old_binding,paths['checkpoint'].parent,paths['checkpoint'])
    if (trained['step']!=100000 or trained['config']!=base['config'] or trained['source_hashes']!=source
            or trained['normalization']!=base['normalization']):
        raise ValueError('Requires matching 100000 experimental checkpoint')
    if not all(torch.isfinite(v).all() for v in trained['ema'].values()):
        raise ValueError('Nonfinite parent EMA')
    metadata=[read_json(p) for p in (ROOT/base['config']['dataset']).glob('*/*/manifest.json')]
    teacher_config=next(m['teacher_config'] for m in metadata if m['split']=='train' and not m['scenario'].get('recover',False))
    binding=dict(parent_checkpoint_sha256=sha256(paths['checkpoint']),formal_base_sha256=old_binding['base_checkpoint_sha256'],
        v1_report_sha256=old_binding['data_report_sha256'],source_hashes=source,v1_tools=old_tools,
        v2_tools=v2_hashes(),teacher_version=VERSION,normalization=base['normalization'])
    return base,trained,parent,v1,metadata,teacher_config,binding


def unchanged(binding):
    if input_hashes()!=binding['source_hashes'] or tool_hashes()!=binding['v1_tools'] or v2_hashes()!=binding['v2_tools']:
        raise ValueError('Runtime inputs/tools changed')


def seed_check(seeds,metadata,v1):
    used={m['seed'] for m in metadata}|{m['seed'] for m in v1['accepted']+v1['rejected']}
    def extract(value):
        if isinstance(value,dict):
            if 'seed' in value: used.add(value['seed'])
            for child in value.values(): extract(child)
        elif isinstance(value,list):
            for child in value: extract(child)
    root=ROOT/'outputs/single_bean/v1/m5/dit'
    for p in root.glob('*calibration*/report.json'): extract(read_json(p))
    # v2 may use other directory names; complete manifests identify actual seeds.
    for p in root.glob('correction_v2_*/*/*/manifest.json'): extract(read_json(p))
    if used.intersection(seeds): raise ValueError('Seed overlaps original/v1/historical calibration evidence')


def scenario_for(seed,config):
    rng=np.random.default_rng(seed)
    return dict(config['scene'],recover=False,head_amp_m=float(rng.uniform(.005,.01)),
        head_freq_hz=float(rng.uniform(.1,.2)),head_phase_rad=float(rng.uniform(-np.pi,np.pi)))


def start_output(args,binding,parent,hardware,mode):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'tool_snapshot';snapshot.mkdir()
    for name in binding['v2_tools']: shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    provenance=dict(schema_version=2,mode=mode,diagnostic=True,weights='ema',training_step=100000,
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=binding['parent_checkpoint_sha256'],
        base_checkpoint=str(Path(args.base_checkpoint).resolve()),v1_report=str(Path(args.v1_report).resolve()),
        binding=binding,parent=parent,hardware=hardware,normalization=binding['normalization'],
        sampler=sampler_settings(read_json(ROOT/'configs/dp_dit.json'),'leading'),physics_precision='fp64',
        optimizer_created=False,optimizer_updates=0,formal_test_run=False,full_validation_run=False,
        m5_status='incomplete',dp_v1='not_frozen',teacher_version=VERSION)
    write_json(output/'provenance.json',provenance)
    return output,provenance


def finish(output,provenance,binding,**result):
    unchanged(binding)
    report=dict(provenance,**result,source_unchanged=True,evidence_sha256=file_hashes(output))
    write_json(output/'report.json',report)
    return report


def paired(baseline,case):
    return all(baseline[k]==case[k] for k in ('initial_signature','initial_physics_sha256','teacher_geometry_sha256'))


def success(m):
    return bool(m['success'] and m['pickup'] and m['delivery'] and m['simulated_s']<=60.
        and not m['abort_reason'] and not m['truncated'] and m['source_unchanged'])


def calibrate(args):
    base,trained,parent,v1,metadata,config,binding=ancestry(args)
    previous=verify_evidence(args.previous_report,binding['parent_checkpoint_sha256'])
    if previous.get('mode')!='descent_coupled_calibration' or previous.get('status')!='passed':
        raise ValueError('Requires passed independent descent calibration')
    seeds=list(range(760001,760009));seed_check(seeds,metadata,v1)
    _,hardware=setup(base['config'],'cpu')
    output,provenance=start_output(args,binding,parent,hardware,'descent_calibration_v2')
    provenance.update(previous_report=str(Path(args.previous_report).resolve()),previous_report_sha256=sha256(args.previous_report),
        training_export=False,model_policy_executed=False)
    pairs=[]
    try:
        for i,seed in enumerate(seeds):
            category=CATEGORIES[i//2];scenario=scenario_for(seed,config)
            baseline_dir=output/'episodes'/f'{seed}_baseline'
            baseline=episode(seed,category,config,scenario,baseline_dir,source='baseline')
            baseline_replay=replay_episode(baseline_dir)
            case_dir=output/'episodes'/f'{seed}_calibration'
            case=episode(seed,category,config,scenario,case_dir,source='calibration') if success(baseline) else None
            replay=replay_episode(case_dir) if case else None
            passed=bool(case and success(case) and case['qualified_correction'] and paired(baseline,case))
            pairs.append(dict(seed=seed,category=category,status='passed' if passed else 'failed',
                baseline={k:v for k,v in baseline.items() if k!='files_sha256'},baseline_replay=baseline_replay,
                case={k:v for k,v in case.items() if k!='files_sha256'} if case else None,replay=replay,
                paired_initial_state=bool(case and paired(baseline,case))))
            print(f'calibration {category} seed={seed}: passed={passed} reason={case and (case["abort_reason"] or case["failure_reason"])}',flush=True)
            if not passed: break
    except Exception as error:
        finish(output,provenance,binding,status='error',pairs=pairs,error=f'{type(error).__name__}: {error}');raise
    return finish(output,provenance,binding,status='passed' if len(pairs)==8 and all(p['status']=='passed' for p in pairs) else 'failed',pairs=pairs)


def calibration_check(path,binding):
    path=Path(path).resolve();r=verify_evidence(path,binding['parent_checkpoint_sha256'])
    if r.get('mode')!='descent_calibration_v2' or r.get('status')!='passed' or r.get('binding')!=binding:
        raise ValueError('Unpassed or mismatched v2 calibration')
    pairs=r['pairs']
    if len(pairs)!=8 or [p['category'] for p in pairs]!=[c for c in CATEGORIES[:4] for _ in range(2)]:
        raise ValueError('Calibration category coverage differs')
    for row in pairs:
        if (row['status']!='passed' or not row['paired_initial_state'] or not success(row['baseline'])
                or not success(row['case']) or not row['case']['qualified_correction']
                or row['replay']['status']!='passed' or row['baseline_replay']['status']!='passed'):
            raise ValueError('Calibration gates/replay differ')
    return r


def collect(args):
    base,trained,parent,v1,metadata,config,binding=ancestry(args)
    calibration_check(args.calibration_report,binding)
    seed_check(range(770001,770025),metadata,v1)
    device,hardware=setup(base['config'],'cuda')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('Host collection requires CUDA BF16; no precision fallback')
    model=ActionDiT(base['config']).to(device).eval().requires_grad_(False);model.load_state_dict(trained['ema'])
    output,provenance=start_output(args,binding,parent,hardware,'correction_dataset_v2')
    data=output/'data';(data/'train').mkdir(parents=True)
    shutil.copyfile(ROOT/base['config']['dataset']/'normalization.json',data/'normalization.json')
    provenance.update(dataset=str(data),calibration_report=str(Path(args.calibration_report).resolve()),
        calibration_report_sha256=sha256(args.calibration_report),precision='bf16_autocast',
        quotas=QUOTAS,maximum_attempts=ATTEMPTS,training_export='experimental_extension_only')
    counts=dict.fromkeys(CATEGORIES,0);accepted=[];rejected=[];seed=770001
    try:
        with spacing_override('leading'):
            for category in CATEGORIES:
                category_seeds=range(seed,seed+ATTEMPTS[category]);seed+=ATTEMPTS[category]
                for case_seed in category_seeds:
                    if counts[category]>=QUOTAS[category]: break
                    scenario=scenario_for(case_seed,config)
                    baseline_dir=output/'attempts'/f'{case_seed}_baseline'
                    baseline=episode(case_seed,category,config,scenario,baseline_dir,source='baseline')
                    baseline_replay=replay_episode(baseline_dir)
                    directory=output/'attempts'/f'{case_seed}_candidate'
                    row=episode(case_seed,category,config,scenario,directory,source='aligned' if category=='aligned' else 'dp',
                        model=model,dp_config=base['config'],normalization=base['normalization'],device=device) if success(baseline) else None
                    if not row or not success(row) or not row['qualified_correction'] or not paired(baseline,row):
                        rejected.append(dict(seed=case_seed,category=category,reason=row and (row['abort_reason'] or row['failure_reason']) or 'category_not_covered',
                            baseline_replay=baseline_replay));print(f'rejected {category} seed={case_seed}',flush=True);continue
                    m,a=load_episode(directory);labels=check_labels(m,a)
                    # Require P3 before accepting; use the existing legal window reader.
                    temporary=output/'window_check';(temporary/'train').mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(data/'normalization.json',temporary/'normalization.json')
                    link=temporary/'train'/directory.name;link.symlink_to(directory,target_is_directory=True)
                    try:
                        windows=ActionWindows(temporary,'train',base['config']['model']['horizon'],base['normalization'])
                        _,intervals=pool_indices(windows)
                    except ValueError as error:
                        rejected.append(dict(seed=case_seed,category=category,reason=str(error),baseline_replay=baseline_replay))
                        continue
                    finally:
                        link.unlink();shutil.rmtree(temporary)
                    replay=replay_episode(directory)
                    destination=data/'train'/f'correction_{case_seed}'
                    shutil.move(str(directory),str(destination));replay['episode']=str(destination)
                    accepted.append(dict(seed=case_seed,category=category,path=str(destination.relative_to(data)),
                        replay=replay,baseline_replay=baseline_replay,pool_intervals=intervals,**labels))
                    counts[category]+=1;print(f'accepted {category} seed={case_seed}: {counts[category]}/{QUOTAS[category]}',flush=True)
    except Exception as error:
        finish(output,provenance,binding,status='error',counts=counts,accepted=accepted,rejected=rejected,error=f'{type(error).__name__}: {error}');raise
    report=finish(output,provenance,binding,status='ready_experimental' if counts==QUOTAS else 'quota_incomplete',
        counts=counts,accepted=accepted,rejected=rejected,dataset_sha256=file_hashes(data))
    return report


def audit(args):
    base,trained,parent,v1,metadata,config,binding=ancestry(args)
    report=audit_corpus(args.data_report,binding,base['normalization'],ROOT/base['config']['dataset'])
    calibration_check(report['calibration_report'],binding)
    if sha256(report['calibration_report'])!=report['calibration_report_sha256']:
        raise ValueError('Calibration report changed')
    device,hardware=setup(base['config'],'cpu')
    mixed=MixedV2(base['config'],base['normalization'],report)
    model=ActionDiT(base['config']).to(device).eval().requires_grad_(False);model.load_state_dict(trained['ema'])
    ema=deepcopy(model)
    rng=np.random.default_rng(base['config']['training']['seed'])
    torch.manual_seed(base['config']['training']['seed'])
    # Audit diffusion loss uses DDPM, unchanged by inference spacing.
    scheduler,_=schedulers(base['config'])
    with torch.no_grad():
        batch=to_device(mixed.batch(rng,64),device);loss=noise_loss(model,scheduler,batch)
        for pools in mixed.pools.values():
            for indices in pools.values():
                item=mixed.extra[indices[0]]
                if not all(torch.isfinite(v).all() for v in item.values()): raise ValueError('Nonfinite category window')
    exact=all(torch.equal(v.cpu(),trained['ema'][k]) for k,v in model.state_dict().items())
    ema_exact=all(torch.equal(v.cpu(),trained['ema'][k]) for k,v in ema.state_dict().items())
    if not torch.isfinite(loss) or not exact or not ema_exact: raise ValueError('Nonfinite audit or EMA inheritance differs')
    output,provenance=start_output(args,binding,parent,hardware,'correction_warmstart_audit_v2')
    # Re-read all evidence after the forward pass to detect concurrent changes.
    audit_corpus(args.data_report,binding,base['normalization'],ROOT/base['config']['dataset'])
    return finish(output,provenance,binding,status='audit_passed',step=100000,precision='fp32',
        model_exact=True,ema_exact=True,optimizer_state_loaded=False,rng_state_loaded=False,
        data_report=str(Path(args.data_report).resolve()),data_report_sha256=sha256(args.data_report),
        forward_loss=float(loss),batch_size=64,data=mixed.summary())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('calibrate','collect','audit'))
    for name in ('base-checkpoint','checkpoint','v1-report','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--previous-report');parser.add_argument('--calibration-report');parser.add_argument('--data-report')
    args=parser.parse_args()
    required={'calibrate':'previous_report','collect':'calibration_report','audit':'data_report'}[args.mode]
    if not getattr(args,required): parser.error('--'+required.replace('_','-')+' is required')
    report={'calibrate':calibrate,'collect':collect,'audit':audit}[args.mode](args)
    print(Path(args.output).resolve()/'report.json',flush=True)
    raise SystemExit(0 if report['status'] in ('passed','ready_experimental','audit_passed') else 1)


if __name__=='__main__': main()
