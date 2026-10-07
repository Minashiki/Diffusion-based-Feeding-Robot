"""Read-only-weight, paired execution diagnostics for the fixed 130k EMA."""

import argparse
from copy import copy
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training'))

from common import check_fork,check_unchanged,compute,evidence,input_arguments
import numpy as np
import torch

from evaluate import select_cases
from train import check_records
from correction_v3.corpus import file_hashes
from acquire_diagnosis import verify_evidence
from feedingrobot.data.episodes import load_episode,write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import sha256
from feedingrobot.policies.dit import ActionDiT
from rollout import VARIANTS,leading_scheduler,rollout


def tool_hashes():
    return {p.name:sha256(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}


def compare_baseline(current,previous):
    old,oa=load_episode(previous);new,na=load_episode(current)
    if sha256(current/'initial_state.pkl')!=sha256(previous/'initial_state.pkl'):
        raise AssertionError('Baseline initial state differs')
    for name in ('seed','scenario','initial_tick','events','success','failure_reason','diagnostic_failure_reason','truncated'):
        if new.get(name)!=old.get(name): raise AssertionError(f'Baseline {name} differs')
    for name in oa:
        if name not in na: raise AssertionError(f'Missing baseline array {name}')
        np.testing.assert_array_equal(na[name],oa[name],err_msg=f'Baseline {name} differs')
    if (current/'commands.json').read_bytes()!=(previous/'commands.json').read_bytes():
        raise AssertionError('Baseline commands differ')
    return dict(status='passed',source=str(previous),source_manifest_sha256=sha256(previous/'manifest.json'))


def summary(rows):
    return dict(attempts=len(rows),autonomous_aligned=sum(r['model_alignment']['success'] for r in rows),
        assisted_aligned=sum(r['hold_gate']['assisted_alignment_pass'] for r in rows),
        alignment_conditions_met=sum(r['alignment_diagnostics']['conditions_met'] for r in rows),
        pickup=sum(r['pickup'] for r in rows),transport=sum(r['transport_completed'] for r in rows),
        delivery=sum(r['delivery'] for r in rows),complete_success=sum(r['success'] for r in rows),
        safety_interventions=sum(len(r['external_braking']['interventions']) for r in rows),
        gate_applications=sum(r['hold_gate']['applications'] for r in rows))


def run(args):
    checkpoint_hash=sha256(args.checkpoint)
    baseline=verify_evidence(args.baseline_report,checkpoint_hash)
    # The immutable training ancestry retains its original CPU budget.
    audit_args=copy(args);audit_args.cpu_threads=baseline['binding']['cpu_budget']
    base,parent,_,data,calibration,binding=evidence(audit_args)
    if (baseline.get('training_step')!=130000 or baseline.get('checkpoint_sha256')!=checkpoint_hash
            or baseline.get('binding')!=binding or baseline.get('noise_seed')!=args.noise_seed
            or baseline.get('weights')!='ema' or baseline.get('sampler')!='leading'
            or baseline.get('status')!='diagnostic_completed'):
        raise ValueError('Requires the matching 130k EMA baseline report and noise seed')
    baseline_hash=sha256(args.baseline_report)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=base['config'];normalization=base['normalization']
    check_fork(checkpoint,binding,config,normalization)
    if checkpoint['step']!=130000: raise ValueError('Requires step 130000')
    metrics=Path(args.checkpoint).resolve().parent/'metrics.jsonl'
    check_records(metrics,130000);metrics_hash=sha256(metrics)
    calibration=dict(calibration,calibration_directory=str(Path(data['calibration_report']).resolve().parent))
    cases=[p for p,c in select_cases(config,calibration,False) if c]
    if [int(p.name.split('_')[0]) for p in cases]!=[780001,780003,780005,780007]:
        raise ValueError('Unexpected independent correction cases')
    device,hardware=compute(config,args.cpu_threads)
    if not compatible_runtime(hardware,baseline['hardware']): raise ValueError('Baseline hardware/runtime differs')
    tools=tool_hashes()
    sampler=leading_scheduler(config)
    sampler.set_timesteps(config['diffusion']['inference_steps'])
    source_files={str(p):sha256(p) for source in cases for p in source.iterdir() if p.is_file()}
    provenance=dict(schema_version=1,diagnostic=True,mode='runtime_probe_small',training_binding=binding,
        diagnostic_binding=dict(tools=tools,origin_evaluate_sha256=sha256(HERE.parent/'training/evaluate.py'),
            source_files=source_files,baseline_report=str(Path(args.baseline_report).resolve()),baseline_report_sha256=baseline_hash,
            checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=checkpoint_hash,
            metrics_sha256=metrics_hash,variants={v:dict(replan_ms=50 if v=='replan50' else 200,
                execution_steps=1 if v=='replan50' else 4,gate='world_vz_negative_to_zero_until_stable_200ms' if v=='holdgate200' else 'none') for v in VARIANTS}),
        hardware=hardware,training_step=130000,weights='ema',noise_seed=args.noise_seed,
        paired_initial_noise=True,sampler=baseline['sampler'],sampler_settings=dict(config=dict(sampler.config),timesteps=sampler.timesteps.tolist(),eta=0.),
        viewer_required=not args.headless,optimizer_created=False,optimizer_updates=0,training_export=False,
        full_validation_run=False,formal_test_run=False,m5_status='incomplete',dp_v1='not_frozen',
        cases=[str(p) for p in cases])
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    for folder,source,names in (('tool_snapshot',HERE,tools),('training_tool_snapshot',HERE.parent/'training',binding['entry_tools'])):
        target=output/folder;target.mkdir()
        for name in names: shutil.copyfile(source/name,target/name)
    write_json(output/'provenance.json',provenance)
    results={variant:{arm:[] for arm in ('parent','v3')} for variant in VARIANTS}
    def verify_unchanged():
        check_unchanged(binding)
        if (tool_hashes()!=tools or sha256(args.checkpoint)!=checkpoint_hash
                or sha256(args.baseline_report)!=baseline_hash or sha256(metrics)!=metrics_hash
                or any(sha256(p)!=h for p,h in source_files.items())):
            raise ValueError('Diagnostic inputs/tools/checkpoint changed')
    def finish(status,**extra):
        report=dict(provenance,status=status,results=results,
            summaries={v:{a:summary(rows) for a,rows in arms.items()} for v,arms in results.items()},**extra)
        report['evidence_sha256']=file_hashes(output)
        write_json(output/'report.json',report)
        return report
    try:
        model=ActionDiT(config).to(device).eval().requires_grad_(False)
        for variant in VARIANTS:
            for arm,weights in (('parent',parent['ema']),('v3',checkpoint['ema'])):
                model.load_state_dict(weights)
                for source in cases:
                    verify_unchanged()
                    directory=output/variant/arm/source.name
                    row=rollout(model,config,normalization,source,directory,device,correction=True,
                        variant=variant,viewer=not args.headless,noise_seed=args.noise_seed)
                    row['replay']=replay_episode(directory)
                    write_json(directory/'result.json',row)
                    results[variant][arm].append(row)
                    if variant=='baseline200':
                        row['baseline_comparison']=compare_baseline(directory,Path(args.baseline_report).resolve().parent/arm/source.name)
                        write_json(directory/'result.json',row)
                    write_json(output/'progress.json',dict(results=results))
                    print(f'{variant} {arm} seed={row["seed"]} autonomous={row["model_alignment"]["success"]} assisted={row["hold_gate"]["assisted_alignment_pass"]} pickup={row["pickup"]} success={row["success"]} reason={row["failure_reason"]}',flush=True)
        evidence(audit_args);verify_unchanged()
        if sum(len(rows) for arms in results.values() for rows in arms.values())!=24:
            raise AssertionError('Incomplete diagnostic')
        return finish('diagnostic_completed',source_unchanged=True,baseline_reproduced=True)
    except Exception as error:
        finish('error',error=f'{type(error).__name__}: {error}')
        raise


def compatible_runtime(current,previous):
    return {k:v for k,v in current.items() if k not in ('cpu_affinity','torch_threads')}=={
        k:v for k,v in previous.items() if k not in ('cpu_affinity','torch_threads')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('small',));input_arguments(parser)
    output_option=parser._option_string_actions['--output']
    output_option.required=False;output_option.default='outputs/single_bean/v1/m5/dit/correction_v3_runtime_probe_001'
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--baseline-report',required=True)
    parser.add_argument('--noise-seed',type=int,default=0)
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args();report=run(args)
    print(Path(args.output).resolve()/'report.json',flush=True)
