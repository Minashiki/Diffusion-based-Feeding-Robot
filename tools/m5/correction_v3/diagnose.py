"""CPU intervention diagnostics from frozen DP states/commands, never new data."""

from pathlib import Path

from correction_v2.controller import VERSION as V2_VERSION
from .controller import MOVING_MIN,MOVING_MAX
from .rollout import episode
from acquire_diagnosis import verify_evidence
from feedingrobot.data.episodes import write_json
from feedingrobot.data.replay import replay_episode
from feedingrobot.policies.audit import read_json,sha256
from feedingrobot.policies.runtime import setup


def diagnose(args):
    from .run import ancestry,finish,start_output,success
    base,trained,parent,v1,metadata,config,binding=ancestry(args)
    old=verify_evidence(args.previous_report,binding['parent_checkpoint_sha256'])
    legacy=dict(binding);legacy.pop('v3_tools');legacy['teacher_version']=V2_VERSION
    if old['mode']!='correction_dataset_v2' or old['binding']!=legacy or old['status']!='quota_incomplete':
        raise ValueError('Requires matching incomplete v2 collection evidence')
    _,hardware=setup(base['config'],'cpu')
    output,provenance=start_output(args,binding,parent,hardware,'intervention_diagnosis_v3')
    provenance.update(input_report=str(Path(args.previous_report).resolve()),input_report_sha256=sha256(args.previous_report),
        model_policy_executed=False,training_export=False,new_independent_samples=0)
    results=[]
    try:
        for seed in (770005,770006,770008,770007,770010):
            source=Path(args.previous_report).resolve().parent/'attempts'/f'{seed}_candidate'
            m=read_json(source/'manifest.json');directory=output/'episodes'/str(seed)
            resumed=seed in (770005,770006,770008)
            case=episode(seed,m['category'],m['teacher_config'],m['scenario'],directory,source='diagnostic',
                **({'resume_from':source} if resumed else {'prefix_from':source}))
            replay=replay_episode(directory);drift=case['alignment_max_drift_m']
            r=case['handover'].get('release_residual',{})
            passed=bool(replay['status']=='passed' and not case['failure_reason'] and case['source_unchanged']
                and (success(case) and case['qualified_correction'] and drift is not None and drift<.0007
                    and MOVING_MIN<r['downward_speed']<=MOVING_MAX if resumed else
                    case['abort_reason']=='unsafe_unqualified_prefix' and case['handover'].get('safety_stop_tick') is not None
                    and case['handover'].get('release_tick') is None and not case['qualified_correction']))
            results.append(dict(seed=seed,category=m['category'],status='passed' if passed else 'failed',
                diagnostic_kind='saved_handover' if resumed else 'recorded_prefix',
                case={k:v for k,v in case.items() if k!='files_sha256'},alignment_max_drift_m=drift,replay=replay))
            print(f'diagnosis seed={seed}: passed={passed} reason={case["abort_reason"] or case["failure_reason"]} drift_mm={None if drift is None else drift*1000}',flush=True)
            write_json(output/'progress.json',dict(results=results))
            if not passed: break
    except Exception as error:
        finish(output,provenance,binding,status='error',results=results,error=f'{type(error).__name__}: {error}');raise
    return finish(output,provenance,binding,status='passed' if len(results)==5 and all(r['status']=='passed' for r in results) else 'failed',results=results)
