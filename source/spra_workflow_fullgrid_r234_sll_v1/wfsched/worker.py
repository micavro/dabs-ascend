"""Spawn-safe worker: original frozen D0, or its budget-generation-free path."""
from __future__ import annotations
import os
import traceback
from pathlib import Path
from .common import atomic

def ready(): return os.getpid()

def process_plan(snapshot, model, cfg, method, deadline):
    if method != 'D' or cfg['base_cap'] != 4096:
        raise ValueError('only fixed-budget workflow scheduling is allowed')
    # Keep module import independent of the native project. Parent runner loads
    # original source paths before actual worker planning.
    from .admission import binding, REVISION as ADMISSION_REVISION
    public = cfg['wf_public_method']
    if public == 'DAG':
        from semantic19.selector import choose
    elif public == 'DAG-A':
        from .v3_selector import choose_adaptive as choose
    elif public == 'DAG-W':
        from .v4_selector import choose_workflow_robust as choose
    elif public in ('DAG-H','Ours'):
        from .v5_selector import choose_heterogeneous as choose
    elif public == 'DAG-V2':
        from .ours_v2_selector import choose_v2 as choose
    elif public == 'DAG-Lite':
        from .selector import choose_lite as choose
    else:
        raise ValueError('baseline must not enter workflow worker')
    try:
        with binding():
            ans = choose(snapshot, model, cfg, 'D', deadline)
    except Exception as exc:
        # Preserve a failing input instead of replacing the action or retrying.
        target = cfg.get('wf_worker_failure_dir')
        if target:
            try:
                atomic(Path(target)/('failure_'+str(os.getpid())+'.json'), {
                    'error': repr(exc), 'traceback': traceback.format_exc(),
                    'snapshot': snapshot, 'config': cfg, 'method': method,
                    'deadline': deadline, 'automatic_retry': False,
                })
            except Exception as save_error:
                # Saving diagnostics must never hide or convert the original failure.
                print('WF_FAILURE_DIAGNOSTIC_WRITE_ERROR='+repr(save_error), flush=True)
        raise
    ans['wf_admission_revision'] = ADMISSION_REVISION
    ans['wf_admission_repair_proposals'] = sum(
        bool(e.get('action', {}).get('admission_projection_repair'))
        for e in ans.get('evaluations', []))
    if ans['action']['caps'] != [4096]*snapshot['replicas']:
        raise ValueError('workflow candidate changed budget')
    if any(e['cell'] not in (('00','10','20','21') if public in ('DAG-H','Ours') else ('00','10','11','20','21') if public=='DAG-W' else ('00','10','20','30') if public=='DAG-A' else ('00','10','20') if public=='DAG-V2' else ('00','10')) for e in ans['evaluations']):
        raise ValueError('non-scheduling cell scored')
    ans['wf_public_method'] = public
    ans['wf_worker_path'] = ('OursV5_heterogeneous_reference_first' if public in ('DAG-H','Ours') else 'OursV4_reference_first_isolated_actions' if public == 'DAG-W' else 'OursV3_V1_first_bounded_candidates' if public == 'DAG-A'
       else 'OursV2_two_destinations' if public == 'DAG-V2'
       else 'D0_admission_repaired' if public == 'DAG' else 'no_budget_candidates')
    return ans
