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
    if any(e['cell'] not in ('00','10') for e in ans['evaluations']):
        raise ValueError('non-scheduling cell scored')
    ans['wf_public_method'] = public
    ans['wf_worker_path'] = 'D0_admission_repaired' if public == 'DAG' else 'no_budget_candidates'
    return ans
