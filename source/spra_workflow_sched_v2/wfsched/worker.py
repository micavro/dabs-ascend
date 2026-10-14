"""Spawn-safe worker: original frozen D0, or its budget-generation-free path."""
from __future__ import annotations
import os

def ready(): return os.getpid()

def process_plan(snapshot, model, cfg, method, deadline):
    if method != 'D' or cfg['base_cap'] != 4096:
        raise ValueError('only fixed-budget workflow scheduling is allowed')
    public = cfg['wf_public_method']
    if public == 'DAG':
        from semantic19.selector import choose
    elif public == 'DAG-Lite':
        from .selector import choose_lite as choose
    else:
        raise ValueError('baseline must not enter workflow worker')
    ans = choose(snapshot, model, cfg, 'D', deadline)
    if ans['action']['caps'] != [4096]*snapshot['replicas']:
        raise ValueError('workflow candidate changed budget')
    if any(e['cell'] not in ('00','10') for e in ans['evaluations']):
        raise ValueError('non-scheduling cell scored')
    ans['wf_public_method'] = public
    ans['wf_worker_path'] = 'frozen_D0' if public == 'DAG' else 'no_budget_candidates'
    return ans
