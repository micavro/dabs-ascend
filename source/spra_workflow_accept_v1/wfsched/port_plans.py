"""Source-only adaptation plans; no historical measured outcomes are stored."""
from __future__ import annotations
from .common import HERE, load
SUITES=('port-smoke','port-cases','port-load','port-scale','port-grid','port-ablation')

def plan(name):
    if name not in SUITES:raise ValueError('unknown port suite')
    value=load(HERE/'plans'/f'{name}.json')
    if value.get('suite')!=name or value.get('schema')!='workflow-grid-plan-2':raise ValueError('invalid plan')
    rows=value['conditions']
    if len(rows)!=value['expected_conditions'] or len({r['label'] for r in rows})!=len(rows):raise ValueError('duplicate/incomplete plan')
    for r in rows:
        if r['replicas'] not in (1,2,3,4) or r['rate'] not in (1.,2.,4.,8.) or r['block'] not in (1,2,3,4):raise ValueError('unsupported original setting')
        if r['method'] not in ('RR','P2C','Affinity','DAG','DAG-Lite') or r['base_cap']!=4096:raise ValueError('not workflow-only')
        if r['packet_slice']!=[(r['block']-1)*16,r['block']*16] or r['workflow_count']!=16:raise ValueError('source Block mismatch')
    return value
