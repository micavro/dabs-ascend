"""Declared workload grid, not a selection over observed performance."""
from __future__ import annotations
import copy
import itertools
from .common import METHODS, digest

SCOPE = 'Same historical 64 MuSiQue development questions, four disjoint 16-question blocks; not unseen holdout'

def orders(methods, rounds):
    methods = list(methods); n = len(methods)
    if rounds == 2:
        return [methods, list(reversed(methods))]
    # Williams-style balanced first-order carryover: n rows for even n, 2n for odd n.
    base = [0]
    for i in range(1, n): base.append((i+1)//2 if i%2 else n-i//2)
    rows = [[methods[(j+i)%n] for j in base] for i in range(n)]
    if n%2: rows += [list(reversed(x)) for x in rows]
    if rounds != len(rows): raise ValueError('unsupported predeclared number of rounds')
    return rows

def make_design(suite):
    if suite == 'smoke':
        cells = [(4,2.)]; blocks=[1]; methods=list(METHODS); seq=[methods]
    elif suite == 'ablation':
        cells = [(4,2.)]; blocks=[1,2,3,4]
        methods=['Affinity','DAG','DAG-Lite']; seq=list(itertools.permutations(methods))
    elif suite in ('grid','confirm'):
        cells = [(r,float(rate)) for r in (1,2,3,4) for rate in (1,2,4,8)]
        blocks=[1,2,3,4]; methods=list(METHODS)
        seq=orders(methods,2 if suite=='grid' else 10)
    else: raise ValueError('unknown suite')
    rows=[]
    # Fixed mirrored cell traversal reduces monotone machine-time confounding.
    for t,order in enumerate(seq,1):
        for R,rate in (cells if t%2 else list(reversed(cells))):
            ci=cells.index((R,rate))
            for b in blocks:
                # Label permutation is fixed by block/cell, never by performance.
                mapping={m:methods[(i+b-1+ci)%len(methods)] for i,m in enumerate(methods)}
                row_order=[mapping[m] for m in order]
                for pos,m in enumerate(row_order,1):
                    rows.append({'label':f'R{R}_L{rate:g}_T{t}_B{b}_{m}',
                      'replicas':R,'rate':rate,'repeat':t,'block':b,'position':pos,
                      'method':m,'seed':190500+b,'router_seed':930000+b+100*t,
                      'packet_slice':[(b-1)*16,b*16], 'workflow_count':16,
                      'base_cap':4096})
    if len({r['label'] for r in rows}) != len(rows): raise AssertionError('duplicate design label')
    return rows

def plan(suite):
    design=make_design(suite)
    return {'schema':'workflow-grid-plan-2','suite':suite,'scope':SCOPE,'conditions':design,
      'expected_conditions':len(design),'methods':list(dict.fromkeys(r['method'] for r in design)),
      'workflows_per_condition':16,'sampling_contract':'original JSON answer-only, natural outputs; no forced equal work',
      'input_namespace_contract':'original model-visible unique UUID per condition; prompts not identical across methods',
      'cache_contract':'cold registered namespace initially; natural within-condition prefix reuse',
      'arrival_contract':'original wave: 1s active at 2*lambda, 1s idle; finite N=16 truncation',
      'replica_mapping':'first R verified dedicated service identities 0..R-1; unused pods remain running',
      'engine_contract':'same native continuous batching, fixed prefill cap and total4096; no adaptive actions',
      'dispatch_clock':'same original 200ms periodic loop for all methods; not unmodified Ray/NGINX end-to-end',
      'primary':'mean of paired within-(cell,repeat,block) percentage changes of mean workflow JCT',
      'ordering':('all 6 permutations per block' if suite=='ablation' else
                  '10-row position and first-order carryover balanced per cell/block' if suite=='confirm' else
                  'two mirrored orders per cell/block, not fully position-balanced' if suite=='grid' else
                  'one smoke order; not performance evidence'),
      'budget_candidate_generation_removed_only_for':'DAG-Lite',
      'no_posthoc_version_selection':True,'all_results_retained':True,'automatic_retry':False,
      'guaranteed_10_percent':False}

def config_for(base, cell):
    cfg=copy.deepcopy(base)
    cfg.update(replicas=cell['replicas'],arrival_rate=cell['rate'],base_cap=4096,
               workflow_count=16,wf_public_method=cell['method'],
               wf_routing_policy=cell['method'],wf_router_seed=cell['router_seed'])
    # The profile is unchanged. These are declared deployment/load interventions,
    # not a claim that the frozen four-replica profile is re-fit at each cell.
    for k,v in [('hard_total',4096),('epoch_s',.2),('max_model_len',4096),
                ('block_size',16),('max_num_seqs',64),('arrival_shape','wave')]:
        if cfg.get(k)!=v: raise ValueError('unexpected original contract: '+k)
    return cfg
