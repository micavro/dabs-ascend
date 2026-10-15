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

def v2_design(suite):
    """Predeclared exploration/confirmation plans; zero result-driven filtering."""
    if suite == 'v2-screen':
        cases=[(1,4.,4),(2,4.,1),(3,2.,4),(4,2.,3),(4,4.,2)]
        methods=['RR','P2C','Affinity','DAG','DAG-V2']
        seq=[methods,list(reversed(methods))]
    elif suite in ('v3-urgent','v3-fixed'):
        # 4 diagnostic cells, predeclared; current goal is rapid engineering screening.
        cases=[(2,4.,1),(3,2.,4),(4,2.,3),(4,4.,2)]
        methods=['RR','P2C','Affinity','DAG','DAG-A']
        seq=[methods,list(reversed(methods))]
    elif suite == 'v4-screen':
        cases=[(2,4.,1),(3,2.,4),(4,2.,3),(4,4.,2)]
        methods=['RR','P2C','Affinity','DAG','DAG-W']
        seq=[methods,list(reversed(methods))]
    elif suite in ('v4-stress-64','v4-hotspot-64'):
        # Declared BEFORE running. Exactly 64 workflows in every condition;
        # the same arrival offsets, R, input/output counts, and packets for all methods.
        # hotspot repeats the entire B3 16-workflow block four times; disclosed as
        # nonsemantic synthetic demand, NEVER as 64 independent MuSiQue questions.
        cases=[(2,4.,0),(3,4.,0),(4,4.,0),(4,8.,0)] if suite=='v4-stress-64' else [
               (2,4.,3),(3,4.,3),(4,4.,3),(4,8.,3)]
        methods=['RR','P2C','Affinity','DAG','DAG-W']
        seq=[methods,list(reversed(methods))]
    elif suite in ('v5-mix-balanced-64','v5-mix-skew-64'):
        cases=[(4,8.,0),(4,4.,0),(3,4.,0),(2,4.,0)]
        methods=['RR','P2C','Preble','DAG','DAG-H']
        seq=[methods,list(reversed(methods))]
    elif suite in ('v5-ayo-balanced-64','v5-ayo-skew-64',
                   'v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64'):
        # Predeclared Ayo Algorithm-2 scheduling proxy comparison. Both profiles
        # have exactly the same graph-shape histogram and output-token budget.
        if '-grid-' in suite:
            cases=[(r,float(rate),0) for r in (1,2,3,4) for rate in (1,2,4,8)]
        else:
            cases=[(4,8.,0),(4,4.,0),(3,4.,0),(2,4.,0)]
        methods=['RR','P2C','Preble','Ayo','DAG','DAG-H']
        seq=[methods,list(reversed(methods))]
    elif suite == 'dabs-r234-balanced-64':
        # The preregistered COMPLETE experiment is 3 replica counts x 4 rates x
        # 2 mirror-order repeats x 4 methods = 96 accepted GPU conditions.
        # R4 lambda8 executes first solely as a GPU integration smoke, not
        # a selection or filtering rule. ALL conditions remain mandatory.
        cases=[(r,float(rate),0) for r in (4,3,2) for rate in (8,4,12,16)]
        methods=['RR','SLL','Ayo','Ours']
        seq=[methods,list(reversed(methods))]
    elif suite == 'v3-grid':
        cases=[(r,float(rate),b) for r in (1,2,3,4)
              for rate in (1,2,4,8) for b in (1,2,3,4)]
        methods=['RR','P2C','Affinity','DAG-A']
        seq=[methods,list(reversed(methods))]
    elif suite == 'v2-grid':
        cases=[(r,float(rate),b) for r in (1,2,3,4)
              for rate in (1,2,4,8) for b in (1,2,3,4)]
        methods=['RR','P2C','Affinity','DAG-V2']
        seq=[methods,list(reversed(methods))]
    elif suite == 'v2-b3':
        cases=[(4,2.,3)]
        methods=['RR','P2C','Affinity','DAG-V2']
        seq=orders(methods,4)*2
    else: raise ValueError('not a v2 plan')
    rows=[]
    for t,order in enumerate(seq,1):
        cells=cases if t%2 else list(reversed(cases))
        for ci,(R,rate,b) in enumerate(cells):
            for pos,m in enumerate(order,1):
                rows.append({'label':f'R{R}_L{rate:g}_T{t}_B{b}_{m}',
                    'replicas':R,'rate':rate,'repeat':t,'block':b,
                    'position':pos,'method':m,'seed':190500+b,
                    'router_seed':950000+b+100*t,
                    'packet_slice':([0,64] if suite in ('dabs-r234-balanced-64','v4-stress-64') or suite.startswith('v5-ayo-') or suite in ('v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64') else
                                    [32,48] if suite=='v4-hotspot-64' else [(b-1)*16,b*16]),
                    'packet_repetitions':(4 if suite=='v4-hotspot-64' else 1),
                    'heterogeneous_workload':suite.startswith('v5-ayo-') or suite=='dabs-r234-balanced-64' or suite in ('v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64'),
                    'popularity_profile':('skew' if suite.endswith('-skew-64') else 'balanced' if suite.endswith('-balanced-64') else None),
                    'workflow_count':(64 if suite in ('dabs-r234-balanced-64','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64') else 16),
                    'base_cap':4096,
                    'measurement_mode':('fixed_token_replay' if suite in ('dabs-r234-balanced-64','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64') else 'natural_json')})
    assert len({x['label'] for x in rows})==len(rows)
    return rows


def make_design(suite):
    if suite == 'smoke':
        cells = [(4,2.)]; blocks=[1]; methods=list(METHODS); seq=[methods]
    elif suite == 'agecheck':
        cells=[(1,4.),(1,8.)];blocks=[1]
        methods=['Affinity','DAG','DAG-Lite','RR','P2C'];seq=[methods]
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
    if suite == 'dabs-r234-balanced-64':
        # Fully immutable preregistered measurement scope, no SOTA cache-affinity
        # comparator; Ayo Algorithm 2 is a topology-only native-controller proxy.
        inherited=plan('grid')
        design=v2_design(suite)
        inherited.update(
            schema='dabs-r234-balanced-64-plan-1',suite=suite,
            conditions=design,expected_conditions=len(design),
            methods=['RR','SLL','Ayo','Ours'],
            scope='predeclared complete multi-replica high-load experiment; not unseen MuSiQue holdout',
            ordering='two mirrored method and cell orders, full 3x4 balanced workload matrix',
            load_matrix={'replicas':[2,3,4],'arrival_rates_wf_per_s':[4,8,12,16],
                'arrival_shape':'1s ON at 2lambda; 1s OFF',
                'expected_conditions':96},
            fixed_workload_replay=True,fixed_prompt_tokens=3584,
            fixed_output_tokens=None,
            fixed_per_DAG_output_tokens={'2hop':16,'3hop':[32,64],'4hop':[48,64,128]},
            sampling_contract='64 original public MuSiQue workflow exemplars reweighted as balanced; original graph edges; all node inputs 3584; 13184 output tokens per condition; nonsemantic DAG replay',
            true_native_kv_skip_logged=True,
            cross_workflow_kv_source_observed_by_engine=False,
            kv_attribution='exclusive/ambiguous eligible prior producer by same-replica exact token-prefix and time, not GPU-block source tracing',
            trace_analysis='per-condition native skips, workflow internal/cross candidate attribution, critical-path latency decomposition, replica occupancy, decision overhead',
            candidate_revision='unchanged DAG-H V5 guarded workflow scheduling implementation',
            ayo_comparator='Ayo Algorithm2 controller-level topology priority + normalized reserved-KV placement proxy; not complete paper/system',
            sampled_least_loaded_definition='sample two pseudorandom distinct replicas (k=2), choose fewer unfinished HTTP requests, same routing as original P2C',
            no_adaptive_batching=True,base_cap=4096,original_DAG_edges_modified=False,
            semantic_quality_applicable=False,
            repetitions=2,no_posthoc_method_switch=True,
            phase_calibration_applied=False,heldout_questions=False,
            source_pool='same frozen 64 public pilot questions and predefined 24 source exemplars as V5 balanced',
            completion_contract='all 96 conditions accepted AND all raw audit files verified; no cherry-picked subset')
        return inherited
    if suite in ('v2-screen','v2-grid','v2-b3','v3-urgent','v3-fixed','v3-grid','v4-screen','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64'):
        inherited=plan('grid')
        design=v2_design(suite)
        inherited.update(suite=suite,conditions=design,expected_conditions=len(design),
           methods=list(dict.fromkeys(r['method'] for r in design)),
           scope=('predeclared exploratory comparison; not final model selection' if suite in ('v2-screen','v3-urgent','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64')
                  else 'predeclared B3 repeat verification' if suite=='v2-b3'
                  else 'complete 4x4x4 R/rate/Block matrix after freezing selected algorithm'),
           ordering='fixed order followed by its reverse; each measured cell has paired method/arrival seed',
           v2_algorithm_revision='one_extra_destination-1',
           v3_algorithm_revision='v1_first_two_opportunistic_candidates-1',
           fixed_workload_replay=(suite in ('dabs-r234-balanced-64','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64')),
           current_reference_is_not_published_preble=True,
           reference_implementation_label=('Preble (internal Affinity-based cache+load approximation, NOT published E2)' if suite.startswith('v5-') else 'CacheAffinity-LB (internal Affinity; not Preble E2)'),
           reference_implementation_source='joint18v2.search.reference -> joint18v2.actions.affinity',
           ayo_comparator=('Algorithm 2 topology-aware per-workflow buckets and descending DAG depth; native-controller 200ms, KV-reservation occupancy routing proxy; no official graph transformations, no partial-prefill/decoding pipelining' if suite.startswith('v5-ayo-') else None),
           candidate_revision=('DAG-H V5 pure-action guarded decision' if suite.startswith('v5-') else 'DAG-W V4 reference-first split route/order/admission, full JCT guards'),
           hotspot_semantics=('six original DAG shapes; four original prompts per shape; fixed same shape histogram and output work, change popularity only' if suite.startswith('v5-') else 'synthetic four-fold repetition of B3 packets' if suite=='v4-hotspot-64'
                             else '64 distinct pilot packets' if suite=='v4-stress-64' else None),
           sampling_contract=('nonsemantic heterogeneous DAG replay; exactly 3584 prompt, output length by predeclared DAG shape' if suite.startswith('v5-') else 'nonsemantic static DAG replay; exactly 3584 prompt and 16 generated tokens per node'
              if suite in ('dabs-r234-balanced-64','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64') else 'natural typed JSON Agent; output EOS is not forced'),
           semantic_quality_applicable=(suite not in ('v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64')),
           fixed_per_DAG_output_tokens=({'2hop':16,'3hop':[32,64],'4hop':[48,64,128]} if suite.startswith('v5-') else None),
           original_DAG_edges_modified=False,
           fixed_prompt_tokens=(3584 if suite in ('dabs-r234-balanced-64','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64') else None),
           fixed_output_tokens=(16 if suite in ('dabs-r234-balanced-64','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64') else None),
           v2_exploratory=(suite in ('v2-screen','v3-urgent','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64','v5-mix-balanced-64','v5-mix-skew-64','v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64')),
           v2_predeclared_all_conditions=True,
           v2_reference_preserved=True,
           no_posthoc_method_switch=True,
           token_protocol=('fixed 3584 prompt, DAG-family output 16/32/48/64/128, no semantic answer' if suite.startswith('v5-') else 'per-node fixed 3584 prompt, 16 generated, no semantic answer' if suite in ('dabs-r234-balanced-64','v3-fixed','v4-screen','v4-stress-64','v4-hotspot-64')
             else 'natural JSON; actual token counts audited and not normalized'))
        return inherited
    if suite.startswith('port-'):
        from .port_plans import plan as port_plan
        return port_plan(suite)
    if suite == 'case-confirm':
        from .acceptance import case_plan
        return case_plan()
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
      'admission_intent_revision':'wf-admission-intent-repair-1',
      'original_frozen_source_modified':False,
      'no_posthoc_version_selection':True,'all_results_retained':True,'automatic_retry':False,
      'guaranteed_10_percent':False}

def config_for(base, cell):
    cfg=copy.deepcopy(base)
    cfg.update(replicas=cell['replicas'],arrival_rate=cell['rate'],base_cap=4096,
               workflow_count=cell['workflow_count'],wf_public_method=cell['method'],
               wf_routing_policy=cell['method'],wf_router_seed=cell['router_seed'])
    # The profile is unchanged. These are declared deployment/load interventions,
    # not a claim that the frozen four-replica profile is re-fit at each cell.
    for k,v in [('hard_total',4096),('epoch_s',.2),('max_model_len',4096),
                ('block_size',16),('max_num_seqs',64),('arrival_shape','wave')]:
        if cfg.get(k)!=v: raise ValueError('unexpected original contract: '+k)
    return cfg
