"""CPU-only synthetic-state selector check. Never used as GPU JCT evidence."""
from __future__ import annotations
import copy
import statistics
import time
from .common import atomic
from .legacy import inspect,load_original,load_profile

def snapshot(R,workflows=1):
    nodes={};ready=[]
    for w in range(workflows):
        keys=[f'w{w}/{i}' for i in range(3)]
        for i,k in enumerate(keys):
            nodes[k]={'id':k,'wid':f'w{w}','name':k,'parents':[] if i<2 else keys[:2],
              'prompt':512+64*w,'output':8,'computed':0,'output_done':0,'ready_at':.0,
              'state':'ready' if i<2 else 'blocked','replica':None,'share':0,'prefix_hash':None,
              'compiled_groups':[keys[:2]],'compiled_successors':[keys[2]] if i<2 else [],'max_output_tokens':96}
            if i<2:ready.append(k)
    return {'nodes':nodes,'ready':ready,'replicas':R,'pressure':[4]+[0]*(R-1),
      'now':.1,'current_caps':[4096]*R,'capacity_blocks':[4096]*R,'reserved_blocks':[0]*R,
      'prefix_history':{},'cache_evidence':[],'decision_index':1,'state_ages_s':[0.]*R,'cache_unknown_count':0}

def run_benchmark(root,out):
    inspect(root,require_data=False)
    original=load_original(root);_,model,_,mf=load_profile(original,root)
    from semantic19.selector import choose
    from .selector import choose_lite
    from joint18v2.common import action_key
    rows=[]
    for R in (1,2,3,4):
        for nw in (1,2,4):
            cfg=copy.deepcopy(mf['profile_configuration'])
            cfg.update(replicas=R,base_cap=4096,planning_soft_limit_s=5.,apply_delay_ms=0.)
            s=snapshot(R,nw)
            for repeat in range(4):
                result={};elapsed={}
                # Alternate CPU ordering; fixed generous deadlines isolate semantic equivalence.
                order=(('DAG',choose),('DAG-Lite',choose_lite))
                if repeat%2:order=tuple(reversed(order))
                for name,fn in order:
                    start=time.perf_counter()
                    result[name]=fn(copy.deepcopy(s),model,cfg,'D',deadline=time.monotonic()+5.)
                    elapsed[name]=1000*(time.perf_counter()-start)
                if action_key(result['DAG']['action'])!=action_key(result['DAG-Lite']['action']):
                    raise ValueError('D and Lite actions differ in ample-deadline synthetic check')
                for name in result:
                    if not result[name]['model_selection']:raise ValueError('CPU check did not obtain complete reference')
                rows.append({'replicas':R,'synthetic_workflows':nw,'repeat':repeat+1,
                             'DAG_cpu_ms':elapsed['DAG'],'DAG_Lite_cpu_ms':elapsed['DAG-Lite'],
                             'same_action_with_generous_deadline':True,
                             'score_cells':[e['cell'] for e in result['DAG-Lite']['evaluations']]})
    value={'scope':'local CPU, synthetic states, frozen real service model; not GPU performance',
      'pairs':len(rows),'all_actions_matched':True,'GPU_requests':0,
      'DAG_mean_cpu_ms':statistics.mean(r['DAG_cpu_ms'] for r in rows),
      'DAG_Lite_mean_cpu_ms':statistics.mean(r['DAG_Lite_cpu_ms'] for r in rows),'rows':rows,
      'JCT_gain_established':False,'real_deadline_trace_equivalence_claimed':False}
    atomic(out,value);return value
