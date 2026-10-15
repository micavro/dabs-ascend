"""Post-run, read-only interpretation of original native vLLM evidence.

Only `model_input.skipped_query_tokens` is a native-observed cache reuse value.
Native records do NOT identify the workflow that populated a cached block.
Token-prefix and causality analysis below describes *eligible prior producers*,
not ground-truth block ownership. Ambiguity is preserved and never assigned.
No service, kernel, scheduler or timing source modification is required.
"""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from .common import atomic

BLOCK = 16
PHASE_KEYS=('dependency_release','central_queue_and_admission','input_prepare',
 'header_wait','postprocess','first_content_wait','content_receive','response_tail')


def common_prefix(a,b):
    length=0
    for x,y in zip(a,b):
        if x!=y: break
        length+=1
    return length


def classify_interval(start: int, skipped: int, same_length: int, other_length: int):
    """Partition ONLY natively skipped positions; both producer types may qualify.

    For prefixes, producers cover [0,same_length) and [0,other_length).
    Excluded sources receive no fabricated credit. This is not an ownership trace.
    """
    end=start+skipped
    common=max(0,min(end,same_length,other_length)-start)
    same_only=max(0,min(end,same_length)-max(start,other_length))
    cross_only=max(0,min(end,other_length)-max(start,same_length))
    unattributed=skipped-common-same_only-cross_only
    if min(same_only,cross_only,common,unattributed)<0 or sum((same_only,cross_only,common,unattributed))!=skipped:
        raise ValueError('causal token attribution does not partition native skip')
    return same_only,cross_only,common,unattributed


def csv_write(path,rows,columns):
    path=Path(path)
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,columns,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def capture_evidence(folder, summary, nodes, cell):
    """Run after native completion, before RAW evidence fingerprint freezes.

    Preserve all model_input events; a single native request can have several
    chunked-Prefill events. `mono_ns` and workflow `finish` share host monotonic clock.
    """
    folder=Path(folder)
    if not (folder/'backend.jsonl').is_file():raise ValueError('missing original native backend events')
    events=[]
    with (folder/'backend.jsonl').open(encoding='utf-8') as f:
        for l in f:
            if l.strip():events.append(json.loads(l))
    if not events:raise ValueError('empty native backend evidence')
    byhash={n['hash']:(k,n) for k,n in nodes.items()}
    if len(byhash)!=len(nodes):raise ValueError('native prompt hash not unique')
    inputs=[e for e in events if e.get('event')=='model_input']
    if not inputs:raise ValueError('native model_input events missing')
    accepted={e['prompt_hash']:e for e in events if e.get('event')=='accept'}
    if len(accepted)!=len(nodes) or any(h not in accepted for h in byhash):
        raise ValueError('native accept request identity/coverage invalid')
    totals=defaultdict(int)
    rows=[]
    memo={}
    node_rows={k:{'node':k,'wid':str(n['wid']),'replica':int(n['replica']),
        'input_tokens':int(n['prompt']),'output_tokens':int(n['output']),
        'source_graph_shape':str(n['graph_shape']),'native_skip_tokens':0,
        'intra_only_eligible_tokens':0,'inter_only_eligible_tokens':0,
        'both_eligible_tokens':0,'unattributed_tokens':0,
        'intra_eligible_potential_tokens':0,'inter_eligible_potential_tokens':0,
        'native_input_events':0} for k,n in nodes.items()}
    for i,e in enumerate(inputs):
        h=e.get('prompt_hash')
        if h not in byhash:raise ValueError('foreign native model input')
        key,n=byhash[h]
        if 'mono_ns' not in e: raise ValueError('native model-input timestamp missing')
        skip=e.get('skipped_query_tokens')
        start=e.get('context_before_cache')
        before=e.get('query_tokens_before_cache')
        if any(type(x)!=int for x in (skip,start,before)) or not (0<=skip<=before and start>=0):
            raise ValueError('invalid native cache skip or chunk context')
        if start+skip>len(n['tokens']):
            raise ValueError('native cached token segment exceeds prompt token count')
        if int(e['replica'])!=int(n['replica']):raise ValueError('native placement mismatch')
        t=float(e['mono_ns'])/1e9
        # Exact *same replica* and *earlier completed request* candidate pool.
        # No speculative cache state is treated as observed. This deliberately
        # ignores in-flight producers and may increase unattributed amount.
        intra=inter=0
        intra_candidates=inter_candidates=0
        if skip:
            for j,prev in nodes.items():
                if j==key or int(prev.get('replica',-1))!=int(n['replica']):continue
                done=prev.get('finish')
                if not isinstance(done,(int,float)) or done>=t:continue
                pair=tuple(sorted((key,j)))
                lp=memo.get(pair)
                if lp is None:
                    lp=common_prefix(n['tokens'],prev['tokens'])
                    memo[pair]=lp
                if lp<=start:continue
                if n['wid']==prev['wid']:
                    intra=max(intra,lp);intra_candidates+=1
                else:
                    inter=max(inter,lp);inter_candidates+=1
        same_only,cross_only,both,unknown=classify_interval(start,skip,intra,inter)
        nr=node_rows[key]
        nr['native_skip_tokens']+=skip
        nr['intra_only_eligible_tokens']+=same_only
        nr['inter_only_eligible_tokens']+=cross_only
        nr['both_eligible_tokens']+=both
        nr['unattributed_tokens']+=unknown
        nr['intra_eligible_potential_tokens']+=same_only+both
        nr['inter_eligible_potential_tokens']+=cross_only+both
        nr['native_input_events']+=1
        totals['native_skip_tokens']+=skip
        totals['intra_only_eligible_tokens']+=same_only
        totals['inter_only_eligible_tokens']+=cross_only
        totals['both_eligible_tokens']+=both
        totals['unattributed_tokens']+=unknown
        rows.append({'native_input_event':i,'node':key,'wid':n['wid'],'replica':int(n['replica']),
           'timestamp_mono_ns':int(e['mono_ns']),'segment_start_token':start,
           'native_skipped_tokens':skip,'same_wf_max_eligible_prefix':intra,
           'other_wf_max_eligible_prefix':inter,'prior_same_wf_candidates':intra_candidates,
           'prior_other_wf_candidates':inter_candidates,
           'intra_only_eligible_tokens':same_only,'inter_only_eligible_tokens':cross_only,
           'both_eligible_tokens':both,'unattributed_tokens':unknown})
    if int(summary['native_skipped_query_tokens'])!=totals['native_skip_tokens']:
        raise ValueError('native skipped-query token checksum mismatch')
    if sum(node['native_skip_tokens'] for node in node_rows.values())!=totals['native_skip_tokens']:
        raise ValueError('per node cache skip checksum mismatch')
    cache=dict(totals)
    cache.update({'schema':'dabs-native-KV-source-eligibility-1',
        'workflow_intra_actual_source_verified':False,
        'workflow_inter_actual_source_verified':False,
        'native_skip_verified':True,
        'attribution':'exclusive eligible prior completed same-replica input-token prefix vs cross-workflow prefix; NOT true GPU KV block writer attribution',
        'both_eligible_kept_separate':True,
        'unattributed_kept_separate':True,
        'native_model_input_records':len(inputs),
        'native_requests':len(nodes),
        'total_prompt_tokens':sum(int(n['prompt']) for n in nodes.values()),
        'root_to_branch_cache_links_verified':int(summary.get('root_to_branch_cache_links_verified',0)),
        'native_cache_skipped_fraction':totals['native_skip_tokens']/max(1,sum(int(n['prompt']) for n in nodes.values())),
        'intra_eligible_upper_bound_tokens':totals['intra_only_eligible_tokens']+totals['both_eligible_tokens'],
        'inter_eligible_upper_bound_tokens':totals['inter_only_eligible_tokens']+totals['both_eligible_tokens']})
    # Native resource occupancy != cross/intra reuse. Preserve separately.
    capacities={int(s['pod'].split('-')[-1]):int(s['num_gpu_blocks']) for s in
         json.loads((folder/'initial_cache.json').read_text())['status'] if 'pod' in s and 'num_gpu_blocks' in s and s['pod'].split('-')[-1].isdigit()}
    occupancy=defaultdict(list)
    for e in events:
        if e.get('event')!='iteration' or not e.get('scheduled'):continue
        r=int(e.get('replica',-1));free=e.get('free_blocks_before')
        cap=capacities.get(r)
        if cap is not None and cap>0 and isinstance(free,int) and 0<=free<=cap:
            occupancy[r].append((cap-free)/cap)
    occ=[{'replica':r,'sampled_iterations':len(v),
          'mean_kv_block_occupancy_fraction':statistics.mean(v),
          'peak_kv_block_occupancy_fraction':max(v)} for r,v in sorted(occupancy.items())]
    if not occ:cache['kv_occupancy_status']='not_available_in_original_backend_events'
    else:cache['kv_occupancy_status']='derived_from_free_gpu_blocks_before_each_scheduled_iteration'
    breakdown=summary.get('mean_critical_path_components_ms')
    if not isinstance(breakdown,dict) or not all(k in breakdown for k in PHASE_KEYS):
        raise ValueError('missing native workflow critical-path components')
    decomp={'schema':'dabs-mean-workflow-critical-path-components-1','mean_jct_ms':summary['mean_jct_ms'],
            'component_ms':{k:breakdown[k] for k in PHASE_KEYS},
            'mean_ttft_ms':summary.get('mean_request_ttft_ms'),
            'p95_ttft_ms':summary.get('p95_request_ttft_ms'),
            'mean_node_time_to_first_content_ms':summary.get('mean_node_time_to_first_content_ms'),
            'controller_mean_decision_ms':summary.get('decision_wall_ms',{}).get('mean'),
            'controller_p95_decision_ms':summary.get('decision_wall_ms',{}).get('p95'),
            'cohort_workflows_per_s':summary.get('cohort_workflows_s'),
            'cohort_output_tokens_per_s':summary.get('cohort_tokens_s'),
            'warning':'critical-path components are mean per-workflow aggregates, not a GPU kernel time breakdown'}
    atomic(folder/'kv_evidence_summary.json',cache)
    atomic(folder/'latency_decomposition.json',decomp)
    atomic(folder/'kv_replica_occupancy.json',occ)
    columns=['node','wid','replica','input_tokens','output_tokens','source_graph_shape','native_skip_tokens',
        'intra_only_eligible_tokens','inter_only_eligible_tokens','both_eligible_tokens',
        'unattributed_tokens','intra_eligible_potential_tokens','inter_eligible_potential_tokens','native_input_events']
    csv_write(folder/'kv_nodes.csv',node_rows.values(),columns)
    csv_write(folder/'kv_model_input_events.csv',rows,list(rows[0]) if rows else [])
    return {'cache':cache,'latency':decomp,'replica_occupancy':occ}
