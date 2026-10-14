"""Request-level client metrics, separate from summed workflow critical paths.

All client durations are seconds in the source, returned in ms (TPOT: ms/token).
No tokenization, GPU/network calls or latency subtraction are used. Counts come
from recorded server usage, never the requested max_tokens.
"""
from __future__ import annotations
import math, statistics
from collections import Counter
from .common import quantile
SCHEMA='stage18-request-latency-1'

def finite(v):
    return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)

def distribution(values):
    xs=[float(x) for x in values if finite(x)]
    return {'n':len(xs),'mean':statistics.mean(xs) if xs else None,
            'p50':quantile(xs,.5) if xs else None,'p95':quantile(xs,.95) if xs else None,
            'max':max(xs) if xs else None}

def eligible_time(node, nodes):
    """Dependency/input-ready boundary; not when a scanning loop noticed it."""
    at=node.get('arrival')
    if not finite(at):return None,'missing_workflow_arrival'
    parents=node.get('parents')
    if not isinstance(parents,list):return None,'missing_parent_list'
    times=[at]
    for k in parents:
        p=nodes.get(k)
        if not isinstance(p,dict) or not finite(p.get('finish')):
            return None,'missing_parent_finish'
        times.append(p['finish'])
    # A real application may explicitly include input/tool readiness later than
    # dependency satisfaction. Never infer it from ready-discovery timestamps.
    if finite(node.get('input_eligible_at')):times.append(node['input_eligible_at'])
    return max(times),None

def request_metrics(nodes):
    rows=[];failures=Counter();tpot_exclusions=Counter()
    for key,node in nodes.items():
        x=node.get('stream') or {}
        row={'node':key,'wid':node.get('wid'),'name':node.get('name'),
             'request_ttft_ms':None,'node_time_to_first_content_ms':None,
             'request_latency_ms':None,'client_tpot_proxy_ms':None,
             'raw_tail_inclusive_tpot_proxy_ms':None,'visible_content_span_ms':None,
             'tpot_content_boundary_proxy_ms':None,'completion_tokens':None,
             'content_event_count':x.get('text_events'),'eligible_at':None,
             'request_valid':False,'tpot_exclusion':None,'node_first_content_exclusion':None}
        if not x or x.get('error') or node.get('state') in ('failed','cancelled'):
            row['invalid_reason']='failed_or_missing_request';failures[row['invalid_reason']]+=1
            rows.append(row);continue
        t0,end,first,last=(x.get(k) for k in ('request_start','return','first','last'))
        if not (finite(t0) and finite(end) and end>=t0):
            row['invalid_reason']='invalid_request_timeline';failures[row['invalid_reason']]+=1;rows.append(row);continue
        row['request_valid']=True;row['request_latency_ms']=(end-t0)*1000
        use=(x.get('usage') or {}).get('completion_tokens')
        if type(use) is int and use>=0:row['completion_tokens']=use
        if not (finite(first) and t0<=first<=end):
            row['invalid_reason']='no_valid_visible_content_time';failures[row['invalid_reason']]+=1
            row['tpot_exclusion']=row['invalid_reason'];tpot_exclusions[row['tpot_exclusion']]+=1;rows.append(row);continue
        row['request_ttft_ms']=(first-t0)*1000
        eligible,reason=eligible_time(node,nodes)
        if eligible is not None and eligible<=first:
            row['eligible_at']=eligible;row['node_time_to_first_content_ms']=(first-eligible)*1000
        else:row['node_first_content_exclusion']=reason or 'eligibility_after_first_content'
        if finite(last) and first<=last<=end:row['visible_content_span_ms']=(last-first)*1000
        else:row['content_boundary_invalid']=True
        if type(use) is not int or use<0:reason='missing_or_invalid_output_usage'
        elif use<=1:reason='output_tokens_le_one'
        else:
            # Raw conventional completion-boundary quotient retained for audit.
            # With one coalesced text event it is chiefly response-tail latency,
            # so it is excluded from the headline TPOT proxy distribution.
            row['raw_tail_inclusive_tpot_proxy_ms']=(end-first)*1000/(use-1)
            ne=x.get('text_events')
            if type(ne) is not int:reason='missing_content_event_count'
            elif ne<2:reason='single_visible_event_no_decode_span'
            elif not (finite(last) and last>first):reason='unresolved_visible_event_time_span'
            else:
                reason=None;row['client_tpot_proxy_ms']=row['raw_tail_inclusive_tpot_proxy_ms']
                row['tpot_content_boundary_proxy_ms']=(last-first)*1000/(use-1)
        if reason:row['tpot_exclusion']=reason;tpot_exclusions[reason]+=1
        # Exact ITL/token generation timestamps are intentionally not invented.
        rows.append(row)
    fields=('request_ttft_ms','node_time_to_first_content_ms','request_latency_ms',
            'client_tpot_proxy_ms','raw_tail_inclusive_tpot_proxy_ms','tpot_content_boundary_proxy_ms')
    d={f:distribution([r[f] for r in rows]) for f in fields}
    summary={'schema':SCHEMA,'requests_total':len(rows),'requests_with_valid_return':sum(r['request_valid'] for r in rows),
             'request_exclusions':dict(failures),'tpot_exclusions':dict(tpot_exclusions),'distributions':d,
             'weighting':'one completed LLM request per sample; not critical-path sums and not token-weighted',
             'definitions':{'request_ttft_ms':'1000*(first_nonempty_content-request_start)',
                'node_time_to_first_content_ms':'1000*(first_nonempty_content-max(workflow_arrival,parent_finishes,input_eligible_at_if_explicit))',
                'client_tpot_proxy_ms':'1000*(request_return-first_nonempty_content)/(usage.completion_tokens-1); N>1 and >=2 resolved visible events',
                'raw_tail_inclusive_tpot_proxy_ms':'same quotient without coalescing exclusion; diagnostic only',
                'client_tpot_end_boundary':'request_return, includes stream termination and response tail',
                'server_output_count_source':'recorded usage; requested output length is never substituted'},
             'exact_client_token_itl_available':False,'gpu_token_generation_timestamps_available':False,
             'workflow_user_visible_ttft_ms':None,'user_visible_reason':'internal workflow outputs are not declared user-visible',
             'planned_compute_cost_subtracted':False}
    flat={}
    for f in ('request_ttft_ms','node_time_to_first_content_ms','request_latency_ms','client_tpot_proxy_ms'):
        flat['mean_'+f]=d[f]['mean'];flat['p95_'+f]=d[f]['p95'];flat[f+'_samples']=d[f]['n']
    return {'request_latency_metrics':summary,'request_latency_rows':rows,**flat}

def backend_progress_metrics(nodes,events):
    """HOST-observed output-count progression, never labelled exact GPU ITL.

    New hook output_progress records retain counts after an engine step. Cumulative
    counts can jump (e.g. async output processing). Do not spread a multi-token jump
    uniformly or treat it as zero token latency.
    """
    byhash={n.get('hash'):k for k,n in nodes.items() if n.get('hash')};series={}
    for e in events:
        if e.get('event')!='output_progress':continue
        k=byhash.get(e.get('prompt_hash'))
        if k is not None:series.setdefault(k,[]).append(e)
    rows=[];intervals=[]
    for k,node in nodes.items():
        xs=sorted(series.get(k,[]),key=lambda x:(x.get('observed_ns',0),x.get('output_tokens',0)))
        expected=((node.get('stream') or {}).get('usage') or {}).get('completion_tokens')
        r={'node':k,'observations':len(xs),'count_reaches_usage':False,'all_increments_single_token':False,
           'host_output_tpot_proxy_ms':None,'host_observed_single_step_intervals_ms':[]}
        if not xs:rows.append(r);continue
        counts=[x.get('output_tokens') for x in xs];ts=[x.get('observed_ns') for x in xs]
        valid=all(type(x) is int and x>0 for x in counts) and all(type(x) is int for x in ts)
        valid=valid and all(b>a for a,b in zip(counts,counts[1:])) and all(b>=a for a,b in zip(ts,ts[1:]))
        r['monotone_progress']=valid
        r['count_reaches_usage']=bool(valid and type(expected) is int and counts[-1]==expected)
        r['all_increments_single_token']=bool(valid and counts[0]==1 and all(b-a==1 for a,b in zip(counts,counts[1:])))
        if r['count_reaches_usage'] and counts[0]==1 and expected>1:
            r['host_output_tpot_proxy_ms']=(ts[-1]-ts[0])/1e6/(expected-1)
        if r['count_reaches_usage'] and r['all_increments_single_token']:
            gaps=[(b-a)/1e6 for a,b in zip(ts,ts[1:])]
            if all(g>0 for g in gaps):r['host_observed_single_step_intervals_ms']=gaps;intervals+=gaps
        rows.append(r)
    return {'schema':'stage18-host-output-progress-1','rows':rows,'requests_total':len(nodes),
            'requests_with_progress':sum(bool(r['observations']) for r in rows),
            'requests_reaching_usage':sum(r['count_reaches_usage'] for r in rows),
            'host_output_tpot_proxy_ms':distribution([r['host_output_tpot_proxy_ms'] for r in rows]),
            'host_observed_interval_ms':distribution(intervals),'exact_gpu_itl':None,
            'clock':'engine-host monotonic at completed-step observation; not CUDA execution time',
            'multi_token_delta_interpolated':False}
