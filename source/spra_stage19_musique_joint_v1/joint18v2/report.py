from __future__ import annotations
import statistics
from .common import quantile

def comparison(rows):
    blocks=sorted({x['block'] for x in rows});paired=[]
    for b in blocks:
        g={r['method']:r['summary'] for r in rows if r['block']==b};A=g['A'];J=g['Joint'];C=g['Decoupled']
        gains={m:100*(1-z['mean_jct_ms']/A['mean_jct_ms']) for m,z in g.items()}
        paired.append({'block':b,'jct_gain_vs_A_pct':gains,'Joint_vs_Decoupled_pct':100*(1-J['mean_jct_ms']/C['mean_jct_ms']),
             'Joint_p95_gain_pct':100*(1-J['p95_jct_ms']/A['p95_jct_ms']),
             'Joint_cohort_tokens_gain_pct':100*(J['cohort_tokens_s']/A['cohort_tokens_s']-1),
             'equal_output_work':len({z['output_tokens'] for z in g.values()})==1})
    return {'paired':paired,'mean_Joint_gain_pct':statistics.mean(x['jct_gain_vs_A_pct']['Joint'] for x in paired),
      'mean_Joint_vs_Decoupled_pct':statistics.mean(x['Joint_vs_Decoupled_pct'] for x in paired),
      'mean_Joint_p95_gain_pct':statistics.mean(x['Joint_p95_gain_pct'] for x in paired),
      'mean_Joint_cohort_tokens_gain_pct':statistics.mean(x['Joint_cohort_tokens_gain_pct'] for x in paired),
      'development_blocks':len(blocks),'singleton_positive_gate':False,'maximum_capacity_established':False,
      'formal_10pct_claim':False,'mean_JCT_is_primary':True,'source':'same-version paired real executions'}

def mechanism_report(rows):
    paired=[]
    for b in sorted({x['block'] for x in rows}):
        g={(x['placement'],x['budget']):x['summary'] for x in rows if x['block']==b}
        cp=g['concentrate','protect']['mean_jct_ms'];ca=g['concentrate','aggressive']['mean_jct_ms']
        sp=g['split','protect']['mean_jct_ms'];sa=g['split','aggressive']['mean_jct_ms']
        delta_protect=cp-sp;delta_aggressive=ca-sa
        paired.append({'block':b,'L_ms':{'concentrate_protect':cp,'concentrate_aggressive':ca,'split_protect':sp,'split_aggressive':sa},
            'placement_gain_ms_protect':delta_protect,'placement_gain_ms_aggressive':delta_aggressive,
            'interaction_difference_ms':delta_protect-delta_aggressive,
            'placement_ranking_reverses':delta_protect*delta_aggressive<0,
            'all_background_workflows_counted':True,
            'trigger_satisfied_in_all_cells':all((x.get('mechanism_trigger') or {}).get('trigger_satisfied',False) for x in g.values()),
            'forced_action_had_prefill_in_all_cells':all(x.get('forced_window_activity',{}).get('prefill_logical_tokens',0)>0 for x in g.values()),
            'forced_window_activity':{pl+'_'+bp:x.get('forced_window_activity') for (pl,bp),x in g.items()}})
    return {'paired':paired,'no_reversal_requirement':True,'scope':'predeclared two-placement/two-vector physical intervention; not online-policy efficacy',
            'snapshot_exact_restore_claimed':False,'fresh_namespace_rule':'equal token work and dependency graph; condition salt differs'}

# Metric fields are per-request distributions within each condition. Reporting
# condition P95s is not a pooled request or per-token percentile.
LATENCY_FIELDS=('mean_request_ttft_ms','p95_request_ttft_ms',
                'mean_client_tpot_proxy_ms','p95_client_tpot_proxy_ms',
                'mean_node_time_to_first_content_ms','p95_node_time_to_first_content_ms')
_original_comparison=comparison
_original_mechanism_report=mechanism_report

def _latencies(s):
    return {k:s.get(k) for k in LATENCY_FIELDS}|{
        'request_ttft_samples':s.get('request_ttft_ms_samples'),
        'client_tpot_proxy_samples':s.get('client_tpot_proxy_ms_samples'),
        'tpot_exclusions':s.get('request_latency_metrics',{}).get('tpot_exclusions',{}),
        'exact_gpu_itl_available':False}

def _gain(a,b):
    return 100*(a-b)/a if isinstance(a,(float,int)) and a>0 and isinstance(b,(float,int)) else None

def comparison(rows):
    report=_original_comparison(rows)
    for block in report['paired']:
        g={r['method']:r['summary'] for r in rows if r['block']==block['block']}
        block['request_latency_by_method']={m:_latencies(s) for m,s in g.items()}
        block['Joint_request_latency_gain_vs_A_pct']={k:_gain(g['A'].get(k),g['Joint'].get(k)) for k in LATENCY_FIELDS}
    report['measurement']='TTFT and TPOT proxy are request-level; full JCT includes all waits; no planning-time subtraction'
    return report

def mechanism_report(rows):
    report=_original_mechanism_report(rows)
    for block in report['paired']:
        cells={r['placement']+'_'+r['budget']:r['summary'] for r in rows if r['block']==block['block']}
        block['request_latency_by_cell']={k:_latencies(s) for k,s in cells.items()}
        block['online_search_calls_by_cell']={k:s.get('online_search_calls') for k,s in cells.items()}
    report['mode']='predeclared_complete_actions_with_no_online_solver'
    report['cost_boundary']='no search in the measured path; publication, communication, event/epoch waits and policy admission remain measured'
    report['zero_total_control_cost_claimed']=False
    report['metric_boundary']='all requests including background; separate workflow-JCT and request-TTFT/TPOT-proxy distributions'
    return report
