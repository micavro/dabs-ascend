"""Complete four-method R=2..4 x lambda=4/8/12/16 report.

Produces RAW-independent derived tables, never pools different GPU run identities.
Outputs partial rows honestly while suite is paused. Percent improvements are
condition/repeat PAIRS first; each paired-rate cell is averaged only afterwards.
"""
from __future__ import annotations
import csv
import os
import uuid
import statistics
from collections import defaultdict
from pathlib import Path
from .common import atomic, load, finite
from .evidence import PHASE_KEYS, csv_write

METHODS=('RR','SLL','Ayo','Ours')


def write_extended(out, rows, design, status):
    out=Path(out);ana=out/'analysis';ana.mkdir(exist_ok=True)
    if set(x['method'] for x in design)!=set(METHODS) or len(design)!=96:
        raise ValueError('full grid experiment contract changed')
    if len({(x['replicas'],x['rate'],x['repeat'],x['method']) for x in design})!=96:
        raise ValueError('duplicated or missing R/rate/repeat/method specification')
    decomp=[];cache=[];occupancy=[];native_node=[];means=defaultdict(list)
    for item in rows:
        spec={k:item[k] for k in ('replicas','rate','repeat','block','method','label')}
        folder=out/item['folder'];s=load(folder/'summary.json')
        for name in ('kv_evidence_summary.json','latency_decomposition.json','kv_replica_occupancy.json','kv_nodes.csv'):
            if not (folder/name).is_file():raise ValueError('accepted result missing forensic artifact '+str(folder/name))
        kv=load(folder/'kv_evidence_summary.json')
        br=load(folder/'latency_decomposition.json')
        oc=load(folder/'kv_replica_occupancy.json')
        decomp.append({**spec,'mean_jct_ms':s['mean_jct_ms'], 'p95_jct_ms':s['p95_jct_ms'],
              'mean_request_ttft_ms':s['mean_request_ttft_ms'],
              'p95_request_ttft_ms':s['p95_request_ttft_ms'],
              'cohort_workflows_s':s['cohort_workflows_s'],
              'cohort_tokens_s':s['cohort_tokens_s'],
              'decision_wall_mean_ms':s['decision_wall_ms']['mean'],
              'decision_wall_p95_ms':s['decision_wall_ms']['p95'],
              'backend_effective_delay_mean_ms':s.get('backend_effective_delay_ms',{}).get('mean'),
              **{k+'_ms':br['component_ms'][k] for k in PHASE_KEYS}})
        cache.append({**spec,**kv})
        means[(item['replicas'],item['rate'],item['method'])].append(decomp[-1])
        for v in oc:occupancy.append({**spec,**v})
        with (folder/'kv_nodes.csv').open(encoding='utf-8',newline='') as f:
            for v in csv.DictReader(f):native_node.append({**spec,**v})
    def write(name,data,cols=None):
        if cols is None:cols=list(dict.fromkeys(k for v in data for k in v))
        # Derived analyses can be rebuilt after a pause/resume or an explicit
        # report command. Original per-attempt evidence remains strictly immutable.
        temporary=ana/('.'+name+'.staging-'+uuid.uuid4().hex)
        try:
            csv_write(temporary,data,cols)  # exclusive create of a new staging file
            os.replace(temporary,ana/name)  # atomic publish, derived directory ONLY
        finally:
            if temporary.exists():temporary.unlink()
    write('latency_decomposition_by_condition.csv',decomp)
    write('kv_by_condition.csv',cache)
    write('kv_by_replica.csv',occupancy)
    write('kv_by_node.csv',native_node)
    numeric=('mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','p95_request_ttft_ms',
      'cohort_workflows_s','cohort_tokens_s','decision_wall_mean_ms','decision_wall_p95_ms')
    numeric+=tuple(k+'_ms' for k in PHASE_KEYS)
    agg=[]
    for (r,rate,m),vs in sorted(means.items()):
        a={'replicas':r,'rate':rate,'method':m,'repeat_count':len(vs),'expected_repeats':2,
           'complete_two_repeats':len(vs)==2}
        for k in numeric:
            x=[v[k] for v in vs if finite(v.get(k))]
            a[k]=statistics.mean(x) if x else None
        x=[c for c in cache if c['replicas']==r and c['rate']==rate and c['method']==m]
        for k in ('native_skip_tokens','intra_only_eligible_tokens','inter_only_eligible_tokens',
                  'both_eligible_tokens','unattributed_tokens',
                  'intra_eligible_upper_bound_tokens','inter_eligible_upper_bound_tokens',
                  'native_cache_skipped_fraction','root_to_branch_cache_links_verified'):
            a[k]=statistics.mean(float(v.get(k,0)) for v in x) if x else None
        agg.append(a)
    write('metric_grid.csv',agg)
    pairs=[];group=defaultdict(dict)
    for v in decomp:group[(v['replicas'],v['rate'],v['repeat'])][v['method']]=v
    metric=('mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','cohort_workflows_s','cohort_tokens_s')
    for (r,rate,t),grouped in sorted(group.items()):
        if 'Ours' not in grouped:continue
        for baseline in ('RR','SLL','Ayo'):
            if baseline not in grouped:continue
            a=grouped['Ours'];b=grouped[baseline]
            v={'replicas':r,'rate':rate,'repeat':t,'candidate':'Ours','baseline':baseline}
            for k in metric:
                if not finite(a[k]) or not finite(b[k]) or b[k]<=0:v[k+'_gain_pct']=None
                elif k in ('cohort_workflows_s','cohort_tokens_s'):
                    v[k+'_gain_pct']=100*(a[k]/b[k]-1)
                else:v[k+'_gain_pct']=100*(1-a[k]/b[k])
            pairs.append(v)
    write('paired_ours_gains.csv',pairs)
    psummary=[]
    for r in (2,3,4):
      for rate in (4,8,12,16):
        for baseline in ('RR','SLL','Ayo'):
          rows2=[v for v in pairs if v['replicas']==r and v['rate']==rate and v['baseline']==baseline]
          a={'replicas':r,'rate':rate,'candidate':'Ours','baseline':baseline,
             'observed_pairs':len(rows2),'expected_pairs':2,'complete_pair':len(rows2)==2}
          for k in metric:
            z=[v[k+'_gain_pct'] for v in rows2 if finite(v.get(k+'_gain_pct'))]
            a[k+'_gain_pct']=statistics.mean(z) if z else None
            a[k+'_gain_min_pct']=min(z) if z else None
            a[k+'_gain_max_pct']=max(z) if z else None
          psummary.append(a)
    write('paired_grid_summary.csv',psummary)
    complete=len(rows)==96 and status=='complete' and all(len(v)==4 for v in group.values())
    receipt={'schema':'dabs-full-balanced64-detailed-report-1',
      'status':status,'complete':complete,'accepted':len(rows),'expected':96,
      'two_repeats_per_cell':True,'all_workload_cells':12,
      'methods':list(METHODS),'no_SOTA_preble_in_this_suite':True,
      'native_KV_reuse': 'skipped_query_tokens from original vLLM model_input events',
      'cross_intra_source_is_inferred':'eligible prior completed request exact-token-prefix; not native block-owner provenance',
      'intra_exclusive_source_evidence':'eligible predecessor within the same workflow, not proof of cache block ownership',
      'inter_exclusive_source_evidence':'eligible predecessor in a different workflow, not proof of cache block ownership',
      'ambiguous_and_unattributed_retained':True,
      'JCT_not_normalized_by_tokens':True,
      'finite_arrival_cohort_throughput_not_steady_state_capacity':True,
      'exact_backend_token_itl_available':False,'real_GPU_kernel_time_available':False}
    atomic(ana/'extended_receipt.json',receipt)
    lines=[
      '# Full Balanced-64 workflow scheduling experiment',
      '',f"Status: {status}; accepted: {len(rows)}/96; complete: {complete}.",
      '', 'Methods: RR; SLL = sample two replicas and choose fewer outstanding requests; Ayo = topology-aware controller-level proxy; Ours = V5 DAG-H.',
      'Grid: replicas R=2,3,4; lambda=4,8,12,16 workflows/s; two mirror-ordered rounds per cell.',
      'Fixed workload: 64 original-MuSiQue-derived workflows, 216 nodes, 3584 input tokens per node, 13184 total output tokens, six natural DAG shapes; Balanced occurrence profile.',
      'Wave arrival: 1 s ON at 2*lambda + 1 s OFF. High lambda relative to R=2 may be overload; max capacity NOT established.',
      '', '## Per-cell Ours paired improvements (%)',
      '', '|R|lambda|baseline|paired rounds|Mean JCT|P95 JCT|TTFT|WF throughput|',
      '|---:|---:|---|---:|---:|---:|---:|---:|']
    for x in psummary:
        if not x['observed_pairs']:continue
        def fmt(k):return 'NA' if x[k] is None else f"{x[k]:.2f}"
        lines.append(f"|{x['replicas']}|{x['rate']}|{x['baseline']}|{x['observed_pairs']}/2|"+
          '|'.join(fmt(k+'_gain_pct') for k in ('mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','cohort_workflows_s'))+'|')
    lines+=['','## Raw and derived evidence',
      '', '- `conditions.csv` and `metric_grid.csv`: native accepted workflow and request performance.',
      '- `latency_decomposition_by_condition.csv`: native critical-path component means, decision time, effective control delay.',
      '- `kv_by_condition.csv`, `kv_by_node.csv`, `kv_by_replica.csv`: native saved tokens, candidate within/cross workflow reuse, ambiguity and replica memory pressure.',
      '- `paired_ours_gains.csv` and `paired_grid_summary.csv`: paired method comparisons, not a best-run choice.',
      '', '**Caution:** native vLLM records actual skipped input tokens, NOT which workflow populated a KV block. The intra/inter breakdown is a conservative observed-predecessor token-prefix eligibility analysis; dual-eligible/unknown tokens are never force-assigned. All values labeled as potential producer, not exact cache ownership.',
      '', '**Caution:** Ayo is the controller-level topology proxy (not the complete published Ayo system).',
      '', '**Caution:** TTFT is per LLM request, not declared workflow user-visible TTFT. No empirical statistical significance claim from only two repeats.',
      '', '**Caution:** a paused/failed subset is NOT a complete experimental result.']
    (ana/'FULL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return receipt
