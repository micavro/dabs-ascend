"""All paired metrics, no best-run or per-metric version selection."""
from __future__ import annotations
import csv
import io
import statistics
from collections import defaultdict
from pathlib import Path
from .common import atomic, finite, load

FIELDS=('mean_jct_ms','p95_jct_ms','planned_arrival_mean_jct_ms','mean_request_ttft_ms',
        'p95_request_ttft_ms','mean_node_time_to_first_content_ms','mean_client_tpot_proxy_ms',
        'p95_client_tpot_proxy_ms','cohort_workflows_s','cohort_tokens_s')
HIGHER={'cohort_workflows_s','cohort_tokens_s'}

def summarize(rows, design):
    expected={r['label']:r for r in design};seen=set();g=defaultdict(dict)
    for row in rows:
        label=row['label']
        if label in seen or label not in expected: raise ValueError('duplicate/unknown accepted condition')
        seen.add(label)
        if any(row[k]!=expected[label][k] for k in expected[label]):raise ValueError('condition specification changed')
        g[(row['replicas'],row['rate'],row['repeat'],row['block'])][row['method']]=row
    pairs=[]
    targets=('DAG','DAG-Lite')
    for (R,rate,t,b),methods in sorted(g.items()):
        for candidate in targets:
            if candidate not in methods:continue
            refs=('RR','P2C','Affinity')+(('DAG',) if candidate=='DAG-Lite' else ())
            for ref in refs:
                if ref not in methods:continue
                c,a=methods[candidate],methods[ref]
                if c['external_work_sha256']!=a['external_work_sha256']:
                    raise ValueError('paired external questions/arrivals differ')
                record={'replicas':R,'rate':rate,'repeat':t,'block':b,'candidate':candidate,'reference':ref}
                for field in FIELDS:
                    cv,av=c['summary'].get(field),a['summary'].get(field)
                    record[field+'_gain_pct']=(100*((cv/av-1) if field in HIGHER else (1-cv/av))
                        if finite(cv) and finite(av) and av>0 else None)
                for k in ('final_em','final_f1'):
                    record[k+'_delta_pp']=100*(c['summary']['semantic_quality'][k]-a['summary']['semantic_quality'][k])
                record['candidate_output_tokens']=c['summary']['output_tokens']
                record['reference_output_tokens']=a['summary']['output_tokens']
                record['candidate_mean_jct_ms']=c['summary']['mean_jct_ms']
                record['reference_mean_jct_ms']=a['summary']['mean_jct_ms']
                pairs.append(record)
    cells=defaultdict(list)
    for p in pairs:cells[(p['replicas'],p['rate'],p['candidate'],p['reference'])].append(p)
    aggregates=[]
    for (R,rate,c,a),ps in sorted(cells.items()):
        ex=sum(r['replicas']==R and r['rate']==rate and r['method']==c for r in design)
        z={'replicas':R,'rate':rate,'candidate':c,'reference':a,'pairs':len(ps),'expected_pairs':ex,
           'complete_cell_comparison':len(ps)==ex}
        for k in (*[f+'_gain_pct' for f in FIELDS],'final_em_delta_pp','final_f1_delta_pp'):
            vals=[p[k] for p in ps if finite(p[k])]
            z[k]=statistics.mean(vals) if vals else None
            z[k+'_n']=len(vals)
        vals=[p['mean_jct_ms_gain_pct'] for p in ps]
        z['jct_positive_pairs']=sum(v>0 for v in vals)
        z['jct_gain_min_pct']=min(vals);z['jct_gain_max_pct']=max(vals)
        aggregates.append(z)
    return {'schema':'workflow-grid-report-2','complete':len(rows)==len(design),
      'accepted_conditions':len(rows),'expected_conditions':len(design),
      'requests':sum(r['summary']['requests'] for r in rows),
      'workflows':sum(r['summary']['workflows'] for r in rows),
      'pairs':pairs,'aggregates':aggregates,
      'primary':'average paired percent reduction, not ratio of grand means',
      'historical_results_pooled':False,'per_condition_p95_not_pooled':True,
      'TPOT_is_client_proxy':True,'capacity_claimed':False,'provenance':'accepted native-run records; see protocol, raw evidence and execution status',
      'confidence_note':'Four historical question blocks and few execution repeats; no iid-workflow significance claim',
      'input_limitation':'Original model-visible per-condition UUID retained; record actual outputs and quality',
      'no_posthoc_version_selection':True,'guaranteed_10_percent':False}

def csv_write(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def write(out, rows, design, status=None):
    out=Path(out);target=out/'analysis';target.mkdir(exist_ok=True)
    result=summarize(rows,design)
    result['execution_status']=status
    # Data contract completeness and session validity are separate.
    result['valid_complete_suite']=result['complete'] and status=='complete'
    atomic(target/'RESULT.json',result)
    csv_write(target/'paired_metrics.csv',result['pairs'])
    csv_write(target/'cell_gains.csv',result['aggregates'])
    absolute=[]
    for row in rows:
        s=row['summary'];z={k:row[k] for k in ('label','replicas','rate','repeat','block','position','method')}
        z.update({k:s.get(k) for k in FIELDS})
        z.update({k:s.get(k) for k in ('requests','output_tokens','max_arrival_lag_ms','budget_switches_total',
              'routing_deviations_from_snapshot_affinity','queue_reorder_iterations','worker_timeout_fallback','commit_fallbacks')})
        z.update(decision_wall_mean_ms=s.get('decision_wall_ms',{}).get('mean'),
                 final_em=s['semantic_quality']['final_em'],final_f1=s['semantic_quality']['final_f1'])
        absolute.append(z)
    csv_write(target/'conditions.csv',absolute)
    lines=['# Workflow请求调度实验结果','',f"状态：{status}；已接受 {len(rows)}/{len(design)} 个条件。",'',
        '各方法独立报告，正数为改善；主统计量为同工况、同轮次、同Block内的百分比改善再平均。',
        'P95是条件P95；TPOT为客户端代理；完成率不是最大吞吐容量。原64题不是未见测试集。',
        '原模型可见UUID隔离方式仍在使用，因此不宣称各方法prompt tokens完全一致。','',
        '|R|λ|方法|对照|配对数|平均JCT改善%|P95 JCT改善%|TTFT改善%|完成率改善%|F1变化pp|',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    def f(x):return 'NA' if not finite(x) else f'{x:.2f}'
    for a in result['aggregates']:
        lines.append('|'+ '|'.join([str(a['replicas']),str(a['rate']),a['candidate'],a['reference'],
          str(a['pairs'])+'/'+str(a['expected_pairs']),*[f(a[k]) for k in ('mean_jct_ms_gain_pct',
          'p95_jct_ms_gain_pct','mean_request_ttft_ms_gain_pct','cohort_workflows_s_gain_pct','final_f1_delta_pp')]])+'|')
    lines+=['','没有根据任何指标选出“最好的一次”；失败条件、未完成计划不能被当成完整成功实验。',
      'R=1下RR与P2C的路由必然相同，是用于诊断时序/输出波动与控制开销的负对照。']
    (target/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
