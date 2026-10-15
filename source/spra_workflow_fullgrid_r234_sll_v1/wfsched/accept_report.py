"""Block-level reports over saved runs; never merge code versions or pick repeats.

Audit here verifies saved protocol, archived source and summary identity. It is
not an independent re-computation of GPU timestamps or backend iterations.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import uuid
import zipfile
from .common import load, sha, digest, finite, write_new
from .report import summarize, csv_write, FIELDS

DROP={'workflow_rows','unfinished_workflow_curve','request_latency_rows','quality_rows','typed_response_rows'}

def safe_file(root: Path, relative: str) -> Path:
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe evidence path')
    q=root
    for part in p.parts:
        q=q/part
        if q.is_symlink():raise ValueError('symlinks are not accepted in evidence')
    if not q.is_file():raise FileNotFoundError(q)
    return q

def read_saved(out: Path):
    out=Path(out).resolve(strict=True)
    protocol=load(out/'protocol.json')
    if sha(out/'protocol.json')!=load(out/'protocol.sha256.json')['sha256']:raise ValueError('protocol hash mismatch')
    status=load(out/'suite_status.json')
    if status.get('status') not in ('paused','complete','failed','prepared'):raise ValueError('report only paused/completed/stopped runs')
    rows=load(out/'index.json');design=protocol['plan']['conditions']
    if len({c['label'] for c in design})!=len(design):raise ValueError('duplicate plan label')
    result=summarize(rows,design)
    if status.get('completed_conditions',len(rows))!=len(rows):raise ValueError('status/index count mismatch')
    if status['status']=='complete' and len(rows)!=len(design):raise ValueError('false complete status')
    for r in rows:
        f=safe_file(out,str(Path(r['folder'])/'summary.json'));s=load(f)
        if sha(f)!=r['files']['summary.json'] or digest(s)!=r['summary_sha256']:raise ValueError('raw summary hash mismatch')
        if {k:v for k,v in s.items() if k not in DROP}!=r['summary']:raise ValueError('index/raw summary mismatch')
        if s.get('condition_valid') is not True or s.get('prefill_caps_applied')!=[4096] or s.get('budget_switches_total')!=0:
            raise ValueError('not a valid fixed-budget scheduling condition')
    source=safe_file(out,'extension_source.zip')
    with zipfile.ZipFile(source) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate source archive member')
        for n,h in protocol['extension_identity'].items():
            if n.startswith('/') or '..' in Path(n).parts or hashlib.sha256(z.read(n)).hexdigest()!=h:
                raise ValueError('archived source identity mismatch')
    return protocol,rows,status,result

def by_block(pairs,design):
    groups=defaultdict(list)
    for p in pairs:groups[(p['replicas'],p['rate'],p['block'],p['candidate'],p['reference'])].append(p)
    result=[]
    for (r,rate,b,c,a),ps in sorted(groups.items()):
        # Expected pairs are the intersection of declared methods, not rows received.
        def want(m):return {d['repeat'] for d in design if (d['replicas'],d['rate'],d['block'],d['method'])==(r,rate,b,m)}
        expected=want(c)&want(a);seen={p['repeat'] for p in ps}
        if len(ps)!=len(seen) or not seen<=expected:raise ValueError('duplicate/unplanned block pair')
        row={'replicas':r,'rate':rate,'block':b,'candidate':c,'reference':a,
             'pairs':len(ps),'expected_pairs':len(expected),'complete_case_comparison':seen==expected and bool(expected),
             'repeat_ids':','.join(map(str,sorted(seen))),
             'scope':'one fixed historical 16-question Block; mean over execution repeats only'}
        for k in [f+'_gain_pct' for f in FIELDS]+['final_em_delta_pp','final_f1_delta_pp']:
            vs=[p[k] for p in ps if finite(p.get(k))]
            row[k]=statistics.mean(vs) if vs else None;row[k+'_n']=len(vs)
        vs=[p['mean_jct_ms_gain_pct'] for p in ps]
        row.update(jct_positive_pairs=sum(v>0 for v in vs),jct_min_gain_pct=min(vs),jct_max_gain_pct=max(vs),
          jct_mean_ms_candidate=statistics.mean(p['candidate_mean_jct_ms'] for p in ps),
          jct_mean_ms_reference=statistics.mean(p['reference_mean_jct_ms'] for p in ps))
        row['point_estimate_at_least_10pct']=row['complete_case_comparison'] and row['mean_jct_ms_gain_pct']>=10
        result.append(row)
    return result

def exploratory_examples(pairs):
    """Same Block for all metrics/baselines; labeled posthoc, never confirmation."""
    grouped=defaultdict(dict)
    for p in pairs:
        if p['candidate']=='DAG' and p['reference'] in ('RR','P2C','Affinity'):
            grouped[(p['replicas'],p['rate'],p['repeat'],p['block'])][p['reference']]=p
    candidates=defaultdict(list)
    for (r,rate,t,b),refs in grouped.items():
        if set(refs)!={'RR','P2C','Affinity'}:continue
        row={'replicas':r,'rate':rate,'repeat':t,'block':b,
             'selection':'posthoc maximum of minimum JCT gain across RR/P2C/Affinity',
             'not_confirmatory':True,'not_overall_mean':True}
        row['minimum_JCT_gain_pct']=min(p['mean_jct_ms_gain_pct'] for p in refs.values())
        for ref,p in refs.items():
            for k in [f+'_gain_pct' for f in FIELDS]+['final_em_delta_pp','final_f1_delta_pp']:
                row[ref+'__'+k]=p.get(k)
        candidates[(r,rate,t)].append(row)
    # Deterministic tie-break: lower Block ID, not a separate metric.
    return [max(rows,key=lambda x:(x['minimum_JCT_gain_pct'],-x['block'])) for _,rows in sorted(candidates.items())]

def write_report(out: Path) -> Path:
    out=Path(out).resolve(strict=True);p,rows,status,result=read_saved(out)
    blocks=by_block(result['pairs'],p['plan']['conditions'])
    selected=exploratory_examples(result['pairs'])
    parent=out/'acceptance';parent.mkdir(exist_ok=True)
    target=parent/('REVIEW_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);target.mkdir()
    csv_write(target/'block_results.csv',blocks)
    csv_write(target/'all_paired_metrics.csv',result['pairs'])
    csv_write(target/'exploratory_selected_blocks.csv',selected)
    write_new(target/'evidence.json',{'run':str(out),'run_uuid':p.get('run_uuid'),'execution_status':status,
        'protocol_sha256':sha(out/'protocol.json'),'source_identity':p['extension_identity'],
        'summary_identity_audited':True,'raw_timestamps_recomputed':False,
        'historical_versions_pooled':False,'selected_examples_are_posthoc':True,
        'block_selection_does_not_choose_individual_questions':True,'block_results':blocks,
        'all_pairs':result['pairs'],'exploratory_selected_blocks':selected})
    lines=['# Block级实验验收登记','',f"来源：`{out}`；状态：{status['status']}；已接受{len(rows)}/{len(p['plan']['conditions'])}条件。",'',
      '每行是固定副本、固定到达率、固定Block和固定对照；只平均执行重复，不强制平均四个Block。',
      '案例是从已观察历史结果中选择的，不是未见问题验证。新确认运行不得再次挑选有利重复。',
      'exploratory_selected_blocks.csv单独列出各轮事后选择案例；不等于总体收益或自动选择算法的实测性能。','',
      '|R|λ|Block|方法|对照|配对|JCT改善%|P95改善%|TTFT改善%|F1差值pp|',
      '|---:|---:|---:|---|---|---:|---:|---:|---:|---:|']
    def fmt(x):return 'NA' if not finite(x) else f'{x:.2f}'
    for b in blocks:
        lines.append('|'+ '|'.join([str(b['replicas']),str(b['rate']),str(b['block']),b['candidate'],b['reference'],
          f"{b['pairs']}/{b['expected_pairs']}",*[fmt(b[k]) for k in ('mean_jct_ms_gain_pct','p95_jct_ms_gain_pct','mean_request_ttft_ms_gain_pct','final_f1_delta_pp')]])+'|')
    lines+=['','10%标记仅是完整计划下的点估计门槛，不是统计显著性、同时置信下界或每次运行保证。',
      'P95仍是条件P95；TPOT为客户端代理；cohort完成率不是最大容量；控制时间不从JCT中扣除。',
      '同一条负载或副本扫描曲线应固定Block；不同Block的案例可以分图、分表，但不能无说明拼成同一应用的扩容曲线。']
    (target/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return target/'REPORT.md'
