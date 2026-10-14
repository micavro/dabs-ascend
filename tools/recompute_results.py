#!/usr/bin/env python3
"""Independently recompute tables from the supplied condition-level evidence.

No model access, no external network, and no mutation of original data.
METHOD TRUTH: RR is round-robin; the published report's Preble label is unsupported.
"""
from __future__ import annotations
import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'evidence' / 'derived'
CAT = ROOT / 'evidence' / 'catalog'
OLD_RUN='spra_workflow_sched_v2/runs/ablation_v2'
NEW_RUN='spra_workflow_sched_v2_1/runs/grid_v2_1'

DONE_RE = re.compile(r'^DONE\s+\+\d+\s+(R(\d+)_L(\d+(?:\.\d+)?)_T(\d+)_B(\d+)_(RR|P2C|Affinity|DAG|DAG-Lite))\s*\|\s*JCT=([\d.]+)ms\s+P95=([\d.]+)ms\s+TTFT=([\d.]+)ms\s+WF/s=([\d.]+)\s+out=(\d+)')

def new_grid_rows(log_path: Path):
    rows=[]
    for line in log_path.read_text(encoding='utf-8').splitlines():
        m=DONE_RE.search(line.strip())
        if not m: continue
        (label,r,rate,repeat,block,method,jct,p95,ttft,wfs,out)=m.groups()
        rows.append(dict(label=label,replicas=int(r),rate=float(rate),repeat=int(repeat),
                         block=int(block),method=method,mean_jct_ms=float(jct),p95_jct_ms=float(p95),
                         mean_request_ttft_ms=float(ttft),cohort_workflows_s=float(wfs),output_tokens=int(out),
                         run_id=NEW_RUN,source='GPU console summary (rounded to 0.01 ms)'))
    return pd.DataFrame(rows)

def pair_gain(a,b):
    return 100.0*(1.0-float(a)/float(b))

def compute_pairs(df, by, ref='RR', cand='DAG'):
    unique_key=list(by)+['method']
    if df.duplicated(unique_key).any():
        raise ValueError('duplicate method per paired condition: '+str(df[df.duplicated(unique_key)].head().to_dict('records')))
    out=[]
    for key,group in df.groupby(list(by),dropna=False):
        if not isinstance(key,tuple):key=(key,)
        methods=group.set_index('method')
        if cand not in methods.index or ref not in methods.index:continue
        ca=methods.loc[cand];ba=methods.loc[ref]
        d={k:v for k,v in zip(by,key)}
        for metric in ['mean_jct_ms','p95_jct_ms','mean_request_ttft_ms']:
            d[metric+'_gain_pct']=pair_gain(ca[metric],ba[metric])
        d['cohort_workflows_s_gain_pct']=100.0*(float(ca['cohort_workflows_s'])/float(ba['cohort_workflows_s'])-1)
        d['candidate']=cand;d['reference']=ref
        out.append(d)
    return pd.DataFrame(out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs'/'offline_reanalysis')
    args=parser.parse_args();out=args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory exists and is not empty; choose a NEW --out path to protect results')
    out.mkdir(parents=True,exist_ok=True)
    catalog=pd.read_csv(CAT/'condition_registry.csv')
    pairs=pd.read_csv(CAT/'paired_metrics.csv')
    plans=ROOT/'source/spra_workflow_sched_v2_1/plans/grid.json'
    planned=len(json.loads(plans.read_text(encoding='utf-8'))['conditions'])
    assert planned==640,planned
    case=pd.read_csv(SRC/'case_confirm_rows.csv')
    assert len(case)==32 and set(case.method)=={'RR','P2C','Affinity','DAG'},'Unexpected fixed-case labels'
    assert sorted(case.groupby('method').size())==[8]*4,'Missing fixed-case repeats'
    case['source']='B3 eight-repeat console-derived condition summaries (not full raw request records)'
    case.to_csv(out/'fixed_B3_condition_rows.csv',index=False)
    case_means=case.groupby('method',sort=True)[['mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','cohort_workflows_s','output_tokens']].mean()
    case_means.to_csv(out/'fixed_B3_method_means.csv')
    gains=[]
    for ref in ['RR','P2C','Affinity']:
        tmp=compute_pairs(case,['repeat'],ref)
        assert len(tmp)==8,(ref,len(tmp))
        tmp.to_csv(out/f'fixed_B3_DAG_vs_{ref}_paired.csv',index=False)
        gains.append(dict(baseline=ref,paired_repeats=len(tmp),positive_repeats=int((tmp['mean_jct_ms_gain_pct']>0).sum()),
                          mean_jct_reduction_pct=tmp['mean_jct_ms_gain_pct'].mean(),
                          p95_jct_reduction_pct=tmp['p95_jct_ms_gain_pct'].mean(),
                          ttft_reduction_pct=tmp['mean_request_ttft_ms_gain_pct'].mean(),
                          workflow_throughput_gain_pct=tmp['cohort_workflows_s_gain_pct'].mean()))
    case_gains=pd.DataFrame(gains)
    case_gains.to_csv(out/'fixed_B3_paired_summary.csv',index=False)
    old=pairs[(pairs['run_id']==OLD_RUN)&(pairs['candidate']=='DAG')&(pairs['reference']=='Affinity')].copy()
    assert len(old)==24,('Historical ablation unexpectedly incomplete',len(old))
    old.to_csv(out/'ablation_all_24_DAG_vs_Affinity.csv',index=False)
    block=old.groupby('block',as_index=False).agg(pairs=('mean_jct_ms_gain_pct','size'),positive=('mean_jct_ms_gain_pct',lambda x:int((x>0).sum())),
                     mean_jct_reduction_pct=('mean_jct_ms_gain_pct','mean'),p95_reduction_pct=('p95_jct_ms_gain_pct','mean'),
                     ttft_reduction_pct=('mean_request_ttft_ms_gain_pct','mean'),wf_throughput_gain_pct=('cohort_workflows_s_gain_pct','mean'))
    block.to_csv(out/'ablation_by_block.csv',index=False)
    old_index=catalog[catalog['run_id']==NEW_RUN]
    latest=new_grid_rows(ROOT/'evidence/original/GPU_R2_second_repeat_console_20261012.txt')
    assert len(latest)==80,('Only partial latest console conditions parsed',len(latest))
    assert set(latest.replicas)=={2} and set(latest.repeat)=={2}
    existing=set(old_index.label)
    if existing.intersection(set(latest.label)):raise ValueError('old checkpoint overlaps newly imported grid console')
    merged=pd.concat([old_index,latest],ignore_index=True)
    assert len(merged)==400,(len(old_index),len(latest),len(merged))
    # historical non-grid/failed records must not enter current main grid
    merged.to_csv(out/'grid_v21_known_400_conditions.csv',index=False)
    gb=[]
    for r,gd in merged.groupby('replicas'):
        for rate,gs in gd.groupby('rate'):
            for ref in ['RR','P2C','Affinity']:
                tmp=compute_pairs(gs,['replicas','rate','repeat','block'],ref)
                gb.append(dict(replicas=int(r),rate=float(rate),reference=ref,pairs=len(tmp),expected_pairs=8,
                               mean_jct_reduction_pct=float(tmp.mean_jct_ms_gain_pct.mean()) if len(tmp) else None,
                               complete=bool(len(tmp)==8)))
    pd.DataFrame(gb).to_csv(out/'grid_coverage_and_paired_gains.csv',index=False)
    mechanisms=pd.read_csv(SRC/'pilot_mechanism_raw.csv')
    mechanisms.groupby(['block','method'],as_index=False).mean(numeric_only=True).to_csv(out/'pilot_mechanism_averages.csv',index=False)
    prefix=pd.read_csv(SRC/'prefix_structure.csv')
    prefix.to_csv(out/'prefix_structure.csv',index=False)
    host=pd.read_csv(SRC/'host_phase_b3.csv')
    host.to_csv(out/'native_host_phase_B3.csv',index=False)
    manifest={
        'basis':'Uploaded historical checkpoint + explicitly logged R2 round two + B3 eight repeats',
        'definitions':{
            'RR':'RR / round-robin only; NOT Preble',
            'DAG':'Main paper method Ours',
            'gain':'mean of paired percentage gains, not ratio of group means',
        },
        'grid_plan':planned,'grid_old_accepted':len(old_index),'grid_extra_console':len(latest),'grid_total_observed':len(merged),
        'grid_missing':planned-len(merged),'B3_confirmation':len(case),'ablation':len(old),
        'pilot_rows':len(mechanisms),'notes':['Historical checkpoint does not contain B3 confirmation full backend logs','Pilot latency/KV measurements are separate from the 8-repeat B3 performance run', 'R2 console values are rounded, not exact source timing records']}
    (out/'SCOPE_AND_COUNTS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output_dir':str(out),'counts':{k:manifest[k] for k in ('grid_old_accepted','grid_extra_console','grid_total_observed','grid_missing','B3_confirmation','ablation')},
                     'case_jct_gains':{r.baseline:round(r.mean_jct_reduction_pct,4) for r in case_gains.itertuples()},
                     'ablation_jct_gain':round(old.mean_jct_ms_gain_pct.mean(),4)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
