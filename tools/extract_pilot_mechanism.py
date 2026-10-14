#!/usr/bin/env python3
"""Regenerate the 24 rows of pilot mechanism data from archived REAL native summaries.

This is a separate three-repeat mechanism experiment, not the eight-repeat B3 confirmation.
"""
from __future__ import annotations
import argparse,io,json,re,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
ZIP=ROOT/'evidence/original/checkpoint_review_20261012.zip'
BASE='repo/spra_stage19_typed_handoff_v1/runs/'
REG=re.compile(r'(PILOT_[^/]+)/B([1-4])_\d+_([AD])/summary\.json$')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/pilot_mechanism_from_raw.csv')
    a=p.parse_args();out=a.out.resolve()
    if out.exists():p.error(f'output exists: {out}')
    rows=[]
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            if not name.startswith(BASE):continue
            m=REG.search(name)
            if not m:continue
            pilot,b,method=m.groups()
            summary=json.loads(z.read(name))
            critical=summary['mean_critical_path_components_ms']
            row={
              'batch':pilot,'block':int(b),'method':'Affinity' if method=='A' else 'Ours',
              'jct_ms':summary['mean_jct_ms'],'p95_jct_ms':summary['p95_jct_ms'],
              'ttft_ms':summary['mean_request_ttft_ms'],'output_tokens':summary['output_tokens'],
              'requests':summary['requests'],
              'native_skipped_query_tokens':summary['native_skipped_query_tokens'],
              'root_to_branch_cache_links_verified':summary['root_to_branch_cache_links_verified'],
              'cache_transition_events':summary['cache_transition_events'],
              'routing_deviations':summary['routing_deviations_from_snapshot_affinity']
            }
            for key,val in critical.items():row['critical_'+key+'_ms']=val
            rows.append(row)
    df=pd.DataFrame(rows).sort_values(['batch','block','method']).reset_index(drop=True)
    if len(df)!=24 or any(df.groupby(['batch','block']).size()!=2):
        raise SystemExit(f'Expected 3 batches x 4 blocks x 2 methods, got {len(df)}')
    reference=pd.read_csv(ROOT/'evidence/derived/pilot_mechanism_raw.csv')
    indices=['batch','block','method']
    merged=df.merge(reference,on=indices,suffixes=('_out','_previous'),validate='one_to_one')
    if len(merged)!=24:raise SystemExit('reference batch alignment failed')
    maxdiff={}
    for col in reference:
        if col not in indices and pd.api.types.is_numeric_dtype(reference[col]):
            maxdiff[col]=float((merged[col+'_out']-merged[col+'_previous']).abs().max())
    if any(v>1e-5 for v in maxdiff.values()):
        raise SystemExit(f'Reconstructed values differ from saved analyses (max abs errors): {maxdiff}')
    out.parent.mkdir(parents=True,exist_ok=True)
    df[reference.columns].to_csv(out,index=False)
    print(f'PILOT_REEXTRACT_OK={out} ROWS={len(df)} MAX_ERROR={max(maxdiff.values())}')
if __name__=='__main__':main()
