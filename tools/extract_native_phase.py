#!/usr/bin/env python3
"""Recompute B3 native host phase timing from real pilot summary files (3 repeats)."""
from __future__ import annotations
import argparse,json,re,zipfile
from collections import defaultdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--out',type=Path,default=ROOT/'outputs/native_phase_B3_reextracted.csv')
    args=a.parse_args(); out=args.out.resolve()
    if out.exists():a.error('output exists; choose new --out')
    d=defaultdict(lambda:[0,0.0,0]);batches=defaultdict(set)
    with zipfile.ZipFile(ROOT/'evidence/original/checkpoint_review_20261012.zip') as z:
        for name in z.namelist():
            m=re.search(r'(PILOT_[^/]+)/B3_\d+_([AD])/summary.json$',name)
            if not m:continue
            batch,mt=m.groups();method='Affinity' if mt=='A' else 'Ours'
            summary=json.loads(z.read(name))
            for phase,v in summary['service_prediction_by_phase'].items():
                key=(method,phase);d[key][0]+=v['n'];d[key][1]+=v['n']*v['mean_actual_host_ms'];batches[key].add(batch)
    records=[dict(method=m,phase=p,n=n,host_mean_ms=total/n,runs=len(batches[(m,p)]))
             for (m,p),(n,total,_) in d.items()]
    df=pd.DataFrame(records).sort_values(['method','phase']).reset_index(drop=True)
    old=pd.read_csv(ROOT/'evidence/derived/host_phase_b3.csv')
    merged=old.merge(df,on=['method','phase'],suffixes=('_old',''),validate='one_to_one')
    if len(merged)!=6 or any(merged.n!=merged.n_old) or any(abs(merged.host_mean_ms-merged.host_mean_ms_old)>1e-7):
        raise SystemExit('native phase mismatch with previously recorded results')
    out.parent.mkdir(exist_ok=True,parents=True)
    df.to_csv(out,index=False)
    print(f'HOST_PHASE_REEXTRACT_OK={out} ROWS={len(df)}')

if __name__=='__main__':main()
