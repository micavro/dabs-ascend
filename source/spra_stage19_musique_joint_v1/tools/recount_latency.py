#!/usr/bin/env python3
"""Read saved Stage18 node records and produce separate TTFT/TPOT-proxy reports.

No writes into the source run, no endpoints, no tokenization, no re-fitting.
Run directory must be explicitly supplied. summaries alone cannot recover metrics.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint18v2.latency_metrics import request_metrics,backend_progress_metrics

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path)
    a=ap.parse_args();src=a.run.resolve(strict=True)
    files=[src] if src.is_file() and src.name=='nodes.json' else sorted(src.rglob('nodes.json'))
    if not files:raise SystemExit('STOP: no full nodes.json; do not reconstruct request metrics from critical-path summary means')
    out=(a.out or ROOT/'reports'/('REQUEST_LATENCY_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])).resolve()
    source_dir=src.parent if src.is_file() else src
    if out==source_dir or source_dir in out.parents:raise SystemExit('STOP: output must not be inside the source run')
    out.mkdir(parents=True,exist_ok=False);results=[];hashes={}
    for i,f in enumerate(files):
        hashes[str(f)]=sha(f);nodes=json.loads(f.read_text())
        if not isinstance(nodes,dict):raise ValueError('nodes.json must map node IDs to records: '+str(f))
        z=request_metrics(nodes);dest=out/f'{i:03d}_{f.parent.name}';dest.mkdir()
        write(dest/'request_latency_summary.json',z['request_latency_metrics']);write(dest/'request_latency_rows.json',z['request_latency_rows'])
        rows=z['request_latency_rows'];columns=list(dict.fromkeys(k for r in rows for k in r))
        with (dest/'request_latency_rows.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        backend=f.parent/'backend.jsonl'
        if backend.is_file():
            hashes[str(backend)]=sha(backend)
            events=[]
            for line in backend.read_text().splitlines():
                if line.strip():
                    e=json.loads(line)
                    if e.get('event')=='output_progress':events.append(e)
            write(dest/'backend_output_progress_metrics.json',backend_progress_metrics(nodes,events))
        row={'source':str(f),**{k:v for k,v in z.items() if k not in ('request_latency_rows','request_latency_metrics')},
             'requests_total':len(nodes),'tpot_exclusions':z['request_latency_metrics']['tpot_exclusions']}
        print('REQUEST_LATENCY='+json.dumps(row,ensure_ascii=False),flush=True);results.append(row)
    unchanged=all(sha(Path(p))==h for p,h in hashes.items())
    write(out/'source_hashes.json',hashes);write(out/'summary.json',results)
    if not unchanged:raise SystemExit('STOP: source files changed during read; output cannot be treated as a fixed archive')
    print('SOURCE_FILES_UNCHANGED=True GPU_REQUESTS=0 NETWORK_REQUESTS=0 LIVE_RUNTIME_READS=0 LIVE_CHANGES=0')
    print('REQUEST_LATENCY_DIRECTORY='+str(out))
if __name__=='__main__':main()
