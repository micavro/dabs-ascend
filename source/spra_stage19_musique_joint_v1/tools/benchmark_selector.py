#!/usr/bin/env python3
"""Synthetic CPU timing, never a GPU speedup or a hard runtime gate."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint18v2.common import load
from joint18v2.workload import build
from joint18v2.search import choose

def main():
    c=load(ROOT/'configs/development.json');ns=build(c,list(range(100,160)),1910,[103]*12)
    for k,n in ns.items():
        n.pop('tokens',None);n['arrival']=0.;n['ready_at']=0.
        if not n['parents']:n.update(state='done',replica=int(n['wid'][1:])%4,computed=n['prompt'],output_done=n['output'],phase='decode')
        elif n['name']!='sink':n.update(state='ready',phase='prefill')
    hs=sorted({n['prefix_hash'] for n in ns.values() if n['prefix_hash']})
    s={'nodes':ns,'ready':[k for k,n in ns.items() if n['state']=='ready'],'replicas':4,'now':0.,'pressure':[0]*4,'current_caps':[1024]*4,'capacity_blocks':[4096]*4,'reserved_blocks':[0]*4,'queue_order':{},'prefix_history':{},'cache_evidence':[{'prefix_hash':h,'replica':r,'tokens':1024 if i==r else 0,'state':'reclaimable_computed' if i==r else 'absent','known':True} for i,h in enumerate(hs) for r in range(4)]}
    m={'model_id':'synthetic_NOT_MEASURED','models':{str(r):{'coef':[15,2,6,.3,.1,.4]} for r in range(4)}}
    rows=[]
    for method in ('A','D','B','Decoupled','Joint'):
        for i in range(3):
            s['decision_index']=i;t=time.monotonic();z=choose(s,m,c,method,t+.100)
            rows.append({'method':method,'repetition':i,'nodes':len(ns),'ready':len(s['ready']),
                'call_wall_ms':1000*(time.monotonic()-t),'complete_evaluations':z['complete_distinct_evaluations'],
                'partial_group_complete':z['partial_group_complete_evaluations'],'cross_complete':z['joint_cross_candidates_evaluated'],
                'reference_unrankable':z.get('reference_unrankable',False),'deadline_reached':z['deadline_reached']})
    out={'scope':'synthetic constant known work; local CPU; no external sockets or GPU requests',
         'python':sys.version,'model':m,'rows':rows,'runtime_gate':False,'online_GPU39_performance_proven':False}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
