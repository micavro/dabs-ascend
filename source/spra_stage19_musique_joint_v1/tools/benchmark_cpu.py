#!/usr/bin/env python3
"""Synthetic service model and CPU only. Measures bounded search, not GPU gains."""
from pathlib import Path
import sys,time,json,copy,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint18v2.search import choose
from joint18v2.common import load,atomic
from joint18v2.workload import build

def main():
    cfg=load(ROOT/'configs/development.json');cases=[]
    for count in (4,16):
        c=dict(cfg,workflow_count=count);nodes=build(c,list(range(200,230)),123,[15]*8)
        for n in nodes.values():
            n.pop('tokens');n['arrival']=0.
            if n['name']=='root':n.update(state='done',finish=0.)
            elif n['name'].startswith('B'):n.update(state='ready',ready_at=0.)
        s={'nodes':nodes,'replicas':4,'ready':[k for k,n in nodes.items() if n['state']=='ready'],'pressure':[0]*4,'capacity_blocks':[4096]*4,'reserved_blocks':[0]*4,'queue_order':{},'now':.01,'prefix_history':{},'current_caps':[1024]*4,'decision_index':2}
        model={'models':{str(r):{'coef':[12,.04,4,.5,.5,.1]} for r in range(4)}}
        for method in ('A','D','B','Decoupled','Joint'):
            for rep in range(3):
                begin=time.monotonic();z=choose(s,model,c,method,begin+c['planning_soft_limit_s']);wall=1000*(time.monotonic()-begin)
                cases.append({'workflows':count,'ready':len(s['ready']),'method':method,'repeat':rep+1,'wall_ms':wall,'soft_deadline_ms':c['planning_soft_limit_s']*1000,
                              'complete_distinct':z.get('complete_distinct_evaluations',0),'complete_crossed':z.get('joint_cross_candidates_evaluated',0),'incomplete':z.get('incomplete_not_ranked',0),
                              'returned_family':z['action']['family'],'model_selection':z['model_selection'],'deadline_reached':z.get('deadline_reached',False)})
    out={'scope':'synthetic constant-input CPU benchmark, no target host or GPU; not a wall-time guarantee','python':platform.python_version(),'cases':cases}
    if '--write' in sys.argv:atomic(ROOT/'analysis/CPU_SEARCH_BENCHMARK.json',out)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
