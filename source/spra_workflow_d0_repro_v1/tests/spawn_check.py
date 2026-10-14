"""CPU-only real native-worker serialization check; no requests or fake benchmark metrics."""
import concurrent.futures
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import workflow_only as w

def main():
    root=Path(os.environ.get('WF_D0_SOURCE_ROOT',str(w.DEFAULT_ROOT)))
    legacy=w.load_original(root)
    art,model,prior,mf=legacy.prepared(SimpleNamespace(prepared=root/'spra_stage19_typed_handoff_v1/prepared'/w.PROFILE))
    from joint18v2.client import process_plan
    cfg=copy.deepcopy(mf['profile_configuration'])
    cfg.update(base_cap=4096,replicas=4,planning_soft_limit_s=1.,apply_delay_ms=0.,action_horizon_ms=200.)
    nodes={}
    for i in range(3):
        k=str(i)
        nodes[k]={'id':k,'wid':'w','name':k,'parents':[] if i<2 else ['0','1'],
            'prompt':256,'output':4,'computed':0,'output_done':0,'ready_at':0.,
            'state':'ready' if i<2 else 'blocked','replica':None,'share':0,'prefix_hash':None,
            'compiled_groups':[['0','1']], 'compiled_successors':['2'] if i<2 else [],'max_output_tokens':96}
    snapshot={'nodes':nodes,'ready':['0','1'],'replicas':4,'pressure':[4,0,0,0],
            'now':.1,'current_caps':[4096]*4,'capacity_blocks':[4096]*4,
            'reserved_blocks':[0]*4,'prefix_history':{},'cache_evidence':[],'decision_index':1}
    with concurrent.futures.ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as ex:
        pid=ex.submit(w.worker_ready).result(timeout=10)
        ans=ex.submit(process_plan,snapshot,model,cfg,'D',time.monotonic()+5).result(timeout=10)
    assert pid!=os.getpid()
    assert ans['action']['caps']==[4096]*4,ans
    assert {x['cell'] for x in ans['evaluations']} <= {'00','10'}
    print(json.dumps({'spawn_ok':True,'native_D_worker_called':True,'same_original_frozen_model':True,
           'fixed_caps':ans['action']['caps'],'evaluated_cells':[x['cell'] for x in ans['evaluations']],
           'gpu_requests':0,'performance_claimed':False},indent=2))
if __name__=='__main__':main()
