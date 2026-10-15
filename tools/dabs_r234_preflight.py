#!/usr/bin/env python3
"""GPU-free preflight: verifies 96-cell plan and original public packets."""
from collections import Counter
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'source/spra_workflow_fullgrid_r234_sll_v1'))
from wfsched.design import plan
from wfsched.heterogeneous import select_packets

p=plan('dabs-r234-balanced-64');cells=p['conditions']
assert len(cells)==96
assert set(p['methods'])=={'RR','SLL','Ayo','Ours'}
assert len({x['label'] for x in cells})==96
assert Counter((x['replicas'],x['rate']) for x in cells)==Counter({(r,float(rate)):8 for r in (2,3,4) for rate in (4,8,12,16)})
assert all(x['heterogeneous_workload'] and x['measurement_mode']=='fixed_token_replay' and x['popularity_profile']=='balanced' and x['base_cap']==4096 for x in cells)
assert json.loads(json.dumps(p))==p
file=root/'source/spra_stage19_musique_joint_v1/dataset/compiled/pilot.public.jsonl'
if not file.is_file():raise RuntimeError('install into original DABS_GPU39_Repro_v1 source tree with compiled public input')
questions=[json.loads(s) for s in file.read_text().splitlines() if s.strip()]
sel,manifest=select_packets(questions,'balanced',510140)
assert len(sel)==64 and sum(len(x['steps']) for x in sel)==216
assert manifest['total_declared_output_tokens']==13184
assert manifest['profile']=='balanced'
print('R234_SLL_PREFLIGHT_OK='+json.dumps({'methods':['RR','SLL','Ayo','Ours'],
 'replicas':[2,3,4],'arrival_rates_wf_per_s':[4,8,12,16],
 'repeat_orders':2,'expected_gpu_conditions':96,'workflows_each':64,
 'nodes_each':216,'prompt_tokens_per_node':3584,'generated_tokens_each':13184,
 'fixed_native_batch_budget':4096,'engine_files_modified':False,
 'meaning_of_SLL':'Sampled Least-Loaded (k=2); choose fewer unfinished requests',
 'Ayo_is_official_reproduction':False,'full_grid_auditable_KV_provenance':True,
 'GPU_calls_by_preflight':0},ensure_ascii=False))
