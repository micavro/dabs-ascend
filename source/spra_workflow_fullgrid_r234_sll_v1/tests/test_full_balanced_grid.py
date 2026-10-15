from __future__ import annotations
import csv,json, tempfile,unittest
from collections import Counter
from pathlib import Path
from wfsched.design import plan
from wfsched.binding import bindings
from wfsched.evidence import classify_interval,capture_evidence,PHASE_KEYS
from wfsched.routing import reference as route_reference

class FullGridContract(unittest.TestCase):
    def test_grid_96_and_all_cells(self):
        p=plan('dabs-r234-balanced-64');cells=p['conditions']
        self.assertEqual(len(cells),96)
        self.assertEqual(set(p['methods']),{'RR','SLL','Ayo','Ours'})
        self.assertEqual(json.loads(json.dumps(p)),p)
        c=Counter((x['replicas'],x['rate'],x['repeat']) for x in cells)
        self.assertEqual(set(c),{(r,float(rate),t) for r in (2,3,4) for rate in (4,8,12,16) for t in (1,2)})
        self.assertTrue(all(v==4 for v in c.values()))
        self.assertTrue(all(x['workflow_count']==64 and x['measurement_mode']=='fixed_token_replay'
             and x['popularity_profile']=='balanced' and x['base_cap']==4096 for x in cells))
        self.assertEqual([x['method'] for x in cells[:4]],['RR','SLL','Ayo','Ours'])
        self.assertEqual([x['method'] for x in cells[-4:]],['Ours','Ayo','SLL','RR'])
        self.assertFalse('Preble' in p['methods'])
    def test_sll_is_original_p2c_routing(self):
        import joint18v2.client as live
        snap={'replicas':3,'ready':['w0/a','w1/a'], 'nodes':{'w0/a':{'ready_at':1.},'w1/a':{'ready_at':2.}},
              'now':2.5,'router_submitted_count':17,'router_ongoing':[3,0,6],
              'reserved_blocks':[0]*3,'capacity_blocks':[4096]*3,'pressure':[0]*3}
        cfg={'wf_router_seed':951000,'wf_routing_policy':'SLL',
             'base_cap':4096,'caps':[4096],'max_admission_wait_s':3.0,'admission_reserve_fraction':.85,'max_num_seqs':64,'block_size':16}
        for k in snap['nodes'].values():k.update({'prompt':3584,'output':16,'state':'ready','parents':[],'max_output_tokens':16})
        with bindings('SLL'):
            self.assertIs(live.reference,route_reference)
            a=live.reference(snap,cfg)
        self.assertEqual(set(a['placement']),set(snap['ready']))
        cfg['wf_routing_policy']='P2C'
        self.assertEqual(a['placement'],route_reference(snap,cfg)['placement'])
    def test_ours_dispatch_preserves_v5_identity(self):
        from wfsched.worker import process_plan
        source=__import__('inspect').getsource(process_plan)
        self.assertIn("('DAG-H','Ours')",source)
        self.assertIn('choose_heterogeneous',source)

class NativeKVAnalysis(unittest.TestCase):
    def test_partition_preserves_ambiguity(self):
        self.assertEqual(classify_interval(0,8,8,0),(8,0,0,0))
        self.assertEqual(classify_interval(0,8,0,8),(0,8,0,0))
        self.assertEqual(classify_interval(0,8,8,8),(0,0,8,0))
        self.assertEqual(classify_interval(0,8,0,0),(0,0,0,8))
        self.assertEqual(classify_interval(2,7,8,5),(3,0,3,1))
    def test_native_skips_and_exclusive_provenance_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            # Producer a creates tokens [1,2,3,4]. b is same-workflow,
            # then c has an eligible OTHER-workflow producer, then d has
            # two possible prior sources (the ambiguity must NOT disappear).
            ns={
                'a':{'wid':'w0','replica':0,'start':1.,'finish':2.,'tokens':[1,2,3,4,9], 'prompt':5,'output':1,'graph_shape':'x','hash':'ha'},
                'b':{'wid':'w0','replica':0,'start':3.,'finish':4.,'tokens':[1,2,3,4,8], 'prompt':5,'output':1,'graph_shape':'x','hash':'hb'},
                'c':{'wid':'w1','replica':0,'start':5.,'finish':6.,'tokens':[1,2,3,4,7], 'prompt':5,'output':1,'graph_shape':'x','hash':'hc'},
                'd':{'wid':'w0','replica':0,'start':7.,'finish':8.,'tokens':[1,2,3,4,6], 'prompt':5,'output':1,'graph_shape':'x','hash':'hd'},
            }
            events=[{'event':'accept','prompt_hash':n['hash'],'replica':0} for n in ns.values()]
            for h,t,skip in [('ha',1.1,0),('hb',3.1,4),('hc',5.1,4),('hd',7.1,4)]:
                events.append({'event':'model_input','prompt_hash':h,'replica':0,'mono_ns':int(t*1e9),
                    'context_before_cache':0,'query_tokens_before_cache':5,'skipped_query_tokens':skip})
            events.append({'event':'iteration','replica':0,'scheduled':[{'prompt_hash':'ha'}],'free_blocks_before':750})
            (folder/'backend.jsonl').write_text('\n'.join(json.dumps(x) for x in events)+'\n')
            (folder/'initial_cache.json').write_text(json.dumps({'status':[{'pod':'spra-vllm-0','num_gpu_blocks':1000}]}))
            summary={'native_skipped_query_tokens':12,'mean_jct_ms':10.,'mean_request_ttft_ms':1.,
                'p95_request_ttft_ms':2.,'decision_wall_ms':{'mean':1.,'p95':2.},
                'mean_critical_path_components_ms':{k:0. for k in PHASE_KEYS}}
            r=capture_evidence(folder,summary,ns,{'method':'RR','replicas':1})
            c=r['cache']
            self.assertEqual(c['native_skip_tokens'],12)
            self.assertEqual(c['intra_only_eligible_tokens'],4)
            self.assertEqual(c['inter_only_eligible_tokens'],4)
            self.assertEqual(c['both_eligible_tokens'],4)
            self.assertEqual(c['unattributed_tokens'],0)
            self.assertFalse(c['workflow_inter_actual_source_verified'])
            with (folder/'kv_nodes.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            self.assertTrue((folder/'kv_evidence_summary.json').exists())
            self.assertEqual(len(r['replica_occupancy']),1)
    def test_unattributed_never_falsely_allocated(self):
        self.assertEqual(classify_interval(3,2,1,1),(0,0,0,2))

if __name__=='__main__':unittest.main()

# Extra test class is deliberately defined after the standard unittest.main()
# guard: unittest discover imports the module and still discovers this class.
class DetailedReport(unittest.TestCase):
    def test_partial_suite_not_misreported_complete(self):
        from wfsched.extended_report import write_extended
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);folder=out/'attempts'/'one'/'RAW_001';folder.mkdir(parents=True)
            spec=plan('dabs-r234-balanced-64')['conditions'][0]
            evidence={'native_skip_tokens':3,'intra_only_eligible_tokens':1,
               'inter_only_eligible_tokens':1,'both_eligible_tokens':0,
               'unattributed_tokens':1,'intra_eligible_upper_bound_tokens':1,
               'inter_eligible_upper_bound_tokens':1,'native_cache_skipped_fraction':3/774144,
               'root_to_branch_cache_links_verified':0}
            (folder/'kv_evidence_summary.json').write_text(json.dumps(evidence))
            (folder/'latency_decomposition.json').write_text(json.dumps({'component_ms':{k:0 for k in PHASE_KEYS}}))
            (folder/'kv_replica_occupancy.json').write_text(json.dumps([]))
            (folder/'kv_nodes.csv').write_text('node,wid,replica,native_skip_tokens\na,w0,0,3\n')
            summ={'mean_jct_ms':1000.,'p95_jct_ms':1500.,
                  'mean_request_ttft_ms':100.,'p95_request_ttft_ms':120.,
                  'cohort_workflows_s':10.,'cohort_tokens_s':1000.,
                  'decision_wall_ms':{'mean':3.,'p95':5.},
                  'backend_effective_delay_ms':{'mean':10.}}
            (folder/'summary.json').write_text(json.dumps(summ))
            row={**spec,'folder':str(folder.relative_to(out)),'summary':summ}
            result=write_extended(out,[row],plan('dabs-r234-balanced-64')['conditions'],'paused')
            self.assertEqual(result['expected'],96)
            self.assertFalse(result['complete'])
            self.assertEqual(result['accepted'],1)
            self.assertTrue((out/'analysis/kv_by_node.csv').is_file())
            self.assertTrue((out/'analysis/metric_grid.csv').is_file())
            self.assertTrue((out/'analysis/paired_grid_summary.csv').is_file())
            # Report after pause must be regenerable without rewriting RAW.
            frozen=(folder/'summary.json').read_bytes()
            receipt2=write_extended(out,[row],plan('dabs-r234-balanced-64')['conditions'],'paused')
            self.assertEqual(receipt2['expected'],96)
            self.assertEqual((folder/'summary.json').read_bytes(),frozen)
            with (out/'analysis/metric_grid.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),1)
