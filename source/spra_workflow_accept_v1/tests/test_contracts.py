from __future__ import annotations
import copy
import json
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from bootstrap import ROOT,CFG,MODEL
from wfsched.common import digest,atomic,load,condition_files,verify_files,own_identity
from wfsched.design import plan,make_design,config_for,orders
from wfsched.routing import pair_for,reference
from wfsched.bench import snapshot
from wfsched.worker import process_plan
from wfsched.report import summarize
from wfsched.legacy import inspect

class DesignTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual([len(make_design(s)) for s in ('smoke','ablation','grid','confirm')],[5,72,640,3200])
    def test_grid_all_cells(self):
        self.assertEqual({(r['replicas'],r['rate']) for r in make_design('grid')},
                         {(r,float(l)) for r in (1,2,3,4) for l in (1,2,4,8)})
    def test_grid_no_duplicates(self):
        d=make_design('grid');self.assertEqual(len({x['label'] for x in d}),len(d))
    def test_exact_four_blocks(self):
        self.assertEqual({tuple(x['packet_slice']) for x in make_design('grid')},{(0,16),(16,32),(32,48),(48,64)})
    def test_mirror_per_cellblock(self):
        d=make_design('grid')
        for R in (1,2,3,4):
            for l in (1,2,4,8):
                for b in (1,2,3,4):
                    x=[[v['method'] for v in d if (v['replicas'],v['rate'],v['block'],v['repeat'])==(R,l,b,t)] for t in (1,2)]
                    self.assertEqual(x[0],list(reversed(x[1])))
    def test_confirmation_position_balance(self):
        rows=orders(list('abcde'),10)
        for i in range(5):self.assertEqual(set(Counter(x[i] for x in rows).values()),{2})
    def test_confirmation_carryover_balance(self):
        c=Counter((a,b) for row in orders(list('abcde'),10) for a,b in zip(row,row[1:]))
        self.assertEqual(len(c),20);self.assertEqual(set(c.values()),{2})
    def test_ablation_all_permutations(self):
        d=make_design('ablation')
        for b in (1,2,3,4):
            x=[tuple(v['method'] for v in d if v['block']==b and v['repeat']==t) for t in range(1,7)]
            self.assertEqual(len(set(x)),6)
    def test_cfg_deep_copy(self):
        c=make_design('grid')[0];cfg=config_for(CFG,c);cfg['caps'].append(999)
        self.assertNotIn(999,CFG['caps']);self.assertEqual(cfg['base_cap'],4096)
    def test_original_clock_guard(self):
        c=make_design('grid')[0];base=copy.deepcopy(CFG);base['epoch_s']=.01
        with self.assertRaises(ValueError):config_for(base,c)
    def test_unknown_suite(self):
        with self.assertRaises(ValueError):plan('unknown')
    def test_plan_no_oracle(self):
        self.assertTrue(plan('grid')['no_posthoc_version_selection'])
        self.assertFalse(plan('grid')['guaranteed_10_percent'])

class RoutingTests(unittest.TestCase):
    def make(self,R=4,count=0):
        s=snapshot(R,2);s['router_submitted_count']=count;s['router_ongoing']=[0]*R
        c=copy.deepcopy(CFG);c.update(replicas=R,wf_router_seed=99,wf_routing_policy='RR')
        return s,c
    def test_round_robin(self):
        s,c=self.make();a=reference(s,c)
        self.assertEqual([a['placement'][k] for k in a['admit']],[0,1,2,3])
    def test_round_robin_resume_count(self):
        s,c=self.make(3,5);a=reference(s,c)
        self.assertEqual([a['placement'][k] for k in a['admit']],[2,0,1,2])
    def test_reference_has_no_cursor_side_effect(self):
        s,c=self.make();self.assertEqual(reference(s,c),reference(s,c));self.assertEqual(s['router_submitted_count'],0)
    def test_no_native_reorder(self):
        s,c=self.make();self.assertEqual(reference(s,c)['order'],[])
    def test_p2c_selects_lower_load(self):
        s,c=self.make();s['ready']=s['ready'][:1];c['wf_routing_policy']='P2C';a,b=pair_for(99,0,4)
        s['router_ongoing'][a]=8
        z=reference(s,c);self.assertEqual(z['placement'][s['ready'][0]],b)
    def test_p2c_idempotence(self):
        s,c=self.make();c['wf_routing_policy']='P2C';self.assertEqual(reference(s,c),reference(s,c))
    def test_p2c_counts_new_placements(self):
        s,c=self.make(2);c['wf_routing_policy']='P2C';a=reference(s,c)
        self.assertEqual(sorted(Counter(a['placement'].values()).values()),[2,2])
    def test_one_replica_same_routes(self):
        s,c=self.make(1);a=reference(s,c);c['wf_routing_policy']='P2C';b=reference(s,c)
        self.assertEqual(a['placement'],b['placement'])
    def test_no_prefix_or_dag_dependency(self):
        s,c=self.make();a=reference(s,c);s['pressure']=[99]*4;s['prefix_history']={'x':[3]}
        for n in s['nodes'].values():n['compiled_successors']=[];n['compiled_groups']=[]
        self.assertEqual(a['placement'],reference(s,c)['placement'])
    def test_no_future_length_ranking(self):
        s,c=self.make();a=reference(s,c);s['nodes'][s['ready'][0]]['output']=999
        # Reservation uses max_output_tokens identically; no output estimate used in route.
        self.assertEqual(a['placement'],reference(s,c)['placement'])
    def test_resource_held_fifo_prefix(self):
        s,c=self.make();s['capacity_blocks'][1]=0;a=reference(s,c)
        self.assertEqual(a['admit'],s['ready'][:1])
        self.assertEqual(len(a['resource_held']),3)
    def test_negative_count(self):
        s,c=self.make();s['router_submitted_count']=-1
        with self.assertRaises(ValueError):reference(s,c)
    def test_unknown_baseline(self):
        s,c=self.make();c['wf_routing_policy']='RayServe'
        with self.assertRaises(ValueError):reference(s,c)
    def test_distinct_pair(self):
        for r in (2,3,4):
            for i in range(30):self.assertEqual(len(set(pair_for(55,i,r))),2)

class SelectorTests(unittest.TestCase):
    def test_budget_function_not_called(self):
        from wfsched.selector import choose_lite
        s=snapshot(4);c=copy.deepcopy(CFG)
        with patch('semantic19.selector.budget_alternative',side_effect=AssertionError('budget call forbidden')):
            ans=choose_lite(s,MODEL,c,'D',time.monotonic()+5)
        self.assertEqual({x['cell'] for x in ans['evaluations']},{'00','10'})
        self.assertTrue(ans['budget_candidate_generation_skipped'])
    def test_reject_adaptive_method(self):
        from wfsched.selector import choose_lite
        with self.assertRaises(ValueError):choose_lite(snapshot(2),MODEL,CFG,'Joint')
    def test_reject_other_cap(self):
        from wfsched.selector import choose_lite
        c=dict(CFG,base_cap=1024)
        with self.assertRaises(ValueError):choose_lite(snapshot(2),MODEL,c,'D')
    def test_expired_retains_reference(self):
        from wfsched.selector import choose_lite
        from joint18v2.search import reference
        s=snapshot(4);ans=choose_lite(s,MODEL,CFG,'D',time.monotonic()-1)
        self.assertEqual(ans['action'],reference(s,CFG));self.assertTrue(ans['reference_unrankable'])
    def test_incomplete_never_ranked(self):
        from wfsched.selector import choose_lite
        with patch('wfsched.selector.evaluate',return_value={'rankable':False,'reason':'TEST'}):
            ans=choose_lite(snapshot(4),MODEL,CFG,'D',time.monotonic()+1)
        self.assertFalse(ans['model_selection']);self.assertIsNone(ans['evaluations'][0]['cost_ms'])
    def test_worker_original_calls_original(self):
        c=dict(CFG,wf_public_method='DAG')
        from semantic19.selector import choose
        with patch('semantic19.selector.choose',wraps=choose) as f:
            process_plan(snapshot(4),MODEL,c,'D',time.monotonic()+1)
        self.assertEqual(f.call_count,1)
    def test_worker_lite_path(self):
        c=dict(CFG,wf_public_method='DAG-Lite')
        ans=process_plan(snapshot(4),MODEL,c,'D',time.monotonic()+1)
        self.assertEqual(ans['wf_worker_path'],'no_budget_candidates')
    def test_baseline_rejected_by_worker(self):
        with self.assertRaises(ValueError):process_plan(snapshot(4),MODEL,dict(CFG,wf_public_method='RR'),'A',time.monotonic()+1)
    def test_binding_restored_on_exception(self):
        import joint18v2.client as live
        from wfsched.binding import bindings
        old=(live.reference,live.make_snapshot,live.stream,live.process_plan,live.compare_iteration)
        with self.assertRaises(RuntimeError):
            with bindings('P2C'):raise RuntimeError('test')
        self.assertEqual(old,(live.reference,live.make_snapshot,live.stream,live.process_plan,live.compare_iteration))
    def test_inspect48_without_data_claim(self):
        x=inspect(ROOT,False);self.assertEqual(x['source_files_verified'],48);self.assertEqual(x['gpu_generation_requests'],0)
    def test_file_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic(p/'x.json',{'a':1});m=condition_files(p);atomic(p/'x.json',{'a':2})
            with self.assertRaises(ValueError):verify_files(p,m)
    def test_extra_file_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic(p/'x.json',{});m=condition_files(p);atomic(p/'y.json',{})
            with self.assertRaises(ValueError):verify_files(p,m)

for R in (1,2,3,4):
    def test(self,R=R):
        from semantic19.selector import choose
        from wfsched.selector import choose_lite
        from joint18v2.common import action_key
        c=dict(CFG,replicas=R);s=snapshot(R,2)
        a=choose(copy.deepcopy(s),MODEL,c,'D',time.monotonic()+5)
        b=choose_lite(copy.deepcopy(s),MODEL,c,'D',time.monotonic()+5)
        self.assertEqual(action_key(a['action']),action_key(b['action']))
        self.assertEqual([e['cost_ms'] for e in a['evaluations']],[e['cost_ms'] for e in b['evaluations']])
    setattr(SelectorTests,'test_complete_objective_equivalence_R'+str(R),test)

class ReportTests(unittest.TestCase):
    def rows(self):
        d=make_design('smoke');rows=[]
        for c in d:
            s={'mean_jct_ms':100. if c['method'] in ('RR','P2C','Affinity') else 110.,
               'workflows':16,'requests':48,'output_tokens':400,
               'semantic_quality':{'final_em':.2,'final_f1':.3}}
            rows.append({**c,'summary':s,'external_work_sha256':'same'})
        return rows,d
    def test_negative_results_retained(self):
        rows,d=self.rows();r=summarize(rows,d)
        self.assertTrue(r['complete']);self.assertEqual(len(r['aggregates']),7)
        self.assertLess(r['aggregates'][0]['mean_jct_ms_gain_pct'],0)
    def test_missing_not_complete(self):
        rows,d=self.rows();self.assertFalse(summarize(rows[:-1],d)['complete'])
    def test_duplicate(self):
        rows,d=self.rows()
        with self.assertRaises(ValueError):summarize(rows+[rows[0]],d)
    def test_external_mismatch(self):
        rows,d=self.rows();rows[-1]['external_work_sha256']='wrong'
        with self.assertRaises(ValueError):summarize(rows,d)
    def test_method_alias_wrong(self):
        rows,d=self.rows();rows[0]['method']='vLLM'
        with self.assertRaises(ValueError):summarize(rows,d)
