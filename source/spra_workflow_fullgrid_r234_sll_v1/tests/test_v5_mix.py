"""Independent V5 high-risk regression tests (no GPU needed)."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import time
import unittest
from unittest.mock import patch
from joint18v2.common import token_hash, action_key, validate_action
from joint18v2.search import reference
from test_ours_v2 import V2Decision
from wfsched.design import plan
from wfsched.heterogeneous import FAMILIES, select_packets, output_budget
from wfsched.fixed_work import convert_nodes, INPUT_TOKENS
from wfsched.v5_selector import project,proposals,choose_heterogeneous,cache_tokens


def dummy_public():
    rows=[]
    for shape, spec in FAMILIES.items():
        parents=[[]]+[[k-1] for k in range(1,len(shape.split(';')))]
        for i in range(10):
            rows.append({'id':f'{shape}/packet{i:02d}',
                         'steps':[{'position':j,'parents':ps} for j,ps in enumerate(parents)],
                         'graph':{'shape':shape,'hop_count':len(parents)}})
    return rows+[{'id':'extra0','steps':[{'position':0,'parents':[]}],
                  'graph':{'shape':'-;0','hop_count':1}} for _ in []] + [
                       {'id':f'extra_{i}','steps':[{'position':0,'parents':[]},{'position':1,'parents':[0]}],
                        'graph':{'shape':'-;0','hop_count':2}} for i in range(4)]


class MixedPlan(unittest.TestCase):
    def test_suite_cardinality_and_identity(self):
        for suite,profile in (('v5-mix-balanced-64','balanced'),('v5-mix-skew-64','skew')):
            rows=plan(suite)['conditions']
            self.assertEqual(len(rows),40)
            self.assertEqual({x['method'] for x in rows},{'RR','P2C','Preble','DAG','DAG-H'})
            self.assertEqual(len(set(x['label'] for x in rows)),40)
            self.assertEqual(rows[0]['label'],'R4_L8_T1_B0_RR')
            self.assertEqual(rows[4]['method'],'DAG-H')
            for x in rows:
                self.assertEqual(x['popularity_profile'],profile)
                self.assertTrue(x['heterogeneous_workload'])
                self.assertEqual(x['workflow_count'],64)
                self.assertEqual(x['measurement_mode'],'fixed_token_replay')
            self.assertEqual([x['method'] for x in rows[:5]],list(reversed([x['method'] for x in rows[-5:]])))

    def test_popularity_varies_only_within_shape(self):
        packets=dummy_public()
        self.assertEqual(len(packets),64)
        a,am=select_packets(packets,'balanced')
        b,bm=select_packets(packets,'skew')
        self.assertEqual(len(a),len(b))
        self.assertEqual([x['graph']['shape'] for x in a],[x['graph']['shape'] for x in b])
        self.assertEqual(am['shape_histogram'],bm['shape_histogram'])
        self.assertEqual(am['total_declared_output_tokens'],bm['total_declared_output_tokens'])
        self.assertEqual(am['total_declared_output_tokens'],13184)
        self.assertGreater(max(bm['source_packet_counts'].values()),max(am['source_packet_counts'].values()))
        self.assertNotEqual(am['manifest_digest'],bm['manifest_digest'])
        self.assertEqual(select_packets(packets,'skew')[1],bm)
        self.assertEqual(sum(am['shape_histogram'].values()),64)

    def test_real_source_no_graph_rewrite(self):
        import os
        source=os.environ.get('WF_SOURCE_ROOT')
        if not source:self.skipTest('real source not installed')
        p=Path(source)/'spra_stage19_musique_joint_v1/dataset/compiled/pilot.public.jsonl'
        if not p.exists():self.skipTest('public pilot missing')
        real=[json.loads(t) for t in p.read_text().splitlines() if t.strip()]
        a,_=select_packets(real,'skew')
        self.assertTrue(all(x['steps'] and x['graph']['hop_count']==len(x['steps']) for x in a))
        self.assertEqual(len(set(x['id'] for x in a)),24)


class FixedHeterogeneous(unittest.TestCase):
    def test_per_shape_output_and_native_unique_hashes(self):
        ns={};source=[11]*32
        for shape in FAMILIES:
            for w in range(2):
                key=f'{shape}/instance{w}'
                ns[key]={'id':key,'wid':f'w{shape}_{w}', 'dataset_id':shape,
                         'step_position':0,'parents':[], 'graph_shape':shape,
                         'tokens':source+[27]*(2800-32), 'share':32,
                         'prefix_hash':token_hash(source)}
        x=convert_nodes(ns,93,heterogeneous=True)
        self.assertIsInstance(x,str)
        self.assertEqual(len({n['hash'] for n in ns.values()}),len(ns))
        for n in ns.values():
            self.assertEqual(len(n['tokens']),INPUT_TOKENS)
            self.assertEqual(n['output'],output_budget(n['graph_shape']))
            self.assertEqual(n['max_output_tokens'],n['output'])
            self.assertLessEqual(n['prompt']+n['output'],4096)
            self.assertEqual(token_hash(n['tokens'][:n['share']]),n['prefix_hash'])


class CandidateSafety(unittest.TestCase):
    def setUp(self):
        self.s,self.m,self.c=V2Decision().fixture(4)
        self.s['pressure']=[8.,2.,1.,1.]
        # Make prefix evidence explicit for all eligible routes.
        self.s['cache_evidence']=[{'prefix_hash':'prefix','replica':i,
                         'state':'active_computed' if i==0 else 'absent',
                         'tokens':256 if i==0 else 0,'known':True} for i in range(4)]
    def test_order_route_no_admission_coupling(self):
        b=reference(self.s,self.c)
        opts=proposals(self.s,self.m,self.c,b)
        self.assertTrue(opts)
        for label,a,origin,affected in opts:
            validate_action(a,self.s,self.c)
            self.assertEqual(a['caps'],b['caps'])
            self.assertEqual(set(a['admit']),set(b['admit']))
            self.assertEqual(set(a.get('resource_held',[])),set(b.get('resource_held',[])))
            if origin=='route_only':
                self.assertEqual(a['order'],b['order'])
                self.assertNotEqual(a['placement'],b['placement'])
            if origin=='order_only':self.assertEqual(a['placement'],b['placement'])
            self.assertTrue(affected)
    def test_unknown_target_distinguished_from_known_absent(self):
        # Source exists; destination missing is NOT an observed zero certificate.
        self.s['cache_evidence']=[e for e in self.s['cache_evidence'] if e['replica']==0]
        self.assertIsNone(cache_tokens(self.s,next(iter(self.s['ready'])),1))
        # The optimizer may PROPOSE a pessimistically-charged migration; it must
        # never report a measured cache-hit or assert unknown==known absent.
        b=reference(self.s,self.c)
        opts=proposals(self.s,self.m,self.c,b)
        for label,a,origin,wids in opts:
            if origin=='route_only':
                self.assertTrue(a['proposal']['destination_cache_unknown'])
                self.assertGreaterEqual(a['proposal']['conservative_cache_loss_tokens'],0)
    def test_weak_gain_uses_local_threshold_not_entire_system(self):
        b=reference(self.s,self.c)
        def simulate(s,a,model,cfg,deadline):
            ref=action_key(a)==action_key(b)
            return {'rankable':True,'sum_remaining_jct_ms':100000 if ref else 99750,
                    'node_completion_ms':{},'workflow_prediction_ms':{n['wid']:2000 for n in self.s['nodes'].values()},
                    'reason':'mock','cpu_wall_ms':.01}
        with patch('wfsched.v5_selector.evaluate',side_effect=simulate),\
             patch('wfsched.v5_selector.dag.guard',return_value=[]):
            result=choose_heterogeneous(self.s,self.m,self.c,'D',time.monotonic()+1)
        self.assertTrue(result['v5_selected_deviation'])
        self.assertIn(result['selected_cell'],('10','20','21'))
    def test_decode_guard_blocks_beneficial_surrogate(self):
        b=reference(self.s,self.c)
        with patch('wfsched.v5_selector.evaluate',return_value={'rankable':True,
             'sum_remaining_jct_ms':1000,'node_completion_ms':{},'workflow_prediction_ms':{}}),\
             patch('wfsched.v5_selector.dag.guard',return_value=[{'kind':'mock'}]):
            result=choose_heterogeneous(self.s,self.m,self.c,'D',time.monotonic()+1)
        self.assertFalse(result['v5_selected_deviation'])
        self.assertEqual(action_key(result['action']),action_key(b))
    def test_preble_proxy_is_exact_existing_native_reference(self):
        import joint18v2.client as client
        from wfsched.binding import bindings
        previous=client.reference
        with bindings('Preble'):
            self.assertIs(client.reference,previous)
            self.assertEqual(action_key(client.reference(self.s,self.c)),
                             action_key(reference(self.s,self.c)))
        self.assertIs(client.reference,previous)

    def test_no_opportunity_reference_identity(self):
        self.s['replicas']=1;self.c['replicas']=1
        self.s['pressure']=[1.]
        b=reference(self.s,self.c)
        result=choose_heterogeneous(self.s,self.m,self.c,'D',time.monotonic()+1)
        self.assertEqual(action_key(result['action']),action_key(b))
        self.assertEqual(result['v5_candidate_generated'],0)

if __name__=='__main__':unittest.main()
