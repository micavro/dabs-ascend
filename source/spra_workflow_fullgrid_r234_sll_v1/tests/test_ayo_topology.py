from __future__ import annotations
import unittest
from unittest.mock import patch

from wfsched.ayo_topology import reverse_topological_depth,topology_priority,reference
from wfsched.design import plan
from wfsched.binding import bindings

class AyoAlgorithm2Tests(unittest.TestCase):
    @staticmethod
    def snapshot():
        ns={}
        def n(key,wid,state,parents,ready,replica=None):
            return {'wid':wid,'state':state,'parents':parents,'ready_at':ready,
                 'replica':replica,'prompt':3584,'output':16,
                 'max_output_tokens':16,'phase':'prefill'}
        ns['w0/a']=n('w0/a','w0','ready',[],1.0)
        ns['w0/b']=n('w0/b','w0','ready',[],1.1)
        ns['w0/c']=n('w0/c','w0','blocked',['w0/a'],0)
        ns['w0/d']=n('w0/d','w0','blocked',['w0/c','w0/b'],0)
        ns['w1/a']=n('w1/a','w1','ready',[],0.9)
        ns['w1/b']=n('w1/b','w1','blocked',['w1/a'],0)
        return {'nodes':ns,'replicas':2,'ready':['w0/b','w1/a','w0/a'],
                'reserved_blocks':[600,0],'capacity_blocks':[4096,4096],
                'pressure':[4,0], 'now':2.0}
    def config(self):return {'base_cap':4096,'max_num_seqs':64,
                  'admission_reserve_fraction':.85,'block_size':16,
                  'caps':[4096],'max_admission_wait_s':3.0}
    def test_depth_longest_to_output(self):
        d=reverse_topological_depth(self.snapshot()['nodes'])
        self.assertEqual((d['w0/a'],d['w0/b'],d['w0/c'],d['w0/d']), (2,1,1,0))
    def test_bucket_arrival_before_depth_between_queries(self):
        order=topology_priority(self.snapshot(),self.snapshot()['ready'])
        self.assertEqual(order, ['w1/a','w0/a','w0/b'])
    def test_native_action_and_occupancy(self):
        a=reference(self.snapshot(),self.config())
        self.assertEqual(a['family'],'Ayo_Alg2_TopologyProxy')
        self.assertEqual(a['caps'],[4096,4096])
        self.assertEqual(set(a['placement']),set(self.snapshot()['ready']))
        self.assertEqual(set(a['admit']),set(self.snapshot()['ready']))
        self.assertEqual(a['placement']['w1/a'],1)
        self.assertTrue(a['ayo_algorithm_2_proxy'])
        self.assertTrue(a['order'])
    def test_safe_capacity_guard(self):
        s=self.snapshot();s['reserved_blocks']=[3410,3410]
        a=reference(s,self.config())
        self.assertFalse(a['admit'])
        self.assertEqual(set(a['resource_held']),set(s['ready']))
    def test_keeps_kv_identity_without_claiming_official_ayo(self):
        a=reference(self.snapshot(),self.config())
        self.assertFalse(a['ayo_exact_engine_batching'])
        self.assertEqual(a['ayo_load_metric'],'normalized_conservative_reserved_KV_blocks')
    def test_plan_and_json_identity(self):
        import json
        for s in ('v5-ayo-balanced-64','v5-ayo-skew-64','v5-ayo-grid-balanced-64','v5-ayo-grid-skew-64'):
            p=plan(s)
            self.assertEqual(json.loads(json.dumps(p)),p)
            self.assertIn('Ayo',p['methods'])
            self.assertEqual(len(p['conditions']),192 if '-grid-' in s else 48)
    def test_baseline_binding_uses_and_restores_ayo_reference(self):
        import joint18v2.client as live
        previous=live.reference
        with bindings('Ayo'):
            self.assertIs(live.reference,reference)
            a=live.reference(self.snapshot(),self.config())
            self.assertEqual(a['family'],'Ayo_Alg2_TopologyProxy')
        self.assertIs(live.reference,previous)
    def test_cycle_rejected(self):
        s=self.snapshot();s['nodes']['w0/a']['parents']=['w0/d']
        with self.assertRaisesRegex(ValueError,'cycle'):
            reverse_topological_depth(s['nodes'])
if __name__=='__main__':unittest.main()
