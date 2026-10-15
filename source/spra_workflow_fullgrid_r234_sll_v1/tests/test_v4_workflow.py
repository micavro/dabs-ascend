from __future__ import annotations
import time,unittest
from unittest.mock import patch
from joint18v2.common import action_key,validate_action
from joint18v2.search import reference
from wfsched.design import plan
from wfsched.v4_selector import proposals,choose_workflow_robust
from test_ours_v2 import V2Decision

class V4Plans(unittest.TestCase):
    def test_declared_contract_and_balanced_order(self):
        for suite in ('v4-screen','v4-stress-64','v4-hotspot-64'):
            p=plan(suite)
            self.assertEqual(len(p['conditions']),40)
            self.assertTrue(p['fixed_workload_replay'])
            self.assertTrue(p['current_reference_is_not_published_preble'])
            self.assertEqual(set(p['methods']),{'RR','P2C','Affinity','DAG','DAG-W'})
            self.assertEqual(len({x['label'] for x in p['conditions']}),40)
            for x in p['conditions']:
                self.assertEqual(x['workflow_count'],64 if suite!='v4-screen' else 16)
                self.assertEqual(x['measurement_mode'],'fixed_token_replay')
                self.assertEqual(x['packet_repetitions'],4 if suite=='v4-hotspot-64' else 1)
                self.assertEqual(x['packet_slice'],[0,64] if suite=='v4-stress-64' else
                                 [32,48] if suite=='v4-hotspot-64' else [(x['block']-1)*16,x['block']*16])
            cell=(x['replicas'],x['rate'],x['block'])
            self.assertEqual(len(set((x['replicas'],x['rate'],x['block'],x['repeat']) for x in p['conditions'])),8)
            first=[x['method'] for x in p['conditions'][:5]]
            last=[x['method'] for x in p['conditions'][-5:]]
            self.assertEqual(first,list(reversed(last)))

class V4Decision(unittest.TestCase):
    def setUp(self):
        self.s,self.m,self.c=V2Decision().fixture(4)
    def _native_evidence(self):
        for r in range(self.s['replicas']):
            self.s['cache_evidence'].append({'prefix_hash':'prefix','replica':r,
                    'known':True,'state':'active_computed' if r==0 else 'absent',
                    'tokens':128 if r==0 else 0})
    def test_reference_when_no_actual_opportunity(self):
        self.s['replicas']=1;self.s['pressure']=[0];self.c['replicas']=1
        base=reference(self.s,self.c)
        ans=choose_workflow_robust(self.s,self.m,self.c,'D',time.monotonic()+1)
        self.assertEqual(action_key(base),action_key(ans['action']))
        self.assertEqual(ans['v4_candidate_generated'],0)
        self.assertFalse(ans['v4_selected_deviation'])
    def test_routes_are_isolated_and_legal(self):
        self._native_evidence()
        base=reference(self.s,self.c)
        opts=proposals(self.s,self.m,self.c,base)
        self.assertTrue(opts)
        for label,a,origin in opts:
            self.assertEqual(a['caps'],[4096]*4)
            validate_action(a,self.s,self.c)
            if origin=='route_only':
                self.assertEqual(a['order'],[])
                self.assertNotIn('bounded_admission',a['family'])
                self.assertEqual(set(a['admit']),set(self.s['ready']))
                self.assertNotEqual(action_key(a),action_key(base))
            if origin=='order_only':self.assertEqual(a['placement'],base['placement'])
    def test_weak_model_gain_stays_reference(self):
        self._native_evidence();base=reference(self.s,self.c)
        def eval_mock(s,a,model,cfg,deadline):
            cost=1000 if a['family']=='reference' else 970
            return {'rankable':True,'sum_remaining_jct_ms':cost,'node_completion_ms':{},'reason':'mock'}
        with patch('wfsched.v4_selector.evaluate',side_effect=eval_mock),\
             patch('wfsched.v4_selector.dag.guard',return_value=[]):
            ans=choose_workflow_robust(self.s,self.m,self.c,'D',time.monotonic()+1)
        self.assertEqual(action_key(ans['action']),action_key(base))
        self.assertTrue(ans['v4_reference_retained'])
    def test_strong_model_gain_can_select(self):
        self._native_evidence()
        def eval_mock(s,a,model,cfg,deadline):
            cost=1000 if a['family']=='reference' else 600
            return {'rankable':True,'sum_remaining_jct_ms':cost,'node_completion_ms':{},'reason':'mock'}
        with patch('wfsched.v4_selector.evaluate',side_effect=eval_mock),\
             patch('wfsched.v4_selector.dag.guard',return_value=[]):
            ans=choose_workflow_robust(self.s,self.m,self.c,'D',time.monotonic()+1)
        self.assertTrue(ans['v4_selected_deviation'])
        self.assertGreaterEqual(ans['v4_candidate_scored'],1)
        self.assertNotEqual(ans['selected_cell'],'00')
    def test_guard_rejection_keeps_reference(self):
        self._native_evidence();base=reference(self.s,self.c)
        with patch('wfsched.v4_selector.evaluate',return_value={'rankable':True,'sum_remaining_jct_ms':100.,'node_completion_ms':{}}),\
             patch('wfsched.v4_selector.dag.guard',return_value=[{'kind':'mock_reject'}]):
            ans=choose_workflow_robust(self.s,self.m,self.c,'D',time.monotonic()+1)
        self.assertFalse(ans['v4_selected_deviation'])
        self.assertEqual(action_key(ans['action']),action_key(base))
    def test_expired_deadline_rejects_new_candidates(self):
        self._native_evidence();base=reference(self.s,self.c)
        ans=choose_workflow_robust(self.s,self.m,self.c,'D',time.monotonic()-1)
        self.assertEqual(action_key(ans['action']),action_key(base))
        self.assertEqual(ans['v4_candidate_scored'],0)
