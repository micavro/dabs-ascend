import json,time,unittest
from pathlib import Path
from unittest.mock import patch
from wfsched.design import plan
from wfsched.v3_selector import choose_adaptive
from wfsched.fixed_work import convert_nodes,work_fingerprint,INPUT_TOKENS,OUTPUT_TOKENS
from wfsched.common import digest
from wfsched.ours_v2_selector import second_destination
from joint18v2.common import action_key
from test_ours_v2 import V2Decision

class V3Design(unittest.TestCase):
    def test_predeclared_workload_counts(self):
        for suite,count in [('v3-urgent',40),('v3-fixed',40),('v3-grid',512)]:
            p=plan(suite);self.assertEqual(len(p['conditions']),count)
            self.assertEqual(len({x['label'] for x in p['conditions']}),count)
            self.assertEqual(set(p['methods']),{'RR','P2C','Affinity','DAG','DAG-A'} if suite!='v3-grid' else {'RR','P2C','Affinity','DAG-A'})
            if suite=='v3-fixed':self.assertTrue(all(x['measurement_mode']=='fixed_token_replay' for x in p['conditions']))

class V3Controls(unittest.TestCase):
    def setUp(self):
        self.s,self.m,self.c=V2Decision().fixture(4)
        self.s.update(router_submitted_count=0,router_ongoing=[0]*4)
        self.c.update(wf_router_seed=950301,wf_routing_policy='DAG-A')
    def test_unavailable_extra_returns_exact_v1_action(self):
        self.s['replicas']=1
        self.s['pressure']=[1]
        self.s['router_ongoing']=[0]
        self.c['replicas']=1
        # Real v1 invoked with its actual choice, both on same frozen input.
        import semantic19.selector as semantic
        with patch('wfsched.v3_selector.semantic.choose',wraps=semantic.choose) as called:
            expected=semantic.choose(self.s,self.m,self.c,'D',time.monotonic()+1.0)
            got=choose_adaptive(self.s,self.m,self.c,'D',time.monotonic()+1.0)
        self.assertTrue(called.called)
        self.assertEqual(action_key(expected['action']),action_key(got['action']))
        self.assertTrue(got['v3_v1_fallback_kept'])
    def test_mock_improvement_can_choose_extra(self):
        import semantic19.selector as semantic
        from joint18v2.search import reference
        base=reference(self.s,self.c)
        incumbent={'action':base,'evaluations':[],'model_selection':True,
             'complete_distinct_evaluations':1,'decode_guard_rejections':0}
        def eval_mock(s,a,m,c,deadline):
            cost=900. if 'second' in a['family'] else 1000.
            return {'rankable':True,'sum_remaining_jct_ms':cost,'node_completion_ms':{},'reason':'mock'}
        with patch('wfsched.v3_selector.semantic.choose',return_value=incumbent),                 patch('wfsched.v3_selector.evaluate',side_effect=eval_mock),                 patch('wfsched.v3_selector.semantic.guard',return_value=[]):
            result=choose_adaptive(self.s,self.m,self.c,'D',time.monotonic()+1.)
        self.assertIn(result['selected_cell'],('20','30'))
        self.assertTrue(result['v3_selected_extra_candidate'])
    def test_mock_worse_extra_keeps_v1(self):
        from joint18v2.search import reference
        base=reference(self.s,self.c)
        incumbent={'action':base,'evaluations':[],'model_selection':True,
             'complete_distinct_evaluations':1,'decode_guard_rejections':0}
        with patch('wfsched.v3_selector.semantic.choose',return_value=incumbent),                 patch('wfsched.v3_selector.evaluate',return_value={'rankable':True,'sum_remaining_jct_ms':1200.,'node_completion_ms':{}}),                 patch('wfsched.v3_selector.semantic.guard',return_value=[]):
            result=choose_adaptive(self.s,self.m,self.c,'D',time.monotonic()+1.)
        self.assertEqual(action_key(result['action']),action_key(base))
        self.assertTrue(result['v3_v1_fallback_kept'])
    def test_fixed_work_is_exact_and_semantics_disabled(self):
        ns={'a':{'dataset_id':'d','step_position':0,'parents':[],
              'tokens':list(range(48)),'prompt':48,'output':9,'share':16,
              'prefix_hash':__import__('joint18v2.common',fromlist=['token_hash']).token_hash(list(range(16)))}}
        receipt=convert_nodes(ns,77)
        self.assertEqual(ns['a']['prompt'],INPUT_TOKENS)
        self.assertEqual(len(ns['a']['tokens']),INPUT_TOKENS)
        self.assertEqual(ns['a']['output'],OUTPUT_TOKENS)
        self.assertFalse(ns['a']['semantic_mode'])
        self.assertFalse(ns['a']['typed_answer_mode'])
        self.assertEqual(receipt,work_fingerprint(ns))
    def test_fixed_work_rejects_oversized_prompts(self):
        ns={'a':{'dataset_id':'d','step_position':0,'parents':[],
              'tokens':[1]*(INPUT_TOKENS+1),'share':16}}
        with self.assertRaises(ValueError):convert_nodes(ns,77)
