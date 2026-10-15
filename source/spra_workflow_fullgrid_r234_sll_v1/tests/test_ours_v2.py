from __future__ import annotations
import copy
import json
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from wfsched.design import plan
from wfsched.ours_v2_selector import choose_v2,second_destination
from joint18v2.search import reference
from joint18v2.common import validate_action
from semantic19.selector import schedule_alternative


class V2Plans(unittest.TestCase):
    def test_declared_suite_counts(self):
        for suite,expected in [('v2-screen',50),('v2-grid',512),('v2-b3',32)]:
            d=plan(suite)
            self.assertEqual(len(d['conditions']),expected)
            self.assertEqual(len({r['label'] for r in d['conditions']}),expected)
            for x in d['conditions']:
                self.assertIn(x['method'],d['methods'])
                self.assertEqual(x['base_cap'],4096)
                self.assertEqual(x['workflow_count'],16)
                self.assertEqual(x['packet_slice'],[(x['block']-1)*16,x['block']*16])
    def test_scanning_covers_full_matrix(self):
        d=plan('v2-grid')
        cells={(x['replicas'],x['rate'],x['block'],x['repeat']) for x in d['conditions']}
        self.assertEqual(len(cells),4*4*4*2)
        for c in cells:
            self.assertEqual({x['method'] for x in d['conditions'] if (x['replicas'],x['rate'],x['block'],x['repeat'])==c},
                             {'RR','P2C','Affinity','DAG-V2'})
    def test_screen_has_v1_reference_and_v2(self):
        methods=set(plan('v2-screen')['methods'])
        self.assertEqual(methods,{'RR','P2C','Affinity','DAG','DAG-V2'})


class V2Decision(unittest.TestCase):
    def fixture(self,R=4):
        ns={}
        for i in range(3):
            k=str(i)
            ns[k]={'id':k,'wid':'w','name':k,'parents':[] if i<2 else ['0','1'],
                   'prompt':256,'output':8,'computed':0,'output_done':0,
                   'ready_at':0.,'state':'ready' if i<2 else 'blocked',
                   'replica':None,'share':128,'prefix_hash':'prefix',
                   'compiled_groups':[['0','1']],'compiled_successors':['2'] if i<2 else [],
                   'max_output_tokens':96}
        s={'nodes':ns,'ready':['0','1'],'replicas':R,'pressure':[3]+[0]*(R-1),
           'now':.1,'current_caps':[4096]*R,'capacity_blocks':[4096]*R,
           'reserved_blocks':[0]*R,'prefix_history':{},'cache_evidence':[],
           'decision_index':1}
        model={'models':{str(r):{'coef':[1.,0.,0.,0.,0.,0.]} for r in range(R)}}
        c=json.loads((Path(__file__).resolve().parents[3]/'profiles/reference/configuration.json').read_text())
        c.update(replicas=R,base_cap=4096,planning_soft_limit_s=1.,
                 apply_delay_ms=0.,action_horizon_ms=200.)
        return s,model,c
    def test_distinct_second_legal(self):
        s,m,c=self.fixture(4)
        base=reference(s,c)
        alt=schedule_alternative(s,m,c,base)
        other=second_destination(s,m,c,base,alt)
        self.assertIsNotNone(other)
        self.assertNotEqual(other['proposal']['destination'],alt['proposal']['destination'])
        self.assertTrue(other['family'].startswith('known_dag_partial_group_second'))
        self.assertEqual(other['caps'],base['caps'])
        validate_action(other,s,c)
    def test_no_extra_destination_r1_r2(self):
        for R in (1,2):
            s,m,c=self.fixture(R);base=reference(s,c)
            alt=schedule_alternative(s,m,c,base)
            self.assertIsNone(second_destination(s,m,c,base,alt))
    def test_full_scores_with_safe_select(self):
        s,m,c=self.fixture(4)
        calls=[]
        def score(snapshot, action, model, cfg, deadline):
            family=action['family']; calls.append(family)
            cost=500. if 'second' in family else 800. if family!='reference' else 1000.
            return {'rankable':True,'sum_remaining_jct_ms':cost,'node_completion_ms':{},
                    'cpu_wall_ms':.1,'reason':'unit_fixture'}
        with patch('wfsched.ours_v2_selector.evaluate',side_effect=score),\
             patch('wfsched.ours_v2_selector.semantic.guard',return_value=[]):
            ans=choose_v2(s,m,c,'D',time.monotonic()+1.)
        self.assertEqual(ans['selected_cell'],'20')
        self.assertTrue(ans['v2_selected_extra_destination'])
        self.assertEqual(len(ans['evaluations']),3)
        self.assertEqual(len(calls),3)
    def test_decode_guard_rejection_keeps_reference_or_old(self):
        s,m,c=self.fixture(4)
        def score(snapshot, action, model, cfg, deadline):
            cost=200 if 'second' in action['family'] else 600 if action['family']!='reference' else 1000
            return {'rankable':True,'sum_remaining_jct_ms':cost,'node_completion_ms':{},'reason':'mock'}
        def guard(s,z0,z,a0,a1,model,cfg):
            return [{'kind':'test_guard'}] if 'second' in a1['family'] else []
        with patch('wfsched.ours_v2_selector.evaluate',side_effect=score),\
             patch('wfsched.ours_v2_selector.semantic.guard',side_effect=guard):
            ans=choose_v2(s,m,c,'D',time.monotonic()+1.)
        self.assertEqual(ans['selected_cell'],'10')
        self.assertEqual(ans['decode_guard_rejections'],1)
    def test_no_future_dependent_output_or_realized_work(self):
        s,m,c=self.fixture(4)
        s['ready']=[]
        for n in s['nodes'].values():n['state']='blocked'
        ans=choose_v2(s,m,c,'D',time.monotonic()+1.)
        self.assertTrue(ans['no_state_opportunity'])
        self.assertEqual(ans['evaluations'],[])
    def test_deadline_never_selects_incomplete(self):
        s,m,c=self.fixture(4)
        ans=choose_v2(s,m,c,'D',time.monotonic()-1.)
        self.assertEqual(ans['action']['family'],'reference')
        self.assertTrue(ans['deadline_reached'])
    def test_zero_output_padding(self):
        from pathlib import Path
        text=(Path(__file__).resolve().parents[1]/'wfsched/ours_v2_selector.py').read_text()
        self.assertNotIn('ignore_eos',text)
        self.assertNotIn('min_tokens',text)
        self.assertNotIn('max_new_tokens',text)

if __name__=='__main__': unittest.main()
