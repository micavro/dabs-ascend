from __future__ import annotations
import copy,json,time,tempfile,unittest,importlib.util,sys
from pathlib import Path
from unittest.mock import patch
from joint18v2.actions import partial_join_candidates,schedules,join_groups
from joint18v2.search import reference,post_budget,choose,route_packet
from joint18v2.model import predict
from joint18v2.service_fit import fit_phase_model,features,read_rows
from joint18v2.common import validate_action,action_key
from core18v2.engine import evaluate
from test_integration_contracts import cfg as smallcfg,node,snapshot,model as constant_model
ROOT=Path(__file__).resolve().parents[1]

def fork_fixture(renamed=False):
    names=['a','b','c','d'] if not renamed else ['alpha','zeta','first','middle']
    ns={'root':node('root','w',state='done',prompt=2048,out=16,r=0)}
    ns['root'].update(computed=2048,output_done=16,prefix_hash='h',share=1024)
    for name,(p,o) in zip(names,[(2816,48),(3328,64),(2176,192),(2304,224)]):
        ns[name]=node(name,'w',['root'],prompt=p,out=o)
        ns[name].update(prefix_hash='h',share=1024)
    ns['join']=node('join','w',names,state='blocked',prompt=256,out=32)
    for i in range(4):
        n=node('bg'+str(i),'bg'+str(i),state='submitted',prompt=512,out=224,r=0)
        n.update(computed=512,phase='decode',output_done=10);ns[n['id']]=n
    s=snapshot(ns,4);s['pressure']=[4,0,0,0]
    s['cache_evidence']=[{'prefix_hash':'h','replica':r,'state':'active_computed' if r==0 else 'absent','known':True,'tokens':1024 if r==0 else 0} for r in range(4)]
    m={'models':{str(r):{'coef':[15,2,6,.3,.1,.4]} for r in range(4)}}
    c=smallcfg();c.update(replicas=4,focus_join_limit=2,route_neighborhood_limit=3,per_candidate_slice_s=.03,planning_soft_limit_s=.12)
    return s,m,c,names

class CandidateRepairTests(unittest.TestCase):
    def test_mechanism_partial_pair_in_complete_library(self):
        s,m,c,n=fork_fixture();base=reference(s,c);aa=schedules(s,m,c,base)
        matching=[a for a in aa if a['placement'][n[0]]==a['placement'][n[1]]==0 and a['placement'][n[2]]==a['placement'][n[3]]==1 and set(a['admit'])==set(n)]
        self.assertTrue(matching)
    def test_renaming_nodes_does_not_erase_partial_pair(self):
        s,m,c,n=fork_fixture(True);aa=partial_join_candidates(s,m,c,reference(s,c))
        self.assertTrue(any(a['placement'][n[2]]==a['placement'][n[3]]==1 and a['placement'][n[0]]==a['placement'][n[1]]==0 for a in aa))
    def test_proper_subsets_and_all_destinations(self):
        s,m,c,n=fork_fixture();aa=partial_join_candidates(s,m,c,reference(s,c))
        self.assertEqual({len(a['proposal']['moved_nodes']) for a in aa},{1,2,3})
        self.assertEqual({a['proposal']['destination'] for a in aa},{1,2,3})
    def test_no_rewrite_of_submitted_nodes(self):
        s,m,c,n=fork_fixture()
        for a in schedules(s,m,c,reference(s,c)):
            self.assertEqual(set(a['placement']),set(s['ready']));validate_action(a,s,c)
    def test_no_join_has_no_invented_join_candidate(self):
        s,m,c,n=fork_fixture();s['nodes']['join']['parents']=[n[-1]]
        self.assertEqual(partial_join_candidates(s,m,c,reference(s,c)),[])
    def test_native_cache_prefix_not_route_history(self):
        s,m,c,n=fork_fixture();s['prefix_history']={'h':[2]}
        self.assertEqual(set(reference(s,c)['placement'].values()),{0})
    def test_packet_contains_proper_pair_without_forced_names(self):
        s,m,c,n=fork_fixture();packet,coverage=route_packet(s,m,c,reference(s,c))
        self.assertTrue(any(a['family'].startswith('partial_join/') for a in packet))
        self.assertGreater(coverage['partial_group_candidates_generated'],0)
    def test_candidate_specific_vectors_match_declared_mechanism(self):
        s,m,c,n=fork_fixture();a=reference(s,c);a['placement'].update({n[2]:1,n[3]:1})
        vv,f=post_budget(s,a,c)
        self.assertIn((256,1024,1024,1024),vv);self.assertIn((4096,4096,1024,1024),vv)
        self.assertEqual(f['resulting_decode'],[4,0,0,0]);self.assertGreater(f['resulting_prefill_query_proxy'][1],0)
    def test_no_prefill_does_not_create_budget_variants(self):
        s,m,c,n=fork_fixture();s['ready']=[]
        for k in n:s['nodes'][k]['state']='blocked'
        a=reference(s,c);vv,_=post_budget(s,a,c);self.assertEqual(vv,[(1024,)*4])
    def test_all_cap_levels_remain_proposable(self):
        s,m,c,n=fork_fixture();a=reference(s,c);vv,f=post_budget(s,a,c)
        self.assertEqual({v[f['focal']] for v in vv},set(c['caps']))

class SearchRepairTests(unittest.TestCase):
    def fake_eval(self,s,a,m,cfg,**kwargs):
        moved=sum(r!=0 for r in a['placement'].values());v=1000+10*moved
        if a['caps'][0]==256:v-=5
        if moved==2 and a['caps'][0]==256:v=900
        return {'sum_remaining_jct_ms':v,'rankable':True,'complete':True,'reason':'fake_complete',
          'cpu_wall_ms':.01,'rollout_steps':10,'unfinished_area_to_horizon_ms':10,
          'common_continuation_value_ms':v-10,'cache_counters':{},'workflow_prediction_ms':{'w':v}}
    def test_joint_selects_cross_only_when_complete_objective_wins(self):
        s,m,c,n=fork_fixture();s['decision_index']=0
        with patch('joint18v2.search.evaluate',side_effect=self.fake_eval):
            z=choose(s,m,c,'Joint',time.monotonic()+2)
        self.assertTrue(z['action']['family'].startswith('joint/'))
        self.assertLess(z['predicted_best_cost_ms'],z['predicted_reference_cost_ms'])
    def test_coordinate_prefix_same_for_decoupled_and_joint(self):
        s,m,c,n=fork_fixture()
        with patch('joint18v2.search.evaluate',side_effect=self.fake_eval):
            d=choose(s,m,c,'Decoupled',time.monotonic()+2);j=choose(s,m,c,'Joint',time.monotonic()+2)
        self.assertEqual(action_key(d['action']),action_key(j['decoupled_action']))
        self.assertTrue({action_key(x['action']) for x in d['evaluations']} <=
                        {action_key(x['action']) for x in j['evaluations']})
    def test_D_keeps_fixed_cap(self):
        s,m,c,n=fork_fixture()
        with patch('joint18v2.search.evaluate',side_effect=self.fake_eval):z=choose(s,m,c,'D',time.monotonic()+2)
        self.assertTrue(all(x['action']['caps']==[1024]*4 for x in z['evaluations']))
    def test_B_keeps_reference_schedule(self):
        s,m,c,n=fork_fixture();base=reference(s,c)
        with patch('joint18v2.search.evaluate',side_effect=self.fake_eval):z=choose(s,m,c,'B',time.monotonic()+2)
        self.assertTrue(all(x['action']['placement']==base['placement'] and x['action']['admit']==base['admit'] and x['action']['order']==[] for x in z['evaluations']))
    def test_incomplete_reference_does_not_allow_other_scores_to_win(self):
        s,m,c,n=fork_fixture()
        def no(s,a,m,cfg,**kw):
            z=self.fake_eval(s,a,m,cfg,**kw);z.update(rankable=False,complete=False,sum_remaining_jct_ms=None);return z
        with patch('joint18v2.search.evaluate',side_effect=no):z=choose(s,m,c,'Joint',time.monotonic()+2)
        self.assertTrue(z['reference_unrankable']);self.assertEqual(z['action']['family'],'reference');self.assertFalse(z['model_selection'])
    def test_expired_worker_returns_reference_without_forged_score(self):
        s,m,c,n=fork_fixture();z=choose(s,m,c,'Joint',time.monotonic()-1)
        self.assertEqual(z['evaluations'],[]);self.assertFalse(z['model_selection'])
    def test_client_reserves_handoff_inside_unchanged_deadline(self):
        text=(ROOT/'joint18v2/client.py').read_text()
        self.assertIn('evaluation_deadline=deadline-handoff',text)
        self.assertIn('timeout=max(.0001,deadline-time.monotonic())',text)
        self.assertIn('scfg,method,evaluation_deadline)',text)
    def test_trace_has_per_workflow_objective_evidence(self):
        s,m,c,n=fork_fixture()
        with patch('joint18v2.search.evaluate',side_effect=self.fake_eval):z=choose(s,m,c,'Joint',time.monotonic()+2)
        self.assertTrue(all('workflow_completion_ms' in x for x in z['evaluations']))

class PhaseModelTests(unittest.TestCase):
    def test_legacy_model_numerics_preserved(self):
        m=constant_model(1);self.assertEqual(predict(m,0,256,2,512,0),1)
    def test_rollout_and_model_use_same_phase(self):
        m=constant_model(1);m['models']['0']['phases']={'prefill_only':{'coef':[3,0,0,0,0,0]},'decode_only':{'coef':[2,0,0,0,0,0]},'mixed':{'coef':[4,0,0,0,0,0]}}
        s=snapshot({'r':node('r','w',out=2)},1);a=reference(s,smallcfg());z=evaluate(s,a,m,dict(smallcfg(),replicas=1))
        self.assertEqual(predict(m,0,256,0,0,0),3);self.assertEqual(z['sum_remaining_jct_ms'],5)
    def make_rows(self):
        rows=[]
        for b in range(1,5):
          for case in range(5):
            for ph in ('prefill_only','mixed','decode_only'):
              for j in range(4):
                p=256*(j+1) if ph!='decode_only' else 0;d=(j+1) if ph!='prefill_only' else 0
                x=features(p,d,512*d,0,0);y=(3 if ph=='prefill_only' else 15)+.5*(p/1024)+.5*d/16
                rows.append({'replica':0,'case':f'B{b}_{case}','block':b,'phase':ph,'p':p,'d':d,'dc':512*d,'pc':0,'skip':0,'x':list(x),'y':y})
        return rows
    def test_block_cv_holds_out_all_methods(self):
        m=constant_model(1);m['models']['0']['coef']=[15,0,0,0,0,0]
        out,rep=fit_phase_model(self.make_rows(),m,1)
        for ph,z in rep['0'].items():
            for f in z['folds']:
                self.assertNotIn(f['held_block'],f['train_blocks']);self.assertFalse(set(f['training_conditions'])&set(f['test_conditions']))
        self.assertTrue(rep['0']['prefill_only']['phase_fit_selected'])
        self.assertFalse(out['counterfactual_rank_certified'])
    def test_no_future_or_JCT_labels_needed(self):
        rows=self.make_rows();a,b=fit_phase_model(rows,constant_model(1),1)
        self.assertIn('phases',a['models']['0']);self.assertFalse(a['fitting_continues_during_measured_run'])
    def test_all_archived_blocks_required(self):
        with self.assertRaises(ValueError):fit_phase_model([x for x in self.make_rows() if x['block']!=4],constant_model(1),1)
    def test_empty_phase_retains_legacy_not_zero_time(self):
        m=constant_model(1);rows=[x for x in self.make_rows() if x['phase']!='prefill_only']
        out,r=fit_phase_model(rows,m,1);self.assertFalse(r['0']['prefill_only']['phase_fit_selected'])
        self.assertEqual(predict(out,0,256,0,0,0),1)
    def test_negative_phase_coefficient_rejected(self):
        m=constant_model(1);m['models']['0']['phases']={'prefill_only':{'coef':[-1,0,0,0,0,0]}}
        with self.assertRaises(ValueError):predict(m,0,256,0,0,0)
    def test_new_duration_uses_native_skips_not_logical_tokens(self):
        m=constant_model(1);m['models']['0']['coef']=[1,10,0,0,0,0]
        self.assertLess(predict(m,0,1024,0,0,0,1023),predict(m,0,1024,0,0,0,0))

class RawExtractionTests(unittest.TestCase):
    def sample(self,folder,include_input=True):
        case='PILOT/'+folder.name;manifest={'case':case};summary={'condition_valid':True}
        (folder/'manifest.json').write_text(json.dumps(manifest));(folder/'summary.json').write_text(json.dumps(summary))
        base={'replica':0,'pid':8,'case':case,'iteration':1}
        scheduled=[{'tokens':256,'prefill':True,'computed_before':0,'context_before':256,'prompt_hash':'p'}]
        ev=[dict(base,event='iteration',scheduled=scheduled,prefill_logical_tokens=256,decode_tokens=0),dict(base,event='engine_step',engine_host_ms=5,engine_error=None)]
        if include_input:ev.append(dict(base,event='model_input',prompt_hash='p',skipped_query_tokens=255))
        (folder/'backend.jsonl').write_text('\n'.join(map(json.dumps,ev))+'\n')
    def test_raw_backend_join(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'B1_1_A';f.mkdir();self.sample(f);rows,rej=read_rows(f)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['x'][1],1/1024);self.assertEqual(rows[0]['y'],5)
    def test_missing_model_input_not_claimed_zero_skip(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'B1_1_A';f.mkdir();self.sample(f,False);rows,rej=read_rows(f)
            self.assertEqual(rows,[]);self.assertEqual(rej['missing_prefill_model_input'],1)

if __name__=='__main__':unittest.main()

class DecodeCoalescingTests(unittest.TestCase):
    def test_coalescing_matches_single_steps_on_fork(self):
        s,m,c,n=fork_fixture();a=reference(s,c)
        a['placement'].update({n[2]:1,n[3]:1});a['caps']=[256,4096,1024,1024]
        z1=evaluate(s,a,m,dict(c,coalesce_decode=False));z2=evaluate(s,a,m,dict(c,coalesce_decode=True))
        self.assertAlmostEqual(z1['sum_remaining_jct_ms'],z2['sum_remaining_jct_ms'],places=6)
        for k,v in z1['node_completion_ms'].items():self.assertAlmostEqual(v,z2['node_completion_ms'][k],places=6)
        self.assertGreater(z2['coalesced_decode_batches'],0)
    def test_random_small_DAGs_agree(self):
        import random
        for seed in range(12):
            rng=random.Random(seed);ns={}
            for w in range(3):
                r=f'{w}/root';b=f'{w}/left';d=f'{w}/right';j=f'{w}/join'
                ns[r]=node(r,str(w),prompt=rng.choice((128,256)),out=rng.randint(1,12))
                ns[b]=node(b,str(w),[r],state='blocked',prompt=rng.choice((128,512)),out=rng.randint(5,25))
                ns[d]=node(d,str(w),[r],state='blocked',prompt=256,out=rng.randint(5,25))
                ns[j]=node(j,str(w),[b,d],state='blocked',prompt=128,out=3)
            s=snapshot(ns,2);c=smallcfg();c.update(action_horizon_ms=20.,apply_delay_ms=1.,continuation_period_ms=10.)
            m={'models':{str(r):{'coef':[rng.uniform(.5,2),rng.uniform(0,3),rng.uniform(0,2),rng.uniform(0,1),.1,.1]} for r in range(2)}}
            a=reference(s,c);a['caps']=[256,512]
            one=evaluate(s,a,m,dict(c,coalesce_decode=False));many=evaluate(s,a,m,dict(c,coalesce_decode=True))
            self.assertTrue(one['rankable'] and many['rankable']);self.assertAlmostEqual(one['sum_remaining_jct_ms'],many['sum_remaining_jct_ms'],places=5)
            for k in ns:self.assertAlmostEqual(one['node_completion_ms'][k],many['node_completion_ms'][k],places=5)
    def test_event_trace_is_not_coalesced(self):
        s,m,c,n=fork_fixture();z=evaluate(s,reference(s,c),m,c,record_events=True)
        self.assertEqual(z['coalesced_decode_batches'],0);self.assertTrue(all(x.get('decode_steps',1)==1 for x in z['event_trace']))
    def test_clamped_short_service_not_coalesced_across_kink(self):
        n=node('r','w',state='submitted',out=120,r=0);n.update(computed=256,phase='decode',output_done=1)
        s=snapshot({'r':n},1);m={'models':{'0':{'coef':[0,0,0,.01,0,0]}}};c=smallcfg();c['replicas']=1
        one=evaluate(s,reference(s,c),m,dict(c,coalesce_decode=False));many=evaluate(s,reference(s,c),m,dict(c,coalesce_decode=True))
        self.assertAlmostEqual(one['sum_remaining_jct_ms'],many['sum_remaining_jct_ms'],places=7)
