from __future__ import annotations
import asyncio,copy,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from joint18v2.common import resolve_control,token_hash,make_budget_class
from joint18v2.cache import NativeCacheObserver,UnsupportedCache
from joint18v2.io import JsonWriter
from joint18v2.batchcheck import predict_next,binding_evidence
from joint18v2.search import post_budget,reference,choose
from core18v2.engine import evaluate

ID={'pod':'spra-vllm-0','pid':1,'boot_id':'same'}
ENV={'hard_total':4096,'block_size':16,'max_num_seqs':64}
def plan():return {'enabled':True,**ID,'case':'test','plan_id':'p','issued_ns':0,'expires_ns':900_000_000,
    'snapshot_ns':0,'not_before_ns':50_000_000,'action_expires_ns':200_000_000,'total':4096,'prefill_cap':256,'order':['x'],
    'continuation':{'total':4096,'prefill_cap':1024}}
def cfg():return {'replicas':2,'hard_total':4096,'block_size':16,'max_num_seqs':64,'base_cap':1024,'caps':[256,512,1024,2048,4096],
   'epoch_s':.2,'action_horizon_ms':200.,'apply_delay_ms':0.,'max_admission_wait_s':.6,'continuation_period_ms':0.,
   'admission_reserve_fraction':.9,'planning_soft_limit_s':.1,'candidate_limit':12}
def node(k,w,parents=(),state='ready',prompt=256,out=1,r=None):
    return {'id':k,'wid':w,'name':k,'parents':list(parents),'state':state,'prompt':prompt,'output':out,'computed':0,
      'output_done':0,'phase':'prefill','prefix_hash':None,'share':0,'replica':r,'ready_at':0,'accepted_at':None}
def snapshot(nodes,R=2):return {'nodes':nodes,'ready':[k for k,n in nodes.items() if n['state']=='ready'],'replicas':R,'now':0.,
    'queue_order':{},'prefix_history':{},'capacity_blocks':[4096]*R,'reserved_blocks':[0]*R,'current_caps':[1024]*R,'pressure':[0]*R}
def action(s,caps=None):return {'family':'ref','placement':{k:i%s['replicas'] for i,k in enumerate(s['ready'])},'admit':s['ready'][:],'order':[], 'caps':caps or [1024]*s['replicas']}
def model(R=2):return {'models':{str(r):{'coef':[1,0,0,0,0,0]} for r in range(R)}}

class TimeContractTests(unittest.TestCase):
    def test_before_activation_uses_continuation(self):self.assertEqual(resolve_control(plan(),ID,ENV,20_000_000)['prefill_cap'],1024)
    def test_current_action(self):self.assertEqual(resolve_control(plan(),ID,ENV,80_000_000)['prefill_cap'],256)
    def test_action_expiry_not_transport_expiry(self):self.assertEqual(resolve_control(plan(),ID,ENV,200_000_000)['prefill_cap'],1024)
    def test_heartbeat_cannot_renew_action(self):
        p=plan();p.update(issued_ns=300_000_000,expires_ns=1_000_000_000)
        self.assertEqual(resolve_control(p,ID,ENV,350_000_000)['execution_phase'],'common_continuation')
    def test_overlong_action_rejected(self):
        p=plan();p['action_expires_ns']=300_000_000
        with self.assertRaises(ValueError):resolve_control(p,ID,ENV,80_000_000)
    def test_engine_identity(self):
        p=plan();p['pid']=2
        with self.assertRaises(ValueError):resolve_control(p,ID,ENV,80_000_000)
    def test_lease_expiration(self):
        with self.assertRaises(ValueError):resolve_control(plan(),ID,ENV,950_000_000)

class CacheTests(unittest.TestCase):
    def setup_cache(self):
        a=NS(cached_blocks={},evictor=NS(free_table={}))
        o=NativeCacheObserver(NS(gpu_allocator=a),16,lambda t,b,k:list(range(len(t)//b)))
        toks=list(range(32));h=token_hash(toks);o.register({'schema':'joint18v2-prefix-query-1','query_id':'q','prefixes':[{'prefix_hash':h,'tokens':toks}]})
        return a,o,h
    def test_absent_is_measured_zero(self):
        a,o,h=self.setup_cache();r=o.sample()['records'][0];self.assertTrue(r['known']);self.assertEqual(r['tokens'],0);self.assertEqual(r['state'],'absent')
    def test_completed_freed_blocks_remain_reusable(self):
        a,o,h=self.setup_cache()
        for i in range(2):a.evictor.free_table[i]=NS(block_hash=i,computed=True,block_number=i,ref_count=0)
        r=o.sample()['records'][0];self.assertEqual(r['tokens'],32);self.assertEqual(r['state'],'reclaimable_computed')
    def test_uncomputed_block_not_credited(self):
        a,o,h=self.setup_cache();a.cached_blocks[0]=NS(block_hash=0,computed=False,block_number=0,ref_count=1)
        self.assertEqual(o.sample()['records'][0]['tokens'],0)
    def test_prefix_gap_not_skipped(self):
        a,o,h=self.setup_cache();a.cached_blocks[1]=NS(block_hash=1,computed=True,block_number=1,ref_count=1)
        self.assertEqual(o.sample()['records'][0]['tokens'],0)
    def test_eviction_observed(self):
        a,o,h=self.setup_cache();a.evictor.free_table[0]=NS(block_hash=0,computed=True,block_number=0,ref_count=0)
        o.sample();a.evictor.free_table.clear();z=o.sample();self.assertEqual(z['records'][0]['tokens'],0);self.assertEqual(z['changes'][0]['change'],'eviction_or_replacement')
    def test_unknown_allocator_not_reported_cold(self):
        with self.assertRaises(UnsupportedCache):NativeCacheObserver(NS(),16)
    def test_query_hash_verified(self):
        a,o,h=self.setup_cache()
        with self.assertRaises(ValueError):o.register({'schema':'joint18v2-prefix-query-1','query_id':'wrong','prefixes':[{'prefix_hash':'bad','tokens':[1]*16}]})
    def test_does_not_change_refcounts(self):
        a,o,h=self.setup_cache();b=NS(block_hash=0,computed=True,block_number=0,ref_count=2);a.cached_blocks[0]=b;o.sample();self.assertEqual(b.ref_count,2)

class CausalTests(unittest.TestCase):
    def test_relabel(self):
        values=[]
        for root_r in (0,1):
            ns={'r':node('r','w',state='submitted',r=root_r),'c':node('c','w',parents=['r'],state='blocked'),
                'x':node('x','x',state='submitted',r=1-root_r)}
            s=snapshot(ns);a=action(s);c=cfg();c.update(continuation_period_ms=0.)
            values.append(evaluate(s,a,model(),c)['sum_remaining_jct_ms'])
        self.assertEqual(values,[3.,3.])
    def test_shared_capacity(self):
        s=snapshot({str(i):node(str(i),str(i)) for i in range(10)},1);a=action(s,[256]);c=cfg();c.update(replicas=1,base_cap=256,continuation_period_ms=0)
        z=evaluate(s,a,model(1),c);self.assertEqual(z['sum_remaining_jct_ms'],55)
    def test_cutoff_no_rank(self):
        s=snapshot({str(i):node(str(i),str(i)) for i in range(10)},1);c=cfg();c.update(base_cap=256,max_surrogate_events=4)
        z=evaluate(s,action(s,[256]),model(1),c);self.assertFalse(z['rankable']);self.assertIsNone(z['sum_remaining_jct_ms'])
    def test_expiry_preserves_placement_and_prefix(self):
        ns={'r':node('r','w',prompt=256,out=3),'c':node('c','w',parents=['r'],state='blocked',prompt=512,out=1)}
        for n in ns.values():n.update(prefix_hash='h',share=256)
        s=snapshot(ns);a=action(s);a['placement']['r']=1;c=cfg();c.update(action_horizon_ms=1.5,continuation_period_ms=0)
        z=evaluate(s,a,model(),c,record_events=True)
        self.assertTrue(z['rankable']);self.assertEqual(z['admissions'][0]['replica'],1)
        self.assertTrue(any(x['replica']==1 and x['kind']=='common_continuation' for x in z['admissions']))
        self.assertGreater(z['cache_counters'].get('logical_overlap_tokens',0),0)
    def test_future_only_cap_has_no_current_effect(self):
        r=node('r','w',state='submitted',out=1000,r=0);r.update(phase='decode',computed=256,output_done=1)
        s=snapshot({'r':r,'c':node('c','w',parents=['r'],state='blocked',prompt=2560)},1);c=cfg();c.update(continuation_period_ms=0)
        values=[evaluate(s,action(s,[cap]),model(1),c)['sum_remaining_jct_ms'] for cap in c['caps']]
        self.assertEqual(values,[1002.]*5)
    def test_inflight_not_restarted(self):
        n=node('r','w',state='submitted',r=0);s=snapshot({'r':n},1);s['inflight_batches']=[{'replica':0,'prefill':[('r',256)],'decode':[],'remaining_ms':.25}]
        z=evaluate(s,action(s,[1024]),model(1),cfg());self.assertEqual(z['sum_remaining_jct_ms'],.25)
    def test_incomplete_search_cannot_choose_bad_score(self):
        s=snapshot({'r':node('r','w')});z=choose(s,model(),cfg(),'Joint',deadline=time.monotonic()-1)
        self.assertFalse(z['model_selection']);self.assertEqual(z['action']['family'],'reference')
    def test_post_placement_budget_follows_destination(self):
        s=snapshot({'r':node('r','w',prompt=3328)});a=action(s);a['placement']['r']=1
        vectors,f=post_budget(s,a,cfg());self.assertEqual(f['focal'],1);self.assertTrue(any(v[1]==4096 for v in vectors))

class AuditTests(unittest.TestCase):
    def test_writer_records_all(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.jsonl';w=JsonWriter()
            for i in range(100):w.submit(p,{'i':i})
            w.close();self.assertEqual(len(p.read_text().splitlines()),100)
    def test_writer_atomic_replace(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.json';w=JsonWriter();w.submit(p,{'n':1},True);w.submit(p,{'n':2},True);w.close();self.assertEqual(json.loads(p.read_text())['n'],2)
    def test_decode_does_not_consume_prefill_cap(self):
        before={'running':[{'request_id':'d','prefill':False,'prompt':16,'computed':16}],
                'waiting':[{'request_id':'p','prefill':True,'prompt':512,'computed':0}]}
        self.assertEqual(predict_next(before,4096,256),{'d':1,'p':256})
    def test_decode_only_not_binding(self):
        z=binding_evidence({'prefill_logical_tokens':0,'decode_tokens':20,'prefill_cap':256,'total':4096,'remaining_prefill_in_before':0})
        self.assertFalse(z['binding_candidate'])
    def test_cap_near_but_total_tighter(self):
        z=binding_evidence({'prefill_logical_tokens':4080,'decode_tokens':16,'prefill_cap':4096,'total':4096,'remaining_prefill_in_before':8000})
        self.assertFalse(z['binding_candidate'])
    def test_real_prefill_competition_binding_candidate(self):
        z=binding_evidence({'prefill_logical_tokens':256,'decode_tokens':16,'prefill_cap':256,'total':4096,'remaining_prefill_in_before':8000})
        self.assertTrue(z['binding_candidate'])
if __name__=='__main__':unittest.main()
