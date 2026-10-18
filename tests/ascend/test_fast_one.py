"""CPU contract tests. Fake engine timing is NOT an Ascend performance result."""
from __future__ import annotations
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_ascend import fast_policy as policy
from dabs_ascend.fast_condition import run_condition, atomic_commit, wait_for_work
from dabs_ascend import fast_one


def config(R=2):
    return {**fast_one.configure({}), 'replicas': R, 'block_size': 128, 'base_cap': 1024,
            'hard_total': 1024, 'max_model_len': 4096, 'max_num_seqs': 10,
            'client_max_inflight': 2, 'admission_reserve_fraction': .9,
            'condition_wall_s': 5., 'fast_harvest_s': .1,
            'minimum_predicted_gain_ms': 1., 'local_model_margin_ms': 1.}


def model(R=2):
    value = {'models': {}, 'validation': {}, 'frozen': True, 'predictor': 'phase6_prefill_startup_v2',
             'timing_target': 'async_host_completion_accounting_v1', 'host_step_equivalence': False}
    for r in range(R):
        value['models'][str(r)] = {'phases': {p: {'coef': [1., 8., 3., .02, .01, .01],
            'observed_regimes': ['startup', 'pipeline'], 'startup_extra_ms': 1.} for p in ('prefill_only', 'decode_only', 'mixed')}}
        value['validation'][str(r)] = {p: {'range': {'p': [0, 1024], 'd': [0, 10], 'dc': [0, 40000], 'pc': [0, 4096]}}
                                      for p in ('prefill_only', 'decode_only', 'mixed')}
    return value


def node(k, wid=None, parents=(), status='ready', now=None, output=32, replica=None):
    now = time.monotonic() if now is None else now
    return {'id': k, 'wid': wid or k, 'parents': list(parents), 'state': status,
            'ready_at': now - .1 if status == 'ready' else None,
            'arrival': now - .1, 'arrival_offset': 0., 'scheduled_arrival': now - .1,
            'prompt': 3584, 'tokens': [1] * 3584, 'output': output, 'max_output_tokens': output,
            'output_done': 0, 'computed': 0, 'phase': 'prefill', 'prefix_hash': 'prefix-' + k,
            'share': 16, 'replica': replica, 'start': now - .05 if status == 'submitted' else None,
            'fixed_workload_replay': True, 'fixed_output_budget_predeclared': output}


def state(nodes, R=2):
    now = time.monotonic()
    active = {k: {f: copy.deepcopy(v) for f, v in n.items() if f not in ('tokens', 'stream')}
              for k, n in nodes.items() if n.get('arrival') is not None}
    reservations = [sum(policy.need_blocks(n, config(R)) for n in active.values() if n['state'] == 'submitted' and n.get('replica') == r) for r in range(R)]
    return {'nodes': active, 'ready': [k for k, n in active.items() if n['state'] == 'ready'],
            'replicas': R, 'now': now, 'pressure': [0] * R,
            'reserved_blocks': reservations, 'capacity_blocks': [543] * R,
            'queue_order': {str(r): [k for k,n in active.items() if n['state']=='submitted' and n.get('replica')==r] for r in range(R)},
            'cache_evidence': [], 'native_observation_errors': [], 'native_prediction_errors': [],
            'native_snapshots': [{'snapshot_id': f's-{r}-{now}', 'captured_ns': int(now * 1e9), 'expires_ns': int((now + 1.5) * 1e9)} for r in range(R)]}


def reference(s, cfg, method='Reference'):
    used = list(s['reserved_blocks'])
    counts = [sum(n['state'] == 'submitted' and n.get('replica') == r for n in s['nodes'].values()) for r in range(s['replicas'])]
    p, ad = {}, []
    for k in sorted(s['ready'], key=lambda k:(s['nodes'][k]['ready_at'], k)):
        r = min(range(s['replicas']), key=lambda r:(counts[r],r))
        p[k] = r
        need = policy.need_blocks(s['nodes'][k], cfg)
        if counts[r] < cfg['client_max_inflight'] and used[r]+need <= int(s['capacity_blocks'][r]*cfg['admission_reserve_fraction']):
            ad.append(k); counts[r] += 1; used[r] += need
    return {'family':'test_reference', 'placement':p, 'admit':ad, 'order':ad[:], 'caps':[cfg['base_cap']]*s['replicas']}


def test_planner(s, m, cfg, base, deadline):
    return policy.choose(s, m, cfg, base, deadline, guard=lambda *a: [])


def late_planner(s, m, cfg, base, deadline):
    time.sleep(.06)
    return {'action': {'bad': 'must never execute'}, 'selected_cell': '1'}


class CostTests(unittest.TestCase):
    def test_chunked_prefill_never_exceeds_real_budget(self):
        s = state({'a':node('a')}); costs=policy.ServiceCosts(s,model(),config(),time.monotonic()+3)
        v=costs.cost('a',0)
        self.assertGreater(v,0); self.assertEqual(costs.stats['primitive_calls'],6) # 4 prefill + 2 decode boundaries
        self.assertEqual(v,costs.cost('a',0)); self.assertEqual(costs.stats['cost_memo_hits'],1)

    def test_decode_affine_sum_and_floor(self):
        n=node('a',status='submitted',output=64,replica=0)
        n.update(phase='decode',computed=3584)
        s=state({'a':n}); m=model()
        m['models']['0']['phases']['decode_only']['coef']=[0.,0.,0.,1.8,0.,0.]
        costs=policy.ServiceCosts(s,m,config(),time.monotonic()+3)
        actual=costs.cost('a',0)
        expected=sum(max(.1,1.8*(3584+i)/65536) for i in range(64))
        self.assertAlmostEqual(actual,expected,places=10)

    def test_out_of_range_is_not_silently_extrapolated(self):
        s=state({'a':node('a')}); m=model()
        m['validation']['0']['prefill_only']['range']['p']=[1,500]
        with self.assertRaises(policy.ModelUnsupported):
            policy.ServiceCosts(s,m,config(),time.monotonic()+3).cost('a',0)

    def test_unknown_is_not_fake_cache_hit(self):
        s=state({'a':node('a')}); s['cache_evidence']=[{'prefix_hash':'prefix-a','replica':0,'known':False,'tokens':1024,'state':'active_computed','valid_until_ms':1000}]
        costs=policy.ServiceCosts(s,model(),config(),time.monotonic()+3)
        a=costs.cost('a',0)
        s['cache_evidence']=[]
        b=policy.ServiceCosts(s,model(),config(),time.monotonic()+3).cost('a',0)
        self.assertEqual(a,b)

    def test_expired_evidence_gets_no_credit(self):
        s=state({'a':node('a')}); s['cache_evidence']=[{'prefix_hash':'prefix-a','replica':0,'known':True,'tokens':1024,'state':'active_computed','valid_until_ms':0}]
        a=policy.ServiceCosts(s,model(),config(),time.monotonic()+3).cost('a',0)
        s['cache_evidence'][0]['valid_until_ms']=1000
        b=policy.ServiceCosts(s,model(),config(),time.monotonic()+3).cost('a',0)
        self.assertGreater(a,b)

    def test_future_continuation_does_not_reuse_old_cache_evidence(self):
        s=state({'a':node('a')}); s['cache_evidence']=[{'prefix_hash':'prefix-a','replica':0,'known':True,'tokens':1024,'state':'active_computed','valid_until_ms':1000}]
        costs=policy.ServiceCosts(s,model(),config(),time.monotonic()+3)
        self.assertGreater(costs.cost('a',0,continuation=True),costs.cost('a',0))


class CandidateTests(unittest.TestCase):
    def fixture(self):
        now=time.monotonic()
        ns={k:node(k,now=now,output=128 if k in ('a','b') else 16) for k in ('a','b','c','d','e','f')}
        for i,k in enumerate(ns): ns[k]['ready_at']=now-.2+i*.001
        s=state(ns); cfg=config(); base=reference(s,cfg)
        return s,cfg,base

    def test_admission_priority_survives_resource_projection(self):
        s,cfg,base=self.fixture(); deadline=time.monotonic()+3
        costs=policy.ServiceCosts(s,model(),cfg,deadline)
        topo=policy.topology(s,costs,deadline)
        candidates,diag=policy.generate_candidates(s,cfg,base,costs,topo[2],deadline)
        self.assertTrue(candidates); self.assertLessEqual(len(candidates),2)
        self.assertTrue(any(set(c['admit'])!=set(base['admit']) for c in candidates))
        for c in candidates: policy.validate_admissions(c,s,cfg)
        self.assertEqual(len(candidates),diag['distinct_candidates'])

    def test_same_order_only_is_not_claimed_as_engine_action(self):
        s,cfg,base=self.fixture(); other=copy.deepcopy(base); other['order'].reverse()
        self.assertEqual(policy.action_key(base),policy.action_key(other))

    def test_target_rejection_does_not_skip_other_replica(self):
        s,cfg,base=self.fixture()
        s['capacity_blocks'][0]=1
        base={'family':'reference','admit':[], 'placement':{k:0 for k in s['ready']},'order':[],'caps':[1024,1024]}
        deadline=time.monotonic()+3; costs=policy.ServiceCosts(s,model(),cfg,deadline)
        topo=policy.topology(s,costs,deadline)
        cs,diag=policy.generate_candidates(s,cfg,base,costs,topo[2],deadline)
        self.assertTrue(any(c['admit'] for c in cs))
        self.assertTrue(all(c['placement'][k]==1 for c in cs for k in c['admit']))
        self.assertGreater(diag['target_rejected'],0)

    def test_no_incumbent_reorder_or_duplicate_admission(self):
        s,cfg,base=self.fixture()
        base['order'].append('not-admitted')
        with self.assertRaises(ValueError): policy.validate_admissions(base,s,cfg)
        base['order']=[];base['admit'].append(base['admit'][0])
        with self.assertRaises(ValueError): policy.validate_admissions(base,s,cfg)

    def test_dependency_violation_rejected(self):
        s,cfg,base=self.fixture(); k=base['admit'][0]
        s['nodes'][k]['parents']=['not-complete']
        with self.assertRaises(ValueError): policy.validate_admissions(base,s,cfg)

    def test_kv_envelope_checked_without_cache_credit(self):
        s,cfg,base=self.fixture(); s['capacity_blocks']=[1,1]
        with self.assertRaises(ValueError): policy.validate_admissions(base,s,cfg)

    def test_full_request_graph_is_scored_without_mutation(self):
        now=time.monotonic()
        ns={'a':node('a',wid='w',now=now), 'b':node('b',wid='w',parents=['a'],status='blocked',now=now)}
        s=state(ns); cfg=config(); base=reference(s,cfg); before=copy.deepcopy(s)
        deadline=time.monotonic()+3; costs=policy.ServiceCosts(s,model(),cfg,deadline)
        topo=policy.topology(s,costs,deadline)
        z=policy.score(s,base,cfg,costs,topo,deadline)
        self.assertTrue(z['complete']);self.assertEqual(z['events'],2)
        self.assertGreater(z['request_completion_ms']['b'],z['request_completion_ms']['a'])
        self.assertEqual(before,s)

    def test_deadline_returns_reference_not_partial_score(self):
        s,cfg,base=self.fixture()
        z=policy.choose(s,model(),cfg,base,time.monotonic()-1)
        self.assertEqual(z['selected_cell'],'00');self.assertFalse(z['model_selection'])

    def test_native_input_failure_does_not_score(self):
        s,cfg,base=self.fixture();s['native_observation_errors']=['lifecycle_changed']
        z=policy.choose(s,model(),cfg,base,time.monotonic()+1)
        self.assertEqual(z['evaluations'],[]); self.assertEqual(z['selected_cell'],'00')

    def test_guard_rejection_is_not_bypassed(self):
        s,cfg,base=self.fixture()
        z=policy.choose(s,model(),cfg,base,time.monotonic()+3,guard=lambda *args:[{'kind':'test_protection'}])
        self.assertEqual(z['selected_cell'],'00');self.assertTrue(z['guard_details'])

    def test_model_failure_preserves_reference(self):
        s,cfg,base=self.fixture();m=model();m['validation']['0']['prefill_only']['range']['p']=[0,20]
        z=policy.choose(s,m,cfg,base,time.monotonic()+3)
        self.assertEqual(z['selected_cell'],'00');self.assertIn('out_of_range',z['stop_reason'])


class EvidenceTests(unittest.TestCase):
    def populate(self,root):
        for name,r in [('02-native-probe-one',1),('03-native-probe-four',4)]:
            d=root/name;d.mkdir();(d/'probe.json').write_text(json.dumps({'replicas':r}));(d/'native_events.jsonl').write_text('')
        d=root/'04-native-calibrate';d.mkdir()
        for n in ('native_profile.json','native_events.jsonl','calibration_samples.json','calibration_plan.json','capabilities.json'):(d/n).write_text('{}')

    def test_relocates_explicit_readonly_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.populate(root)
            (root/'evidence.json').write_text(json.dumps({'paths':{'native-probe-one':'/project/outputs/old/02-native-probe-one','native-probe-four':'/project/outputs/old/03-native-probe-four','native-calibrate':'/project/outputs/old/04-native-calibrate'}}))
            paths=fast_one.resolve_evidence(root)
            self.assertEqual(len(paths),3);self.assertTrue(all(Path(p).is_relative_to(root) for p in paths.values()))

    def test_ambiguous_model_never_chooses_latest(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.populate(root)
            shutil.copytree(root/'04-native-calibrate',root/'another-calibrate')
            with self.assertRaises(ValueError):fast_one.resolve_evidence(root)

    def test_plan_selects_only_one_real_frozen_scenario(self):
        # Reads the actual shipped frozen design without loading any services.
        selected=fast_one.plan()
        self.assertEqual([s['method'] for s in selected], ['RR','SLL','Ayo','Ours'])
        self.assertEqual(len(selected),4)
        self.assertEqual({(s['replicas'],s['rate'],s['repeat']) for s in selected},{(4,8.,1)})
        self.assertEqual({s['seed'] for s in selected},{190500})
        self.assertEqual({s['workflow_count'] for s in selected},{64})


class FakeNative:
    def __init__(self,nodes,cfg):
        self.nodes,self.cfg=nodes,cfg;self.salt='test';self.log=None
        self.receipts={};self.active={};self.calls=[];self.peak=0
    def flush_evidence(self,**kwargs):pass
    async def drain(self,R):
        if self.active:raise ValueError('fake engine not drained')
        return []
    async def harvest(self,R,**kwargs):return []
    async def observe(self,nodes,cfg,index,m):
        await asyncio.sleep(.001)
        return state(nodes,cfg['replicas'])
    def admission(self,n):
        n.setdefault('client_request_id',uuid.uuid4().hex);n['cache_salt']=self.salt
        return {'node_id':n['id'],'client_request_id':n['client_request_id'],'prompt_tokens':n['prompt'],'max_tokens':n['max_output_tokens']}
    async def call(self,r,op,**payload):
        self.calls.append((r,op,payload))
        if op=='receipt':return copy.deepcopy(self.receipts[payload['action_id']])
        if op!='admit':raise ValueError('fast-one must not send a queue mutation')
        if 'action' in payload or 'order' in payload:raise ValueError('unexpected queue mutation')
        aid=uuid.uuid4().hex;allowed=[]
        for a in payload['admissions']:
            active=sum(rr==r for rr in self.active.values())
            if active>=self.cfg['client_max_inflight']:raise ValueError('concurrent duplicate slot use')
            k=a['node_id']
            if k in self.active:raise ValueError('duplicate request')
            self.active[k]=r;allowed.append(a['client_request_id'])
        self.peak=max(self.peak,len(self.active))
        row={'action_id':aid,'status':'applied','admitted_permits':allowed,'engine_admissions':[]}
        self.receipts[aid]=row
        await asyncio.sleep(.001)
        return copy.deepcopy(row)


class FakeBackend:
    def __init__(self,nodes,cfg):self.native=FakeNative(nodes,cfg)
    async def generate(self,r,n,seed,log,schema=None):
        assert schema is None
        begin=time.monotonic();aid=n['native_action_id'];engine='engine-'+n['id']
        binding={'node_id':n['id'],'client_request_id':n['client_request_id'],'server_request_id':'server-'+n['id'],'engine_request_id':engine,'action_id':aid}
        self.native.receipts[aid]['engine_admissions'].append(binding)
        await asyncio.sleep(.005+(n['max_output_tokens']/100000))
        end=time.monotonic();self.native.active.pop(n['id'])
        result={'start':begin,'finish':end,'text':'test','usage':{'prompt_tokens':n['prompt'],'completion_tokens':n['max_output_tokens']},
                'ttft_text_proxy_ms':1.,'native_mapping':{'mappings':[binding]}}
        n.update(stream=result,binding_status='verified',binding_verified_at=begin,http_done_at=end)
        return result


def workload():
    ns={}
    for i in range(6):
        for j in range(2):
            k=f'w{i}/s{j}'
            n=node(k,wid=f'w{i}',parents=[] if j==0 else [f'w{i}/s0'],status='blocked',output=128 if i<2 else 16)
            n.update(arrival=None,arrival_offset=0. if i<3 else .002)
            ns[k]=n
    return ns


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_wakes_without_waiting_full_poll(self):
        e=asyncio.Event();e.set();t=time.monotonic()
        await wait_for_work(e,.2,0.)
        self.assertLess(time.monotonic()-t,.05)

    async def test_fake_native_end_to_end_preserves_work_and_dependencies(self):
        ns=workload();cfg=config();cfg.update(fast_plan_budget_s=.2,fast_commit_reserve_s=.01)
        b=FakeBackend(ns,cfg)
        with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(1) as pool:
            s=await run_condition(b,ns,{'external_work_fingerprint':'fake'},model(),cfg,
                                  {'label':'test','method':'Ours','seed':190500},Path(d)/'run',pool,reference,test_planner)
            self.assertTrue(s['functional_pass']);self.assertEqual(s['completed_requests'],12)
            self.assertEqual(len({k for k in ns if ns[k]['state']=='done'}),12)
            self.assertLessEqual(b.native.peak,4)
            self.assertTrue(all(op in ('admit','receipt') for _,op,_ in b.native.calls))
            self.assertEqual(s['output_tokens'],sum(n['max_output_tokens'] for n in ns.values()))
            self.assertGreater(s['diagnostics']['planner_started'],0)

    async def test_late_planner_never_executes_malformed_old_action(self):
        ns=workload();cfg=config();cfg.update(fast_plan_budget_s=.001,fast_commit_reserve_s=.001,fast_poll_s=.002)
        b=FakeBackend(ns,cfg)
        with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(1) as pool:
            s=await run_condition(b,ns,{},model(),cfg,{'label':'late','method':'Ours','seed':190500},Path(d)/'run',pool,reference,late_planner)
            self.assertTrue(s['functional_pass']);self.assertEqual(s['diagnostics']['nonreference_selected'],0)
            self.assertGreater(s['diagnostics']['planner_timeout'],0)

    async def test_reference_does_not_start_planner(self):
        ns=workload();cfg=config();b=FakeBackend(ns,cfg)
        with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(1) as pool:
            s=await run_condition(b,ns,{},model(),cfg,{'label':'ref','method':'Reference','seed':190500},Path(d)/'run',pool,reference,late_planner)
            self.assertEqual(s['diagnostics'].get('planner_started',0),0)

    async def test_unissued_permit_prevents_http_submission(self):
        ns=workload();cfg=config();k=next(iter(ns));ns[k].update(state='ready',arrival=time.monotonic(),ready_at=time.monotonic())
        s=state(ns);a=reference(s,cfg)
        b=FakeNative(ns,cfg)
        async def wrong(*args,**kwargs):return {'status':'applied','action_id':'bad','admitted_permits':['foreign']}
        b.call=wrong
        with self.assertRaises(ValueError):await atomic_commit(b,a,s,ns,cfg)



# Larger-shape and deployment-facing checks; still entirely CPU/fake native.
class AdditionalPolicyTests(unittest.TestCase):
    def test_candidate_retains_ready_placement_schema(self):
        s,cfg,base=CandidateTests().fixture()
        deadline=time.monotonic()+3
        costs=policy.ServiceCosts(s,model(),cfg,deadline)
        topo=policy.topology(s,costs,deadline)
        choices,_=policy.generate_candidates(s,cfg,base,costs,topo[2],deadline)
        self.assertTrue(choices)
        for a in choices:self.assertEqual(set(a['placement']),set(s['ready']))

    def test_native_guard_uses_no_unexecuted_queue_priority(self):
        ns={'old':node('old',status='submitted',replica=0), 'new':node('new')}
        ns['old'].update(phase='decode',computed=3584)
        s=state(ns);cfg=config();base=reference(s,cfg)
        seen=[]
        def next_batch(s,a,m,r,c):
            seen.append(a.get('order'))
            return {'prefill':0,'decode':1,'decode_context':3584,'cache_credit':0,'step_ms':2.,'decode_only_ms':2.}
        fake=types.ModuleType('dabs_ascend.native_scoring');fake.next_batch=next_batch
        with patch.dict(sys.modules,{'dabs_ascend.native_scoring':fake}):
            self.assertEqual(policy.native_guard(s,base,base,model(),cfg),[])
        self.assertEqual(seen,[[],[]])

    def test_native_guard_failure_rejects_not_bypass(self):
        ns={'old':node('old',status='submitted',replica=0),'new':node('new')}
        ns['old'].update(phase='decode',computed=3584)
        s=state(ns);cfg=config();base=reference(s,cfg)
        def unsupported(*args):raise ValueError('native_calibration_out_of_range:test')
        fake=types.ModuleType('dabs_ascend.native_scoring');fake.next_batch=unsupported
        with patch.dict(sys.modules,{'dabs_ascend.native_scoring':fake}):
            rows=policy.native_guard(s,base,base,model(),cfg)
        self.assertEqual(rows[0]['kind'],'native_guard_model_unusable')

    def test_full_216_node_score_completes_all_64_workflows(self):
        ns=large_workload()
        now=time.monotonic()
        for n in ns.values():
            n['arrival']=now-.1
            if not n['parents']:n.update(state='ready',ready_at=now-.1)
        s=state(ns,4);cfg=config(4);base=reference(s,cfg)
        deadline=time.monotonic()+5
        costs=policy.ServiceCosts(s,model(4),cfg,deadline)
        z=policy.score(s,base,cfg,costs,policy.topology(s,costs,deadline),deadline)
        self.assertEqual(z['events'],216)
        self.assertEqual(len(z['request_completion_ms']),216)
        self.assertEqual(len(z['workflow_completion_ms']),64)
        for k,n in ns.items():
            for parent in n['parents']:
                self.assertGreater(z['request_completion_ms'][k],z['request_completion_ms'][parent])

    def test_report_keeps_negative_performance(self):
        def row(method,jct):
            return {'spec':{'method':method},'jct_mean_ms':jct,'jct_p95_ms':2*jct,
                    'workflows_s':1/jct,'output_tokens_s':100/jct,'completed_requests':216,
                    'output_tokens':13184,'functional_pass':True,'diagnostics':{},'timings_ms':{}}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            fast_one.make_report(out,[row('RR',10),row('SLL',10),row('Ayo',10),row('Ours',12)])
            comparisons=json.loads((out/'comparisons.json').read_text())
            self.assertEqual(len(comparisons),3)
            self.assertEqual({c['baseline'] for c in comparisons},{'RR','SLL','Ayo'})
            self.assertTrue(all(c['jct_mean_improvement_pct']==-20 for c in comparisons))
            self.assertIn('NOT evidence of a performance win',(out/'test_result.txt').read_text())


def large_workload():
    """Synthetic full-size graph: 40 chains of 3 and 24 fork-joins of 4.

    Its graph families and timings are NOT asserted to be the real dataset.
    The 216-node/64-workflow/13184-token totals exercise the runner envelope.
    """
    ns={}
    for w in range(64):
        wid=f'w{w:04d}'
        edges=[[],[0],[1]] if w<40 else [[],[0],[0],[1,2]]
        for j,parents in enumerate(edges):
            k=f'{wid}/step{j+1}'
            n=node(k,wid=wid,parents=[f'{wid}/step{p+1}' for p in parents],status='blocked',output=64)
            n.update(arrival=None,arrival_offset=0.)
            ns[k]=n
    # 216*64 - 40*16 = 13184. Predeclared fake output counts, not fitted data.
    for n in list(ns.values())[:40]:
        n.update(output=48,max_output_tokens=48,fixed_output_budget_predeclared=48)
    return ns


class AdditionalLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_216_request_fake_native_run(self):
        ns=large_workload();cfg=config(4)
        cfg.update(condition_wall_s=15.,fast_plan_budget_s=.5,fast_snapshot_age_s=2.,fast_commit_reserve_s=.01,
                   ready_coalesce_s=.001,fast_poll_s=.003)
        b=FakeBackend(ns,cfg)
        with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(1) as pool:
            s=await run_condition(b,ns,{'external_work_fingerprint':'synthetic-216'},model(4),cfg,
                                 {'label':'full_size_fake','method':'Ours','seed':190500},Path(d)/'run',pool,reference,test_planner)
            self.assertEqual(s['completed_workflows'],64)
            self.assertEqual(s['completed_requests'],216)
            self.assertEqual(s['output_tokens'],13184)
            self.assertTrue(s['functional_pass'])
            self.assertLessEqual(b.native.peak,8)
            self.assertEqual(len({n['client_request_id'] for n in ns.values()}),216)

    async def test_partial_rpc_failure_waits_other_replica_and_stops(self):
        ns={k:node(k) for k in ('a','b')};cfg=config();s=state(ns);a=reference(s,cfg)
        b=FakeNative(ns,cfg);seen=[]
        async def call(r,op,**payload):
            if r==0:
                await asyncio.sleep(.001)
                raise RuntimeError('fake transport failure')
            await asyncio.sleep(.01)
            seen.append(r)
            return {'status':'applied','action_id':'other-replica',
                    'admitted_permits':[x['client_request_id'] for x in payload['admissions']]}
        b.call=call
        with self.assertRaises(RuntimeError):await atomic_commit(b,a,s,ns,cfg)
        self.assertEqual(seen,[1])


class FourMethodTests(unittest.TestCase):
    def test_default_cli_is_exactly_four_methods(self):
        self.assertEqual(fast_one.parser().parse_args(['plan']).methods, 'RR,SLL,Ayo,Ours')
        self.assertEqual(fast_one.METHODS, ('RR','SLL','Ayo','Ours'))

    def test_default_plan_prints_one_scenario_four_conditions(self):
        from contextlib import redirect_stdout
        import io
        stream=io.StringIO()
        with redirect_stdout(stream): self.assertEqual(fast_one.main(['plan']),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result['condition_count'],4)
        self.assertEqual(result['generation_requests'],864)
        self.assertEqual(result['workflows'],256)
        self.assertEqual(result['output_tokens'],52736)
        self.assertFalse(result['automatic_grid'])
        self.assertFalse(result['new_calibration'])
        self.assertEqual(result['methods'],['RR','SLL','Ayo','Ours'])

    def test_reference_is_internal_not_a_fifth_condition(self):
        with self.assertRaises(ValueError): fast_one.plan(['Reference'])
        with self.assertRaises(ValueError): fast_one.plan(['RR','RR'])
        with self.assertRaises(ValueError): fast_one.plan([])
        with self.assertRaises(SystemExit): fast_one.main(['plan','--methods','Reference,Ayo,Ours'])

    def test_method_subset_does_not_expand_scenario(self):
        selected=fast_one.plan(['Ayo','Ours'])
        self.assertEqual([s['method'] for s in selected],['Ayo','Ours'])
        self.assertEqual({(s['replicas'],s['rate'],s['repeat']) for s in selected},{(4,8.,1)})

    def test_baseline_dispatch_preserves_each_native_method(self):
        # Dispatch contract only; this fake does not test native engine behaviour.
        seen=[]
        fake=types.ModuleType('dabs_ascend.native_algorithms')
        fake.baseline=lambda s,c,m: seen.append(m) or {'method':m}
        with patch.dict(sys.modules,{'dabs_ascend.native_algorithms':fake}):
            for method in fast_one.METHODS:
                self.assertEqual(fast_one.baseline({}, {}, method),{'method':method})
            with self.assertRaises(ValueError): fast_one.baseline({}, {}, 'Reference')
        self.assertEqual(seen,['RR','SLL','Ayo','Ours'])

    def test_partial_report_does_not_invent_ours_result(self):
        row={'spec':{'method':'RR'},'jct_mean_ms':10.,'jct_p95_ms':12.,'workflows_s':1.,
             'output_tokens_s':100.,'completed_requests':216,'output_tokens':13184,
             'functional_pass':True,'diagnostics':{},'timings_ms':{}}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fast_one.make_report(out,[row])
            self.assertEqual(json.loads((out/'comparisons.json').read_text()),[])
            self.assertIn('Completed methods: RR',(out/'test_result.txt').read_text())

    def test_frozen_rr_sll_kernel_uses_declared_routing_semantics(self):
        # Byte-preserved baseline kernel; stub only resource/project boundary.
        folder=ROOT/'source/spra_workflow_fullgrid_r234_sll_v1/wfsched'
        spec=importlib.util.spec_from_file_location('_fast_test_routing',folder/'routing.py')
        routing=importlib.util.module_from_spec(spec); spec.loader.exec_module(routing)
        ns={str(i):{'ready_at':float(i)} for i in range(6)}
        snap={'nodes':ns,'ready':list(ns),'replicas':4,
              'router_submitted_count':0,'router_ongoing':[0,2,1,3]}
        cfg={'wf_routing_policy':'RR','wf_router_seed':190500}
        package=types.ModuleType('joint18v2'); package.__path__=[]
        actions=types.ModuleType('joint18v2.actions'); actions.resources=lambda a,s,c:a
        common=types.ModuleType('joint18v2.common'); common.validate_action=lambda a,s,c:None
        with patch.dict(sys.modules,{'joint18v2':package,'joint18v2.actions':actions,'joint18v2.common':common}):
            rr=routing.reference(snap,cfg)
            self.assertEqual([rr['placement'][k] for k in ns],[0,1,2,3,0,1])
            cfg['wf_routing_policy']='SLL'
            first=routing.reference(snap,cfg); again=routing.reference(snap,cfg)
        self.assertEqual(first,again)
        for trace in first['routing_trace']:
            self.assertEqual(len(trace['candidates']),2)
            chosen=trace['candidates'].index(trace['chosen'])
            self.assertEqual(trace['candidate_loads'][chosen],min(trace['candidate_loads']))


class FourMethodLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_rr_sll_ayo_share_fast_loop_without_planner(self):
        for method in ('RR','SLL','Ayo'):
            with self.subTest(method=method):
                ns=workload(); cfg=config(); b=FakeBackend(ns,cfg)
                with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(1) as pool:
                    s=await run_condition(b,ns,{},model(),cfg,
                        {'label':method,'method':method,'seed':190500},Path(d)/'run',pool,reference,late_planner)
                self.assertTrue(s['functional_pass'])
                self.assertEqual(s['completed_requests'],12)
                self.assertEqual(s['diagnostics'].get('planner_started',0),0)


class LogTests(unittest.TestCase):
    def test_logs_reads_existing_summary_without_services(self):
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'status.json').write_text('{"status":"completed","conditions":4}')
            (root/'test_result.txt').write_text('RR SLL Ayo Ours\n')
            output=io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(fast_one.main(['logs','--out',d]),0)
            self.assertIn('RR SLL Ayo Ours',output.getvalue())
            self.assertIn('"conditions":4',output.getvalue())

    def test_logs_keeps_failure_visible_even_with_partial_report(self):
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'status.json').write_text('{"status":"failed"}')
            (root/'test_result.txt').write_text('RR only completed')
            (root/'error_traceback.txt').write_text('test transport failure')
            output=io.StringIO()
            with redirect_stdout(output):self.assertEqual(fast_one.show_logs(root),0)
            self.assertIn('"status":"failed"',output.getvalue())
            self.assertIn('test transport failure',output.getvalue())

    def test_logs_without_summary_is_not_success(self):
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            self.assertEqual(fast_one.show_logs(d),2)

if __name__=='__main__':unittest.main()
