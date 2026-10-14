from __future__ import annotations
import asyncio,copy,contextlib,importlib.util,io,json,os,sys,tempfile,time,unittest,uuid,zipfile
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from semantic19.dataset import compile_record,validate_public,convert,load_public,balanced_select
from semantic19.workflow import bind,extract_answer,fair_prefix_lengths,compile_packets,instantiate,arrival_offsets
from semantic19.quality import normalize,answer_score
from semantic19.selector import choose,schedule_alternative,budget_alternative
from joint18v2.common import load,token_hash,atomic,validate_action
ROOT=Path(__file__).resolve().parents[1]

def raw(i=0,fork=True):
    qs=['What city is mentioned in Document 0?','What person is mentioned in Document 1?','Where does #2 live in #1?'] if fork else ['What city is mentioned?','Who lives in #1?']
    return {'id':('3hop2__' if fork else '2hop__')+str(i),'question':'A fixture, NOT a real dataset example.',
        'answerable':True,'answer':'GOLD_SECRET_NOT_IN_RUNTIME','answer_aliases':['HiddenAlias'],
        'paragraphs':[{'idx':j,'title':'Document '+str(j),'paragraph_text':'Paris is a city. Sam lives in Paris. '+('Filler.'*10),'is_supporting':j==0} for j in range(3)],
        'question_decomposition':[{'id':4000+j,'question':q,'answer':'Gold'+str(j),'paragraph_support_idx':0} for j,q in enumerate(qs)]}

class DatasetTests(unittest.TestCase):
    def test_ordinal_refs_not_source_id(self):
        p,g=compile_record(raw());self.assertEqual(p['steps'][2]['parents'],[0,1]);self.assertEqual(p['steps'][0]['source_subquestion_id'],4000)
    def test_runtime_strips_gold(self):
        p,g=compile_record(raw());s=json.dumps(p)
        for k in ['GOLD_SECRET','HiddenAlias','paragraph_support_idx','is_supporting','"answer"']:self.assertNotIn(k,s)
    def test_native_fork_not_added(self):
        p,_=compile_record(raw());self.assertEqual(len(p['steps']),3);self.assertEqual(p['graph']['max_parallel_width'],2)
    def test_chain_retained(self):
        p,_=compile_record(raw(fork=False));self.assertEqual(p['graph']['max_parallel_width'],1)
    def test_forward_ref_rejected(self):
        r=raw();r['question_decomposition'][0]['question']='#2 is what?'
        with self.assertRaises(ValueError):compile_record(r)
    def test_disconnected_rejected(self):
        r=raw();r['question_decomposition'][2]['question']='Where?'
        with self.assertRaises(ValueError):compile_record(r)
    def test_exact_parent_response_not_gold(self):
        q,u=bind('Where is #1 relative to #2?',{0:'<answer>Paris</answer>',1:'<answer>London</answer>'})
        self.assertEqual(q,'Where is (Paris) relative to (London)?');self.assertEqual(u['1']['response'],'<answer>Paris</answer>')
    def test_missing_parent_rejected(self):
        with self.assertRaises(ValueError):bind('#1',{})
    def test_format_failure_is_not_teacher_forced(self):
        q,u=bind('#1',{0:'I do not know'});self.assertIn('I do not know',q);self.assertFalse(u['1']['parsed_answer_tag'])
    def test_fair_context_budget(self):
        self.assertEqual(fair_prefix_lengths([100,100,2],12),[5,5,2])
    def test_fair_context_no_padding(self):self.assertEqual(fair_prefix_lengths([2,3],100),[2,3])
    def test_answer_metric(self):
        self.assertEqual(answer_score('The United States.',['United States']),{'em':1,'f1':1.})
    def test_public_validation_rejects_injected_gold(self):
        p,g=compile_record(raw());p['steps'][0]['answer']='x'
        with self.assertRaises(ValueError):validate_public(p)
    def test_compile_dataset_count_and_disjoint(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);tr=d/'train.jsonl';de=d/'dev.jsonl'
            tr.write_text(''.join(json.dumps(raw(i,i%2==0))+'\n' for i in range(60)))
            de.write_text(''.join(json.dumps(raw(1000+i,i%2==0))+'\n' for i in range(100)))
            m=convert(tr,de,d/'compiled');self.assertEqual(m['counts'],{'smoke':2,'profile':32,'pilot':64})
            a,_=load_public(d/'compiled','profile');b,_=load_public(d/'compiled','pilot');self.assertFalse({x['id'] for x in a}&{x['id'] for x in b})
    def test_reject_changed_public_data(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);f=p/'pilot.public.jsonl';f.write_text('{}\n');(p/'manifest.json').write_text(json.dumps({'files':{f.name:'bad'}}))
            with self.assertRaises(ValueError):load_public(p,'pilot')
    def test_wave_is_declared_fixed_shape(self):self.assertEqual(arrival_offsets(16,2,1),arrival_offsets(16,2,2))
    def test_no_equalization_real_outputs(self):
        from joint18v2.transport import metrics
        def n(name,out,finish):
            return {'id':name,'wid':name,'name':name,'arrival':0.,'arrival_offset':0.,'parents':[],'state':'done','ready_at':0.,'finish':finish,
                'stream':{'request_start':0.,'header':.1,'first':.2,'last':finish-.1,'return':finish,'usage':{'completion_tokens':out},'text_events':2}}
        ns={'a':n('a',5,1.),'b':n('b',7,2.)};z=metrics(ns,0.,2.2,[])
        self.assertEqual(z['output_tokens'],12);self.assertEqual(z['cohort_duration_s'],2.);self.assertAlmostEqual(z['runner_observation_lag_s'],.2)

class FakeTokenizer:
    def __init__(self):self.calls=0;self.texts=[]
    async def encode(self,text):self.calls+=1;self.texts.append(text);return [ord(c)+100 for c in text] or [132]

class PacketTests(unittest.IsolatedAsyncioTestCase):
    async def test_packet_uses_all_documents_and_bounds(self):
        p,_=compile_record(raw());c=load(ROOT/'configs/semantic.json');c['document_tokens_budget']=256;t=FakeTokenizer()
        packets,info=await compile_packets([p],t,c);self.assertEqual(len(packets[0]['paragraph_coverage']),3)
        self.assertLessEqual(len(packets[0]['evidence_tokens']),256);self.assertNotIn('GOLD_SECRET',''.join(t.texts))
    async def test_dynamic_child_receives_actual_outputs(self):
        p,_=compile_record(raw());c=load(ROOT/'configs/semantic.json');c['document_tokens_budget']=256;t=FakeTokenizer()
        pk,_=await compile_packets([p],t,c);ns,mat,proof=await instantiate(pk,t,c,1)
        ns['w0000/step1']['stream']={'text':'<answer>Paris</answer>'};ns['w0000/step2']['stream']={'text':'<answer>Sam</answer>'}
        await mat('w0000/step3',ns)
        self.assertIn('Paris',ns['w0000/step3']['resolved_question']);self.assertIn('Sam',ns['w0000/step3']['resolved_question'])
        self.assertNotIn('GOLD_SECRET',''.join(t.texts));self.assertTrue(ns['w0000/step3']['prompt_materialized'])
    async def test_dynamic_oversize_never_silently_truncates(self):
        p,_=compile_record(raw());c=load(ROOT/'configs/semantic.json');c['document_tokens_budget']=256;t=FakeTokenizer()
        pk,_=await compile_packets([p],t,c);ns,mat,proof=await instantiate(pk,t,c,1)
        for n in list(ns.values())[:2]:n['stream']={'text':'z'*5000}
        with self.assertRaises(ValueError):await mat('w0000/step3',ns)

class SelectorTests(unittest.TestCase):
    def snapshot(self):
        ns={}
        for i in range(3):
            k=str(i);ns[k]={'id':k,'wid':'w','name':k,'parents':[] if i<2 else ['0','1'],'prompt':256,'output':4,
               'computed':0,'output_done':0,'ready_at':0.,'state':'ready' if i<2 else 'blocked','replica':None,'share':0,'prefix_hash':None,
               'compiled_groups':[['0','1']],'compiled_successors':['2'] if i<2 else [],'max_output_tokens':96}
        s={'nodes':ns,'ready':['0','1'],'replicas':2,'pressure':[4,0],'now':.1,'current_caps':[1024,1024],
           'capacity_blocks':[4096,4096],'reserved_blocks':[0,0],'prefix_history':{},'cache_evidence':[],'decision_index':1}
        m={'models':{str(r):{'coef':[1.,0.,0.,0.,0.,0.]} for r in range(2)}}
        c=load(ROOT/'configs/semantic.json');c.update(replicas=2,planning_soft_limit_s=1.,apply_delay_ms=0.,action_horizon_ms=200.)
        return s,m,c
    def test_four_cells_scored_joint(self):
        s,m,c=self.snapshot();z=choose(s,m,c,'Joint',time.monotonic()+1)
        self.assertTrue(z['paired_packet_complete']);self.assertEqual({e['cell'] for e in z['evaluations']},{'00','10','01','11'})
    def test_B_does_not_build_schedules(self):
        s,m,c=self.snapshot()
        with patch('semantic19.selector.schedule_alternative',side_effect=RuntimeError('must not call')):choose(s,m,c,'B',time.monotonic()+1)
    def test_joint_deadline_retains_reference(self):
        s,m,c=self.snapshot();z=choose(s,m,c,'Joint',time.monotonic()-1);self.assertTrue(z['reference_unrankable'])
    def test_pair_incomplete_never_ranks_partial_value(self):
        s,m,c=self.snapshot();calls=[]
        def ev(*args,**kw):
            calls.append(1)
            return {'rankable':len(calls)==1,'sum_remaining_jct_ms':10 if len(calls)==1 else None,'node_completion_ms':{},'cpu_wall_ms':.1,'reason':'fixture'}
        with patch('semantic19.selector.evaluate',side_effect=ev):z=choose(s,m,c,'Joint',time.monotonic()+1)
        self.assertFalse(z['paired_packet_complete']);self.assertEqual(z['action']['family'],'reference')
    def test_no_output_future_labels_in_selector(self):
        s,m,c=self.snapshot();z=choose(s,m,c,'D',time.monotonic()+1);validate_action(z['action'],s,c)
    def test_decode_guard_can_reject_predicted_harm(self):
        from semantic19.selector import guard
        s,m,c=self.snapshot();n=s['nodes']['0'];n.update(state='submitted',replica=0,phase='decode');s['ready']=['1']
        from joint18v2.search import reference
        a=reference(s,c);v=guard(s,{'node_completion_ms':{'0':100}}, {'node_completion_ms':{'0':200}},a,a,m,c)
        self.assertEqual(v[0]['kind'],'remaining_incumbent_decode')

class SemanticEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def scenario(self,method):
        import httpx
        from fake_native import modules,FakeGroup,hashes
        from joint18v2.cache import NativeCacheObserver
        from joint18v2.client import execute
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);mods,sm,Engine=modules()
            with patch.dict(sys.modules,mods),patch.dict(os.environ,{'SPRA_E1_ROOT':temp,'POD_NAME':'spra-vllm-0','NVIDIA_VISIBLE_DEVICES':'MOCK-NOT-GPU'}):
                spec=importlib.util.spec_from_file_location('semfakehook'+uuid.uuid4().hex,ROOT/'runtime/joint18v2_hook.py');hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
                hook.capabilities=lambda:{'scheduler_source_sha256':'fixture','imported_version':'0.6.3','distribution_version':'0.6.3'}
                hook.NativeCacheObserver=lambda mgr,b:NativeCacheObserver(mgr,b,hashes)
                hook.install();sched=sm.Scheduler();engine=Engine(sched);sched._v2_writer.drain();initial=[load(path/'joint18v2/spra-vllm-0.status.json')]
                c=load(ROOT/'configs/semantic.json');c.update(replicas=1,epoch_s=.04,ready_coalesce_s=.001,planning_soft_limit_s=.02,
                    commit_reserve_s=.003,document_tokens_budget=256,max_output_tokens=8,initial_output_estimate=2,
                    snapshot_max_age_s=2.,cache_evidence_max_age_s=2.,condition_wall_s=10,initial_expected_apply_delay_ms=2.)
                p,_=compile_record(raw());tok=FakeTokenizer();pack,_=await compile_packets([p],tok,c);ns,mat,proof=await instantiate(pack,tok,c,1)
                for n in ns.values():n['arrival_offset']=0.
                sent=[]
                class Stream(httpx.AsyncByteStream):
                    def __init__(self,g):self.g=g
                    async def __aiter__(self):
                        while self.g.done<1:await asyncio.sleep(.001)
                        yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'<answer>Paris'}]})+'\n\n').encode()
                        while self.g.done<2:await asyncio.sleep(.001)
                        yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'</answer><evidence>Document 0</evidence>','finish_reason':'stop'}]})+'\n\n').encode()
                        yield ('data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':len(self.g.tokens),'completion_tokens':2}})+'\n\n').encode()
                        yield b'data: [DONE]\n\n'
                async def handler(req):
                    payload=json.loads(req.content);sent.append(payload)
                    self.assertNotIn('min_tokens',payload);self.assertNotIn('ignore_eos',payload)
                    g=FakeGroup('r'+uuid.uuid4().hex,payload['prompt'],2);sched.add_seq_group(g);return httpx.Response(200,stream=Stream(g))
                alive=True
                async def pump():
                    while alive:
                        if sched.running or sched.waiting:await engine.step_async()
                        await asyncio.sleep(.003)
                task=asyncio.create_task(pump());model=None if method=='A' else {'models':{'0':{'coef':[1.,0.,0.,0.,0.,0.]}}}
                try:
                    with ThreadPoolExecutor(1) as executor:
                        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                            with contextlib.redirect_stdout(io.StringIO()):s,events=await execute(client,path,initial,c,ns,path/'RUN',method,model,executor=executor if model else None,materializer=mat)
                    self.assertEqual(s['request_correlation'],'3/3');self.assertEqual(s['output_tokens'],6)
                    self.assertEqual(s['native_next_batch_matches'],s['native_next_batch_checked']);self.assertTrue(s['control_time_match'])
                    child=ns['w0000/step3'];self.assertEqual(len(child['parent_response_bindings']),2)
                    self.assertIn('Paris',child['resolved_question']);self.assertNotIn('GOLD_SECRET',json.dumps(sent))
                    self.assertLessEqual(max(n['finish'] for n in ns.values()),max(n['finish'] for n in ns.values())+s['runner_observation_lag_s'])
                    self.assertGreater(s['mean_critical_path_components_ms']['input_prepare'],0)
                    self.assertEqual(child['parent_response_bindings']['1']['response'],ns['w0000/step1']['stream']['text'])
                finally:
                    alive=False
                    try:await task
                    finally:sched._v2_stop.set();await asyncio.sleep(.04);sched._v2_writer.close()
    async def test_semantic_A(self):await self.scenario('A')
    async def test_semantic_D(self):await self.scenario('D')
    async def test_semantic_B(self):await self.scenario('B')
    async def test_semantic_Decoupled(self):await self.scenario('Decoupled')
    async def test_semantic_Joint(self):await self.scenario('Joint')
