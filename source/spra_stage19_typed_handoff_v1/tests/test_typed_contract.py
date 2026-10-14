from __future__ import annotations
import asyncio,contextlib,copy,hashlib,importlib.util,io,json,os,sys,tempfile,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from typed19.dependency import activate,SOURCE,identity
activate();sys.path.insert(0,str(SOURCE/'tests'))
from typed19.answers import parse,bind,legacy_refusal_signal,SCHEMA
from typed19.workflow import ServerTokenizer,compile_packets,instantiate,task_text,lcp
from semantic19.dataset import compile_record
from joint18v2.common import load


def fixture():
    return {'id':'SYNTHETIC_TEST','question':'DO_NOT_SEND_OVERALL', 'answerable':True,
      'answer':'GOLD_SECRET','answer_aliases':[],
      'paragraphs':[{'idx':0,'title':'City','paragraph_text':'Paris is a city. Sam lives in Paris.','is_supporting':True},
        {'idx':1,'title':'Other','paragraph_text':'Other material. '*15,'is_supporting':False}],
      'question_decomposition':[{'id':77,'question':'Which city?','answer':'Paris','paragraph_support_idx':0},
        {'id':88,'question':'Which person?','answer':'Sam','paragraph_support_idx':0},
        {'id':99,'question':'Where does #2 live in #1?','answer':'Paris','paragraph_support_idx':0}]}

class FakeTokenizer:
    markers={'<|im_start|>':151644,'<|im_end|>':151645,'system':900001,'user':900002,'assistant':900003}
    def __init__(self):self.calls=self.decode_calls=self.chat_calls=0;self.first_blocks=set();self.sent=[]
    async def encode(self,text):
        self.calls+=1;out=[];i=0
        while i<len(text):
            match=next((x for x in self.markers if text.startswith(x,i)),None)
            if match:out.append(self.markers[match]);i+=len(match)
            else:out.append(ord(text[i])+100);i+=1
        return out or [132]
    async def decode(self,tokens):
        self.decode_calls+=1;r={v:k for k,v in self.markers.items()}
        return ''.join(r[t] if t in r else chr(t-100) for t in tokens)
    async def chat_tokens(self,system,user):
        self.chat_calls+=1;self.sent.append((system,user))
        return await self.encode('<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n')

class AnswerTests(unittest.TestCase):
    def test_valid_json(self):self.assertEqual(parse('{"answer":"Paris"}')['value'],'Paris')
    def test_unknown_status(self):self.assertEqual(parse('{"answer":"UNKNOWN"}')['status'],'abstain')
    def test_unknown_never_entity(self):self.assertIsNone(parse('{"answer":"UNKNOWN"}')['value'])
    def test_wrong_country_not_corrected(self):self.assertEqual(parse('{"answer":"Armenia"}')['value'],'Armenia')
    def test_prose_invalid(self):self.assertEqual(parse('The answer is Paris')['status'],'format_error')
    def test_old_markup_not_silently_repaired(self):self.assertEqual(parse('<answer>Paris</answer>')['status'],'format_error')
    def test_extra_field_invalid(self):self.assertFalse(parse('{"answer":"Paris","evidence":"x"}')['schema_ok'])
    def test_duplicate_invalid(self):self.assertFalse(parse('{"answer":"Paris","answer":"Rome"}')['schema_ok'])
    def test_nonstring_invalid(self):self.assertFalse(parse('{"answer":42}')['schema_ok'])
    def test_null_invalid(self):self.assertFalse(parse('{"answer":null}')['schema_ok'])
    def test_empty_invalid(self):self.assertFalse(parse('{"answer":"  "}')['schema_ok'])
    def test_truncated_invalid(self):self.assertFalse(parse('{"answer":"Paris"','length')['schema_ok'])
    def test_complete_at_length_flagged(self):self.assertFalse(parse('{"answer":"Paris"}','length')['schema_ok'])
    def test_trailing_text_invalid(self):self.assertFalse(parse('{"answer":"Paris"} goodbye')['schema_ok'])
    def test_multiline_invalid(self):self.assertFalse(parse('{"answer":"Paris\\nAnother paragraph"}')['schema_ok'])
    def test_unknown_extra_text_invalid(self):self.assertFalse(parse('{"answer":"UNKNOWN because no evidence"}')['schema_ok'])
    def test_hash_matches_raw(self):self.assertEqual(parse(' raw ')['raw_response_sha256'],hashlib.sha256(b' raw ').hexdigest())
    @unittest.skip('Historical-result-only audit omitted: this delivery deliberately excludes received_results.json')
    def test_refusal_signal_not_quality_rewrite(self):
        report=load(ROOT/'analysis/received_results.json')
        counts={r['cell']:sum(legacy_refusal_signal(x['raw_response']) for x in report['raw_responses'] if x['cell']==r['cell']) for r in report['rows']}
        self.assertEqual(counts,{'P0C0':6,'P1C0':3,'P0C1':7,'P1C1':0})
        self.assertTrue(all(r['semantic_quality_unchanged']['final_em']==0 for r in report['rows']))
    def test_solution_unknown_identified(self):self.assertTrue(legacy_refusal_signal('<solution>UNKNOWN</solution>'))
    def test_nonrefusal_mention(self):self.assertFalse(legacy_refusal_signal('This UNKNOWN claim is false'))

class BindingTests(unittest.TestCase):
    def test_real_values_only(self):
        text,used=bind('What is #1?',{0:{'text':'{"answer":"Paris"}','finish_reason':'stop'}})
        self.assertEqual(text,'What is (Paris)?');self.assertEqual(used['1']['response'],'{"answer":"Paris"}')
    def test_invalid_not_prose(self):
        text,used=bind('What is #1?',{0:{'text':'Actual parent outputs: bad long prose'}})
        self.assertNotIn('bad long prose',text);self.assertIn('UNAVAILABLE_STEP_1',text)
        self.assertIsNone(used['1']['bound_value']);self.assertEqual(used['1']['parse_status'],'format_error')
    def test_abstention_status_not_error(self):
        text,used=bind('What is #1?',{0:{'text':'{"answer":"UNKNOWN"}'}})
        self.assertEqual(used['1']['parse_status'],'abstain');self.assertIn('UNAVAILABLE_STEP_1',text)
    def test_missing_parent_fails(self):
        with self.assertRaises(ValueError):bind('What is #1?',{})
    def test_repeated_reference(self):
        text,used=bind('#1 and #1',{0:{'text':'{"answer":"Paris"}'}});self.assertEqual(len(used),1)
    def test_raw_not_task(self):
        text,u=bind('#1',{0:{'text':'NARRATIVE_SECRET'}})
        self.assertNotIn('NARRATIVE_SECRET',task_text(text,u))

class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def setup_fixture(self):
        c=load(ROOT/'configs/semantic.json');c['document_tokens_budget']=256
        row,_=compile_record(fixture());t=FakeTokenizer();pack,info=await compile_packets([row],t,c)
        ns,mat,proof=await instantiate(pack,t,c,1);return c,t,pack,ns,mat,proof
    async def test_canonical_tokens_not_fragments(self):
        c,t,p,ns,mat,proof=await self.setup_fixture();await mat('w0000/step1',ns)
        q=ns['w0000/step1'];sys_,user=t.sent[-1];self.assertEqual(q['tokens'],await t.chat_tokens(sys_,user))
        self.assertTrue(q['canonical_chat_tokens_used'])
    async def test_no_overall_or_gold(self):
        c,t,p,ns,mat,proof=await self.setup_fixture();await mat('w0000/step1',ns)
        s=json.dumps(t.sent);self.assertNotIn('GOLD_SECRET',s);self.assertNotIn('DO_NOT_SEND_OVERALL',s)
    async def test_dynamic_valid_parent(self):
        c,t,p,ns,mat,proof=await self.setup_fixture()
        ns['w0000/step1']['stream']={'text':'{"answer":"Paris"}'};ns['w0000/step2']['stream']={'text':'{"answer":"Sam"}'}
        await mat('w0000/step3',ns)
        self.assertEqual(ns['w0000/step3']['resolved_question'],'Where does (Sam) live in (Paris)?')
        self.assertEqual(len(ns['w0000/step3']['parent_response_bindings']),2)
    async def test_dynamic_invalid_status_retained(self):
        c,t,p,ns,mat,proof=await self.setup_fixture()
        ns['w0000/step1']['stream']={'text':'RAW_NARRATIVE_SECRET'};ns['w0000/step2']['stream']={'text':'{"answer":"UNKNOWN"}'}
        await mat('w0000/step3',ns)
        self.assertNotIn('RAW_NARRATIVE_SECRET',t.sent[-1][1]);self.assertIn('format_error',t.sent[-1][1]);self.assertIn('abstain',t.sent[-1][1])
    async def test_new_first_block_each_condition(self):
        c,t,p,ns,mat,pr1=await self.setup_fixture();_,_,pr2=await instantiate(p,t,c,1)
        self.assertNotEqual(pr1['first_cache_block_sha256'],pr2['first_cache_block_sha256'])
    async def test_exact_namespace_collision_stops(self):
        c,t,p,ns,mat,pr=await self.setup_fixture()
        from types import SimpleNamespace
        with patch('typed19.workflow.uuid.uuid4',return_value=SimpleNamespace(hex='a'*32)):
            await instantiate(p,t,c,1)
            with self.assertRaises(RuntimeError):await instantiate(p,t,c,1)
    async def test_dynamic_overflow_not_clipped(self):
        c,t,p,ns,mat,proof=await self.setup_fixture()
        ns['w0000/step1']['stream']={'text':json.dumps({'answer':'z'*5000})};ns['w0000/step2']['stream']={'text':'{"answer":"Sam"}'}
        with self.assertRaisesRegex(ValueError,'dynamic canonical'):await mat('w0000/step3',ns)
    async def test_dynamic_prefix_is_verified(self):
        c,t,p,ns,mat,proof=await self.setup_fixture();original=t.chat_tokens
        async def wrong(*args):return [88888]+(await original(*args))[1:]
        t.chat_tokens=wrong
        with self.assertRaisesRegex(RuntimeError,'canonical prefix'):await mat('w0000/step1',ns)
    async def test_native_payload(self):
        import httpx
        seen=[]
        def handler(req):seen.append(json.loads(req.content));return httpx.Response(200,json={'tokens':[7,8]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            t=ServerTokenizer(client,'fixture');self.assertEqual(await t.chat_tokens('system','user'),[7,8])
        self.assertTrue(seen[0]['add_generation_prompt']);self.assertFalse(seen[0]['add_special_tokens']);self.assertNotIn('prompt',seen[0])
    async def test_malformed_token_list_fails(self):
        import httpx
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'tokens':[True]}))) as client:
            with self.assertRaises(ValueError):await ServerTokenizer(client,'fixture').chat_tokens('a','b')

class QualityTests(unittest.TestCase):
    def test_no_labels_before_end(self):
        from typed19.quality import evaluate
        with self.assertRaises(RuntimeError):evaluate({'a':{'state':'running'}},'/missing',1.)
    def test_format_valid_wrong_answer_kept_wrong(self):
        from typed19.quality import evaluate
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);gp=d/'evaluation_gold.jsonl';gp.write_text(json.dumps({'id':'x','answer':'Paris','answer_aliases':[],'subanswers':['Paris']})+'\n')
            (d/'manifest.json').write_text(json.dumps({'files':{gp.name:hashlib.sha256(gp.read_bytes()).hexdigest()}}))
            n={'id':'w/s','wid':'w','state':'done','parents':[],'step_position':0,'dataset_id':'x','graph_shape':'-',
              'stream':{'text':'{"answer":"Rome"}','finish_reason':'stop','usage':{'completion_tokens':6}}}
            with contextlib.redirect_stdout(io.StringIO()):r=evaluate({'w/s':n},d,1.)
            self.assertEqual(r['semantic_quality']['final_em'],0);self.assertEqual(r['semantic_quality']['final_format_rate'],1.)
    def test_unknown_not_filtered(self):
        self.assertEqual(parse('{"answer":"UNKNOWN"}')['status'],'abstain')

class RealCodeFakeEngine(unittest.IsolatedAsyncioTestCase):
    async def scenario(self,method):
        import httpx
        from fake_native import modules,FakeGroup,hashes
        from joint18v2.cache import NativeCacheObserver
        import joint18v2.client as C
        from typed19.transport import stream
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);mods,sm,Engine=modules()
            with patch.dict(sys.modules,mods),patch.dict(os.environ,{'SPRA_E1_ROOT':td,'POD_NAME':'spra-vllm-0','NVIDIA_VISIBLE_DEVICES':'MOCK-NOT-GPU'}):
                spec=importlib.util.spec_from_file_location('typedhook'+uuid.uuid4().hex,SOURCE/'runtime/joint18v2_hook.py')
                hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
                hook.capabilities=lambda:{'scheduler_source_sha256':'fixture','imported_version':'0.6.3','distribution_version':'0.6.3'}
                hook.NativeCacheObserver=lambda mgr,b:NativeCacheObserver(mgr,b,hashes)
                hook.install();sched=sm.Scheduler();engine=Engine(sched);sched._v2_writer.drain()
                initial=[load(path/'joint18v2/spra-vllm-0.status.json')]
                cfg=load(ROOT/'configs/semantic.json');cfg.update(replicas=1,epoch_s=.04,ready_coalesce_s=.001,planning_soft_limit_s=.02,
                    commit_reserve_s=.003,document_tokens_budget=256,max_output_tokens=8,initial_output_estimate=2,
                    snapshot_max_age_s=2.,cache_evidence_max_age_s=2.,condition_wall_s=10,initial_expected_apply_delay_ms=2.)
                public,_=compile_record(fixture());tok=FakeTokenizer();pack,_=await compile_packets([public],tok,cfg)
                ns,mat,proof=await instantiate(pack,tok,cfg,1)
                for n in ns.values():n['arrival_offset']=0.
                sent=[]
                class Stream(httpx.AsyncByteStream):
                    def __init__(self,g):self.g=g
                    async def __aiter__(self):
                        while self.g.done<1:await asyncio.sleep(.001)
                        yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'{"answer":'}]})+'\n\n').encode()
                        while self.g.done<2:await asyncio.sleep(.001)
                        yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'"Paris"}','finish_reason':'stop'}]})+'\n\n').encode()
                        yield ('data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':len(self.g.tokens),'completion_tokens':2}})+'\n\n').encode()
                        yield b'data: [DONE]\n\n'
                async def handler(req):
                    payload=json.loads(req.content);sent.append(payload)
                    self.assertEqual(payload['guided_json'],SCHEMA);self.assertNotIn('min_tokens',payload);self.assertNotIn('ignore_eos',payload)
                    g=FakeGroup('r'+uuid.uuid4().hex,payload['prompt'],2);sched.add_seq_group(g)
                    return httpx.Response(200,stream=Stream(g))
                alive=True
                async def pump():
                    while alive:
                        if sched.running or sched.waiting:await engine.step_async()
                        await asyncio.sleep(.003)
                task=asyncio.create_task(pump());model=None if method=='A' else {'models':{'0':{'coef':[1.,0.,0.,0.,0.,0.]}}}
                try:
                    with ThreadPoolExecutor(1) as executor,patch.object(C,'stream',stream):
                        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,contextlib.nullcontext():
                            with contextlib.redirect_stdout(io.StringIO()):s,e=await C.execute(client,path,initial,cfg,ns,path/'RUN',method,model,executor=executor if model else None,materializer=mat)
                    self.assertEqual(s['request_correlation'],'3/3');self.assertEqual(s['native_next_batch_matches'],s['native_next_batch_checked'])
                    self.assertTrue(s['control_time_match']);self.assertEqual(len(sent),3)
                    self.assertTrue(all(n['answer_contract']['status']=='answer' for n in ns.values()))
                    self.assertNotIn('GOLD_SECRET',json.dumps(proof));self.assertIn('(Paris)',ns['w0000/step3']['resolved_question'])
                finally:
                    alive=False
                    try:await task
                    finally:sched._v2_stop.set();await asyncio.sleep(.04);sched._v2_writer.close()
    async def test_A(self):await self.scenario('A')
    async def test_D(self):await self.scenario('D')
    async def test_B(self):await self.scenario('B')
    async def test_Decoupled(self):await self.scenario('Decoupled')
    async def test_Joint(self):await self.scenario('Joint')

if __name__=='__main__':unittest.main()
