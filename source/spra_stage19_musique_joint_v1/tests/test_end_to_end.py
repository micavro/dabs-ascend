"""Actual v2 hook + controller + HTTP streaming + six-node DAG, on a fake native API.
No GPU, real vLLM import, external socket, live runtime directory or Kubernetes call.
"""
from __future__ import annotations
import asyncio,contextlib,importlib.util,io,json,os,tempfile,time,unittest,uuid,sys
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from fake_native import modules,FakeGroup,hashes
from joint18v2.common import load,token_hash,atomic
from joint18v2.cache import NativeCacheObserver
from joint18v2.client import execute,ForcedMechanism
from joint18v2.workload import build
ROOT=Path(__file__).resolve().parents[1]

class EndToEnd(unittest.IsolatedAsyncioTestCase):
    async def one(self,method,fail=False):
        import httpx
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);mods,sm,Engine=modules()
            with patch.dict(sys.modules,mods),patch.dict(os.environ,{'SPRA_E1_ROOT':temp,'POD_NAME':'spra-vllm-0','NVIDIA_VISIBLE_DEVICES':'FAKE-NOT-GPU'}):
                spec=importlib.util.spec_from_file_location('fakehook_'+uuid.uuid4().hex,ROOT/'runtime/joint18v2_hook.py');hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
                hook.capabilities=lambda:{'scheduler_source_sha256':'fake','imported_version':'0.6.3','distribution_version':'0.6.3'}
                hook.NativeCacheObserver=lambda mgr,b:NativeCacheObserver(mgr,b,hashes)
                hook.install();s=sm.Scheduler();e=Engine(s);s._v2_writer.drain()
                initial=[load(path/'joint18v2/spra-vllm-0.status.json')]
                cfg=load(ROOT/'configs/development.json');cfg.update(replicas=1,workflow_count=1,epoch_s=.02,ready_coalesce_s=.001,planning_soft_limit_s=.008,commit_reserve_s=.004,snapshot_max_age_s=1.,cache_evidence_max_age_s=1.,condition_wall_s=10,initial_expected_apply_delay_ms=2.)
                ns=build(cfg,list(range(100,140)),555,[110]*8)
                for i,n in enumerate(ns.values()):
                    toks=list(range(100,132))+list(range(200+i*16,216+i*16))
                    n.update(tokens=toks,prompt=48,output=2,hash=token_hash(toks),share=32,prefix_hash=token_hash(toks[:32]),arrival_offset=0.)
                class Stream(httpx.AsyncByteStream):
                    def __init__(self,g):self.g=g
                    async def __aiter__(self):
                        seen=0
                        while self.g.done<self.g.output:
                            if self.g.done>seen:seen=self.g.done;yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'x'}]})+'\n\n').encode()
                            await asyncio.sleep(.001)
                        # Deterministic fixture ordering: wait for the idle native-cache
                        # observer to publish completed prefix blocks before ending the fake
                        # parent response. Real runtime checks remain unchanged.
                        await asyncio.sleep(.035)
                        yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'x'}]})+'\n\n').encode()
                        yield ('data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':48,'completion_tokens':2}})+'\n\n').encode()
                        yield b'data: [DONE]\n\n'
                async def handler(req):
                    if fail:return httpx.Response(500,json={'error':'fixture failure'})
                    payload=json.loads(req.content);g=FakeGroup('r'+uuid.uuid4().hex,payload['prompt'],payload['max_tokens']);s.add_seq_group(g);return httpx.Response(200,stream=Stream(g))
                alive=True
                async def pump():
                    while alive:
                        if s.running or s.waiting:await e.step_async()
                        await asyncio.sleep(.002)
                task=asyncio.create_task(pump())
                m=None if method=='Forced' else {'model_id':'CPU-fixture','models':{'0':{'coef':[1,0,0,0,0,0]}}}
                try:
                    with ThreadPoolExecutor(1) as ex:
                        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                            with contextlib.redirect_stdout(io.StringIO()):
                                if fail:
                                    with self.assertRaises(Exception):await execute(client,path,initial,cfg,ns,path/'RUN',method,m,executor=ex)
                                else:summary,events=await execute(client,path,initial,cfg,ns,path/'RUN',method,m,executor=None if method=='Forced' else ex,mechanism=ForcedMechanism('concentrate',[256]) if method=='Forced' else None)
                    if fail:
                        self.assertTrue((path/'RUN/backend.jsonl').is_file());self.assertEqual(load(path/'RUN/status.json')['execution'],'failed')
                    else:
                        self.assertEqual(summary['request_correlation'],'6/6');self.assertEqual(summary['output_tokens'],12);self.assertTrue(summary['budget_invariants']);self.assertTrue(summary['control_time_match']);self.assertEqual(summary['root_to_branch_cache_links_verified'],5, json.dumps(load(path/'RUN/cache_chain_evidence.json')))
                        self.assertEqual(summary['native_next_batch_matches'],summary['native_next_batch_checked'])
                        self.assertTrue(any(x['event']=='model_input' and x['skipped_query_tokens']>0 for x in events))
                        
                        if method=='Forced':self.assertEqual(summary['online_search_calls'],0);self.assertEqual(summary['execution_mode'],'preset_complete_action_no_online_search')
                        self.assertGreater(summary['mean_request_ttft_ms'],0);self.assertEqual(summary['request_ttft_ms_samples'],6);self.assertTrue((path/'RUN/request_latency.json').is_file());self.assertTrue(any(x['event']=='output_progress' for x in events));self.assertTrue((path/'RUN/plans.jsonl').is_file());self.assertGreaterEqual(summary['ready_discovery_ms']['mean'],0.0)
                finally:
                    alive=False
                    try:await task
                    finally:s._v2_stop.set();await asyncio.sleep(.04);s._v2_writer.close()
    async def test_Forced_no_model_whole_DAG(self):await self.one('Forced')
    async def test_A_whole_DAG(self):await self.one('A')
    async def test_D_whole_DAG(self):await self.one('D')
    async def test_B_whole_DAG(self):await self.one('B')
    async def test_Decoupled_whole_DAG(self):await self.one('Decoupled')
    async def test_Joint_whole_DAG(self):await self.one('Joint')
    async def test_failure_keeps_source_trace(self):await self.one('A',True)
