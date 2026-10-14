"""Actual retained executor/hook on an API-shaped CPU fake; NEVER a GPU benchmark."""
from __future__ import annotations
import asyncio
import contextlib
import copy
import importlib.util
import io
import json
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from bootstrap import CFG,MODEL,SOURCE
from synthetic import fixture,FakeTokenizer
from wfsched.binding import bindings
from wfsched.common import atomic,load,sha,METHODS
from wfsched.audit import validate_condition,inactive_offsets,inactive_proof
from wfsched.worker import ready,process_plan
from wfsched.bench import snapshot

class NativePath(unittest.IsolatedAsyncioTestCase):
    async def scenario(self,method,R):
        import httpx
        from fake_native import modules,FakeGroup,hashes
        from joint18v2.cache import NativeCacheObserver
        from semantic19.dataset import compile_record
        from typed19.workflow import compile_packets,instantiate
        from typed19.answers import SCHEMA
        from typed19.quality import evaluate as quality_evaluate
        import joint18v2.client as C
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);mods,sm,Engine=modules();scheds=[];engines=[]
            with patch.dict(sys.modules,mods),patch.dict(os.environ,{'SPRA_E1_ROOT':td,
                  'POD_NAME':'spra-vllm-0','NVIDIA_VISIBLE_DEVICES':'MOCK-NOT-GPU'}):
                spec=importlib.util.spec_from_file_location('TEST_CPU_HOOK_'+uuid.uuid4().hex,SOURCE/'runtime/joint18v2_hook.py')
                hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
                hook.capabilities=lambda:{'scheduler_source_sha256':'SYNTHETIC','imported_version':'0.6.3','distribution_version':'0.6.3'}
                hook.NativeCacheObserver=lambda mgr,b:NativeCacheObserver(mgr,b,hashes)
                with contextlib.redirect_stdout(io.StringIO()):
                    hook.install()
                    for r in range(4):
                        os.environ['POD_NAME']=f'spra-vllm-{r}';os.environ['NVIDIA_VISIBLE_DEVICES']=f'SYNTHETIC-CPU-{r}'
                        s=sm.Scheduler();scheds.append(s);engines.append(Engine(s));s._v2_writer.drain()
                initial=[load(path/'joint18v2'/f'spra-vllm-{r}.status.json') for r in range(R)]
                cfg=copy.deepcopy(CFG)
                cfg.update(replicas=R,workflow_count=2,epoch_s=.04,ready_coalesce_s=.001,
                  planning_soft_limit_s=.025,commit_reserve_s=.003,max_output_tokens=8,initial_output_estimate=2,
                  snapshot_max_age_s=2.,cache_evidence_max_age_s=2.,condition_wall_s=15,
                  document_tokens_budget=256,initial_expected_apply_delay_ms=2.,
                  wf_public_method=method,wf_routing_policy=method,wf_router_seed=88)
                records=[];gold=[]
                for i in range(2):
                    raw=fixture();raw['id']=f'SYNTHETIC_{i}';raw['paragraphs'][0]['title']=f'City{i}'
                    p,g=compile_record(raw);records.append(p);gold.append(g)
                dataset=path/'test_dataset';dataset.mkdir()
                gp=dataset/'evaluation_gold.jsonl';gp.write_text(''.join(json.dumps(x)+'\n' for x in gold))
                atomic(dataset/'manifest.json',{'files':{gp.name:sha(gp)}})
                tok=FakeTokenizer();pack,_=await compile_packets(records,tok,cfg)
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
                    r=req.url.port-8000;self.assertLess(r,R)
                    payload=json.loads(req.content);sent.append((r,payload))
                    self.assertEqual(payload['guided_json'],SCHEMA)
                    self.assertNotIn('min_tokens',payload);self.assertNotIn('ignore_eos',payload)
                    g=FakeGroup(uuid.uuid4().hex,payload['prompt'],2);scheds[r].add_seq_group(g)
                    return httpx.Response(200,stream=Stream(g))
                alive=True
                async def pump():
                    while alive:
                        for s,e in zip(scheds,engines):
                            if s.running or s.waiting:await e.step_async()
                        await asyncio.sleep(.003)
                task=asyncio.create_task(pump());off=inactive_offsets(path,R)
                try:
                    with ThreadPoolExecutor(1) as pool:
                        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                            with bindings(method),contextlib.redirect_stdout(io.StringIO()):
                                native='D' if method.startswith('DAG') else 'A'
                                s,events=await C.execute(client,path,initial,cfg,ns,path/'RAW',native,MODEL,
                                    executor=pool if native=='D' else None,materializer=mat)
                                q=quality_evaluate(ns,dataset,s['cohort_duration_s']);s.update(q)
                    self.assertEqual(len(sent),6);self.assertEqual(s['request_correlation'],'6/6')
                    self.assertTrue(s['control_time_match']);self.assertEqual(s['budget_switches_total'],0)
                    self.assertEqual(s['prefill_caps_applied'],[4096]);self.assertTrue(inactive_proof(path,off)['pass'])
                    audit=validate_condition(path/'RAW',s,ns,{'replicas':R,'workflow_count':2,'method':method})
                    self.assertTrue(audit['native_fixed4096'])
                    if method=='RR':self.assertEqual([r for r,_ in sent],[i%R for i in range(6)])
                    self.assertIn('(Paris)',ns['w0000/step3']['resolved_question'])
                    self.assertNotIn('GOLD_SECRET',json.dumps([p for r,p in sent]))
                finally:
                    alive=False
                    try:await task
                    finally:
                        for s in scheds:s._v2_stop.set()
                        await asyncio.sleep(.05)
                        for s in scheds:s._v2_writer.close()

for m in METHODS:
    for R in (1,2,3,4):
        async def test(self,m=m,R=R):await self.scenario(m,R)
        setattr(NativePath,'test_'+m.replace('-','_')+'_R'+str(R),test)

class SpawnPath(unittest.TestCase):
    def test_both_worker_versions_in_actual_spawn(self):
        with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as pool:
            pid=pool.submit(ready).result(timeout=15);self.assertNotEqual(pid,os.getpid())
            for method in ('DAG','DAG-Lite'):
                c=dict(CFG,wf_public_method=method)
                ans=pool.submit(process_plan,snapshot(4),MODEL,c,'D',time.monotonic()+5).result(timeout=15)
                self.assertEqual(ans['wf_public_method'],method)
                self.assertEqual(ans['action']['caps'],[4096]*4)
