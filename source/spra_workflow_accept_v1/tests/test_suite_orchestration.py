"""Runner transaction/resume tests with an explicit CPU-only mocked executor.

These synthetic results are deleted with TemporaryDirectory and are never shipped
as performance evidence. Real client/hook invocation is tested separately.
"""
from __future__ import annotations
import contextlib
import copy
import io
import json
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
from concurrent.futures import ThreadPoolExecutor
from bootstrap import ORIGINAL,MODEL,PRIORS,MANIFEST
from synthetic import fixture,FakeTokenizer
from wfsched.common import atomic,load,sha
from wfsched import runner

class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from semantic19.dataset import compile_record
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.repo=self.base/'source';self.runtime=self.base/'runtime';self.runtime.mkdir()
        self.dataset=self.repo/'spra_stage19_musique_joint_v1/dataset/compiled';self.dataset.mkdir(parents=True)
        self.records=[];gold=[]
        for i in range(64):
            raw=fixture();raw['id']=f'SYNTHETIC_SUITE_{i}'
            p,g=compile_record(raw);self.records.append(p);gold.append(g)
        gp=self.dataset/'evaluation_gold.jsonl';gp.write_text(''.join(json.dumps(x)+'\n' for x in gold))
        atomic(self.dataset/'manifest.json',{'files':{gp.name:sha(gp)}})
        self.states=[]
        for r in range(4):
            s={k:'SYNTHETIC' for k in runner.HARDWARE_FIELDS}
            s.update(pod=f'spra-vllm-{r}',hook_source_sha256=ORIGINAL.HOOK,pid=100+r,
                     engine_boot_id=f'MOCK-{r}',nvidia_visible_devices=f'MOCK-{r}')
            self.states.append(s)
        self.identity={'source_files_verified':48,'test_only':True}
        self.args=SimpleNamespace(repo=self.repo,runtime=self.runtime,out=self.base/'results',suite='smoke',
              resume=False,replicas=None,max_conditions=2,confirm_gpu_use=True)
        self.calls=0;self.fail=False
        self.stack=contextlib.ExitStack()
        for target,value in (
            ('wfsched.runner.freeze',lambda *a,**k:{'archive_sha256':'SYNTHETIC'}),
            ('wfsched.runner.inspect',lambda *a,**k:self.identity),
            ('wfsched.runner.load_original',lambda *a:ORIGINAL),
            ('wfsched.runner.load_profile',lambda *a:(None,MODEL,PRIORS,MANIFEST)),
            ('semantic19.dataset.load_public',lambda *a:(self.records,{})),
            ('typed19.workflow.ServerTokenizer',lambda *a:FakeTokenizer()),
            ('wfsched.runner.all_gate',AsyncMock(side_effect=lambda *a:copy.deepcopy(self.states))),
            ('wfsched.runner.warmup',AsyncMock()),
            ('joint18v2.client.same_engines',lambda *a:None),
            ('joint18v2.client.wait_drained',AsyncMock(return_value=True)),
            ('joint18v2.client.execute',self.execute),
            ('wfsched.runner.ProcessPoolExecutor',lambda *a,**k:ThreadPoolExecutor(1)),
        ):self.stack.enter_context(patch(target,new=value))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.stack.close();self.tmp.cleanup()

    async def execute(self,client,runtime,initial,cfg,nodes,folder,method,model,**kwargs):
        self.calls+=1
        if self.fail:raise RuntimeError('synthetic execution failure retained')
        self.assertIsNone(kwargs['force_caps']);self.assertEqual(cfg['base_cap'],4096)
        self.assertEqual(method,'D' if cfg['wf_public_method'].startswith('DAG') else 'A')
        for i,(k,n) in enumerate(nodes.items()):
            n['arrival']=100.+n['arrival_offset']
            n['ready_at']=max([n['arrival']]+[nodes[p]['finish'] for p in n['parents']])
            await kwargs['materializer'](k,nodes)
            n.update(start=n['ready_at']+.01,finish=n['ready_at']+.11,state='done',replica=i%cfg['replicas'])
            n['stream']={'text':'{"answer":"Paris"}','finish_reason':'stop','usage':{'completion_tokens':2}}
        by={}
        for n in nodes.values():by.setdefault(n['wid'],[]).append(n)
        j=[1000*(max(n['finish'] for n in ns)-min(n['arrival'] for n in ns)) for ns in by.values()]
        s={'condition_valid':True,'budget_invariants':True,'prefill_caps_applied':[4096],
           'budget_switches_total':0,'native_next_batch_matches':len(nodes),'native_next_batch_checked':len(nodes),
           'requests':len(nodes),'request_correlation':f'{len(nodes)}/{len(nodes)}','workflows':len(by),
           'control_time_match':True,'actual_commit_route_mismatches':0,'mean_jct_ms':statistics.mean(j),
           'p95_jct_ms':max(j),'output_tokens':2*len(nodes),'cohort_duration_s':10.,
           'online_search_calls':1 if method=='D' else 0}
        folder.mkdir(parents=True)
        atomic(folder/'nodes.json',nodes)
        atomic(folder/'controls.json',[{'caps':[4096]*cfg['replicas'],'totals':[4096]*cfg['replicas']}])
        p={'action':{'order':[]},'evaluations':[]}
        if cfg['wf_public_method']=='DAG-Lite':p.update(wf_worker_path='no_budget_candidates')
        (folder/'plans.jsonl').write_text(json.dumps(p)+'\n')
        return s,[]

    async def first(self):
        await runner.run(self.args)
        self.assertEqual(load(self.args.out/'suite_status.json')['status'],'paused')
        self.assertEqual(self.calls,2)

    async def test_pause_resume_all_methods(self):
        await self.first();self.args.resume=True;self.args.max_conditions=None
        await runner.run(self.args)
        p,rows,status=runner.read_run(self.args.out)
        self.assertEqual(status,'complete');self.assertEqual(len(rows),5);self.assertEqual(self.calls,5)
        self.assertTrue(load(self.args.out/'analysis/RESULT.json')['valid_complete_suite'])
        await runner.run(self.args);self.assertEqual(self.calls,5)

    async def test_existing_output_cannot_overwrite(self):
        await self.first()
        with self.assertRaisesRegex(ValueError,'output exists'):await runner.run(self.args)
        self.assertEqual(self.calls,2)

    async def test_protocol_tamper(self):
        await self.first();p=load(self.args.out/'protocol.json');p['run_uuid']='changed';atomic(self.args.out/'protocol.json',p)
        self.args.resume=True
        with self.assertRaisesRegex(ValueError,'protocol changed'):await runner.run(self.args)

    async def test_raw_time_tamper(self):
        await self.first();row=load(self.args.out/'index.json')[0]
        f=self.args.out/row['folder']/'nodes.json';ns=load(f);next(iter(ns.values()))['finish']+=1;atomic(f,ns)
        self.args.resume=True
        with self.assertRaisesRegex(ValueError,'file set/hash changed'):await runner.run(self.args)

    async def test_index_summary_tamper(self):
        await self.first();rows=load(self.args.out/'index.json');rows[0]['summary']['mean_jct_ms']=1;atomic(self.args.out/'index.json',rows)
        self.args.resume=True
        with self.assertRaisesRegex(ValueError,'index summary differs'):await runner.run(self.args)

    async def test_failed_attempt_not_retried(self):
        self.fail=True
        with self.assertRaisesRegex(RuntimeError,'synthetic execution failure'):await runner.run(self.args)
        self.assertEqual(load(self.args.out/'suite_status.json')['status'],'failed')
        self.assertEqual(len(list((self.args.out/'attempts').glob('*/status.json'))),1)
        self.args.resume=True;self.fail=False
        with self.assertRaisesRegex(ValueError,'failed/interrupted'):await runner.run(self.args)
        self.assertEqual(self.calls,1)

    async def test_runtime_change_stops_resume(self):
        await self.first();self.args.resume=True;self.args.runtime=self.base/'OTHER'
        with self.assertRaisesRegex(ValueError,'source/profile/runtime changed'):await runner.run(self.args)

    async def test_replica_filter_keeps_full_plan(self):
        self.args.suite='grid';self.args.replicas=[2]
        await self.first();p,rows,status=runner.read_run(self.args.out)
        self.assertEqual(len(p['plan']['conditions']),640)
        self.assertEqual({r['replicas'] for r in rows},{2})
        self.assertEqual(status,'paused')
