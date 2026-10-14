from __future__ import annotations
import asyncio, contextlib, copy, importlib.util, io, json, os, sys, tempfile, time, unittest, uuid
from pathlib import Path
from unittest.mock import patch
from joint18v2.lifetime_probe import finite_window_evidence, reuse_cache_chain
from joint18v2.common import VERSION, load, atomic, token_hash
ROOT = Path(__file__).resolve().parents[1]


def evidence_fixture():
    start=600_000_000; expiry=800_000_000
    rec={'plan_id':'pulse','snapshot_ns':start,'not_before_ns':start+1_000_000,
         'action_expires_ns':expiry,'caps':[256],'orders':[[]],'totals':[4096]}
    trigger={'admit':['root'],'control':rec,'continuation_cap':1024}
    def row(at,cap,phase,p=256):
        return {'plan_id':'pulse','plan_reason':'lease_ok','execution_phase':phase,
                'begin_ns':at,'effective_ns':at,'snapshot_ns':start,'action_expires_ns':expiry,
                'prefill_cap':cap,'total':4096,'prefill_logical_tokens':p,'decode_tokens':1,
                'remaining_prefill_in_before':2048,'scheduled':[{'tokens':p}]}
    rows=[row(620_000_000,256,'current_action'), row(810_000_000,1024,'common_continuation'),
          row(1_000_000_000,1024,'common_continuation',0)]
    flushes=[dict(rec,cause='publication',issued_ns=start+1_000_000),
             dict(rec,cause='heartbeat',issued_ns=950_000_000)]
    return trigger,[rec],flushes,rows,{'root':{'ready_at':.5}}

class EvidenceTests(unittest.TestCase):
    def check_modified(self,index,change,reason):
        args=list(evidence_fixture());change(args[index]);out=finite_window_evidence(*args)
        self.assertFalse(out['passed']);self.assertIn(reason,out['failure_reasons'])
    def test_valid_one_shot(self):self.assertTrue(finite_window_evidence(*evidence_fixture())['passed'])
    def test_absent_trigger_does_not_pass(self):
        args=list(evidence_fixture());args[0]=None
        self.assertFalse(finite_window_evidence(*args)['passed'])
    def test_all_continuation_not_accepted(self):
        self.check_modified(3,lambda xs:xs.pop(0),'pulse_has_scheduled_prefill')
    def test_bootstrap_frames_cannot_substitute(self):
        self.check_modified(3,lambda xs:xs[0].update(plan_id='bootstrap'),'pulse_has_scheduled_prefill')
    def test_idle_frames_cannot_substitute(self):
        self.check_modified(3,lambda xs:xs[0].update(scheduled=[],prefill_logical_tokens=0),'pulse_has_scheduled_prefill')
    def test_unbounded_window_rejected(self):
        self.check_modified(0,lambda x:x['control'].update(action_expires_ns=900_000_000),'window_positive_and_at_most_200ms')
    def test_started_before_ready_rejected(self):
        self.check_modified(4,lambda x:x['root'].update(ready_at=.7),'no_pre_admission_setup_pulse')
    def test_cap_after_expiry_rejected(self):
        self.check_modified(3,lambda xs:xs[-1].update(prefill_cap=256),'expired_cap_and_phase_match')
    def test_renewed_deadline_rejected(self):
        self.check_modified(2,lambda xs:xs[-1].update(action_expires_ns=1_150_000_000),'heartbeat_keeps_original_deadline')
    def test_no_heartbeat_evidence_rejected(self):
        self.check_modified(2,lambda xs:xs.pop(),'transport_heartbeat_continues_after_expiry')
    def test_window_not_republished(self):
        self.check_modified(1,lambda xs:xs.append(copy.deepcopy(xs[0])),'exactly_one_pulse_publication')
    def test_native_fallback_not_continuation(self):
        self.check_modified(3,lambda xs:xs[-1].update(plan_reason='lease_expired'),'expired_cap_and_phase_match')
    def test_binding_required(self):
        self.check_modified(3,lambda xs:xs[0].update(prefill_logical_tokens=32),'pulse_logically_binds_prefill')
    def test_arrival_regression(self):
        from joint18v2.workload import arrivals
        self.assertEqual(arrivals(1,1,'steady',42),[.5])
        self.assertGreater(arrivals(1,1,'steady',42)[0],.2)

class ReuseTests(unittest.TestCase):
    def files(self,root):
        identity={'pod':'spra-vllm-0','pid':94,'boot_id':'b','launch_id':'launch',
                  'hook_source_sha256':'hook','recovery_revision':'boot1'}
        cfg={k:None for k in ('model','replicas','workflow_count','hard_total','base_cap','max_model_len','block_size','max_num_seqs','share_tokens','input_profile','topology')}
        cfg.update(replicas=1,workflow_count=1,base_cap=1024,hard_total=4096)
        summary={'condition_valid':True,'budget_invariants':True,'control_time_match':True,
                 'request_correlation':'6/6','requests':6,'workflows':1,'output_tokens':576,
                 'native_next_batch_checked':100,'native_next_batch_matches':100,
                 'root_to_branch_cache_links_verified':4}
        atomic(root/'environment.json',[identity]);atomic(root/'canary_result.json',{'version':VERSION,'hook_hash':'hook','pass':False})
        atomic(root/'cleanup.json',{'backend_drained':True})
        atomic(root/'CACHE_CHAIN/manifest.json',{'config':cfg})
        atomic(root/'CACHE_CHAIN/summary.json',summary)
        atomic(root/'CACHE_CHAIN/status.json',{'execution':'complete'})
        atomic(root/'CACHE_CHAIN/cache_chain_evidence.json',[{'root_to_branch_evidence_pass':True}]*4)
        return [identity],cfg
    def test_passed_subcondition_of_failed_suite_reused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);initial,cfg=self.files(p);before={str(f):f.read_bytes() for f in p.rglob('*.json')}
            summary,evidence=reuse_cache_chain(p,initial,cfg)
            self.assertFalse(evidence['source_overall_canary_pass']);self.assertTrue(evidence['cache_subcondition_passed'])
            self.assertEqual(before,{str(f):f.read_bytes() for f in p.rglob('*.json')})
    def test_changed_process_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);initial,cfg=self.files(p);initial[0]['pid']=95
            with self.assertRaises(ValueError):reuse_cache_chain(p,initial,cfg)
    def test_changed_hook_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);initial,cfg=self.files(p);initial[0]['hook_source_sha256']='another'
            with self.assertRaises(ValueError):reuse_cache_chain(p,initial,cfg)
    def test_changed_launch_rejected_even_same_pid(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);initial,cfg=self.files(p);initial[0]['launch_id']='new'
            with self.assertRaises(ValueError):reuse_cache_chain(p,initial,cfg)
    def test_incomplete_cache_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);initial,cfg=self.files(p)
            atomic(p/'CACHE_CHAIN/cache_chain_evidence.json',[{'root_to_branch_evidence_pass':True}]*3)
            with self.assertRaises(ValueError):reuse_cache_chain(p,initial,cfg)
    def test_changed_config_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);initial,cfg=self.files(p);cfg['base_cap']=2048
            with self.assertRaises(ValueError):reuse_cache_chain(p,initial,cfg)

async def native_fixture():
    """Run the real client/Publisher/hook with .5s arrival against an API-shaped fake.

    Real clock, in-memory HTTP transport, no sockets/GPU/vLLM/Kubernetes access.
    Original6 token work is preserved. This function also drives the old-code
    reproduction using the same API and deterministic request work.
    """
    import httpx
    from fake_native import modules, FakeGroup, hashes
    from joint18v2.client import execute
    from joint18v2.cache import NativeCacheObserver
    from joint18v2.workload import build
    with tempfile.TemporaryDirectory() as t:
        path=Path(t);mods,sm,Engine=modules()
        with patch.dict(sys.modules,mods),patch.dict(os.environ,{'SPRA_E1_ROOT':t,'POD_NAME':'spra-vllm-0','NVIDIA_VISIBLE_DEVICES':'CPU-FAKE-NOT-GPU'}):
            spec=importlib.util.spec_from_file_location('lifetime_fakehook_'+uuid.uuid4().hex,ROOT/'runtime/joint18v2_hook.py')
            hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
            hook.capabilities=lambda:{'scheduler_source_sha256':'fake','imported_version':'0.6.3','distribution_version':'0.6.3'}
            hook.NativeCacheObserver=lambda mgr,b:NativeCacheObserver(mgr,b,hashes)
            hook.install();s=sm.Scheduler();engine=Engine(s);s._v2_writer.drain()
            initial=[load(path/'joint18v2/spra-vllm-0.status.json')]
            cfg=load(ROOT/'configs/development.json');cfg.update(replicas=1,workflow_count=1,prefix_families=1,arrival_rate=1.,arrival_shape='steady',condition_wall_s=20,snapshot_max_age_s=1.,cache_evidence_max_age_s=1.)
            nodes=build(cfg,list(range(100,140)),182002,[110]*8)
            class Stream(httpx.AsyncByteStream):
                def __init__(self,g):self.g=g
                async def __aiter__(self):
                    seen=0
                    while self.g.done<self.g.output:
                        if self.g.done>seen:
                            seen=self.g.done
                            yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'x'}]})+'\n\n').encode()
                        await asyncio.sleep(.001)
                    yield ('data: '+json.dumps({'id':self.g.request_id,'choices':[{'text':'x'}]})+'\n\n').encode()
                    yield ('data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':len(self.g.tokens),'completion_tokens':self.g.output}})+'\n\n').encode()
                    yield b'data: [DONE]\n\n'
            async def handler(req):
                payload=json.loads(req.content);g=FakeGroup('r'+uuid.uuid4().hex,payload['prompt'],payload['max_tokens'])
                s.add_seq_group(g);return httpx.Response(200,stream=Stream(g))
            alive=True
            async def pump():
                while alive:
                    if s.running or s.waiting:await engine.step_async()
                    await asyncio.sleep(.004)
            task=asyncio.create_task(pump())
            try:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary,events=await execute(client,path,initial,cfg,nodes,path/'RUN','A',lifetime_probe=True)
                controls=load(path/'RUN/controls.json')
                return summary,events,nodes,controls
            finally:
                alive=False
                try:await task
                finally:s._v2_stop.set();await asyncio.sleep(.05);s._v2_writer.close()

class RealClientFixture(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_arrival_exercises_pulse_then_expires(self):
        summary,events,nodes,controls=await native_fixture()
        self.assertEqual(summary['request_correlation'],'6/6')
        self.assertEqual(summary['output_tokens'],576)
        self.assertEqual(summary['online_search_calls'],0)
        self.assertEqual(set(summary['prefill_caps_applied']),{256,1024})
        self.assertTrue(summary['control_time_match'])
        self.assertTrue(summary['finite_lifetime_probe_pass'],summary['finite_lifetime_evidence'])
        pulse=[c for c in controls if c['caps']==[256]]
        self.assertEqual(len(pulse),1)
        self.assertGreaterEqual(pulse[0]['snapshot_ns']/1e9,nodes['w0000/root']['ready_at'])
        self.assertEqual(summary['native_next_batch_matches'],summary['native_next_batch_checked'])

if __name__=='__main__':unittest.main()

class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, valid):
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('lifetime_run_'+uuid.uuid4().hex,ROOT/'tools/run.py')
        run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);runtime=p/'runtime';runtime.mkdir();source=p/'OLD_CANARY'
            initial,cfg=ReuseTests().files(source)
            real_cfg=load(ROOT/'configs/development.json');real_cfg.update(replicas=1,workflow_count=1,prefix_families=1,arrival_rate=1,arrival_shape='steady')
            atomic(source/'CACHE_CHAIN/manifest.json',{'config':real_cfg});atomic(p/'config.json',real_cfg)
            calls=[]
            async def gate(*args):return initial
            async def pool(*args):return list(range(100,140))
            async def drain(*args):return True
            async def execute(*args,**kwargs):
                calls.append((args,kwargs))
                summary=load(source/'CACHE_CHAIN/summary.json')
                summary.update(prefill_caps_applied=[256,1024],finite_lifetime_probe_pass=valid,
                               finite_lifetime_evidence={'failure_reasons':[] if valid else ['pulse_has_scheduled_prefill']})
                return summary,[]
            a=SimpleNamespace(command='canary',runtime=runtime,config=p/'config.json',blocks=None,
                              reuse_cache_chain=source,model=None,score_model=False)
            with patch.object(run,'ROOT',p),patch.object(run,'gate',gate),patch.object(run,'token_pool',pool),patch.object(run,'execute',execute),patch.object(run,'wait_drained',drain),contextlib.redirect_stdout(io.StringIO()):
                if valid:await run.live(a)
                else:
                    with self.assertRaises(RuntimeError):await run.live(a)
            self.assertEqual(len(calls),1)
            self.assertTrue(calls[0][1]['lifetime_probe'])
            self.assertEqual(calls[0][0][5].name,'FINITE_LIFETIME')
            result=load(next((p/'runs').glob('CANARY_*/canary_result.json')))
            self.assertEqual(result['pass'],valid);self.assertEqual(result['new_generation_requests'],6)
            self.assertFalse(load(source/'canary_result.json')['pass'])
    async def test_reuses_cache_only_runs_one_failed_probe(self):await self.exercise(True)
    async def test_caps_alone_cannot_pass_without_lifetime_evidence(self):await self.exercise(False)
