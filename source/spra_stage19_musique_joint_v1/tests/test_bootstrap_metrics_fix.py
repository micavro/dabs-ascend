from __future__ import annotations
import asyncio,copy,importlib.util,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from joint18v2.latency_metrics import request_metrics,backend_progress_metrics
from joint18v2.client import gate,BackendNotReady
from joint18v2.common import load,atomic
ROOT=Path(__file__).resolve().parents[1]

class MetricTests(unittest.TestCase):
    def node(self,**changes):
        n={'wid':'w','name':'root','parents':[],'arrival':10.,'ready_at':10.15,'finish':11.01,'state':'done','hash':'h',
           'stream':{'request_start':10.2,'header':10.3,'first':10.5,'last':10.9,'return':11.,'text_events':3,'error':None,'usage':{'completion_tokens':3}}}
        n.update(changes);return n
    def test_request_ttft_includes_header(self):
        r=request_metrics({'r':self.node()});self.assertAlmostEqual(r['mean_request_ttft_ms'],300.)
    def test_node_start_is_dependency_eligible_not_discovered(self):
        r=request_metrics({'r':self.node()});self.assertAlmostEqual(r['mean_node_time_to_first_content_ms'],500.)
    def test_tail_inclusive_tpot_and_content_boundary_are_separate(self):
        r=request_metrics({'r':self.node()})['request_latency_rows'][0]
        self.assertAlmostEqual(r['client_tpot_proxy_ms'],250.);self.assertAlmostEqual(r['tpot_content_boundary_proxy_ms'],200.)
    def test_child_eligibility_from_max_parent_finish(self):
        n=self.node(name='child',parents=['p'],ready_at=10.49);p=self.node(wid='w',finish=10.4)
        r=request_metrics({'p':p,'c':n})['request_latency_rows'][1];self.assertAlmostEqual(r['node_time_to_first_content_ms'],100.)
    def test_one_token_tpot_not_zero(self):
        n=self.node();n['stream']['usage']['completion_tokens']=1;r=request_metrics({'r':n})
        self.assertIsNone(r['mean_client_tpot_proxy_ms']);self.assertEqual(r['request_latency_rows'][0]['tpot_exclusion'],'output_tokens_le_one')
    def test_single_event_is_not_free_decode(self):
        n=self.node();n['stream'].update(text_events=1,last=10.5)
        r=request_metrics({'r':n})['request_latency_rows'][0]
        self.assertIsNone(r['client_tpot_proxy_ms']);self.assertGreater(r['raw_tail_inclusive_tpot_proxy_ms'],0)
        self.assertEqual(r['tpot_exclusion'],'single_visible_event_no_decode_span')
    def test_no_usage_does_not_use_requested_count(self):
        n=self.node(output=999);n['stream']['usage']=None;r=request_metrics({'r':n})
        self.assertIsNone(r['mean_client_tpot_proxy_ms']);self.assertAlmostEqual(r['mean_request_ttft_ms'],300)
    def test_missing_visible_content_is_missing(self):
        n=self.node();n['stream'].update(first=None,last=None,text_events=0);r=request_metrics({'r':n})
        self.assertIsNone(r['mean_request_ttft_ms']);self.assertIsNone(r['mean_client_tpot_proxy_ms'])
    def test_failed_request_not_latency_zero(self):
        n=self.node();n['stream']['error']='HTTP error';r=request_metrics({'r':n})
        self.assertIsNone(r['mean_request_ttft_ms']);self.assertEqual(r['request_ttft_ms_samples'],0)
    def test_missing_parent_does_not_use_ready_at(self):
        n=self.node(parents=['missing']);r=request_metrics({'r':n})['request_latency_rows'][0]
        self.assertIsNone(r['node_time_to_first_content_ms']);self.assertEqual(r['node_first_content_exclusion'],'missing_parent_finish')
    def test_request_weighting_not_workflow_path_sum(self):
        a=self.node();b=self.node();b['stream']['first']=10.6
        r=request_metrics({'a':a,'b':b});self.assertAlmostEqual(r['mean_request_ttft_ms'],350.)
    def test_negative_timeline_not_clamped(self):
        n=self.node();n['stream']['first']=10.0;r=request_metrics({'r':n})
        self.assertIsNone(r['mean_request_ttft_ms'])
    def test_empty_input_missing_not_zero(self):
        r=request_metrics({});self.assertIsNone(r['mean_request_ttft_ms']);self.assertEqual(r['request_ttft_ms_samples'],0)
    def test_token_output_progress_host_label(self):
        n=self.node();e=[{'event':'output_progress','prompt_hash':'h','observed_ns':10_000_000+i*20_000_000,'output_tokens':i+1} for i in range(3)]
        r=backend_progress_metrics({'r':n},e);self.assertEqual(r['requests_reaching_usage'],1);self.assertEqual(r['host_output_tpot_proxy_ms']['mean'],20.);self.assertIsNone(r['exact_gpu_itl'])
    def test_multitoken_jump_not_interpolated(self):
        n=self.node();e=[{'event':'output_progress','prompt_hash':'h','observed_ns':10_000_000,'output_tokens':1},{'event':'output_progress','prompt_hash':'h','observed_ns':50_000_000,'output_tokens':3}]
        r=backend_progress_metrics({'r':n},e);self.assertEqual(r['host_output_tpot_proxy_ms']['mean'],20.);self.assertEqual(r['host_observed_interval_ms']['n'],0)
    def test_old_backend_trace_has_no_token_times(self):
        r=backend_progress_metrics({'r':self.node()},[{'event':'iteration'}]);self.assertEqual(r['requests_with_progress'],0);self.assertIsNone(r['host_output_tpot_proxy_ms']['mean'])
    def test_recount_keeps_original_files(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);src=p/'run';src.mkdir();f=src/'nodes.json';atomic(f,{'r':self.node()});before=f.read_bytes()
            z=subprocess.run([sys.executable,'-B',str(ROOT/'tools/recount_latency.py'),'--run',str(src),'--out',str(p/'report')],capture_output=True,text=True)
            self.assertEqual(z.returncode,0,z.stderr);self.assertEqual(f.read_bytes(),before);self.assertIn('SOURCE_FILES_UNCHANGED=True',z.stdout)
    def test_recount_rejects_output_inside_source(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);atomic(p/'nodes.json',{'r':self.node()})
            z=subprocess.run([sys.executable,'-B',str(ROOT/'tools/recount_latency.py'),'--run',str(p),'--out',str(p/'new')],capture_output=True,text=True)
            self.assertNotEqual(z.returncode,0);self.assertFalse((p/'new').exists())

class GateTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_status_does_not_send_request(self):
        import httpx
        calls=[]
        def handler(req):calls.append(req);return httpx.Response(200)
        with tempfile.TemporaryDirectory() as t:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(BackendNotReady,'status missing'):await gate(client,Path(t),1)
        self.assertEqual(calls,[])

class SpawnTests(unittest.TestCase):
    def tree(self,path,failed=False):
        import shutil
        shutil.copytree(ROOT/'joint18v2',path/'joint18v2')
        (path/'launcher.py').write_bytes((ROOT/'runtime/launcher.py').read_bytes())
        for d in ['vllm','vllm/engine','vllm/engine/multiprocessing','vllm/entrypoints','vllm/entrypoints/openai']:
            (path/d).mkdir(parents=True,exist_ok=True);(path/d/'__init__.py').write_text('')
        (path/'joint18v2_hook.py').write_text('import os,json\ndef install():\n os.environ["FIXTURE_HOOK_PID"]=str(os.getpid())\n')
        (path/'vllm/engine/multiprocessing/engine.py').write_text('import os,json\nfrom pathlib import Path\ndef run_mp_engine(*args,**kwargs):\n assert os.environ.get("FIXTURE_HOOK_PID")==str(os.getpid()),"child hook missing"\n '+('raise RuntimeError("fixture engine error")' if failed else 'Path(os.environ["FIXTURE_OUTPUT"]).write_text(json.dumps({"pid":os.getpid(),"hook_pid":os.environ["FIXTURE_HOOK_PID"]}))')+'\n')
        (path/'vllm/entrypoints/openai/api_server.py').write_text('''import argparse,asyncio,multiprocessing,os
from types import SimpleNamespace
from vllm.engine.multiprocessing.engine import run_mp_engine
FlexibleArgumentParser=argparse.ArgumentParser
uvloop=SimpleNamespace(run=asyncio.run)
def make_arg_parser(p):
 p.add_argument('--disable-frontend-multiprocessing',action='store_true');return p
def validate_parsed_serve_args(args):pass
async def run_server(args):
 if args.disable_frontend_multiprocessing:
  from vllm.engine.multiprocessing.engine import run_mp_engine as direct
  return direct()
 p=multiprocessing.get_context('spawn').Process(target=run_mp_engine)
 p.start();p.join(8)
 if p.is_alive():p.terminate();p.join();raise RuntimeError('fixture child stalled')
 if p.exitcode!=0:raise RuntimeError('fixture child failed')
''')
    def run_fixture(self,args=(),failed=False):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);self.tree(p,failed)
            env=dict(os.environ,SPRA_E1_ROOT=t,POD_NAME='fixture',FIXTURE_OUTPUT=str(p/'child.json'),PYTHONDONTWRITEBYTECODE='1')
            z=subprocess.run([sys.executable,'-B',str(p/'launcher.py'),*args],env=env,capture_output=True,text=True,timeout=15)
            output=load(p/'child.json') if (p/'child.json').exists() else None
            return z,output
    def test_spawn_has_single_server_and_child_hook(self):
        z,output=self.run_fixture();self.assertEqual(z.returncode,0,z.stderr);self.assertEqual(output['pid'],int(output['hook_pid']))
        phases=[json.loads(l.split('=',1)[1])['phase'] for l in z.stderr.splitlines() if l.startswith('JOINT18V2_STARTUP=')]
        self.assertEqual(phases.count('api_server_enter'),1);self.assertEqual(phases.count('engine_child_hook_installed'),1)
    def test_direct_engine_flag_preserved(self):
        z,_=self.run_fixture(['--disable-frontend-multiprocessing']);self.assertEqual(z.returncode,0,z.stderr);self.assertNotIn('engine_child_enter',z.stderr)
    def test_engine_failure_keeps_traceback(self):
        z,out=self.run_fixture(failed=True);self.assertNotEqual(z.returncode,0);self.assertIn('fixture engine error',z.stderr);self.assertIn('engine_child_failed',z.stderr);self.assertIsNone(out)
    def test_importing_launcher_does_not_install(self):
        spec=importlib.util.spec_from_file_location('not_main_launcher',ROOT/'runtime/launcher.py');m=importlib.util.module_from_spec(spec)
        with patch('joint18v2.bootstrap.run_server',side_effect=AssertionError('must not start')):spec.loader.exec_module(m)
    def test_old_runpy_launcher_reenters_under_spawn(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);self.tree(p)
            (p/'launcher.py').write_text("import sys,runpy\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nimport joint18v2_hook\njoint18v2_hook.install()\nrunpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')\n")
            api=p/'vllm/entrypoints/openai/api_server.py';api.write_text(api.read_text()+"\nif __name__=='__main__':\n uvloop.run(run_server(SimpleNamespace(disable_frontend_multiprocessing=False)))\n")
            env=dict(os.environ,SPRA_E1_ROOT=t,POD_NAME='fixture',FIXTURE_OUTPUT=str(p/'child.json'),PYTHONDONTWRITEBYTECODE='1')
            z=subprocess.run([sys.executable,'-B',str(p/'launcher.py')],env=env,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(z.returncode,0);self.assertIn('bootstrapping phase',z.stderr)

class MechanismNoSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_mechanism_cli_needs_no_model_or_worker(self):
        import types
        spec=importlib.util.spec_from_file_location('fixture_run_cli',ROOT/'tools/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'runtime').mkdir();cfg=load(ROOT/'configs/development.json');cfg['mechanism_blocks']=2;atomic(p/'config.json',cfg)
            args=types.SimpleNamespace(command='mechanism',runtime=p/'runtime',config=p/'config.json',blocks=None,model=p/'missing.json',score_model=False)
            seen=[]
            async def fake_execute(*args,**kwargs):
                self.assertIsNone(args[7]);self.assertIsNone(kwargs.get('executor'));seen.append(args[6])
                return {'mean_jct_ms':100.,'p95_jct_ms':110.,'output_tokens':10,'online_search_calls':0,'cohort_tokens_s':1.,'mechanism_trigger':{'trigger_satisfied':False},'forced_window_activity':{}},[]
            async def fake_gate(*args):return [{'pod':'fixture','hook_source_sha256':'fixture'}]
            async def fake_pool(*args):return list(range(100,150))
            async def drained(*args):return True
            with patch.object(run,'ROOT',p),patch.object(run,'gate',fake_gate),patch.object(run,'token_pool',fake_pool),patch.object(run,'require_integrated_canary',lambda x:'fixture-canary'),patch.object(run,'execute',fake_execute),patch.object(run,'wait_drained',drained),patch.object(run,'ProcessPoolExecutor',side_effect=AssertionError('no worker allowed')),patch.object(run,'model_path',side_effect=AssertionError('no model required')):
                await run.live(args)
            self.assertEqual(seen,['Forced']*8)
            report=next((p/'runs').glob('MECHANISM_*/mechanism_report.json'));self.assertEqual(load(report)['mode'],'predeclared_complete_actions_with_no_online_solver')
if __name__=='__main__':unittest.main()
