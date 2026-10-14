from __future__ import annotations
import contextlib,importlib.util,io,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from typed19.dependency import activate,SOURCE,identity
activate()

def runner():
    spec=importlib.util.spec_from_file_location('typedrunner_fixture',ROOT/'tools/run_typed.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

class RunnerTests(unittest.TestCase):
    def test_plan_cli_has_no_runtime_access(self):
        env=dict(os.environ,STAGE19_SOURCE=str(SOURCE));p=subprocess.run([sys.executable,'-B',str(ROOT/'tools/run_typed.py'),'plan','--runtime','/not/a/runtime'],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('guided JSON',p.stdout)
    def test_source_unchanged(self):self.assertTrue(identity())
    def test_no_joint_codegen_changes(self):
        import semantic19.selector
        self.assertEqual(Path(semantic19.selector.__file__).resolve(),SOURCE/'semantic19/selector.py')
    def test_all_methods_keep_null_tpot(self):
        m=runner();rows=[]
        for name in m.METHODS:
            s={k:100. for k in ['mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','p95_request_ttft_ms','mean_node_time_to_first_content_ms','mean_client_tpot_proxy_ms','p95_client_tpot_proxy_ms']}
            s.update(mean_client_tpot_proxy_ms=None,p95_client_tpot_proxy_ms=None,output_tokens=10,cohort_tokens_s=20.,semantic_quality={'final_em':0.,'final_f1':0.})
            rows.append({'block':1,'method':name,'summary':s})
        r=m.progress_summary(rows)
        self.assertIsNone(r['paired'][0]['methods']['Joint']['gain_vs_A_pct']['mean_client_tpot_proxy_ms'])
        self.assertFalse(r['paired'][0]['methods']['Joint']['metric_guard_pass'])
    def test_old_profile_rejected(self):
        m=runner()
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'manifest.json').write_text('{"files":{}}')
            with self.assertRaisesRegex(ValueError,'exact typed'):m.prepared(SimpleNamespace(prepared=p))
    def test_incomplete_profile_rejected(self):
        m=runner()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'index.json').write_text('[]')
            with self.assertRaises(ValueError):m.freeze_profile(p,p/'old',p/'ds')
    def test_profile_freeze_shared_contract(self):
        m=runner()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);run=p/'run';run.mkdir();ds=p/'ds';ds.mkdir();(ds/'manifest.json').write_text('{}')
            old=p/'old.json';old.write_text('{"models":{}}');idx=[]
            (run/'frozen_protocol.json').write_text('{"config":{"arrival_rate":2.0}}')
            for b in range(1,5):
                for cap in (256,512,1024,2048,4096):
                    f=run/f'{b}_{cap}';f.mkdir();s={'condition_valid':True,'mean_jct_ms':cap,'semantic_quality':{'final_em':.5,'final_f1':.5}}
                    (f/'nodes.json').write_text('{"n":{"hop_count":2,"stream":{"usage":{"completion_tokens":6}}}}')
                    for name,text in [('summary.json',json.dumps(s)),('manifest.json','{}'),('backend.jsonl','')]: (f/name).write_text(text)
                    idx.append({'block':b,'cap':cap,'folder':str(f),'summary':s})
            (run/'index.json').write_text(json.dumps(idx))
            with patch.object(m,'ROOT',p),patch('joint18v2.service_fit.read_rows',return_value=([],{})),patch('joint18v2.service_fit.fit_phase_model',return_value=({'models':{}},{})),contextlib.redirect_stdout(io.StringIO()):
                out=m.freeze_profile(run,old,ds)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['application_revision'],'stage19-typed-handoff-1')
    def test_metric_clock_unchanged(self):
        from joint18v2.transport import metrics
        self.assertEqual(Path(sys.modules[metrics.__module__].__file__).resolve(),SOURCE/'joint18v2/transport.py')

class SchemaWarmupTest(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_four_replicas_no_semantic_dataset(self):
        m=runner();sent=[]
        from test_typed_contract import FakeTokenizer
        from typed19.answers import parse
        async def execute(client,runtime,initial,cfg,ns,folder,method,model,mechanism=None):
            sent.extend(ns.values())
            for n in ns.values():n.update(replica=n['warm_replica'],answer_contract=parse('{"answer":"READY"}'))
            return {'output_tokens':24},[]
        with tempfile.TemporaryDirectory() as td,patch.object(m,'execute',execute):
            cfg=json.loads((ROOT/'configs/semantic.json').read_text());p=Path(td)
            with contextlib.redirect_stdout(io.StringIO()):await m.warm_schema(None,p,[{}]*4,cfg,p,FakeTokenizer(),list(range(1000,2000)))
        self.assertEqual(len(sent),4);self.assertTrue(all(n['typed_answer_mode'] for n in sent));self.assertTrue(all(n['share']==0 for n in sent))

if __name__=='__main__':unittest.main()
