"""CLI preparation and suite orchestration using temporary archives/mock IO."""
from __future__ import annotations
import asyncio,contextlib,copy,importlib.util,io,json,tempfile,unittest,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from joint18v2.common import load,atomic
ROOT=Path(__file__).resolve().parents[1]

def module():
    spec=importlib.util.spec_from_file_location('experiment_'+uuid.uuid4().hex,ROOT/'tools/experiment.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class PreparationTests(unittest.TestCase):
    def test_all_twenty_temp_conditions_fit_and_freeze_without_network(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);old=p/'old';old.mkdir();atomic(old/'suite_status.json',{'status':'complete'})
            for block in range(1,5):
                for pos,method in enumerate(('A','D','B','Decoupled','Joint'),1):
                    f=old/f'B{block}_{pos}_{method}';f.mkdir();case=old.name+'/'+f.name
                    atomic(f/'manifest.json',{'case':case,'method':method})
                    atomic(f/'summary.json',{'condition_valid':True,'requests':96,'output_tokens':9216})
                    events=[]
                    for r in range(4):
                        for it,(pp,dd) in enumerate([(256,0),(512,0),(256,1),(512,2),(0,1),(0,2)],1):
                            common={'replica':r,'case':case,'iteration':it,'pid':10+r}
                            scheduled=[]
                            if pp:scheduled.append({'prefill':True,'tokens':pp,'prompt_hash':'p','context_before':0,'computed_before':0})
                            for j in range(dd):scheduled.append({'prefill':False,'tokens':1,'prompt_hash':'d'+str(j),'context_before':256,'computed_before':256})
                            events.append(dict(common,event='iteration',scheduled=scheduled,prefill_logical_tokens=pp,decode_tokens=dd))
                            if pp:events.append(dict(common,event='model_input',prompt_hash='p',skipped_query_tokens=0))
                            events.append(dict(common,event='engine_step',engine_host_ms=(4 if pp and not dd else 16)+pp/1024+dd,engine_error=None))
                    (f/'backend.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
            legacy=p/'legacy.json';atomic(legacy,{'model_id':'synthetic-old','models':{str(r):{'coef':[15,0,0,0,0,0]} for r in range(4)}})
            root=p/'new';root.mkdir();args=SimpleNamespace(source=old,pilot_run=old,legacy_model=legacy)
            initial=legacy.read_bytes()
            with patch.object(m,'ROOT',root),patch.object(m,'source_check',return_value={'pass':True}),contextlib.redirect_stdout(io.StringIO()):
                out=m.prepare(args);frozen,model,mf=m.read_prepared()
            self.assertEqual(out,frozen);self.assertEqual(model['model_id'],mf['model_id'])
            self.assertTrue(mf['source_files_unchanged']);self.assertEqual(mf['GPU_REQUESTS'],0)
            self.assertFalse(mf['fit_uses_JCT_labels']);self.assertEqual(initial,legacy.read_bytes())
            self.assertEqual(len(mf['training_source_hashes']),60)
    def test_prepared_hash_rejects_mutation(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);atomic(p/'phase_model.json',{'model_id':'x'});atomic(p/'manifest.json',{'model_id':'x','model_sha256':'wrong'})
            with self.assertRaises(RuntimeError):m.read_prepared(p)

class SuiteTests(unittest.IsolatedAsyncioTestCase):
    async def scenario(self,failure=False):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root=p/'new';root.mkdir();(root/'configs').mkdir();atomic(root/'configs/development.json',load(ROOT/'configs/development.json'))
            runtime=p/'runtime';runtime.mkdir();old=p/'original';old.mkdir()
            legacy={'model_id':'fake','models':{str(r):{'coef':[1,0,0,0,0,0]} for r in range(4)}}
            calls=[]
            async def gate(*args):return [{'pod':f'p{r}','hook_source_sha256':m.HOOK_SHA} for r in range(4)]
            async def pool(*args):return list(range(100,140))
            async def drained(*args):return True
            async def execute(client,rt,initial,cfg,ns,folder,method,model=None,**kw):
                folder.mkdir(parents=True)
                if method=='Forced':
                    for n in ns.values():n['replica']=n['warm_replica']
                if failure and method=='B':raise RuntimeError('injected condition failure')
                calls.append(method)
                return {'mean_jct_ms':1000,'p95_jct_ms':1100,'cohort_tokens_s':500,'output_tokens':sum(n['output'] for n in ns.values()),
                    'condition_valid':True,'requests':len(ns),'selector_coverage':{'fake':True}},[]
            args=SimpleNamespace(source=old,runtime=runtime,prepared=None,blocks=4)
            with patch.object(m,'ROOT',root),patch.object(m,'source_check',return_value={'pass':True}),patch.object(m,'read_prepared',return_value=(p,legacy,{})),patch.object(m,'gate',gate),patch.object(m,'canary_receipt',return_value='saved-canary'),patch.object(m,'token_pool',pool),patch.object(m,'execute',execute),patch.object(m,'wait_drained',drained),patch.object(m,'ProcessPoolExecutor',side_effect=lambda *args,**kw:ThreadPoolExecutor(1)),contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(RuntimeError):await m.run(args)
                else:await m.run(args)
            out=next((root/'runs').glob('SELECTOR_PILOT_*'));status=load(out/'suite_status.json')
            self.assertTrue(load(out/'cleanup.json')['backend_drained'])
            self.assertFalse(load(runtime/'joint18v2/control.json')['enabled'])
            self.assertEqual(calls[:2],['Forced','Forced'])
            if failure:self.assertEqual(status['status'],'failed');self.assertFalse(status['automatic_retry'])
            else:
                self.assertEqual(status['status'],'complete');self.assertEqual(status['measured_conditions'],20)
                for method in ('A','D','B','Decoupled','Joint'):self.assertEqual(calls.count(method),4)
                self.assertFalse(load(out/'joint_result.json')['old_results_pooled'])
    async def test_suite_warmup_then_twenty_conditions(self):await self.scenario()
    async def test_suite_failure_keeps_partial_results_and_disables_control(self):await self.scenario(True)

if __name__=='__main__':unittest.main()
