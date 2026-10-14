import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import workflow_only as w
from tail_check import prefill_chunk, compare_iteration


def summary(jct=100., **extra):
    result = {f: 100. for f in w.FIELDS}
    result.update(mean_jct_ms=jct, condition_valid=True, budget_invariants=True,
                  prefill_caps_applied=[4096], budget_switches_total=0,
                  native_next_batch_matches=4, native_next_batch_checked=4,
                  requests=1, request_correlation='1/1',
                  semantic_quality={'final_em': .5, 'final_f1': .5}, output_tokens=8)
    result.update(extra)
    return result


class Reports(unittest.TestCase):
    def test_paired_not_ratio_of_means(self):
        rows=[{'block':1,'method':'A','summary':summary(100)},
              {'block':1,'method':'D','summary':summary(50)},
              {'block':2,'method':'A','summary':summary(1000)},
              {'block':2,'method':'D','summary':summary(1000)}]
        r=w.paired_report(rows,2)
        self.assertEqual(r['D_vs_A']['mean_jct_ms']['mean_gain_pct'],25.)
        self.assertTrue(r['complete'])
    def test_negative_retained(self):
        r=w.paired_report([{'block':1,'method':'A','summary':summary()},
                           {'block':1,'method':'D','summary':summary(120)}],1)
        self.assertAlmostEqual(r['D_vs_A']['mean_jct_ms']['mean_gain_pct'],-20.)
    def test_missing_auxiliary_null(self):
        r=w.paired_report([{'block':1,'method':'A','summary':summary(mean_client_tpot_proxy_ms=None)},
                           {'block':1,'method':'D','summary':summary()}],1)
        self.assertIsNone(r['D_vs_A']['mean_client_tpot_proxy_ms']['mean_gain_pct'])
    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            w.paired_report([{'block':1,'method':'D','summary':summary()}]*2)
    def test_incomplete_not_success(self):
        self.assertFalse(w.paired_report([{'block':1,'method':'A','summary':summary()}])['complete'])
    def test_adaptive_method_rejected(self):
        with self.assertRaises(ValueError):w.paired_report([{'block':1,'method':'Joint','summary':summary()}])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):w.paired_report([{'block':1,'method':m,'summary':summary(float('nan'))} for m in ('A','D')])
    def test_quality_percentage_points(self):
        r=w.paired_report([{'block':1,'method':'A','summary':summary()},
             {'block':1,'method':'D','summary':summary(semantic_quality={'final_em':.4,'final_f1':.6})}],1)
        self.assertAlmostEqual(r['paired_blocks'][0]['quality_delta_pp']['final_em_delta_pp'],-10.)

class FixedCheck(unittest.TestCase):
    def test_valid(self):w.validate_fixed_summary(summary(),1)
    def test_cap_change_rejected(self):
        with self.assertRaises(ValueError):w.validate_fixed_summary(summary(prefill_caps_applied=[1024,4096]))
    def test_switch_rejected(self):
        with self.assertRaises(ValueError):w.validate_fixed_summary(summary(budget_switches_total=1))
    def test_missing_batches_rejected(self):
        with self.assertRaises(ValueError):w.validate_fixed_summary(summary(native_next_batch_checked=0,native_next_batch_matches=0))
    def test_correlation_rejected(self):
        with self.assertRaises(ValueError):w.validate_fixed_summary(summary(request_correlation='0/1'),1)
    def test_bad_validity_rejected(self):
        with self.assertRaises(ValueError):w.validate_fixed_summary(summary(condition_valid=False))
    def test_tail_fits(self):self.assertEqual(prefill_chunk(31,31),31)
    def test_partial_aligns(self):self.assertEqual(prefill_chunk(100,31),16)
    def test_tail_smaller_than_block(self):self.assertEqual(prefill_chunk(13,1024),13)
    def test_no_space(self):self.assertEqual(prefill_chunk(32,0),0)
    def test_bad_int_rejected(self):
        with self.assertRaises(ValueError):prefill_chunk(True,16)

class ThinBridge(unittest.IsolatedAsyncioTestCase):
    async def test_original_executor_used_without_force_caps(self):
        calls=[];atomic=[]
        async def execute(client,runtime,initial,cfg,nodes,folder,method,model=None,**kw):
            calls.append((method,kw))
            Path(folder).mkdir()
            (Path(folder)/'controls.json').write_text(json.dumps([{'caps':[4096]*4,'totals':[4096]*4}]))
            return summary(),[]
        m=SimpleNamespace(execute=execute,atomic=lambda p,v:atomic.append((p,v)),
                          native_client=SimpleNamespace(compare_iteration=lambda row:None))
        state,*_=w.configure_remeasurement(m,'historical-relative',{})
        self.assertEqual(m.METHODS,['A','D'])
        with tempfile.TemporaryDirectory() as t:
            nodes={'n':{'state':'done','finish':2.}}
            await m.execute(None,None,[{}]*4,{'base_cap':4096,'replicas':4},nodes,Path(t)/'cond','D',{},executor='native-worker')
        self.assertEqual(calls,[('D',{'executor':'native-worker'})])
    async def test_force_caps_guard_prevents_bypass(self):
        async def never(*a,**k):raise AssertionError('must stop before execution')
        m=SimpleNamespace(execute=never,atomic=lambda *a:None,native_client=SimpleNamespace(compare_iteration=None))
        w.configure_remeasurement(m,'historical-relative',{})
        with self.assertRaisesRegex(ValueError,'bypasses'):
            await m.execute(None,None,[{}]*4,{'base_cap':4096,'replicas':4},{},Path('/unused'),'D',{},force_caps=[4096]*4)
    async def test_joint_never_runs(self):
        async def never(*a,**k):raise AssertionError('must stop before execution')
        m=SimpleNamespace(execute=never,atomic=lambda *a:None,native_client=SimpleNamespace(compare_iteration=None))
        w.configure_remeasurement(m,'historical-relative',{})
        with self.assertRaises(ValueError):await m.execute(None,None,[{}]*4,{}, {},Path('/unused'),'Joint')
    async def test_mirrored_order(self):
        m=SimpleNamespace(execute=lambda:None,atomic=lambda *a:None,native_client=SimpleNamespace(compare_iteration=None))
        w.configure_remeasurement(m,'reverse',{})
        self.assertEqual(m.ORDERS,[list(reversed(x)) for x in w.HISTORICAL_ORDERS])
        self.assertEqual(w.HISTORICAL_ORDERS[0],['A','D'])
    async def test_protocol_marks_remeasurement(self):
        saved=[]
        m=SimpleNamespace(execute=lambda:None,atomic=lambda p,x:saved.append(x),native_client=SimpleNamespace(compare_iteration=None))
        state,*_=w.configure_remeasurement(m,'historical-relative',{'model':'pinned'})
        m.atomic(Path('/new/run/frozen_protocol.json'),{'old':True})
        self.assertEqual(saved[0]['workflow_only_remeasurement']['methods'],['A','D'])
        self.assertFalse(saved[0]['workflow_only_remeasurement']['original_20_condition_suite_replayed_exactly'])
    async def test_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.json';w.save_new(p,{'a':1})
            with self.assertRaises(FileExistsError):w.save_new(p,{'a':2})

class RealSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ.get('WF_D0_SOURCE_ROOT',str(w.DEFAULT_ROOT)))
        if not (cls.root/'spra_stage19_typed_handoff_v1').is_dir():
            raise unittest.SkipTest('original source not mounted; set WF_D0_SOURCE_ROOT')
    def test_48_frozen_source_hashes(self):
        self.assertEqual(w.inspect_sources(self.root)['source_files_verified'],48)
    def test_original_native_import_and_prepared(self):
        m=w.load_original(self.root)
        out=m.prepared(SimpleNamespace(prepared=self.root/'spra_stage19_typed_handoff_v1/prepared'/w.PROFILE))
        self.assertEqual(out[3]['best_fixed_cap'],4096)
        import semantic19.selector as sel
        self.assertEqual(Path(sel.__file__).resolve(),self.root/'spra_stage19_musique_joint_v1/semantic19/selector.py')
    def test_inspect_cli_no_runtime(self):
        p=subprocess.run([sys.executable,str(HERE/'workflow_only.py'),'inspect','--repo',str(self.root),'--runtime','/does/not/exist'],capture_output=True,text=True,timeout=15)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertFalse(json.loads(p.stdout)['backend_accessed'])


class HistoryAudit(unittest.TestCase):
    def create_history(self, p):
        run=p/'old';run.mkdir()
        (run/'frozen_protocol.json').write_text(json.dumps({'config':{'replicas':4,'arrival_rate':2.,'base_cap':4096}}))
        for b in range(1,5):
            for m in ('A','D'):
                f=run/f'B{b}_1_{m}';f.mkdir()
                (f/'summary.json').write_text(json.dumps(summary(90. if m=='D' else 100.)))
                (f/'manifest.json').write_text(json.dumps({'method':m,'input_nodes':{'x':{}}}))
        return run
    def test_saved_summary_audit(self):
        import contextlib,io
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);r=self.create_history(p)
            old={str(x):w.sha(x) for x in r.rglob('*') if x.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):w.audit_history(p,r,p/'out')
            q=w.load(p/'out/HISTORICAL_AD_AUDIT.json')
            self.assertTrue(q['complete']);self.assertFalse(q['raw_backend_recomputed'])
            self.assertAlmostEqual(q['D_vs_A']['mean_jct_ms']['mean_gain_pct'],10.)
            self.assertEqual(old,{str(x):w.sha(x) for x in r.rglob('*') if x.is_file()})
    def test_no_imputed_history(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);r=self.create_history(p);(r/'B4_1_D/summary.json').unlink()
            with self.assertRaisesRegex(ValueError,'incomplete'):w.audit_history(p,r,p/'out')
    def test_missing_history_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            with self.assertRaisesRegex(FileNotFoundError,'not present'):w.audit_history(p,p/'missing',p/'out')
    def test_nonfixed_history_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);r=self.create_history(p)
            (r/'B2_1_D/summary.json').write_text(json.dumps(summary(prefill_caps_applied=[1024])))
            with self.assertRaises(ValueError):w.audit_history(p,r,p/'out')

if __name__=='__main__':unittest.main()
