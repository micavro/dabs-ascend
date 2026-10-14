from __future__ import annotations
import json,subprocess,sys,tempfile,unittest,time
from pathlib import Path
from joint18v2.common import load,atomic
from joint18v2.native_contract import _version_decision,NativeContractError,KNOWN_DEPLOYED_SCHEDULER_SHA256
from joint18v2.client import Publisher
from joint18v2.report import comparison,mechanism_report
ROOT=Path(__file__).resolve().parents[1]

class DeploymentTests(unittest.TestCase):
    def call(self,root,*args):return subprocess.run([sys.executable,'-B',str(ROOT/'tools/stage.py'),*args,'--runtime',str(root)],capture_output=True,text=True)
    def test_stage_and_restore_exact_launcher(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);old='import joint18_hook\njoint18_hook.install()\n';(p/'launcher.py').write_text(old)
            z=self.call(p,'stage');self.assertEqual(z.returncode,0,z.stderr);m=json.loads(z.stdout.split('SOURCE_STAGED=',1)[1])
            self.assertFalse(m['automatic_restart']);self.assertFalse(m['live_processes_changed']);self.assertTrue((p/'joint18v2_hook.py').is_file())
            self.assertEqual(self.call(p,'stage').returncode!=0,True)
            z=self.call(p,'restore','--backup',m['backup']);self.assertEqual(z.returncode,0,z.stderr);self.assertEqual((p/'launcher.py').read_text(),old)
    def test_staging_refuses_recorded_work(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'launcher.py').write_text('import joint18_hook\n');atomic(p/'joint18/x.status.json',{'running':[1]})
            z=self.call(p,'stage');self.assertNotEqual(z.returncode,0);self.assertFalse((p/'joint18v2_hook.py').exists())
    def test_cli_plan_no_runtime(self):
        z=subprocess.run([sys.executable,'-B',str(ROOT/'tools/run.py'),'plan','--runtime','/definitely/not/a/runtime'],capture_output=True,text=True)
        self.assertEqual(z.returncode,0,z.stderr);self.assertFalse(json.loads(z.stdout)['CPU_REPLAY_upload_required'])

class NativeVersionTests(unittest.TestCase):
    def facts(self):return {'imported_version':'dev','distribution_version':'0.6.3','scheduler_source_sha256':KNOWN_DEPLOYED_SCHEDULER_SHA256,'distribution_scheduler_matches_loaded':True}
    def test_dev_evidence_supported(self):self.assertIn('dev_metadata',_version_decision(self.facts()))
    def test_dev_wrong_hash_rejected(self):
        f=self.facts();f['scheduler_source_sha256']='wrong'
        with self.assertRaises(NativeContractError):_version_decision(f)
    def test_other_version_not_silently_accepted(self):
        f=self.facts();f['imported_version']='0.6.4'
        with self.assertRaises(NativeContractError):_version_decision(f)
    def test_dev_metadata_absent_rejected(self):
        f=self.facts();f['distribution_version']=None
        with self.assertRaises(NativeContractError):_version_decision(f)

class PublisherTests(unittest.TestCase):
    def test_model_before_activation_uses_actual_continuation(self):
        p=Publisher('/not_accessed',[{}],'x',{'base_cap':1024});p.record={'not_before_ns':50,'action_expires_ns':200,'caps':[4096]}
        self.assertEqual(p.caps_at(25),[1024]);self.assertEqual(p.caps_at(100),[4096]);self.assertEqual(p.caps_at(200),[1024])

class ReportTests(unittest.TestCase):
    def summ(self,x):return {'mean_jct_ms':x,'p95_jct_ms':x+1,'cohort_tokens_s':1000/x,'output_tokens':576,'mechanism_trigger':{'trigger_satisfied':True},'forced_window_activity':{'prefill_logical_tokens':512}}
    def test_pair_comparison(self):
        rows=[{'block':1,'method':m,'summary':self.summ(v)} for m,v in [('A',100),('D',98),('B',99),('Decoupled',96),('Joint',90)]]
        r=comparison(rows);self.assertAlmostEqual(r['mean_Joint_gain_pct'],10);self.assertAlmostEqual(r['mean_Joint_vs_Decoupled_pct'],6.25);self.assertFalse(r['formal_10pct_claim'])
    def test_mechanism_reports_reversal_not_requires_it(self):
        vals=[('concentrate','protect',100),('split','protect',90),('concentrate','aggressive',80),('split','aggressive',90)]
        rows=[{'block':1,'placement':p,'budget':b,'summary':self.summ(v)} for p,b,v in vals]
        r=mechanism_report(rows);self.assertTrue(r['paired'][0]['placement_ranking_reverses']);self.assertEqual(r['paired'][0]['interaction_difference_ms'],20)
        self.assertTrue(r['no_reversal_requirement'])
    def test_mechanism_missing_trigger_retained(self):
        rows=[{'block':1,'placement':p,'budget':b,'summary':self.summ(100)} for p in ('concentrate','split') for b in ('protect','aggressive')]
        rows[0]['summary']['mechanism_trigger']['trigger_satisfied']=False
        r=mechanism_report(rows);self.assertFalse(r['paired'][0]['trigger_satisfied_in_all_cells']);self.assertFalse(r['paired'][0]['placement_ranking_reverses'])
if __name__=='__main__':unittest.main()
