import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ops

class Files(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.base=Path(self.t.name);self.root=self.base/'repo';self.root.mkdir()
    def tearDown(self):self.t.cleanup()
    def fixture(self):
        app=self.root/'spra_workflow_sched_v2_1';run=app/'runs/grid_v2_1';run.mkdir(parents=True)
        rt=self.base/'runtime';rt.mkdir();(rt/'experiment.lock').write_text('')
        specs=[];rows=[]
        for i,m in enumerate(['RR','P2C','Affinity','DAG'],1):
            label=f'R2_L2_T1_B1_{m}'
            spec=dict(label=label,replicas=2,rate=2.,repeat=1,block=1,position=i,method=m,base_cap=4096,
                seed=1,router_seed=2,packet_slice=[0,16])
            specs.append(spec);folder=run/'attempts'/label/'RAW';folder.mkdir(parents=True)
            s=dict(mean_jct_ms=80. if m=='DAG' else 100.,p95_jct_ms=120.,requests=1,workflows=1,output_tokens=8,
                condition_valid=True,budget_switches_total=0,prefill_caps_applied=[4096],semantic_quality={'final_em':.5,'final_f1':.6},
                native_method='D' if m=='DAG' else 'A')
            ops.put_json(folder/'summary.json',s);(folder/'backend.jsonl').write_text('{"test":true}\n')
            rows.append({**spec,'summary':s,'folder':str(folder.relative_to(run)),
                'summary_sha256':ops.digest(s),'external_work_sha256':'same',
                'files':{'summary.json':ops.file_sha(folder/'summary.json'),'backend.jsonl':ops.file_sha(folder/'backend.jsonl')}})
        p={'plan':{'suite':'grid','conditions':specs},'runtime':str(rt),'extension_identity':{},'run_uuid':'fixture'}
        ops.put_json(run/'protocol.json',p);ops.put_json(run/'protocol.sha256.json',{'sha256':ops.file_sha(run/'protocol.json')})
        ops.put_json(run/'suite_status.json',{'status':'complete','completed_conditions':len(rows)})
        ops.put_json(run/'index.json',rows)
        (app/'run.sh').write_text('#!/bin/bash\necho fixture\n')
        return app,run
    def alter(self,path,obj):path.write_text(json.dumps(obj))
    def test_discovery(self):
        a,r=self.fixture();self.assertEqual(ops.discover(self.root),([a],[r]))
    def test_source_inventory_keeps_failed_attempt(self):
        a,r=self.fixture();(r/'attempts/failure').mkdir();(r/'attempts/failure/failure.json').write_text('{}')
        files,_,_=ops.source_inventory(self.root,[a],[r],{})
        self.assertIn(str((r/'attempts/failure/failure.json').relative_to(self.root)),files)
    def test_source_inventory_keeps_operator_logs(self):
        a,r=self.fixture();p=self.root/'wf_ops_output/log';p.mkdir(parents=True);(p/'full.log').write_text('x')
        files,_,_=ops.source_inventory(self.root,[a],[r],{});self.assertIn('wf_ops_output/log/full.log',files)
    def test_safe_path(self):
        for n in ('../x','/etc/passwd'):
            with self.assertRaises(ValueError):ops.safe_file(self.root,n)
    def test_symlink(self):
        (self.root/'s').symlink_to('/etc/passwd')
        with self.assertRaises(ValueError):ops.safe_file(self.root,'s')
    def test_inspect(self):
        a,r=self.fixture();d=ops.inspect_run(r);self.assertEqual(d['coverage'][0]['accepted'],4)
    def test_hash_mismatch(self):
        a,r=self.fixture();(r/'protocol.json').write_text('{}')
        with self.assertRaises(ValueError):ops.inspect_run(r)
    def test_raw_summary_tamper(self):
        a,r=self.fixture();f=next((r/'attempts').glob('*/RAW/summary.json'));f.write_text('{}')
        with self.assertRaises(ValueError):ops.inspect_run(r)
    def test_duplicate_label(self):
        a,r=self.fixture();rows=ops.load(r/'index.json');rows.append(rows[0]);self.alter(r/'index.json',rows)
        with self.assertRaises(ValueError):ops.inspect_run(r)
    def test_changed_spec(self):
        a,r=self.fixture();rows=ops.load(r/'index.json');rows[0]['rate']=4;self.alter(r/'index.json',rows)
        with self.assertRaises(ValueError):ops.inspect_run(r)
    def test_running_refused(self):
        a,r=self.fixture();self.alter(r/'suite_status.json',{'status':'running'})
        with self.assertRaises(ValueError):ops.inspect_run(r)
    def test_false_complete(self):
        a,r=self.fixture();rows=ops.load(r/'index.json')[:-1];self.alter(r/'index.json',rows)
        self.alter(r/'suite_status.json',{'status':'complete','completed_conditions':len(rows)})
        with self.assertRaises(ValueError):ops.inspect_run(r)
    def test_comparisons(self):
        a,r=self.fixture();p,c=ops.comparisons('x',ops.inspect_run(r))
        self.assertEqual(len(c),3);self.assertTrue(all(x['jct_at_least_10_pct'] for x in c))
        self.assertAlmostEqual(c[0]['mean_jct_ms_gain_pct'],20.)
    def test_missing_pair_not_qualified(self):
        a,r=self.fixture();d=ops.inspect_run(r)
        for spec in list(d['protocol']['plan']['conditions']):
            other=dict(spec);other['label']+='extra';other['repeat']=2;d['protocol']['plan']['conditions'].append(other)
        d['status']['status']='paused';_,c=ops.comparisons('x',d)
        self.assertFalse(any(x['jct_at_least_10_pct'] for x in c))
    def test_external_work_mismatch(self):
        a,r=self.fixture();d=ops.inspect_run(r);d['rows'][0]['external_work_sha256']='bad'
        with self.assertRaises(ValueError):ops.comparisons('x',d)
    def test_failed_suite_not_qualified(self):
        a,r=self.fixture();d=ops.inspect_run(r);d['status']['status']='failed'
        _,c=ops.comparisons('x',d);self.assertFalse(any(x['jct_at_least_10_pct'] for x in c))
    def test_smoke_not_performance(self):
        a,r=self.fixture();d=ops.inspect_run(r);d['protocol']['plan']['suite']='smoke'
        _,c=ops.comparisons('x',d);self.assertFalse(any(x['jct_at_least_10_pct'] for x in c))
    def test_negative_results(self):
        a,r=self.fixture();d=ops.inspect_run(r);d['rows'][-1]['summary']['mean_jct_ms']=110
        _,c=ops.comparisons('x',d);self.assertTrue(all(x['mean_jct_ms_gain_pct']<0 for x in c))
    def test_full_checkpoint(self):
        a,r=self.fixture();out=self.base/'out';before={str(p):ops.file_sha(p) for p in ops.walk_regular(self.root)}
        with patch('sys.stdout',io.StringIO()):receipt=ops.checkpoint(self.root,out,True);ops.verify_archive(out/'full_raw.zip');ops.verify_archive(out/'review.zip')
        self.assertTrue(receipt['full_archive_created']);self.assertTrue((out/'ten_percent_candidates.csv').is_file())
        after={str(p):ops.file_sha(p) for p in ops.walk_regular(self.root)};self.assertEqual(before,after)
        with zipfile.ZipFile(out/'full_raw.zip') as z:self.assertTrue(any(n.endswith('backend.jsonl') for n in z.namelist()))
        with zipfile.ZipFile(out/'review.zip') as z:self.assertFalse(any(n.endswith('backend.jsonl') for n in z.namelist()))
    def test_existing_checkpoint_refused(self):
        self.fixture();out=self.base/'out';out.mkdir()
        with self.assertRaises(FileExistsError):ops.checkpoint(self.root,out,True)
    def test_review_only_explicit(self):
        self.fixture();out=self.base/'out'
        with patch('sys.stdout',io.StringIO()):r=ops.checkpoint(self.root,out,False)
        self.assertFalse(r['full_archive_created']);self.assertFalse((out/'full_raw.zip').exists())
    def test_verify_zip(self):
        p=self.root/'a';p.write_text('abc');z=self.base/'a.zip';ops.write_zip_stream(z,{'a':p})
        with patch('sys.stdout',io.StringIO()):self.assertEqual(ops.verify_archive(z),1)
    def test_stream_checks_recorded_hash(self):
        p=self.root/'a';p.write_text('abc')
        with self.assertRaises(ValueError):ops.write_zip_stream(self.base/'a.zip',{'a':p},{'a':'bad'})
    def test_logged_real_subprocess(self):
        app=self.root/'app';app.mkdir();event={'label':'x','replicas':2,'rate':2.,'repeat':1,'block':1,'method':'DAG',
            'mean_jct_ms':100.,'p95_jct_ms':140.,'mean_request_ttft_ms':20.,'cohort_workflows_s':2.,'requests':1,'output_tokens':8}
        raw='TYPED_RESPONSE={"answer":"hidden detail"}\nWF_CONDITION_ACCEPTED='+json.dumps(event)+'\nSTOP: test failure\n'
        (app/'payload').write_text(raw);(app/'run.sh').write_text('#!/bin/bash\ncat payload\nexit 7\n')
        logroot=self.base/'logs'
        with patch('sys.stdout',io.StringIO()):rc=ops.logged(app,logroot,['run','--confirm-gpu-use'])
        self.assertEqual(rc,7);d=next(logroot.iterdir());self.assertEqual((d/'full.log').read_text(),raw)
        self.assertNotIn('hidden detail',(d/'summary.log').read_text());self.assertIn('STOP:',(d/'summary.log').read_text())
        self.assertEqual(ops.load(d/'receipt.json')['accepted_lines_this_invocation'],1)
    def test_no_confirm_no_run(self):
        a,r=self.fixture()
        with self.assertRaises(ValueError):ops.logged(a,self.base/'logs',['run'])
    def test_log_outside_app(self):
        a,r=self.fixture()
        with self.assertRaises(ValueError):ops.logged(a,a/'logs',['gate'])
    def test_console_single_pair(self):
        screen=io.StringIO();events=io.StringIO();csvf=io.StringIO();c=ops.Console(screen,events,csvf)
        base=dict(label='x',replicas=3,rate=2.,repeat=1,block=1,p95_jct_ms=110.,mean_request_ttft_ms=10.,cohort_workflows_s=2.)
        for m,jct in [('DAG',90.),('Affinity',100.)]:c.feed(('WF_CONDITION_ACCEPTED='+json.dumps({**base,'method':m,'mean_jct_ms':jct})).encode())
        self.assertIn('+10.00%',screen.getvalue());self.assertEqual(len(events.getvalue().splitlines()),2)
    def test_bad_json_preserved_warning(self):
        sc=io.StringIO();c=ops.Console(sc,io.StringIO(),io.StringIO());c.feed(b'WF_CONDITION_ACCEPTED={bad}')
        self.assertEqual(c.errors,1);self.assertIn('WARNING',sc.getvalue())
    def test_weight_exclusion_explicit(self):
        a,r=self.fixture();(a/'model.safetensors').write_bytes(b'x');files,missing,omitted=ops.source_inventory(self.root,[a],[r],{})
        self.assertFalse(any(n.endswith('.safetensors') for n in files));self.assertEqual(len(omitted),1)
    def test_cli_help(self):
        p=subprocess.run([sys.executable,str(Path(ops.__file__)),'--help'],capture_output=True)
        self.assertEqual(p.returncode,0)

if __name__=='__main__':unittest.main()
