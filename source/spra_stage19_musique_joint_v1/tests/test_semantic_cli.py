"""Acquisition and run/report contracts using generated fixtures, never real data claims."""
from __future__ import annotations
import contextlib,copy,importlib.util,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from test_semantic19 import raw
from semantic19.dataset import compile_record,convert,sha
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class AcquisitionTests(unittest.TestCase):
    def test_official_archive_names_imported_and_labels_separated(self):
        m=module('fetch19test',ROOT/'tools/fetch_musique.py')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);archive=d/'fixture.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('data/musique_ans_v1.0_train.jsonl',''.join(json.dumps(raw(i,i%2==0))+'\n' for i in range(40)))
                z.writestr('data/musique_ans_v1.0_dev.jsonl',''.join(json.dumps(raw(1000+i,i%2==0))+'\n' for i in range(70)))
            with patch.object(sys,'argv',['fetch','--archive',str(archive),'--download-dir',str(d/'source'),'--out',str(d/'compiled')]),contextlib.redirect_stdout(io.StringIO()) as out:
                m.main()
            self.assertIn('DATASET_READY=',out.getvalue());man=json.loads((d/'compiled/manifest.json').read_text())
            self.assertEqual(man['counts'],{'smoke':2,'profile':32,'pilot':64})
            self.assertNotIn('GOLD_SECRET',(d/'compiled/pilot.public.jsonl').read_text())
            self.assertIn('GOLD_SECRET',(d/'compiled/evaluation_gold.jsonl').read_text())
    def test_ambiguous_archive_rejected(self):
        m=module('fetch19bad',ROOT/'tools/fetch_musique.py')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);archive=d/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('a/musique_ans_v1.0_train.jsonl','{}');z.writestr('b/musique_ans_v1.0_train.jsonl','{}')
            with patch.object(sys,'argv',['fetch','--archive',str(archive),'--download-dir',str(d/'source'),'--out',str(d/'out')]):
                with self.assertRaises(ValueError):m.main()
    def test_confirmation_form(self):
        m=module('fetch19form',ROOT/'tools/fetch_musique.py');f=m.Form()
        f.feed('<form id="download-form" action="https://drive.usercontent.google.com/download"><input name="id" value="known"><input name="confirm" value="t"></form>')
        self.assertEqual(f.fields['confirm'],'t');self.assertEqual(f.action,'https://drive.usercontent.google.com/download')
    def test_duplicate_paragraph_rejected(self):
        r=raw();r['paragraphs'][1]['idx']=r['paragraphs'][0]['idx']
        with self.assertRaises(ValueError):compile_record(r)
    def test_no_overwrite_existing_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'manifest.json').write_text('{}')
            with self.assertRaises(FileExistsError):convert(d/'missing1',d/'missing2',d)

class ReportAndFreezeTests(unittest.TestCase):
    def test_all_methods_reported_with_variable_outputs(self):
        m=module('run19reports',ROOT/'tools/run_semantic.py');rows=[]
        for i,name in enumerate(m.METHODS):
            s={x:float(100+i) for x in ['mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','p95_request_ttft_ms','mean_node_time_to_first_content_ms','mean_client_tpot_proxy_ms','p95_client_tpot_proxy_ms']}
            s.update(output_tokens=100+i,cohort_tokens_s=100.,semantic_quality={'final_em':.5,'final_f1':.75})
            rows.append({'block':1,'method':name,'summary':s})
        result=m.progress_summary(rows)
        self.assertFalse(result['paired'][0]['actual_work_identical']);self.assertEqual(len(result['paired'][0]['methods']),5)
        self.assertFalse(result['formal_nonregression_established'])
    def test_incomplete_profile_not_frozen(self):
        m=module('run19incomplete',ROOT/'tools/run_semantic.py')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'index.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'20 unique'):m.freeze_profile(d,d/'missing',d)
    def test_profile_guard_preserves_quality_and_uses_all_20(self):
        m=module('run19freeze',ROOT/'tools/run_semantic.py')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);run=d/'run';run.mkdir();ds=d/'dataset';ds.mkdir();(ds/'manifest.json').write_text('{}')
            legacy=d/'legacy.json';legacy.write_text('{"models":{}}');idx=[]
            (run/'frozen_protocol.json').write_text(json.dumps({'config':{'arrival_rate':2.0}}))
            for b in range(1,5):
                for c in (256,512,1024,2048,4096):
                    f=run/f'B{b}_{c}';f.mkdir();q=.1 if c==4096 else .8
                    s={'condition_valid':True,'mean_jct_ms':10 if c==4096 else 900 if c==512 else 1000,'semantic_quality':{'final_em':q,'final_f1':q}}
                    (f/'nodes.json').write_text(json.dumps({'a':{'hop_count':2,'stream':{'usage':{'completion_tokens':23}}}}))
                    for name,val in [('summary.json',json.dumps(s)),('backend.jsonl',''),('manifest.json','{}')]: (f/name).write_text(val)
                    idx.append({'block':b,'cap':c,'folder':str(f),'summary':s})
            (run/'index.json').write_text(json.dumps(idx))
            with patch.object(m,'ROOT',d),patch('joint18v2.service_fit.read_rows',return_value=([],{})),patch('joint18v2.service_fit.fit_phase_model',return_value=({'models':{}},{})),contextlib.redirect_stdout(io.StringIO()):
                out=m.freeze_profile(run,legacy,ds)
            manifest=json.loads((out/'manifest.json').read_text());self.assertEqual(manifest['best_fixed_cap'],512)
            self.assertEqual(len(manifest['fixed_candidates']),5);self.assertEqual(manifest['profile_configuration']['arrival_rate'],2.)
    def test_prepared_mutation_rejected(self):
        m=module('run19prepared',ROOT/'tools/run_semantic.py')
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'manifest.json').write_text(json.dumps({'files':{'model.json':'wrong'}}));(d/'model.json').write_text('{}')
            from types import SimpleNamespace
            with self.assertRaises(ValueError):m.prepared(SimpleNamespace(prepared=d))
    def test_wave_low_rate_does_not_hang(self):
        from semantic19.workflow import arrival_offsets
        with self.assertRaises(ValueError):arrival_offsets(2,.01,0)
