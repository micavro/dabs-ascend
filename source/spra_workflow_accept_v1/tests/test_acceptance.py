from __future__ import annotations
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from wfsched import acceptance as a
from wfsched import accept_report as rep
from wfsched.common import HERE,sha,load

class CaseDesignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name)
        self.basis=self.d/'evidence.txt';self.basis.write_text('synthetic selection provenance, not performance')
        self.reg=self.d/'cases.json'
    def tearDown(self):self.tmp.cleanup()
    def regplan(self,cases=('4:2:3',)):
        a.register(list(cases),self.basis,'test selection only',self.reg)
        return a.case_plan(self.reg)
    def test_32_conditions_one_case(self):
        p=self.regplan();self.assertEqual(p['expected_conditions'],32)
        self.assertEqual({r['block'] for r in p['conditions']},{3})
    def test_96_conditions_three_cases(self):
        p=self.regplan(('4:2:3','3:2:4','2:4:1'));self.assertEqual(len(p['conditions']),96)
    def test_exact_four_methods(self):self.assertEqual(set(self.regplan()['methods']),{'DAG','Affinity','P2C','RR'})
    def test_position_balance(self):
        c=Counter((r['method'],r['position']) for r in self.regplan()['conditions'])
        self.assertEqual(set(c.values()),{2})
    def test_adjacent_pair_balance(self):
        rows=defaultdict(list)
        for r in self.regplan()['conditions']:rows[r['repeat']].append(r['method'])
        c=Counter(x for v in rows.values() for x in zip(v,v[1:]))
        self.assertEqual(len(c),12);self.assertEqual(set(c.values()),{2})
    def test_paired_inputs_and_fixed_budget(self):
        rows=defaultdict(list)
        for r in self.regplan()['conditions']:
            self.assertEqual(r['base_cap'],4096);self.assertEqual(r['packet_slice'],[32,48]);rows[r['repeat']].append(r)
        for rs in rows.values():self.assertEqual(len({r['seed'] for r in rs}),1)
    def test_no_duplicate_labels(self):
        p=self.regplan(('4:2:3','3:2:4'));self.assertEqual(len({r['label'] for r in p['conditions']}),64)
    def test_selection_provenance_recorded(self):
        p=self.regplan();self.assertTrue(p['selection']['selected_after_observing_historical_results'])
        self.assertEqual(p['selection']['selection_basis']['sha256'],sha(self.basis));self.assertFalse(p['held_out_questions'])
    def test_no_registration_overwrite(self):
        self.regplan()
        with self.assertRaises(FileExistsError):a.register(['4:2:2'],self.basis,'second',self.reg)
    def test_no_duplicate_cell_blocks(self):
        with self.assertRaises(ValueError):a.register(['4:2:1','4:2:3'],self.basis,'test',self.reg)
    def test_bad_cases(self):
        for s in ('0:2:1','3:nan:1','4:3:1','4:2:5','4:2','x:2:1'):
            with self.subTest(s=s),self.assertRaises(ValueError):a.parse_case(s)
    def test_no_empty_reason(self):
        with self.assertRaises(ValueError):a.register(['4:2:3'],self.basis,'',self.reg)
    def test_missing_evidence(self):
        with self.assertRaises(FileNotFoundError):a.register(['4:2:3'],self.d/'none','test',self.reg)
    def test_stale_registration_edits_fail(self):
        self.regplan();j=load(self.reg);j['rounds']=1;self.reg.write_text(json.dumps(j))
        with self.assertRaises(ValueError):a.case_plan(self.reg)
    def test_original_algorithm_bytes(self):
        j=load(HERE/'review_accept/UNCHANGED_IMPLEMENTATION.json')
        for name,h in j['files'].items():self.assertEqual(sha(HERE/name),h,name)

class BlockReportTests(unittest.TestCase):
    def pairs(self):
        ps=[]
        for t in (1,2):
            for b in (1,2):
                for ref in ('RR','P2C','Affinity'):
                    v=12 if b==1 else (20 if t==1 else -20)
                    p={'replicas':3,'rate':2.,'repeat':t,'block':b,'candidate':'DAG','reference':ref,
                       'candidate_mean_jct_ms':100,'reference_mean_jct_ms':110,
                       'mean_jct_ms_gain_pct':v,'final_em_delta_pp':0,'final_f1_delta_pp':-1}
                    for f in rep.FIELDS:p.setdefault(f+'_gain_pct',v)
                    ps.append(p)
        return ps
    def design(self):
        return [dict(label=f'{t}_{b}_{m}',replicas=3,rate=2.,repeat=t,block=b,method=m)
            for t in (1,2) for b in (1,2) for m in ('DAG','Affinity','P2C','RR')]
    def test_no_four_block_average(self):
        rows=rep.by_block(self.pairs(),self.design());self.assertEqual(len(rows),6)
        self.assertEqual({r['mean_jct_ms_gain_pct'] for r in rows},{12,0})
    def test_keep_negative_repeat(self):
        r=[r for r in rep.by_block(self.pairs(),self.design()) if r['block']==2][0]
        self.assertEqual(r['jct_min_gain_pct'],-20);self.assertEqual(r['jct_positive_pairs'],1)
    def test_missing_repeat_is_incomplete(self):
        r=rep.by_block([p for p in self.pairs() if p['repeat']==1],self.design())
        self.assertFalse(any(z['complete_case_comparison'] for z in r))
        self.assertFalse(any(z['point_estimate_at_least_10pct'] for z in r))
    def test_duplicate_pair_rejected(self):
        ps=self.pairs()
        with self.assertRaises(ValueError):rep.by_block(ps+[ps[0]],self.design())
    def test_exploratory_is_separate_and_labeled(self):
        s=rep.exploratory_examples(self.pairs());self.assertEqual([x['block'] for x in s],[2,1])
        self.assertTrue(all(x['not_confirmatory'] and x['not_overall_mean'] for x in s))
    def test_missing_baseline_no_selected_example(self):
        ps=[p for p in self.pairs() if p['reference']!='RR'];self.assertEqual(rep.exploratory_examples(ps),[])
    def test_path_traversal(self):
        with self.assertRaises(ValueError):rep.safe_file(Path('/tmp'),'../outside')

class FetchTests(unittest.TestCase):
    def run_fetch(self,fail=False,authfail=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cpu=root/'cpu';cpu.mkdir();bin=root/'bin';bin.mkdir();key=root/'key';key.write_text('synthetic, not a private key')
            source=root/'source';source.mkdir();(source/'review.zip').write_bytes(b'synthetic archive')
            (source/'review.zip.sha256').write_text(sha(source/'review.zip')+'  review.zip\n')
            log=root/'trace'
            stub='''#!/usr/bin/env python3
import os,sys,pathlib,shutil
name=pathlib.Path(sys.argv[0]).name
with open(os.environ['MOCK_LOG'],'a') as f:f.write(name+' '+' '.join(sys.argv[1:])+'\\n')
if name=='ssh':
 if '-fN' in sys.argv and os.environ.get('AUTHFAIL')=='1':sys.exit(255)
 sys.exit(0)
remote,dest=sys.argv[-2:]
base=remote.rsplit('/',1)[-1]
if base=='review.zip' and os.environ.get('COPYFAIL')=='1':sys.exit(255)
shutil.copyfile(pathlib.Path(os.environ['MOCK_SOURCE'])/base,dest)
'''
            for n in ('ssh','scp'):(bin/n).write_text(stub);(bin/n).chmod(0o755)
            env=dict(os.environ,WF_CPU_DIR=str(cpu),WF_SSH_KEY=str(key),MOCK_SOURCE=str(source),MOCK_LOG=str(log),
                     COPYFAIL=str(int(fail)),AUTHFAIL=str(int(authfail)))
            env['PATH']=str(bin)+os.pathsep+env['PATH']
            p=subprocess.run(['bash',str(HERE/'WF_fetch_saved_review.sh')],env=env,text=True,capture_output=True,timeout=15)
            return p,(log.read_text() if log.exists() else '')
    def test_transfer_success_one_master(self):
        p,t=self.run_fetch();self.assertEqual(p.returncode,0,p.stderr);self.assertIn('TRANSFER_VERIFIED=',p.stdout);self.assertEqual(t.count(' -fN '),1)
    def test_missing_archive_stops_before_checksum(self):
        p,t=self.run_fetch(fail=True);self.assertNotEqual(p.returncode,0);self.assertNotIn('TRANSFER_VERIFIED=',p.stdout)
        self.assertEqual(len([x for x in t.splitlines() if x.startswith('scp ')]),1)
    def test_failed_auth_starts_no_copy(self):
        p,t=self.run_fetch(authfail=True);self.assertNotEqual(p.returncode,0);self.assertNotIn('scp ',t)
