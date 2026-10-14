"""Actual frozen source, synthetic pressure: no model service or GPU request."""
from __future__ import annotations
import copy
import json
import multiprocessing
import os
import tempfile
import subprocess
import sys
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from bootstrap import CFG, MODEL
from wfsched.bench import snapshot
from wfsched.admission import schedule_alternative, binding, _ORIGINAL, REVISION
from wfsched.worker import process_plan, ready
from joint18v2.search import reference
from joint18v2.actions import resources
from joint18v2.common import validate_action, action_key

class AdmissionRepairTests(unittest.TestCase):
    def inputs(self, R=1, index=1, capacity=50):
        s=snapshot(R,2)
        s.update(now=1.,decision_index=index,capacity_blocks=[capacity]*R,reserved_blocks=[0]*R)
        c=dict(CFG,replicas=R,base_cap=4096)
        return s,c,reference(s,c)

    def test_original_failure_reproduced_without_gpu(self):
        s,c,a=self.inputs()
        self.assertTrue(a['resource_held'])
        validate_action(a,s,c)
        self.assertFalse(resources(a,s,c)['resource_held'])
        with self.assertRaisesRegex(ValueError,'^age_admission$'):_ORIGINAL(s,MODEL,c,a)

    def test_repaired_candidate_keeps_genuine_resource_holds(self):
        s,c,a=self.inputs();z=schedule_alternative(s,MODEL,c,a)
        self.assertTrue(validate_action(z,s,c))
        self.assertEqual(set(z['admit'])|set(z['resource_held']),set(s['ready']))
        self.assertEqual(len(z['admit']),1)
        self.assertEqual(z['caps'],[4096])

    def test_original_no_pressure_action_is_exactly_unchanged(self):
        for R in (1,2,3,4):
            for index in (1,3):
                s,c,a=self.inputs(R,index,4096)
                self.assertEqual(schedule_alternative(s,MODEL,c,a),_ORIGINAL(s,MODEL,c,a))

    def test_input_snapshots_and_base_not_mutated(self):
        s,c,a=self.inputs();before=copy.deepcopy((s,c,a))
        schedule_alternative(s,MODEL,c,a)
        self.assertEqual((s,c,a),before)

    def test_partial_partition_is_not_hidden(self):
        s,c,a=self.inputs();a['resource_held'].pop()
        with self.assertRaisesRegex(ValueError,'partition'):schedule_alternative(s,MODEL,c,a)

    def test_duplicate_or_overlapping_partition_rejected(self):
        for overlap in (False,True):
            s,c,a=self.inputs();a['resource_held'].append(a['admit'][0] if overlap else a['resource_held'][0])
            with self.assertRaisesRegex(ValueError,'partition'):schedule_alternative(s,MODEL,c,a)

    def test_age_validator_is_still_enforced(self):
        s,c,a=self.inputs();a['resource_held']=[]
        with self.assertRaisesRegex(ValueError,'^age_admission$'):validate_action(a,s,c)

    def test_sequence_limit_remains_enforced(self):
        s,c,_=self.inputs(capacity=4096);c['max_num_seqs']=1;a=reference(s,c)
        z=schedule_alternative(s,MODEL,c,a);validate_action(z,s,c)
        self.assertEqual(len(z['admit']),1);self.assertEqual(len(z['resource_held']),3)

    def test_zero_capacity_never_forces_an_aged_admission(self):
        s,c,a=self.inputs(capacity=0);z=schedule_alternative(s,MODEL,c,a)
        self.assertEqual(z['admit'],[]);self.assertEqual(set(z['resource_held']),set(s['ready']))
        validate_action(z,s,c)

    def test_rerouted_previously_held_request_is_reconsidered(self):
        s,c,a=self.inputs(R=2,capacity=50)
        s['capacity_blocks']=[0,4096]
        for k in a['placement']:a['placement'][k]=0
        a['admit']=[];a['resource_held']=list(s['ready'])
        z=schedule_alternative(s,MODEL,c,a)
        self.assertTrue(z['admit'])
        self.assertTrue(all(z['placement'][k]==1 for k in z['admit']))
        validate_action(z,s,c)

    def test_bounded_admission_policy_is_not_overridden(self):
        s,c,_=self.inputs(index=3,capacity=50)
        # One incumbent is enough to activate the original soft rule in this test.
        n=copy.deepcopy(s['nodes']['w0/0']);n.update(id='inc',wid='old',state='submitted',replica=0,phase='decode',compiled_groups=[],compiled_successors=[])
        s['nodes']['inc']=n;s['now']=.1;c['admission_soft_active']=1
        a=reference(s,c);z=schedule_alternative(s,MODEL,c,a)
        self.assertTrue(a['resource_held'])
        self.assertIn('bounded_admission',z['family'])
        self.assertLess(len(set(z['admit'])|set(z['resource_held'])),len(s['ready']))
        validate_action(z,s,c)

    def test_old_resources_function_is_unchanged(self):
        import joint18v2.actions as A
        f=A.resources
        with binding():self.assertIs(A.resources,f)
        self.assertIs(A.resources,f)

    def test_binding_restores_original_even_on_failure(self):
        import semantic19.selector as S
        saved=S.schedule_alternative
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with binding():
                self.assertIs(S.schedule_alternative,schedule_alternative)
                raise RuntimeError('synthetic')
        self.assertIs(S.schedule_alternative,saved)

    def test_many_resource_limits_and_decision_indices(self):
        for R in (1,2,3,4):
            for cap in (0,50,100,4096):
                for index in (1,2,3,6):
                    s,c,a=self.inputs(R,index,cap)
                    z=schedule_alternative(s,MODEL,c,a)
                    validate_action(z,s,c)
                    self.assertEqual(z['caps'],[4096]*R)
                    # Original projector must agree that final admitted set fits.
                    self.assertEqual(resources(z,s,c)['admit'],z['admit'])

    def test_both_worker_versions_complete_under_pressure(self):
        for m in ('DAG','DAG-Lite'):
            s,c,a=self.inputs();c['wf_public_method']=m
            ans=process_plan(s,MODEL,c,'D',time.monotonic()+5)
            validate_action(ans['action'],s,c)
            self.assertEqual(ans['wf_admission_revision'],REVISION)
            self.assertTrue(all(e['cell'] in ('00','10') for e in ans['evaluations']))
            self.assertGreater(ans['wf_admission_repair_proposals'],0)

    def test_worker_failure_snapshot_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            s,c,a=self.inputs();c.update(wf_public_method='DAG',wf_worker_failure_dir=td)
            with patch('semantic19.selector.choose',side_effect=ValueError('test diagnostic')):
                with self.assertRaisesRegex(ValueError,'test diagnostic'):process_plan(s,MODEL,c,'D',time.monotonic()+5)
            files=list(Path(td).glob('failure_*.json'));self.assertEqual(len(files),1)
            v=json.loads(files[0].read_text());self.assertEqual(v['snapshot'],s)
            self.assertIn('ValueError',v['traceback']);self.assertFalse(v['automatic_retry'])

    def test_real_spawn_process_pressure_case(self):
        with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertNotEqual(pool.submit(ready).result(timeout=15),os.getpid())
            for m in ('DAG','DAG-Lite'):
                s,c,_=self.inputs();c['wf_public_method']=m
                ans=pool.submit(process_plan,s,MODEL,c,'D',time.monotonic()+5).result(timeout=15)
                self.assertEqual(ans['wf_admission_revision'],REVISION)
                validate_action(ans['action'],s,c)

    def test_agecheck_is_not_a_performance_suite(self):
        from wfsched.design import plan
        p=plan('agecheck')
        self.assertEqual(len(p['conditions']),10)
        self.assertEqual({(r['replicas'],r['rate']) for r in p['conditions']},{(1,4.),(1,8.)})
        self.assertEqual({r['block'] for r in p['conditions']},{1})
        self.assertEqual(p['admission_intent_revision'],REVISION)

    def test_runner_imports_before_native_source_activation(self):
        app=Path(__file__).resolve().parents[1]
        script="import sys;sys.path.insert(0,"+repr(str(app))+");import wfsched.runner;print('BOOTSTRAP_OK')"
        v=subprocess.run([sys.executable,'-I','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(v.returncode,0,v.stderr)
        self.assertIn('BOOTSTRAP_OK',v.stdout)
