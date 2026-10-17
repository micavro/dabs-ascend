"""Screen selection, raw scoring audit and workflow doubles; no NPU claims."""
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dabs_ascend import r234
from dabs_ascend.cli import prepare_args, make_plan
from dabs_ascend.common import append_json, read_json, write_json
from dabs_ascend.native_algorithms import choose
from dabs_ascend.native_planning import CONTRACT, run_planner
from dabs_ascend.planning_audit import audit
from dabs_ascend.r234_report import make_report, paired_rows
from dabs_ascend.r234_run import stable_arguments
from test_native import model
from test_native_planning import opportunity
import test_native_workflow as workflow_tests

entry = workflow_tests.entry


def screening_fixture(directory, group='all', scored=True):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    plan = r234.screening_plan(group)
    cfg, s = opportunity(); r234.planner_ready(True)
    window = {'contract': CONTRACT, 'prepared_at': time.monotonic(), 'deadline': time.monotonic()+10}
    result = run_planner(choose, s, model(), cfg, window) if scored else {
        'evaluations': [], 'v5_stop_reason': 'no_distinct_executable_candidate', 'planning_finished_at': time.monotonic()}
    summaries = []
    for spec in plan:
        summaries.append({'spec': spec, 'functional_pass': True, 'fixed_work_pass': True,
                          'dependency_handoff_pass': True, 'cpu_fixture_only': True,
                          'external_work_fingerprint': 'fixture', 'arrival_offsets': [0, 1],
                          'workflow_rows': [], 'jct_mean_ms': 100, 'jct_p95_ms': 120, 'workflows_s': 1,
                          'output_tokens': 13184, 'selector_scored_decisions': int(scored),
                          'v5_candidate_scored': int(scored), 'nonreference_actions': 0})
        if spec['method'] == 'Ours':
            raw = directory/'conditions'/spec['label']/'decisions.jsonl'
            raw.parent.mkdir(parents=True, exist_ok=True)
            append_json(raw,
                        {'snapshot': s, 'planning_window': window, 'result': result})
    write_json(directory/'index.json', summaries)
    write_json(directory/'protocol.json', {'screening': group, 'plan': plan})
    write_json(directory/'status.json', {'status': 'complete', 'completed_conditions': len(plan), 'functional_test_pass': True})
    return summaries


class ScreeningTests(unittest.TestCase):
    def test_exact_three_groups_preserve_every_spec_and_method_order(self):
        full = r234.plan(); saved = copy.deepcopy(full)
        screen = r234.screening_plan()
        self.assertEqual(len(screen), 12)
        self.assertEqual([(s['replicas'], s['rate']) for s in screen[::4]], [(3, 8), (4, 4), (3, 4)])
        self.assertEqual([s['method'] for s in screen[:4]], ['SLL', 'Ayo', 'Ours', 'RR'])
        self.assertEqual([s['method'] for s in screen[4:8]], ['Ayo', 'Ours', 'RR', 'SLL'])
        for group in r234.SCREENING_CASES:
            subset = r234.screening_plan(group)
            self.assertEqual(len(subset), 4)
            self.assertEqual({s['method'] for s in subset}, set(r234.METHODS))
            self.assertTrue(all(s['repeat'] == 1 and s in full for s in subset))
        self.assertEqual(r234.plan(), saved)
        screen[0]['seed'] = -1
        self.assertEqual(r234.plan(), saved)

    def test_cli_selection_is_explicit_and_cannot_resume_or_change_work(self):
        base = ['plan', '--suite', 'r234', '--devices', '0,1,2,3']
        args = prepare_args(base, existing=True)
        self.assertEqual(len(make_plan(args)), 96)
        self.assertNotIn('screening', stable_arguments(args))
        self.assertEqual(len(make_plan(prepare_args(base+['--screening'], existing=True))), 12)
        self.assertEqual(len(make_plan(prepare_args(base+['--screening', 'r3-l8'], existing=True))), 4)
        for extra in (['--resume', 'old'], ['--max-conditions', '4'], ['--rate', '16'],
                      ['--execution-scope', 'http-limited'], ['--exploratory-calibration']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                prepare_args(base+['--screening']+extra, existing=True)

    def test_subset_audit_requires_every_group_and_keeps_failures(self):
        for group in ('all', *r234.SCREENING_CASES):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                summaries = screening_fixture(tmp, group)
                coverage = audit(tmp, len(summaries), screening=group)
                self.assertTrue(coverage['pass'], coverage)
                self.assertEqual(len(coverage['conditions']), len(summaries)//4)
                self.assertFalse(coverage['full_matrix_acceptance'])
                self.assertFalse(audit(tmp, len(summaries))['pass'])
                write_json(Path(tmp)/'index.json', summaries[:-1])
                self.assertFalse(audit(tmp, len(summaries), screening=group)['pass'])
        with tempfile.TemporaryDirectory() as tmp:
            summaries = screening_fixture(tmp, scored=False)
            coverage = audit(tmp, 12, screening='all')
            self.assertFalse(coverage['pass'])
            self.assertEqual(len(coverage['conditions']), 3)
            self.assertEqual(len(paired_rows(summaries)), 9)

    def test_subset_report_cannot_claim_full_matrix_or_performance_acceptance(self):
        for group in ('all', 'r3-l8'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp); summaries = screening_fixture(out, group)
                write_json(out/'environment.json', {'passed': True})
                write_json(out/'native_gate.json', {'pass': True, 'live_verified': True})
                write_json(out/'ascend_profile.json', {'original_gpu_timing_equivalent': True})
                write_json(out/'canary_summary.json', [{'passed': True, 'replica': r, 'fixed_output_budget': n}
                    for r in range(4) for n in (16, 32, 48, 64, 128)])
                report = make_report(out, r234.screening_plan(group), summaries, 'complete')
                self.assertTrue(report['functional_test_pass'])
                self.assertTrue(report['screening_completed'])
                for field in ('formal_acceptance', 'full_matrix_complete', 'strict_native_reproduction_pass'):
                    self.assertFalse(report[field], field)
                self.assertEqual(report['declared_conditions'], len(summaries))
                text = (out/'TEST_REPORT.md').read_text(encoding='utf-8')
                self.assertIn(f"{len(summaries)}/{len(summaries)}", text)
                self.assertIn('诊断筛查', text)
                self.assertNotIn(' / 24', text)


class ScreeningWorkflowTests(unittest.TestCase):
    setUp = workflow_tests.NativeWorkflowTests.setUp
    fake_launch = workflow_tests.NativeWorkflowTests.fake_launch
    run_workflow = workflow_tests.NativeWorkflowTests.run_workflow

    def screen_launch(self, command, logfile):
        if command[2] != 'benchmark' or self.failure:
            return self.fake_launch(command, logfile)
        self.commands.append(command)
        target = self.current.host_path(command[command.index('--out')+1])
        group = command[command.index('--screening')+1]
        screening_fixture(target, group, scored=False)
        logfile.write_text('CPU-only screening double', encoding='utf-8')
        return 0

    def test_blocked_verify_raw_calibration_reused_all_groups_finish_without_scoring_pass(self):
        tested, code = self.run_workflow()
        self.assertEqual(code, 0)
        verified = self.root/'outputs/native_verify_fixture'
        write_json(verified/'run.json', {'mode': 'verify', 'status': 'blocked', 'acceptance': tested.record['acceptance']})
        write_json(self.root/'outputs/native-entry-last-verify.json', {'output': verified.relative_to(self.root).as_posix()})
        before = (verified/'run.json').read_bytes()
        self.commands = []
        self.current = entry.Workflow(self.root/'new-checkout', 'screen', input_path=verified)
        with patch.object(self.current, 'launch', side_effect=self.screen_launch), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 0, self.current.record)
        run = self.current
        self.assertEqual([c[2] for c in self.commands], ['benchmark'])
        self.assertTrue(run.record['screening_completed'])
        self.assertFalse(run.record['planning_pass'])
        self.assertFalse(run.record['formal_acceptance'])
        self.assertEqual(run.record['status'], 'completed')
        self.assertEqual(run.record['completed_conditions'], 12)
        coverage = read_json(run.out/'planning_audit.json')
        self.assertEqual(len(coverage['conditions']), 3)
        self.assertEqual((verified/'run.json').read_bytes(), before)
        self.assertFalse((run.root/'outputs/native-entry-last-verify.json').exists())
        text = ''.join(p.read_text(encoding='utf-8') for p in sorted((run.out/'copy_logs').glob('COPY_00_overview_*.txt')))
        self.assertIn('planning_pass=False', text)
        self.assertIn('SCREEN_PLAN', text)

    def test_single_group_and_functional_failure_do_not_become_scoring_pass(self):
        tested, _ = self.run_workflow()
        self.commands = []
        self.current = entry.Workflow(self.root, 'screen', input_path=tested.out, screen_group='r4-l4')
        with patch.object(self.current, 'launch', side_effect=self.screen_launch), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 0, self.current.record)
        self.assertEqual(self.current.record['completed_conditions'], 4)
        self.assertFalse(self.current.record['planning_pass'])
        self.failure = 'benchmark'
        self.current = entry.Workflow(self.root, 'screen', input_path=tested.out)
        with patch.object(self.current, 'launch', side_effect=self.screen_launch), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 2)
        self.assertFalse(self.current.record['screening_completed'])
        self.assertEqual(self.current.record['status'], 'blocked')

    def test_default_screen_uses_latest_verify_but_experiment_cannot_import_it(self):
        tested, _ = self.run_workflow()
        verified = self.root/'outputs/native_verify_fixture'
        write_json(verified/'run.json', {'mode': 'verify', 'status': 'blocked', 'acceptance': tested.record['acceptance']})
        write_json(self.root/'outputs/native-entry-last-verify.json', {'output': verified.relative_to(self.root).as_posix()})
        self.current = entry.Workflow(self.root, 'screen')
        with patch.object(self.current, 'launch', side_effect=self.screen_launch), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 0)
        self.assertEqual(self.current.record['imported_test_source'], str(verified.resolve()))
        experiment = entry.Workflow(self.root, 'experiment', input_path=verified)
        with self.assertRaises(entry.Stopped):
            experiment.import_test()
