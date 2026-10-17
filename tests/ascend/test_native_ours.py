"""Bounded Ours entry/report tests, using CPU doubles rather than NPU claims."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from dabs_ascend import cli, r234, r234_run, native_ours_report
from dabs_ascend.common import read_json, write_json
from dabs_ascend.copy_log import ConditionLog, enable_parts
from dabs_ascend.native_shadow_timing import profile_path
from test_native import model, resources
from test_native_simplification import capability
from test_port import FakeTokenizer
import test_native_workflow as workflows


def summary(spec):
    return dict(spec=spec, functional_pass=True, fixed_work_pass=True, dependency_handoff_pass=True,
                completed_requests=216, output_tokens=13184, jct_mean_ms=123, jct_p95_ms=150,
                external_work_fingerprint='same-fixed-work', arrival_offsets=[],
                engine_applied_nonreference_actions=0,
                runtime_diagnostics={'stages': {'decision_opportunities': 10, 'planner_dispatched': 8}})


def canaries():
    return [dict(replica=r, fixed_output_budget=n, passed=True)
            for r in range(4) for n in (16, 32, 48, 64, 128)]


class OursEntryTests(unittest.TestCase):
    def test_ours_prefix_is_exactly_the_existing_24_specs(self):
        original = [s for s in r234.plan() if s['method'] == 'Ours']
        self.assertEqual(len(original), 24)
        for n in (None, 1, 6, 12, 24):
            argv = ['plan', '--suite', 'r234', '--devices', '0,1,2,3', '--ours']
            if n is not None:
                argv += ['-n', str(n)]
            args = cli.prepare_args(argv, existing=True)
            self.assertEqual(cli.make_plan(args), original[:n])
            with redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(r234_run.main(args), 0)
            report = json.loads(printed.getvalue())
            self.assertTrue(report['ours_only'])
            self.assertFalse(report['formal_acceptance'])
            self.assertEqual(report['plan'], original[:n])

    def test_ours_rejects_shadow_resume_exploratory_and_out_of_range(self):
        base = ['plan', '--suite', 'r234', '--devices', '0,1,2,3', '--ours']
        for extra in (['-n', '0'], ['-n', '25'], ['--ours-timing'], ['--screening'],
                      ['--resume', 'old'], ['--max-conditions', '4'],
                      ['--timing-profile', 'old.json'], ['--exploratory-calibration'],
                      ['--execution-scope', 'http-limited']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.prepare_args(base+extra, existing=True)
        for argv in (['ours', '-n', '0'], ['ours', '-n', '25'], ['ours', '--group', 'all'], ['ours-12', '-n', '6']):
            with patch.object(sys, 'argv', ['run_native_workflow.py', *argv]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                workflows.entry.main()

    def test_report_requires_live_gate_canaries_exact_work_and_no_error(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            plan = r234.ours_timing_plan(limit=6)
            write_json(out/'protocol.json', {'ours_only': True, 'timing_count': 6})
            rows = [summary(s) for s in plan]
            self.assertFalse(native_ours_report.make_report(out, plan, rows, 'complete')['functional_test_pass'])
            write_json(out/'native_gate.json', {'pass': True, 'live_verified': True})
            write_json(out/'environment.json', {'passed': True})
            write_json(out/'canary_summary.json', canaries())
            result = native_ours_report.make_report(out, plan, rows, 'complete')
            self.assertTrue(result['functional_test_pass'])
            self.assertFalse(result['formal_acceptance'])
            self.assertFalse(result['full_matrix_complete'])
            self.assertFalse(result['default_only_execution'])
            self.assertEqual(result['counts']['planner_dispatched'], 48)
            self.assertEqual(result['engine_applied_nonreference'], 0)
            for invalid in (rows[:-1], rows[::-1], [{**rows[0], 'fixed_work_pass': False}, *rows[1:]]):
                self.assertFalse(native_ours_report.make_report(out, plan, invalid, 'complete')['functional_test_pass'])
            self.assertFalse(native_ours_report.make_report(out, plan, rows, 'failed', 'fixture failure')['functional_test_pass'])
            self.assertTrue((out/'test_result.txt').is_file())

    def test_diagnostics_do_not_count_late_result_as_new_dispatch_or_application(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            spec = r234.ours_timing_plan(limit=1)[0]
            enable_parts(out, [spec['label']])
            log = ConditionLog(out, spec['label'])
            scored = {'v5_candidate_generated': 1, 'selected_cell': '20', 'wall_ms': 1100,
                      'planning_from_ready_ms': 1120, 'planning_return_delay_ms': 5,
                      'evaluations': [{'cell': c, 'complete': True} for c in ('00', '20')]}
            log.decision({'index': 1, 'planner_dispatched': True, 'discarded_expired': True})
            log.decision({'event': 'expired_plan_diagnostic', 'index': 1,
                          'discarded_expired': True, 'result': scored})
            log.decision({'index': 2, 'planner_dispatched': True, 'result': scored})
            log.complete(summary(spec))
            result = log.summary()
            self.assertEqual(result['stages']['decision_opportunities'], 2)
            self.assertEqual(result['stages']['planner_dispatched'], 2)
            self.assertEqual(result['stages']['all_scores_complete'], 2)
            self.assertEqual(result['stages']['all_scores_complete_on_time'], 1)
            self.assertEqual(result['stages']['nonreference_selected_on_time'], 1)
            self.assertEqual(result['ms']['complete_planner_ms']['n'], 2)
            self.assertEqual(result['ms']['planning_return_delay_ms']['max'], 5)
            overview = ''.join(p.read_text(encoding='utf-8') for p in (out/'copy_logs').glob('COPY_00_*'))
            self.assertIn('OURS_BRIEF', overview)
            self.assertIn('applied=0', overview)

    def test_copy_keeps_linkable_selection_and_receipt_without_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            log = ConditionLog(out, 'Ours')
            log.decision({'index': 7, 'planner_dispatched': True, 'reference_action': {'placement': {'a': 0}},
                'planning_window': {'deadline': 2.2}, 'result': {
                    'planning_finished_at': 2., 'planning_received_at': 2.1, 'v5_stop_reason': 'complete',
                    'v5_candidate_generated': 1, 'selected_cell': '20', 'action': {'placement': {'a': 1}}},
                'native_receipts': [{'replica': 1, 'receipt': {'action_id': 'action-seven', 'status': 'applied'}}]})
            # Scoring detail may have used its allowance; terminal receipts retain space.
            log.detail_bytes = 12*1024
            log.receipt(7, 1, {'action_id': 'action-seven', 'status': 'applied',
                'mutation_at_ns': 2100000000, 'action_expires_ns': 2200000000,
                'engine_admissions': [{'node_id': 'a', 'client_request_id': 'c-a', 'server_request_id': 's-a',
                                       'engine_request_id': 'e-a', 'action_id': 'action-seven'}]})
            text = (out/'copy_logs/COPY.txt').read_text(encoding='utf-8')
            self.assertIn(' SELECT ', text)
            self.assertIn(' RECEIPT condition=Ours i=7 replica=1', text)
            for value in ('action-seven', 'e-a', 'reference=', 'chosen=', 'deadline=', 'mutation_at_ns='):
                self.assertIn(value, text)
            self.assertNotIn('conclusion=', text)
            self.assertEqual(log.summary()['stages']['planner_returned'], 1)
            self.assertEqual(log.summary()['planner_stop_reasons'], {'complete': 1})

    def test_retry_log_distinguishes_recovered_errors_and_counts_both_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ConditionLog(Path(directory), 'Ours')
            log.decision({'index': 1, 'snapshot': {
                'native_observation_errors': [], 'native_query_budget': {'ready_requests': 2},
                'native_observation_retry': {'attempts': 2, 'recovered': True,
                    'initial_errors': ['http_done_binding_pending'],
                    'initial_query_budget': {'ready_requests': 1}}}})
            report = log.summary()
            self.assertEqual(report['observation_errors'], {})
            self.assertEqual(report['retry_initial_errors'], {'http_done_binding_pending': 1})
            self.assertEqual(report['observation_retries']['recovered'], 1)
            self.assertEqual(log.query_totals['ready_requests'], 3)

    def test_timing_can_reuse_new_ours_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/'r234_ours/ascend_profile.json'
            write_json(target, model())
            self.assertEqual(profile_path(root), target)


class OursWorkflowTests(unittest.TestCase):
    setUp = workflows.NativeWorkflowTests.setUp
    fake_launch = workflows.NativeWorkflowTests.fake_launch
    run_workflow = workflows.NativeWorkflowTests.run_workflow

    def test_normal_ours_uses_current_test_evidence_and_copies_report(self):
        self.assertEqual(self.run_workflow()[1], 0)
        self.commands = []
        self.current = workflows.entry.Workflow(self.root, 'ours', timing_count=6)
        def launch(command, logfile):
            self.commands.append(command)
            out = self.current.host_path(command[command.index('--out')+1])
            write_json(out/'status.json', {'status': 'complete', 'functional_test_pass': True, 'completed_conditions': 6})
            write_json(out/'index.json', [summary(s) for s in r234.ours_timing_plan(limit=6)])
            (out/'test_result.txt').write_text('bounded-ours-fixture', encoding='utf-8')
            logfile.write_text('fixture', encoding='utf-8')
            return 0
        with patch.object(self.current, 'launch', side_effect=launch), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 0)
        self.assertEqual([c[2] for c in self.commands], ['benchmark'])
        command = self.commands[0]
        self.assertIn('--ours', command)
        self.assertIn('--native-evidence', command)
        self.assertNotIn('--ours-timing', command)
        self.assertNotIn('--timing-profile', command)
        self.assertEqual(command[command.index('-n')+1], '6')
        self.assertEqual(self.current.record['completed_conditions'], 6)
        self.assertEqual(self.current.record['status'], 'completed')
        self.assertFalse(self.current.record['formal_acceptance'])
        self.assertEqual((self.current.out/'test_result.txt').read_text(), 'bounded-ours-fixture')

    def test_ours12_only_runs_fixed_prefix_and_leaves_judgement_offline(self):
        self.current = workflows.entry.Workflow(self.root, 'ours-12')
        def ours():
            self.assertEqual(self.current.timing_count, 12)
            plan = r234.ours_timing_plan(limit=12)
            report_dir = self.current.out/'r234_ours'
            report_dir.mkdir()
            write_json(report_dir/'protocol.json', {'ours_only': True, 'timing_count': 12})
            write_json(report_dir/'native_gate.json', {'pass': True, 'live_verified': True})
            write_json(report_dir/'environment.json', {'passed': True})
            write_json(report_dir/'canary_summary.json', canaries())
            rows = [summary(s) for s in plan]  # No positive scheduling-effect requirement.
            result = native_ours_report.make_report(report_dir, plan, rows, 'complete')
            self.assertEqual(result['engine_applied_nonreference'], 0)
            for field in ('conclusion', 'effective_scheduling_verified', 'mechanism_pass'):
                self.assertNotIn(field, result)
            (self.current.out/'test_result.txt').write_text(
                (report_dir/'test_result.txt').read_text(encoding='utf-8'), encoding='utf-8')
            self.current.record['next_commands'] = ['bash run_native.sh logs']
        with patch.object(self.current, 'ours', side_effect=ours), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 0)
        self.assertEqual(self.current.record['status'], 'completed')
        self.assertEqual(self.current.record['next_commands'], ['bash run_native.sh logs'])
        self.assertNotIn('effective_scheduling_verified', self.current.record)
        self.assertFalse((self.current.out/'ours_execution_check.json').exists())
        self.assertIn('不自动判断调度有效性', (self.current.out/'test_result.txt').read_text(encoding='utf-8'))

    def test_missing_test_does_not_use_shadow_model_as_acceptance(self):
        self.current = workflows.entry.Workflow(self.root, 'ours')
        with patch.object(self.current, 'launch', side_effect=AssertionError('no traffic')), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 2)
        self.assertEqual(self.current.record['status'], 'blocked')
        self.assertTrue((self.current.out/'test_result.txt').exists())


class OursExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_ours_runs_gate_canaries_and_six_single_method_conditions(self):
        args = SimpleNamespace(service_mode='existing', execution_scope='native', request_timeout=300,
            http_keepalive=0, ours=True, ours_timing=None, resume=None, max_inflight=2, condition_timeout=1800,
            max_conditions=None, devices=[0, 1, 2, 3], timing_count=6)
        caps = [capability() for _ in range(4)]
        native = SimpleNamespace(inspect=AsyncMock(return_value=caps), drain=AsyncMock(), harvest=AsyncMock(return_value=[]))
        servers = SimpleNamespace(start=AsyncMock(), endpoints=['http://unused']*4, check_alive=lambda: None, aclose=AsyncMock())
        backend = SimpleNamespace(inspect=AsyncMock(return_value={}), fixed_canaries=AsyncMock(return_value=canaries()))
        seen = []
        async def condition(b, tok, packets, m, cfg, spec, out, pool, alive):
            seen.append(spec)
            self.assertFalse(cfg.get('ours_shadow_timing', False))
            self.assertEqual((cfg['epoch_s'], cfg['snapshot_max_age_s'], cfg['planning_soft_limit_s']), (1.5, 1.5, 1.2))
            out.mkdir(parents=True)
            for name in ('input_proof.json', 'initial_nodes.json', 'nodes.json', 'summary.json', 'quality.json',
                         'status.json', 'requests.jsonl', 'decisions.jsonl', 'native_events.jsonl'):
                write_json(out/name, {})
            return summary(spec)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            paths = {'native-calibrate': str(out/'calibration')}
            write_json(out/'calibration/native_profile.json', {**model(), 'timing_target': 'fixture',
                'identities': [c['identity'] for c in caps], 'resources': [c['resources'] for c in caps]})
            write_json(out/'protocol.json', {'ours_only': True, 'timing_count': 6})
            write_json(out/'repair_protocol.json', {})
            write_json(out/'environment.json', {'passed': True})
            gate = {'pass': True, 'live_verified': True, 'paths': paths, 'artifacts': {}}
            summaries = []
            with patch('dabs_ascend.existing.ExistingServers', return_value=servers), \
                 patch('dabs_ascend.backend.Backend', return_value=backend), \
                 patch('dabs_ascend.backend.Tokenizer', return_value=FakeTokenizer()), \
                 patch('dabs_ascend.native_client.NativeClient', return_value=native), \
                 patch('dabs_ascend.native_gate.verify_live', AsyncMock(return_value=gate)) as verify, \
                 patch('dabs_ascend.runner.run_condition', side_effect=condition), \
                 patch.object(r234_run, 'native_paths', return_value=paths), \
                 patch.object(r234_run, 'ProcessPoolExecutor', side_effect=lambda *a, **k: ThreadPoolExecutor(1)), \
                 patch.object(r234_run, 'source_identity', return_value={}):
                with redirect_stdout(io.StringIO()):
                    await r234_run.execute(args, out, r234.ours_timing_plan(limit=6), summaries, {})
            verify.assert_awaited_once()
            backend.fixed_canaries.assert_awaited_once()
            self.assertEqual(seen, r234.ours_timing_plan(limit=6))
            self.assertEqual(len(summaries), 6)
            self.assertFalse((out/'group_audits.jsonl').exists())
            self.assertFalse(read_json(out/'protocol.json')['formal_acceptance'])
