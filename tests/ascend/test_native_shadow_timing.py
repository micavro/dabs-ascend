"""Developer CPU regressions; never part of the onsite ours-timing entry."""
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from dabs_ascend import cli, r234, r234_run
from dabs_ascend.common import DATASET, read_json, write_json, configuration
from dabs_ascend.native_algorithms import adapted, choose, executable_key
from dabs_ascend import native_shadow_timing as shadow
from dabs_ascend.runner import run_condition
from test_native import model, resources
from test_native_planning import opportunity
from test_native_runner import BackendDouble, NativeDouble
from test_port import FakeTokenizer
import test_native_workflow as workflows
from semantic19.dataset import load_public
from typed19.workflow import compile_packets


def context():
    now = time.monotonic()
    return {'window': {'prepared_at': now-.01, 'deadline': now-.005, 'budget_ms': 5},
            'front_ms': 100., 'earliest_snapshot_at': now-.05}


class ShadowUnitTests(unittest.TestCase):
    def test_only_ours_from_original_matrix_and_prefixes(self):
        plan = r234.ours_timing_plan()
        self.assertEqual(len(plan), 24)
        self.assertEqual(plan, [s for s in r234.plan() if s['method'] == 'Ours'])
        for n in range(1, 25):
            self.assertEqual(r234.ours_timing_plan(limit=n), plan[:n])
        old = r234.ours_timing_plan('screening')
        self.assertEqual([s['label'] for s in old], ['R3_L8_T1_B0_Ours', 'R4_L4_T1_B0_Ours', 'R3_L4_T1_B0_Ours'])
        self.assertEqual(r234.ours_timing_plan('r3-l8'), old[:1])
        for n in (0, -1, 25, True):
            with self.assertRaises(ValueError): r234.ours_timing_plan(limit=n)
        with self.assertRaises(ValueError): r234.ours_timing_plan('r3-l8', 2)
        args = cli.prepare_args(['plan', '--suite', 'r234', '--devices', '0,1,2,3', '--ours-timing'], existing=True)
        self.assertEqual(cli.make_plan(args), plan)
        args = cli.prepare_args(['plan', '--suite', 'r234', '--devices', '0,1,2,3', '--ours-timing', '-n', '6'], existing=True)
        self.assertEqual(cli.make_plan(args), plan[:6])
        with redirect_stdout(io.StringIO()) as printed:
            r234_run.main(args)
        self.assertEqual(json.loads(printed.getvalue())['plan'], plan[:6])

    def test_count_and_part_flags_reject_wrong_scope_before_running(self):
        for argv in (['ours-timing', '-n', '0', '--input', 'old'], ['ours-timing', '-n', '25', '--input', 'old'],
                     ['ours-timing', '--group', 'r3-l8', '-n', '2', '--input', 'old'],
                     ['screen', '-n', '4'], ['screen', '--group', 'screening'], ['ours-timing', '--part', 'x']):
            with self.subTest(argv=argv), patch.object(sys, 'argv', ['run_native_workflow.py', *argv]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                workflows.entry.main()
        for extra in (['-n', '0', '--ours-timing'], ['-n', '25', '--ours-timing'], ['-n', '6']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.prepare_args(['plan', '--suite', 'r234', '--devices', '0,1,2,3', *extra], existing=True)

    def test_cli_separates_diagnostic_from_normal_and_disallows_resume(self):
        base = ['plan', '--suite', 'r234', '--devices', '0,1,2,3', '--ours-timing']
        for extra in (['--screening'], ['--resume', 'old'], ['--max-conditions', '1'], ['--execution-scope', 'http-limited']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                cli.prepare_args(base+extra, existing=True)
        with self.assertRaises(SystemExit):
            cli.prepare_args(['plan', '--suite', 'r234', '--devices', '0,1,2,3', '--timing-profile', 'model.json'], existing=True)

    def test_actual_selector_and_every_scorer_have_no_cpu_deadline(self):
        cfg, state = opportunity()
        saved = copy.deepcopy(state)
        engine = adapted('engine')
        original = engine.evaluate
        deadlines = []
        def score(*args, **kwargs):
            deadlines.append(kwargs['deadline'])
            time.sleep(.015)  # Exceeds original saved deadline; not a fake score.
            return original(*args, **kwargs)
        with patch.object(engine, 'evaluate', side_effect=score):
            result = shadow.run_unbounded(state, model(), cfg, context())
        self.assertGreater(len(deadlines), 1)
        self.assertTrue(all(d is None for d in deadlines))
        timing = result['shadow_timing']
        self.assertTrue(timing['exceeded_original_deadline'])
        self.assertTrue(timing['all_scores_complete'], result)
        self.assertFalse(timing['action_usable'])
        self.assertNotIn('action', result)
        self.assertTrue(all('action' not in e for e in result['evaluations']))
        self.assertTrue(all(e['diagnostic']['slice_budget_ms'] is None for e in result['evaluations']))
        self.assertGreater(timing['snapshot_need_estimate_ms'], timing['worker_wall_ms'])
        json.dumps(result, allow_nan=False)  # No infinity leaked from selector.
        self.assertEqual(state, saved)

    def test_normal_deadline_and_model_support_behavior_unchanged(self):
        cfg, state = opportunity()
        old = choose(state, model(), cfg, time.monotonic()-1)
        self.assertEqual(old['evaluations'], [])
        state['native_prediction_errors'] = ['recorded unsupported observation']
        result = shadow.run_unbounded(state, model(), cfg, context())
        self.assertEqual(result['v5_stop_reason'], 'native_prediction_unusable')
        self.assertFalse(result['shadow_timing']['all_scores_complete'])

    def test_worker_errors_retained_without_executable_action(self):
        cfg, state = opportunity()
        with patch('dabs_ascend.native_algorithms.choose', side_effect=ValueError('fixture error')):
            result = shadow.run_unbounded(state, model(), cfg, context())
        self.assertIn('fixture error', result['worker_error'])
        self.assertNotIn('action', result)
        self.assertFalse(result['shadow_timing']['all_scores_complete'])

    def test_return_delivery_is_counted_once_in_deadline_and_budget_estimates(self):
        future = Future()
        future.set_result({'shadow_timing': {'worker_finished_at': 10., 'window_prepared_at': 9.95,
            'original_deadline': 10.05, 'cycle_need_estimate_ms': 200., 'snapshot_need_estimate_ms': 150.}})
        with patch.object(shadow.time, 'monotonic', return_value=10.1):
            shadow.record_return(future)
            shadow.record_return(future)
        t = future.result()['shadow_timing']
        self.assertAlmostEqual(t['return_delay_ms'], 100)
        self.assertAlmostEqual(t['from_window_ready_ms'], 150)
        self.assertAlmostEqual(t['cycle_need_estimate_ms'], 300)
        self.assertAlmostEqual(t['snapshot_need_estimate_ms'], 250)
        self.assertTrue(t['exceeded_original_deadline'])

    def test_profile_resolution_reuses_only_model_without_pass_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root/'r234_screening/ascend_profile.json', {'models': {}, 'pass': False})
            self.assertEqual(shadow.profile_path(root), root/'r234_screening/ascend_profile.json')
            self.assertEqual(shadow.profile_path(root/'r234_screening/ascend_profile.json'), root/'r234_screening/ascend_profile.json')
            write_json(root/'r234_native/ascend_profile.json', {})
            with self.assertRaises(ValueError): shadow.profile_path(root)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root/'r234_ours_timing/ascend_profile.json', model())
            self.assertEqual(shadow.profile_path(root), root/'r234_ours_timing/ascend_profile.json')
            write_json(root/'timing_input/ascend_profile.json', model())
            self.assertEqual(shadow.profile_path(root), root/'timing_input/ascend_profile.json')

    def test_split_copy_keeps_all_records_and_each_file_under_64kib(self):
        from dabs_ascend.copy_log import enable_parts, emit
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root/'outputs/run'
            labels = [s['label'] for s in r234.ours_timing_plan()]
            emit(run, 'START', mode='fixture')
            enable_parts(run, labels)
            for label in labels:
                emit(run, 'DETAIL', condition=label, text='字'*30000)
                emit(run, 'SHADOW_SUMMARY', condition=label, returned=617)
                emit(run, 'SHADOW_BRIEF', label=label, counts={'returned': 617})
            manifest = read_json(run/'copy_logs/parts.json')
            texts = []
            for f in manifest['files']:
                path = run/'copy_logs'/f['name']
                self.assertEqual(len(path.read_bytes()), f['bytes'])
                self.assertLessEqual(f['bytes'], 64*1024)
                texts.append(path.read_text(encoding='utf-8'))
            joined = ''.join(texts)
            self.assertEqual(joined.count('字'), 24*30000)
            self.assertEqual(joined.count(' SHADOW_SUMMARY '), 24)
            self.assertEqual(joined.count(' SHADOW_BRIEF '), 24)
            self.assertIn(' START ', joined)
            self.assertNotIn('SHADOW_SUMMARY', (run/'copy_logs/COPY.txt').read_text(encoding='utf-8'))
            with patch.object(workflows.entry, 'ROOT', root), patch.object(sys, 'argv',
                    ['run_native_workflow.py', 'logs', '--part', 'COPY_00_overview_01.txt']), redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(workflows.entry.main(), 0)
            self.assertIn('SHADOW_BRIEF', printed.getvalue())
            with patch.object(workflows.entry, 'ROOT', root), patch.object(sys, 'argv',
                    ['run_native_workflow.py', 'logs', '--part', '../COPY.txt']), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                workflows.entry.main()

    def test_compact_statistics_separate_candidate_calls_and_preserve_maxima(self):
        cfg, state = opportunity()
        result = shadow.run_unbounded(state, model(), cfg, context())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            log = shadow.ShadowLog(root, root/'condition', 'Ours')
            for i in range(100):
                current = copy.deepcopy(result)
                t = current['shadow_timing']
                t['worker_wall_ms'] = i+1
                if i < 95:
                    current['evaluations'] = []
                    t.update(generated=0, all_scores_complete=False, candidate_started=0, candidate_complete=0)
                log.started()
                log.add({'index': i, 'result': current})
            log.busy()
            summary = log.finish()
            self.assertEqual(summary['counts']['returned'], 100)
            self.assertEqual(summary['counts']['unfinished'], 0)
            self.assertEqual(summary['ms']['with_candidates']['worker_wall_ms']['n'], 5)
            self.assertEqual(summary['ms']['with_candidates']['worker_wall_ms']['max'], 100)
            self.assertEqual(summary['ms']['all_scores_complete']['worker_wall_ms']['p50'], 98)
            self.assertEqual(summary['slowest_candidate_calls'][0]['i'], 99)
            text = (root/'copy_logs/COPY.txt').read_text(encoding='utf-8')
            self.assertLess(len(text.encode('utf-8')), 16000)
            self.assertEqual(text.count(' SHADOW_SLOW '), 3)
            self.assertIn('SHADOW_SUMMARY', text)
            self.assertIn('SHADOW_TIMING', text)

    def test_report_never_claims_four_method_or_native_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = r234.ours_timing_plan('r3-l8')
            summaries = [{'spec': plan[0], 'functional_pass': True, 'completed_requests': 216, 'output_tokens': 13184}]
            report = shadow.make_report(root, plan, summaries, 'complete')
            self.assertFalse(report['formal_acceptance'])
            self.assertTrue(report['functional_test_pass'])
            self.assertEqual(report['planner_actions_applied'], 0)
            self.assertTrue((root/'test_result.txt').is_file())

    def test_workflow_runs_only_benchmark_and_keeps_old_model_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'old/r234_native/ascend_profile.json'
            write_json(source, model()); saved = source.read_bytes()
            run = workflows.entry.Workflow(root, 'ours-timing', input_path=root/'old', timing_count=6)
            commands = []
            def launch(command, logfile):
                commands.append(command)
                out = run.host_path(command[command.index('--out')+1]); out.mkdir(parents=True)
                (out/'test_result.txt').write_text('fixture timing', encoding='utf-8')
                logfile.write_text('fixture', encoding='utf-8')
                return 0
            with patch.object(run, 'launch', side_effect=launch), redirect_stdout(io.StringIO()):
                self.assertEqual(run.run(), 0)
            self.assertEqual([c[2] for c in commands], ['benchmark'])
            self.assertIn('--ours-timing', commands[0])
            self.assertNotIn('--native-evidence', commands[0])
            self.assertEqual(run.record['requested_conditions'], 6)
            self.assertEqual(commands[0][commands[0].index('-n')+1], '6')
            self.assertEqual(run.record['status'], 'completed')
            self.assertFalse(run.record['formal_acceptance'])
            self.assertEqual(saved, source.read_bytes())
            self.assertEqual((run.out/'test_result.txt').read_text(), 'fixture timing')
            self.assertFalse((root/'outputs/native-entry-last-test.json').exists())


class ShadowAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_natural_drain_keeps_running_through_multiple_wait_timeouts(self):
        release = asyncio.Event()
        async def slow():
            await release.wait()
            return 123
        task = asyncio.create_task(slow())
        callbacks = []
        def progress():
            callbacks.append(True)
            if len(callbacks) == 3:
                release.set()
        result = await shadow.await_natural_completion(task, progress, interval=.001)
        self.assertEqual(result, 123)
        self.assertEqual(len(callbacks), 3)
        self.assertFalse(task.cancelled())

    async def test_live_216_node_run_uses_default_only_skips_busy_and_drains_all(self):
        cfg = configuration(4, resources())
        cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                   epoch_s=.03, planning_soft_limit_s=.01, commit_reserve_s=.003,
                   condition_wall_s=60., ours_shadow_timing=True)
        tok = FakeTokenizer(); rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
        original = shadow.run_unbounded
        counts = {'started': 0, 'finished': 0}
        def slow(state, m, c, ctx):
            counts['started'] += 1
            time.sleep(.18)  # Must survive many old 10ms deadlines.
            result = original(state, m, c, ctx)
            result['selected_cell'] = 'would-have-been-a-different-action'
            counts['finished'] += 1
            return result
        class DefaultOnly(NativeDouble):
            async def commit(self, action, state, nodes, fallback=False):
                base = r234.baseline(state, cfg, 'Ours')
                if executable_key(action, state) != executable_key(base, state):
                    raise AssertionError('shadow result reached live engine')
                return await super().commit(action, state, nodes, fallback)
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(1) as pool:
            backend = BackendDouble(); backend.native = DefaultOnly()
            out = Path(d)/'condition'
            with patch.object(shadow, 'run_unbounded', side_effect=slow), \
                 patch.object(r234, 'plan_ours', side_effect=AssertionError('bounded planner used in shadow mode')):
                summary = await run_condition(backend, tok, packets, model(), cfg, r234.plan()[3], out, pool)
            self.assertEqual(summary['completed_requests'], 216)
            self.assertEqual(summary['output_tokens'], 13184)
            self.assertTrue(summary['functional_pass'])
            self.assertEqual(counts['started'], counts['finished'])
            stats = summary['shadow_timing']['counts']
            self.assertGreater(stats['skipped_busy_no_queue'], 0)
            self.assertEqual(stats['unfinished'], 0)
            records = [json.loads(line) for line in (out/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
            timings = [r for r in records if r.get('event') == 'unbounded_planner_timing']
            self.assertEqual(len(timings), counts['started'])
            self.assertTrue(any(r['discarded_expired'] for r in timings))
            self.assertTrue(all(not r.get('client_applied_nonreference') for r in records))
            self.assertEqual(summary['engine_applied_nonreference_actions'], 0)
            self.assertEqual(summary['selected_nonreference_actions'], 0)

    async def test_executor_entry_skips_canaries_gate_and_recalibration(self):
        args = SimpleNamespace(service_mode='existing', execution_scope='native', request_timeout=300,
            http_keepalive=0, ours_timing='r3-l8', resume=None, max_inflight=2, condition_timeout=1800,
            max_conditions=None, devices=[0, 1, 2, 3])
        caps = [{'resources': resources(), 'identity': {'engine_instance': str(i)}} for i in range(4)]
        native = SimpleNamespace(inspect=AsyncMock(return_value=caps), drain=AsyncMock(), harvest=AsyncMock(return_value=[]))
        servers = SimpleNamespace(start=AsyncMock(), endpoints=['http://unused']*4, check_alive=lambda: None, aclose=AsyncMock())
        backend = SimpleNamespace(inspect=AsyncMock(return_value={}), fixed_canaries=AsyncMock(side_effect=AssertionError('no canaries')))
        async def condition(b, tok, packets, m, cfg, spec, out, pool, alive):
            self.assertTrue(cfg['ours_shadow_timing'])
            self.assertEqual(spec['method'], 'Ours')
            out.mkdir(parents=True)
            for name in ('input_proof.json', 'initial_nodes.json', 'nodes.json', 'summary.json', 'quality.json',
                         'status.json', 'requests.jsonl', 'decisions.jsonl', 'native_events.jsonl'):
                write_json(out/name, {})
            return {'spec': spec, 'functional_pass': True, 'external_work_fingerprint': 'fixture',
                    'arrival_offsets': [], 'jct_mean_ms': 100, 'output_tokens': 13184, 'completed_requests': 216}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            args.timing_profile = out/'input.json'
            write_json(args.timing_profile, {**model(), 'timing_target': 'fixture'})
            write_json(out/'protocol.json', {'ours_timing': 'r3-l8'})
            write_json(out/'repair_protocol.json', {})
            with patch('dabs_ascend.existing.ExistingServers', return_value=servers), \
                 patch('dabs_ascend.backend.Backend', return_value=backend), \
                 patch('dabs_ascend.backend.Tokenizer', return_value=FakeTokenizer()), \
                 patch('dabs_ascend.native_client.NativeClient', return_value=native), \
                 patch('dabs_ascend.native_gate.verify_live', side_effect=AssertionError('no validation gate')), \
                 patch('dabs_ascend.calibration.calibrate', side_effect=AssertionError('no recalibration')), \
                 patch('dabs_ascend.runner.run_condition', side_effect=condition), \
                 patch.object(r234_run, 'ProcessPoolExecutor', side_effect=lambda *a, **k: ThreadPoolExecutor(1)), \
                 patch.object(r234_run, 'source_identity', return_value={}):
                await r234_run.execute(args, out, r234.ours_timing_plan('r3-l8'), [], {})
            backend.fixed_canaries.assert_not_called()
            self.assertFalse((out/'canary_summary.json').exists())
            self.assertFalse(read_json(out/'protocol.json')['formal_acceptance'])
            self.assertTrue((out/'test_result.txt').exists())
