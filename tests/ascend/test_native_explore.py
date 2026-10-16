"""CPU synthetic evidence only; exploratory execution must not certify bad fits."""
import asyncio
from contextlib import redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from dabs_ascend.common import read_json, write_json
from dabs_ascend.native_calibration import calibration_plan, fit_model
from dabs_ascend.native_explore import evaluate, verify
from dabs_ascend.native_gate import evaluate_local, events
from dabs_ascend.native_timing import reconstruct
from dabs_ascend.native_scoring import check_model_range
from test_native_async_calibration import pipeline
from test_native_evidence import probe_fixture
from test_native_simplification import capability


def fixture(root):
    probe_fixture(root / 'one', 1); probe_fixture(root / 'four', 4)
    out = root / 'cal'; out.mkdir()
    plan = calibration_plan()
    # Two independently identified synthetic episodes per declared workload group.
    rows = pipeline([g for g in plan for _ in range(2)])
    cutoff = 2801 * 1_000_000
    def clock(t):
        return t if t < cutoff else cutoff + 2 * (t - cutoff)
    for event in rows:
        b = event['response']['batches'][0]
        for c in b['host_calls']:
            c['begin_ns'], c['end_ns'] = clock(c['begin_ns']), clock(c['end_ns'])
            c['engine_host_ms'] = (c['end_ns'] - c['begin_ns']) / 1e6
        for k in ('host_begin_ns', 'scheduled_ns'):
            b[k] = clock(b[k])
        if b['completion_predecessor']:
            b['completion_predecessor']['end_ns'] = clock(b['completion_predecessor']['end_ns'])
        b['engine_host_ms'] = b['host_calls'][-1]['engine_host_ms']
        b['host_batch_span_ms'] = (b['host_calls'][-1]['end_ns'] - b['host_begin_ns']) / 1e6
    samples, excluded = reconstruct(rows, plan)
    assert not excluded, excluded
    caps = [capability()]
    model = fit_model(samples, caps)
    assert not model['pass']
    start = {'snapshot_id': 'start', 'step': 0}
    rows.insert(0, {'op': 'snapshot', 'replica': 0, 'response': {
        **start, 'identity': caps[0]['identity'], 'running_count': 0, 'waiting_count': 0, 'inflight_batches': []}})
    for name, value in [('native_profile.json', model), ('calibration_samples.json', samples),
                        ('capabilities.json', caps), ('calibration_plan.json', {
                            'plan': plan, 'diagnostic_only': False, 'collection_start': {'0': start}})]:
        write_json(out / name, value)
    (out / 'native_events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in rows), encoding='utf8')
    return {'native-calibrate': str(out), 'native-probe-one': str(root / 'one'), 'native-probe-four': str(root / 'four')}


class ExploreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = fixture(self.root)

    def live(self):
        ledger = {}
        for path in self.paths.values():
            for e in events(path):
                row = e['response']
                if e['op'] in ('apply', 'admit', 'receipt') and row.get('action_id'):
                    ledger[e['replica'], row['action_id']] = row
        async def call(r, op, **args):
            return ledger[r, args['action_id']]
        caps = [capability() for _ in range(4)]
        return SimpleNamespace(capabilities=caps, identities=[c['identity'] for c in caps],
                               inspect=AsyncMock(), call=call)

    def test_bad_error_can_run_only_as_explicit_exploration(self):
        self.assertFalse(evaluate_local(self.paths)['pass'])
        gate, model = evaluate(self.paths)
        self.assertTrue(gate['exploratory_allowed'])
        self.assertFalse(gate['pass']); self.assertFalse(model['frozen'])
        self.assertTrue(gate['calibration_error_deferred'])
        gate, model = asyncio.run(verify(self.live(), self.paths))
        self.assertTrue(gate['mechanisms_live_verified'])
        self.assertFalse(gate['live_verified'])
        self.assertEqual(set(model['models']), {'0', '1', '2', '3'})
        self.assertEqual(set(model['validation']), {'0'})
        check_model_range(model, 3, 1024, 0, 0, 3072)
        self.assertFalse(evaluate_local(self.paths)['pass'])

    def test_missing_samples_and_forged_model_cannot_be_ignored(self):
        path = self.root / 'cal/calibration_samples.json'
        samples = read_json(path); write_json(path, samples[1:])
        with self.assertRaisesRegex(ValueError, 'raw mechanism/timing'):
            evaluate(self.paths)
        write_json(path, samples)
        path = self.root / 'cal/native_profile.json'
        model = read_json(path); model['models']['0']['phases']['prefill_only']['coef'][0] += 5
        write_json(path, model)
        with self.assertRaisesRegex(ValueError, 'differs from raw refit'):
            evaluate(self.paths)

    def test_failed_queue_mechanism_is_not_prediction_error(self):
        path = self.root / 'one/native_events.jsonl'
        rows = events(path.parent)
        for row in rows:
            if row['op'] == 'batches':
                row['response']['batches'][0]['scheduled_tokens'] = {}
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'priority_reached_execution'):
            evaluate(self.paths)

    def test_restart_and_resource_change_still_stop_exploration(self):
        native = self.live(); native.identities[0]['engine_instance'] = 'restarted'
        with self.assertRaisesRegex(ValueError, 'restarted'):
            asyncio.run(verify(native, self.paths))
        native = self.live(); native.capabilities[3]['resources']['step_tokens'] = 4096
        with self.assertRaisesRegex(ValueError, 'identical effective resources'):
            asyncio.run(verify(native, self.paths))

    def test_report_keeps_exploration_separate_from_acceptance(self):
        from dabs_ascend.r234_report import make_report
        from dabs_ascend.r234 import plan
        gate, model = asyncio.run(verify(self.live(), self.paths))
        write_json(self.root / 'native_gate.json', gate)
        write_json(self.root / 'ascend_profile.json', model)
        report = make_report(self.root, plan(), [], 'paused')
        self.assertTrue(report['exploratory']); self.assertFalse(report['formal_acceptance'])
        self.assertFalse(report['strict_native_reproduction_pass'])
        self.assertTrue(report['native_cache_observed'])
        self.assertIn('探索性试跑', (self.root / 'TEST_REPORT.md').read_text(encoding='utf8'))

    def test_cli_allows_more_conditions_but_keeps_exploration_separate(self):
        from dabs_ascend.cli import prepare_args
        argv = ['benchmark', '--suite', 'r234', '--devices', '0,1,2,3',
                '--deployment-manifest', 'fixture.json', '--native-evidence', 'evidence.json',
                '--max-conditions', '4', '--exploratory-calibration']
        self.assertTrue(prepare_args(argv, existing=True).exploratory_calibration)
        for count in (1, 24, 96):
            self.assertEqual(prepare_args(argv + ['--max-conditions', str(count)], existing=True).max_conditions, count)
        for extra in (['--max-conditions', '97'], ['--resume', 'old'], ['--execution-scope', 'http-limited']):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                prepare_args(argv + extra, existing=True)
