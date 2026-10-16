"""CPU predictor regressions, not field calibration evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

from dabs_ascend.common import read_json
from dabs_ascend.native_calibration import calibration_plan, fit_model, refit_existing
from dabs_ascend.native_prediction import PREDICTOR, predict, pending_startup, check_startup_inputs
from dabs_ascend.native_timing import reconstruct
from dabs_ascend.native_algorithms import adapted, baseline
from dabs_ascend.native_scoring import next_batch
from test_native import state, model
from test_native_async_calibration import pipeline
from test_native_simplification import capability
from test_native_explore import fixture


class PredictionTests(unittest.TestCase):
    def samples(self):
        plan = [{'split': split, 'seed': base + i} for split, base in [('train', 0), ('holdout', 100)] for i in range(20)]
        samples, excluded = reconstruct(pipeline(plan), plan)
        self.assertFalse(excluded)
        for s in samples:
            if s['phase'] == 'prefill_only':
                s['regime'] = 'startup' if s['group']['seed'] % 2 else 'pipeline'
                s['y'] = 100. if s['regime'] == 'startup' else 50.
        return samples

    def test_fit_separates_startup_without_relaxing_errors_or_learning_holdout(self):
        samples = self.samples()
        old = fit_model(samples, [capability()], predictor='phase6_v1')
        new = fit_model(samples, [capability()])
        self.assertFalse(old['pass']); self.assertTrue(new['pass'], new['failures'])
        self.assertEqual(old['thresholds_predeclared'], new['thresholds_predeclared'])
        self.assertAlmostEqual(predict(new, 0, 1024, 0, 0, 3072, startup=True), 100., places=5)
        self.assertAlmostEqual(predict(new, 0, 1024, 0, 0, 3072, startup=False), 50., places=5)
        for s in samples:
            if s['split'] == 'holdout':
                s['y'] *= 10
        failed = fit_model(samples, [capability()])
        self.assertFalse(failed['pass']); self.assertEqual(failed['models'], new['models'])

    def test_unknown_or_unobserved_startup_is_explicit_not_zero(self):
        fitted = fit_model(self.samples(), [capability()])
        with self.assertRaisesRegex(ValueError, 'unknown prefill startup'):
            predict(fitted, 0, 1024, 0, 0, 3072)
        fitted['models']['0']['phases']['prefill_only']['observed_regimes'] = ['startup']
        with self.assertRaisesRegex(ValueError, 'unobserved prefill regime'):
            predict(fitted, 0, 1024, 0, 0, 3072, startup=False)
        b = {'batch_id': 'B', 'host_calls': [{'scheduled_batch_ids': ['B'], 'pending_before': ['A']}]}
        self.assertFalse(pending_startup(b))
        b['host_calls'][0]['pending_before'] = []
        self.assertTrue(pending_startup(b))
        b['host_calls'][0].pop('pending_before')
        self.assertIsNone(pending_startup(b))

    def test_training_startup_must_be_available_at_submission(self):
        plan = [{'split': 'train', 'seed': 1}]
        rows = pipeline(plan); samples, _ = reconstruct(rows, plan)
        check_startup_inputs(samples, rows)
        samples[0]['regime'] = 'pipeline'
        with self.assertRaisesRegex(ValueError, 'not reproducible from submission'):
            check_startup_inputs(samples, rows)

    def test_real_scoring_uses_startup_once_then_pipeline_and_preserves_legacy(self):
        cfg, snapshot = state()
        fitted = model(); fitted['predictor'] = PREDICTOR
        for phases in fitted['models'].values():
            phases['phases']['prefill_only'].update(startup_extra_ms=30., observed_regimes=['startup', 'pipeline'])
        action = baseline(snapshot, cfg, 'RR')
        first = next_batch(snapshot, action, fitted, 0, cfg)
        legacy = next_batch(snapshot, action, model(), 0, cfg)
        self.assertAlmostEqual(first['step_ms'] - legacy['step_ms'], 30.)
        result = adapted('engine').evaluate(snapshot, action, fitted, cfg, record_events=True)
        self.assertTrue(result['rankable'], result)
        for replica in (0, 1):
            prefill = [b for b in result['event_trace'] if b['replica'] == replica and b['prefill']]
            self.assertTrue(prefill[0]['prefill_startup_predicted'])
            self.assertTrue(all(not b['prefill_startup_predicted'] for b in prefill[1:]))
        original = model()
        from core18v2.service import predict as original_predict
        self.assertEqual(predict(original, 0, 1024, 0, 0, 0), original_predict(original, 0, 1024, 0, 0, 0))
        # A successor admitted after an empty engine starts another pipeline.
        snapshot['nodes']['b'].update(state='blocked', parents=['a'])
        snapshot['ready'] = ['a']
        result = adapted('engine').evaluate(snapshot, baseline(snapshot, cfg, 'RR'), fitted, cfg, record_events=True)
        for key in ('a', 'b'):
            batches = [b for b in result['event_trace'] if any(k == key for k, _ in b['prefill'])]
            self.assertTrue(batches[0]['prefill_startup_predicted'])

    def test_old_data_refit_is_readonly_and_never_freezes_examined_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            before = {p.name: p.read_bytes() for p in (root / 'cal').iterdir()}
            out = root / 'refit'; out.mkdir()
            result = refit_existing(root / 'cal', out)
            self.assertFalse(result['frozen']); self.assertFalse(result['acceptance_granted'])
            self.assertTrue(result['fresh_holdout_required'])
            self.assertFalse(read_json(out / 'native_profile.json')['frozen'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / 'cal').iterdir()})
        old_seeds = {g['seed'] for g in calibration_plan(predictor='phase6_v1')}
        self.assertFalse(old_seeds & {g['seed'] for g in calibration_plan()})
