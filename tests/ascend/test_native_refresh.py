"""Fault injection for the site's stale *refreshed* snapshot. CPU only."""
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from dabs_ascend import r234
from dabs_ascend.common import DATASET, configuration
from dabs_ascend.runner import run_condition
from dabs_ascend.repair import admission_check
from semantic19.dataset import load_public
from typed19.workflow import compile_packets
from test_native import resources, model
from test_native_runner import BackendDouble, NativeDouble, SlowObservationDouble
from test_port import FakeTokenizer


class ExpiringRefresh(SlowObservationDouble):
    persistent = False

    async def observe(self, nodes, cfg, index, model, **kwargs):
        s = await super().observe(nodes, cfg, index, model, **kwargs)
        if not kwargs.get('cache_queries', True) and (self.persistent or self.query_refreshes % 5 == 1):
            # Equivalent to projection/planning consuming the configured snapshot
            # lease after an otherwise fast RPC, without slow wall-clock sleeps.
            s['now'] -= cfg['snapshot_max_age_s'] + 1
        return s

    async def commit(self, action, state, nodes, fallback=False):
        if time.monotonic() - state['now'] > .12:
            raise AssertionError('stale refreshed selection reached admission')
        return await super().commit(action, state, nodes, fallback=fallback)


class RejectedThenExpired(NativeDouble):
    refreshes = 0
    rejected = False

    async def observe(self, nodes, cfg, index, model, **kwargs):
        s = await super().observe(nodes, cfg, index, model, **kwargs)
        s['native_observation_errors'] = []
        if not kwargs.get('cache_queries', True) and self.rejected:
            self.refreshes += 1
            if self.refreshes == 1:
                s['now'] -= cfg['snapshot_max_age_s'] + 1
        return s

    async def commit(self, action, state, nodes, fallback=False):
        if not self.rejected and not fallback and action['admit']:
            self.rejected = True
            return [], [{'replica': 0, 'receipt': {'status': 'expired', 'accepted': False}}]
        if time.monotonic() - state['now'] > .12:
            raise AssertionError('stale secondary fallback reached admission')
        return await super().commit(action, state, nodes, fallback=fallback)


class RefreshIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = configuration(4, resources())
        self.cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                        epoch_s=.03, planning_soft_limit_s=.01, commit_reserve_s=.003, condition_wall_s=60.)
        self.tok = FakeTokenizer()
        rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, self.tok, self.cfg)
        self.packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')

    async def run_case(self, native, spec, out, pool):
        backend = BackendDouble(); backend.native = native
        return await run_condition(backend, self.tok, self.packets, model(), self.cfg, spec, out, pool)

    def decisions(self, out):
        return [json.loads(x) for x in (out / 'decisions.jsonl').read_text().splitlines()]

    def assert_deferred(self, out):
        rows = [d for d in self.decisions(out) if d.get('native_deferred_reason')]
        self.assertTrue(rows)
        for d in rows:
            self.assertEqual(d['native_deferred_reason'], 'stale_snapshot')
            self.assertEqual(d['applied_submission_order'], [])
            self.assertFalse(d['client_applied_nonreference'])
            self.assertGreater(d['fallback_validation']['snapshot_age_ms'], self.cfg['snapshot_max_age_s'] * 1000)

    async def test_stale_refresh_recovers_for_all_four_methods_without_duplicate_work(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            for spec in r234.plan()[:4]:
                with self.subTest(method=spec['method']):
                    out = Path(directory) / spec['label']
                    native = ExpiringRefresh()
                    result = await self.run_case(native, spec, out, pool)
                    self.assertTrue(result['functional_pass'])
                    self.assertEqual((result['completed_requests'], result['output_tokens']), (216, 13184))
                    self.assertEqual(native.admitted, 216)
                    self.assert_deferred(out)
                    submitted = [k for d in self.decisions(out) for k in d.get('applied_submission_order', [])]
                    self.assertEqual(len(submitted), len(set(submitted)))

    async def test_engine_rejection_then_stale_secondary_refresh_also_defers(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            out = Path(directory) / 'secondary'
            native = RejectedThenExpired()
            result = await self.run_case(native, r234.plan()[0], out, pool)
            self.assertTrue(result['functional_pass'])
            self.assertEqual(result['completed_requests'], 216)
            self.assert_deferred(out)
            deferred = next(d for d in self.decisions(out) if d.get('native_deferred_reason'))
            self.assertEqual(deferred['native_receipts'][0]['receipt']['status'], 'expired')

    async def test_permanently_stale_refresh_stops_with_recorded_stall_not_endless_retry(self):
        class PersistentAfterAttempts(ExpiringRefresh):
            persistent = True
            async def observe(self, nodes, cfg, *args, **kwargs):
                result = await super().observe(nodes, cfg, *args, **kwargs)
                if self.query_refreshes >= 3:
                    # Trigger the short watchdog after exercising the refresh
                    # failures; slow Windows file setup must not preempt them.
                    cfg['native_idle_stall_s'] = .3
                return result
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            out = Path(directory) / 'persistent'
            native = PersistentAfterAttempts()
            with self.assertRaisesRegex(ValueError, 'native admission stalled'):
                await self.run_case(native, r234.plan()[0], out, pool)
            self.assertEqual(native.admitted, 0)
            self.assertGreaterEqual(native.query_refreshes, 3)
            self.assert_deferred(out)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'failed')

    async def test_dependency_failure_is_not_treated_as_transient_expiry(self):
        class InvalidRefresh(NativeDouble):
            refreshed_at = None
            async def observe(self, nodes, cfg, index, model, **kwargs):
                s = await super().observe(nodes, cfg, index, model, **kwargs)
                if not kwargs.get('cache_queries', True):
                    self.refreshed_at = s['now']
                return s
            async def commit(self, *args, **kwargs):
                raise AssertionError('invalid admission reached engine')
        native = InvalidRefresh()
        def check(action, nodes, cfg, snapshot_at, now):
            if snapshot_at == native.refreshed_at:
                return 'dependency_not_complete'
            return admission_check(action, nodes, cfg, snapshot_at, now)
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            out = Path(directory) / 'dependency'
            with patch('dabs_ascend.runner.admission_check', side_effect=check), self.assertRaisesRegex(
                    ValueError, 'Fresh native fallback admission failed: dependency_not_complete'):
                await self.run_case(native, r234.plan()[0], out, pool)
            last = self.decisions(out)[-1]
            self.assertEqual(last['commit_reject_reason'], 'fresh_native_fallback:dependency_not_complete')
            self.assertNotIn('native_deferred_reason', last)
