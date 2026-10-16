"""216-node native controller branch integration on a CPU double, NOT NPU evidence."""
import copy
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dabs_ascend import r234
from dabs_ascend.common import DATASET, append_json, configuration
from dabs_ascend.runner import run_condition
from dabs_ascend.scheduler import snapshot
from test_native import resources, model
from test_port import FakeTokenizer
from test_r234 import SimulatedFixedBackend
from semantic19.dataset import load_public
from typed19.workflow import compile_packets


class NativeDouble:
    log = None
    def flush_evidence(self, **kwargs):
        pass  # This double has no native RPC evidence buffer.
    async def drain(self, replicas):
        append_json(self.log, {'cpu_double': True, 'drained': True})
    async def harvest(self, replicas, **kwargs):
        return []
    async def observe(self, nodes, cfg, index, model, **kwargs):
        s = snapshot(nodes, {}, cfg['replicas'], cfg, index)
        s.update(native_snapshots=[{}]*cfg['replicas'], native_mutable=[],
            native_queue_groups={str(r): {'running_prefill': [], 'waiting': []} for r in range(cfg['replicas'])},
            physical_active_blocks=[0]*cfg['replicas'], capacity_blocks=[1000]*cfg['replicas'],
            native_observation_errors=['CPU double has no native evidence'])
        return s
    async def commit(self, action, state, nodes, fallback=False):
        for k in action['admit']:
            nodes[k]['native_action_id'] = 'cpu-only-' + k
        return r234.submission_order(action, nodes), []


class BackendDouble(SimulatedFixedBackend):
    def __init__(self):
        self.native = NativeDouble()
    async def generate(self, replica, node, seed, log, schema=None):
        result = await super().generate(replica, node, seed, log, schema)
        result['native_mapping'] = {'mappings': [{'action_id': node['native_action_id']}], 'cpu_double': True}
        return result


class SlowObservationDouble(NativeDouble):
    def __init__(self):
        self.admitted, self.atomic_calls, self.query_refreshes = 0, 0, 0

    async def observe(self, nodes, cfg, index, model, **kwargs):
        if kwargs.get('cache_queries', True):
            # Reproduce observation taking longer than the fixed action window.
            await asyncio.sleep(cfg['epoch_s'] * 1.5)
        else:
            self.query_refreshes += 1
        s = await super().observe(nodes, cfg, index, model)
        s['native_observation_errors'] = []  # expiration itself must trigger fallback
        return s

    async def commit(self, action, state, nodes, fallback=False):
        if not fallback:
            raise AssertionError('expired window reached queue-apply instead of atomic admission')
        self.atomic_calls += 1
        self.admitted += len(action['admit'])
        return await super().commit(action, state, nodes, fallback=True)


class NativeRunnerIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_late_nonreference_selection_is_diagnostic_not_engine_application(self):
        cfg = configuration(4, resources())
        cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                   epoch_s=.1, planning_soft_limit_s=.04, commit_reserve_s=.01, condition_wall_s=60.)
        def late(s, m, c, deadline):
            time.sleep(.06)
            return {'action': r234.baseline(s, c, 'Ours'), 'evaluations': [], 'selected_cell': 'late-cpu-fixture'}
        tok = FakeTokenizer(); rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(1) as pool:
            out = Path(d) / 'late'
            with patch.object(r234, 'plan_ours', side_effect=late):
                summary = await run_condition(BackendDouble(), tok, packets, model(), cfg, r234.plan()[3], out, pool)
            rows = [json.loads(line) for line in (out / 'decisions.jsonl').read_text().splitlines()]
            expired = [r for r in rows if r.get('event') == 'expired_plan_diagnostic']
            self.assertTrue(expired)
            self.assertTrue(all(r['result']['selected_cell'] == 'late-cpu-fixture' and not r['client_applied_nonreference'] for r in expired))
            self.assertEqual(summary['engine_applied_nonreference_actions'], 0)
            self.assertTrue(summary['functional_pass'])

    async def test_real_runner_dispatches_after_110ms_observation_with_original_fixed_deadline(self):
        cfg = configuration(4, resources())
        cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                   ready_coalesce_s=.001, condition_wall_s=30.)
        class SlowButFresh(NativeDouble):
            async def observe(self, *args, **kwargs):
                await asyncio.sleep(.11)
                return await super().observe(*args, **kwargs)
        class Reached(Exception): pass
        deadlines = []
        def planner(s, m, c, deadline):
            deadlines.append((deadline, time.monotonic(), len(s['nodes'])))
            raise Reached('post-observation planner reached')
        tok = FakeTokenizer(); rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
        with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(1) as pool:
            backend = BackendDouble(); backend.native = SlowButFresh()
            with patch.object(r234, 'plan_ours', side_effect=planner), self.assertRaisesRegex(Reached, 'post-observation'):
                await run_condition(backend, tok, packets, model(), cfg, r234.plan()[3], Path(d) / 'slow', pool)
        self.assertTrue(deadlines)
        deadline, dispatched, nodes = deadlines[0]
        self.assertGreater(deadline, dispatched)  # old runner never reaches this
        self.assertLessEqual(deadline - dispatched, .08)
        self.assertEqual(nodes, 216)

    async def test_rejected_admission_cannot_silently_idle_until_condition_timeout(self):
        cfg = configuration(4, resources())
        cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                   epoch_s=.03, planning_soft_limit_s=.01, commit_reserve_s=.003, condition_wall_s=30.)
        class Refused(NativeDouble):
            attempts = 0
            async def commit(self, action, state, nodes, fallback=False):
                self.attempts += 1
                cfg['native_idle_stall_s'] = .02
                await asyncio.sleep(.03)
                return [], []
        tok = FakeTokenizer()
        rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            backend = BackendDouble(); backend.native = Refused()
            out = Path(directory) / 'refused'
            with self.assertRaisesRegex(ValueError, 'native admission stalled'):
                await run_condition(backend, tok, packets, model(), cfg, r234.plan()[0], out, pool)
            self.assertGreater(backend.native.attempts, 0)
            status = json.loads((out / 'status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertIn('no inflight/completion/admission', status['error'])
            self.assertTrue((out / 'decisions.jsonl').is_file())

    async def test_slow_observation_cannot_starve_any_method_or_apply_expired_queue(self):
        cfg = configuration(4, resources())
        cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                   epoch_s=.03, planning_soft_limit_s=.01, commit_reserve_s=.003, condition_wall_s=60.)
        tok = FakeTokenizer()
        rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            for spec in r234.plan()[:4]:
                backend = BackendDouble(); backend.native = SlowObservationDouble()
                out = Path(directory) / spec['label']
                result = await run_condition(backend, tok, packets, model(), cfg, spec, out, pool)
                self.assertTrue(result['functional_pass'])
                self.assertEqual((result['completed_requests'], result['output_tokens']), (216, 13184))
                self.assertEqual(backend.native.admitted, 216)
                self.assertGreater(backend.native.query_refreshes, 0)
                decisions = [json.loads(line) for line in (out / 'decisions.jsonl').read_text().splitlines()]
                applied = [d for d in decisions if d.get('applied_submission_order')]
                self.assertTrue(applied)
                self.assertTrue(all(d.get('discarded_expired') and d.get('native_fallback_reason') == 'expired_decision_window' for d in applied))
                self.assertTrue(all(d['client_applied_nonreference'] is False for d in applied))

    async def test_all_methods_complete_216_nodes_without_changing_work_or_dependencies(self):
        cfg = configuration(4, resources())
        cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                   epoch_s=.03, planning_soft_limit_s=.01, commit_reserve_s=.003, condition_wall_s=60.)
        tok = FakeTokenizer()
        rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
        fingerprints = set()
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            for spec in r234.plan()[:4]:
                summary = await run_condition(BackendDouble(), tok, packets, model(), cfg, spec,
                                              Path(directory) / spec['label'], pool)
                self.assertTrue(summary['functional_pass'])
                self.assertEqual((summary['completed_workflows'], summary['completed_requests'], summary['output_tokens']), (64, 216, 13184))
                fingerprints.add(summary['external_work_fingerprint'])
        self.assertEqual(len(fingerprints), 1)
