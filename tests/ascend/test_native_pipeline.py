"""Regression for 1610 field failures. Synthetic tests are never NPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from dabs_ascend.native_client import NativeClient
from dabs_ascend.native_algorithms import baseline, choose, adapted
from dabs_ascend.native_observation import prefix_queries, prefix_identities
from dabs_ascend.native_planning import aligned_start, decision_record
from dabs_ascend.native_scoring import select_batch, check_model_range, cache_model_class
from dabs_ascend.native_calibration import calibration_plan, fit_model
from dabs_ascend.native_coverage import coverage_check, CONTRACT
from dabs_native.contracts import prefix_key
from test_native import model, state
from test_native_observation import backlog
from test_native_simplification import observation, capability


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_state_survives_prediction_failure_but_ours_cannot_score(self):
        cfg, s = state()
        nodes = s['nodes']
        for n in nodes.values():
            n['tokens'] = [7] * 3584
        nodes['a'].update(state='submitted', replica=0, start=1.)
        native = NativeClient(None, ['r0', 'r1'])
        native.capabilities = [capability(), capability()]
        async def call(r, op, **payload):
            row = observation(); ns = time.monotonic_ns()
            row.update(captured_ns=ns, expires_ns=ns + 120_000_000)
            if r == 0:
                row['running'] = [{'engine_request_id': 'engine-a', 'mapping': {'node_id': 'a'},
                    'completed_computed_tokens': 3583, 'output_tokens': 0, 'accepted_ns': ns,
                    'phase': 'prefill', 'queue_mutable': False}]
                row['running_count'] = 1
                row['inflight_batches'] = [{'batch_id': 'b1', 'step': 1, 'requests': [
                    {'engine_request_id': 'engine-a', 'prefill_tokens': 1,
                     'decode_tokens': 0, 'context_tokens': 3583}]}]
            return row
        native.call = call
        m = model()
        m['validation'] = {str(r): {p: {'range': {'p': [128, 1024], 'd': [0, 2],
            'dc': [0, 8192], 'pc': [0, 3584]}} for p in
            ('prefill_only', 'mixed', 'decode_only')} for r in range(2)}
        observed = await native.observe(nodes, cfg, 1, m)
        self.assertFalse(observed['native_observation_errors'])
        self.assertIn('native_calibration_out_of_range:p:1', observed['native_prediction_errors'][0])
        self.assertEqual(observed['nodes']['a']['computed'], 3583)
        self.assertEqual(observed['pressure'], [1, 0])
        self.assertIsNone(observed['inflight_batches'][0]['remaining_ms'])
        for method in ('RR', 'SLL', 'Ayo'):
            self.assertIn('b', baseline(observed, cfg, method)['admit'])
        result = choose(observed, m, cfg, time.monotonic() + .08)
        self.assertEqual(result['v5_stop_reason'], 'native_prediction_unusable')
        self.assertFalse(result['evaluations'])

    async def test_rpc_logging_is_deferred_and_trace_is_durable_before_ack(self):
        calls = []
        def handler(request):
            op = json.loads(request.content)['op']; calls.append(op)
            return httpx.Response(200, json={**observation(), 'batches': [
                {'step': 9, 'requests': [], 'device': None}] if op == 'batches' else []})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                native = NativeClient(client, ['http://r0'], path)
                native.defer_evidence = True
                await native.call(0, 'snapshot')
                self.assertFalse(path.exists())
                with patch('dabs_ascend.native_evidence_log.os.fsync') as sync:
                    await native.harvest(1)
                    sync.assert_called_once()
                self.assertEqual(calls, ['snapshot', 'batches', 'ack-trace'])
                recorded = [json.loads(l) for l in path.read_text().splitlines()]
                self.assertEqual([r['op'] for r in recorded], ['snapshot', 'batches'])
                native.flush_evidence()
                self.assertEqual(json.loads(path.read_text().splitlines()[-1])['op'], 'ack-trace')

    async def test_failed_raw_write_never_acknowledges_trace(self):
        calls = []
        def handler(request):
            op = json.loads(request.content)['op']; calls.append(op)
            return httpx.Response(200, json={**observation(), 'batches': [
                {'step': 9, 'requests': [], 'device': None}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            native = NativeClient(client, ['http://r0'], Path('unused.jsonl'))
            native.defer_evidence = True
            with patch('pathlib.Path.open', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    await native.harvest(1)
            self.assertEqual(calls, ['batches'])
            self.assertEqual(native.watermarks, {})
            self.assertTrue(native.evidence_buffer.rows)

    def test_queries_cover_critical_backlog_and_subblock_prefix_without_faking_hit(self):
        cfg, nodes = backlog()
        nodes['n095']['output'] = 512
        _, bindings, stats = prefix_queries(nodes, cfg, 'salt', model=model())
        self.assertIn('n095', bindings)  # outside old FIFO first 24 nodes
        self.assertLessEqual(stats['unique_queries'], 8)
        self.assertLessEqual(stats['query_tokens_per_replica'], 8192)
        nodes = {'short': {**nodes['n000'], 'share': 16}}
        queries, bindings, _ = prefix_queries(nodes, cfg, 'salt')
        self.assertEqual(queries[0]['tokens'], nodes['short']['tokens'][:1])
        self.assertIn('short', bindings)  # actual engine query, no invented miss

    def test_identity_cache_hits_and_mutations_do_not_cache_residency(self):
        _, nodes = backlog(); cache = {}
        before = prefix_identities(nodes, 'one', cache)
        with patch('dabs_native.contracts.prefix_key', wraps=prefix_key) as digest:
            self.assertEqual(before, prefix_identities(nodes, 'one', cache))
            self.assertEqual(digest.call_count, 0)
        nodes['n000']['tokens'][0] = 123
        self.assertNotEqual(before['n000'], prefix_identities(nodes, 'one', cache)['n000'])
        self.assertNotEqual(before['n001'], prefix_identities(nodes, 'two', cache)['n001'])

    def test_small_prefill_tail_is_token_granular_at_128_block_size(self):
        cfg, s = state()
        class Cache:
            def length(self, *args): return 0
        for size in (1, 3):
            ns = copy.deepcopy(s['nodes'])
            ns['a'].update(state='submitted', replica=0, computed=3584-size)
            batch = select_batch(ns, ['a'], ['a'], {}, {}, Cache(), 0, 0, cfg)
            self.assertEqual(batch[0], [('a', size)])
            self.assertEqual(batch[2], 3584-size)

    def test_phase_coverage_cannot_be_satisfied_by_other_phase_or_holdout(self):
        samples = []
        for split in ('train', 'holdout'):
            for phase in ('prefill_only', 'mixed', 'decode_only'):
                for low in (True, False):
                    samples.append({'replica': 0, 'split': split, 'phase': phase,
                        'p': 1 if low else 1024, 'pc': 0 if low else 3584,
                        'd': 1 if low else 2, 'dc': 3584 if low else 7936})
        good = coverage_check(samples, 0, 'mixed', 1024)
        self.assertTrue(good['pass'])
        for row in samples:
            if row['phase'] == 'mixed' and row['split'] == 'train':
                row['p'] = 128
        bad = coverage_check(samples, 0, 'mixed', 1024)
        self.assertFalse(bad['pass'])
        self.assertIn('p', bad['splits']['train']['missing'])
        self.assertTrue(bad['splits']['holdout']['pass'])

    def test_new_plan_has_disjoint_splits_and_partial_prefix_cases(self):
        plan = calibration_plan()
        self.assertTrue(all(g['coverage_contract'] == CONTRACT for g in plan))
        self.assertFalse({g['seed'] for g in plan if g['split'] == 'train'} &
                         {g['seed'] for g in plan if g['split'] == 'holdout'})
        self.assertTrue(any(g.get('warm_prefix') == 640 and g['concurrency'] == 3 for g in plan))
        self.assertTrue(any(g['length'] == 1 and g['concurrency'] == 3 for g in plan))

    def test_complete_216_node_candidate_pair_with_strict_model_support(self):
        from dabs_ascend.scheduler import snapshot
        cfg, template = state()
        nodes = copy.deepcopy(template['nodes'])
        nodes['b'].update(state='submitted', replica=0, phase='decode', start=1.,
                          computed=3584, output_done=2)
        # Two complete 108-node chains: blocked work is scored, never truncated.
        for root in ('a', 'b'):
            parent = root
            for i in range(107):
                key = root + str(i)
                nodes[key] = {**copy.deepcopy(template['nodes']['a']), 'id': key, 'wid': root,
                    'parents': [parent], 'compiled_successors': [], 'compiled_groups': [],
                    'compiled_descendants': [], 'state': 'blocked', 'prefix_hash': key}
                nodes[parent]['compiled_successors'] = [key]
                parent = key
        s = snapshot(nodes, {}, 2, cfg, 1, 1.)
        s.update(native_mutable=[], native_queue_groups=template['native_queue_groups'],
                 native_running={'0': ['b'], '1': []}, pressure=[1, 0])
        s['cache_evidence'] = [{'prefix_hash': 'a', 'replica': r, 'tokens': 256 if r == 0 else 0,
            'known': True, 'state': 'active_computed' if r == 0 else 'absent', 'valid_until_ms': 120}
            for r in range(2)]
        m = model()
        # CPU fixture support, not a claimed measured calibration.
        m['validation'] = {str(r): {p: {'range': {'p': [0, 1024], 'd': [0, 2],
            'dc': [0, 8192], 'pc': [0, 4096]}} for p in
            ('prefill_only', 'mixed', 'decode_only')} for r in range(2)}
        result = choose(s, m, cfg, time.monotonic() + 2)
        self.assertGreater(result['v5_candidate_scored'], 0)
        self.assertTrue(any(e['cell'] == '00' and e['complete'] for e in result['evaluations']))
        self.assertEqual(len(s['nodes']), 216)
        full = adapted('engine').evaluate(s, baseline(s, cfg, 'Ours'), m, cfg)
        self.assertTrue(full['complete'])
        self.assertEqual(len(full['node_completion_ms']), 216)

    def test_decode_coalescing_checks_later_context_not_only_first_batch(self):
        cfg, s = state()
        s['nodes'].pop('b'); s['ready'] = []
        s['nodes']['a'].update(state='submitted', phase='decode', replica=0,
                              computed=3584, output_done=0, start=1.)
        s['native_running'] = {'0': ['a'], '1': []}
        s['queue_order']['0'] = ['a']
        m = model()
        m['validation'] = {str(r): {'decode_only': {'range': {'p': [0, 0], 'd': [1, 1],
            'dc': [3584, 3585], 'pc': [0, 0]}}} for r in range(2)}
        with self.assertRaisesRegex(ValueError, 'native_calibration_out_of_range:dc:'):
            adapted('engine').evaluate(s, baseline(s, cfg, 'Ours'), m, cfg)

    def test_cache_eviction_optimization_preserves_evictions_and_pinned_prefixes(self):
        from core18v2.engine import CacheModel
        cfg, s = state()
        s['cache_evidence'] = [{'prefix_hash': str(i), 'replica': 0, 'tokens': 256,
            'state': 'active_computed', 'known': True, 'valid_until_ms': 120} for i in range(80)]
        for capacity in (1000, 35):
            s['capacity_blocks'][0] = capacity
            old, new = CacheModel(s, cfg), cache_model_class()(s, cfg)
            active = [{**s['nodes']['a'], 'prefix_hash': '5', 'computed': 128}]
            old.evict_for_reservations(0, active, 0)
            new.evict_for_reservations(0, active, 0)
            self.assertEqual(old.entries, new.entries)
            self.assertEqual(old.known, new.known)
            self.assertEqual(old.counts, new.counts)
            self.assertIn((0, '5'), new.entries)

    def test_missed_tick_skips_window_without_extending_old_deadline(self):
        self.assertAlmostEqual(aligned_start(10., 10.2, 10.27, .2, .025), 10.4)
        self.assertAlmostEqual(aligned_start(10., 10.2, 10.201, .2, .025), 10.2)

    def test_log_projection_retains_semantics_and_snapshot_link_not_duplicate_payload(self):
        _, s = state()
        s['native_snapshots'] = [{'snapshot_id': 'engine:7', 'identity': {'engine_instance': 'e'},
                                 'cache': [{'tokens': [42] * 3584}]}]
        record = decision_record({'index': 7, 'snapshot': s})
        self.assertEqual(record['snapshot']['nodes'], s['nodes'])
        self.assertEqual(record['snapshot']['native_snapshot_refs'][0]['snapshot_id'], 'engine:7')
        self.assertNotIn('native_snapshots', record['snapshot'])
        self.assertIn('native_snapshots', s)  # memory used by final queue receipt audit is unchanged
