"""CPU regression for the field's 96-ready observation overload; no NPU claim."""
import asyncio
import copy
import time
import unittest
from unittest.mock import AsyncMock, patch

from dabs_ascend.native_client import NativeClient
from dabs_ascend.native_observation import prefix_queries, prefix_identities
from dabs_native.contracts import prefix_key
from test_native import state, model
from test_native_simplification import observation


def backlog():
    cfg, s = state()
    nodes = {}
    for i in range(96):
        n = copy.deepcopy(s['nodes']['a'])
        key = f'n{i:03d}'
        n.update(id=key, wid=key, tokens=[i // 3] + [7] * 3583)
        nodes[key] = n
    return cfg, nodes


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    def test_input_identity_deduplicates_exact_prefixes_and_invalidates_on_change(self):
        _, nodes = backlog()
        with patch('dabs_native.contracts.prefix_key', wraps=prefix_key) as hashed:
            keys = prefix_identities(nodes, 'salt')
            self.assertEqual(hashed.call_count, 32)
        for k, n in nodes.items():
            self.assertEqual(keys[k], prefix_key(n['tokens'][:n['share']], 'salt'))
        self.assertNotEqual(keys, prefix_identities(nodes, 'new-salt'))
        nodes['n000']['tokens'][0] = 99
        changed = prefix_identities(nodes, 'salt')
        self.assertNotEqual(keys['n000'], changed['n000'])
        self.assertEqual(keys['n001'], changed['n001'])

    async def test_projection_elapsed_expires_evidence_without_faking_a_fresh_timestamp(self):
        from dabs_ascend.scheduler import snapshot
        cfg, nodes = backlog()
        native = NativeClient(None, ['r0', 'r1'])
        native.capabilities = [{'resources': observation()['resources']}] * 2
        async def call(r, op, **payload):
            row = observation(); now = time.monotonic_ns()
            row.update(captured_ns=now, expires_ns=now + 120_000_000)
            row['cache'] = [{'query_id': q['query_id'], 'state': 'hit', 'reusable_tokens': 256}
                            for q in payload.get('queries', [])]
            return row
        def slow_projection(*args):
            time.sleep(.13)
            return snapshot(*args)
        native.call = call
        with patch('dabs_ascend.scheduler.snapshot', side_effect=slow_projection), patch('dabs_ascend.native_observation.query_priority', return_value=list(nodes)):
            s = await native.observe(nodes, cfg, 1, model())
        self.assertGreater(time.monotonic() - s['now'], .12)
        self.assertGreater(s['native_projection_ms'], 120)
        self.assertIn('snapshot_expired_before_observation_return', s['native_observation_errors'])
        self.assertTrue(all(not e['known'] and e['tokens'] is None for e in s['cache_evidence']))

    async def test_prefix_preparation_happens_before_rpc_capture(self):
        cfg, nodes = backlog()
        native = NativeClient(None, ['r0', 'r1'])
        native.capabilities = [{'resources': observation()['resources']}] * 2
        prepared = False
        def prepare(*args, **kwargs):
            nonlocal prepared
            result = prefix_identities(*args, **kwargs)
            prepared = True
            return result
        async def call(r, op, **payload):
            self.assertTrue(prepared)
            row = observation(); now = time.monotonic_ns()
            row.update(captured_ns=now, expires_ns=now + 120_000_000)
            return row
        native.call = call
        with patch('dabs_ascend.native_observation.prefix_identities', side_effect=prepare):
            s = await native.observe(nodes, cfg, 1, model())
        self.assertAlmostEqual(s['native_observe_ms'],
            s['native_prepare_ms'] + s['native_rpc_ms'] + s['native_projection_ms'], places=5)

    async def test_expired_queue_is_never_sent_by_atomic_fallback(self):
        cfg, nodes = backlog()
        node = nodes['n000']; node['client_request_id'] = 'client-a'
        native = NativeClient(None, ['r0'])
        old = {'running': [], 'waiting': [{'mapping': {'node_id': 'old'},
                'engine_request_id': 'old-engine', 'queue_mutable': True}]}
        state = {'native_snapshots': [old], 'action_deadline_ns': 0}
        action = {'admit': ['n000'], 'placement': {'n000': 0}, 'order': ['old', 'n000']}
        native.call = AsyncMock(return_value={'action_id': 'fresh', 'status': 'applied',
                                               'admitted_permits': ['client-a']})
        order, receipts = await native.commit(action, state, nodes, fallback=True)
        self.assertEqual(order, ['n000'])
        self.assertEqual(native.call.call_args.args, (0, 'admit'))
        self.assertEqual(set(native.call.call_args.kwargs), {'admissions'})
        self.assertEqual(receipts[0]['receipt']['action_id'], 'fresh')
        native.call = AsyncMock(return_value={'action_id': 'refused', 'status': 'rejected',
                                               'admitted_permits': []})
        order, receipts = await native.commit(action, state, nodes, fallback=True)
        self.assertEqual(order, [])  # no HTTP submission without an engine permit

    def test_prefix_payload_is_bounded_deduplicated_and_block_correct(self):
        cfg, nodes = backlog()
        queries, bindings, stats = prefix_queries(nodes, cfg, 'scope')
        self.assertEqual(len(queries), 8)
        self.assertEqual(stats['query_tokens_per_replica'], 8 * 257)
        self.assertEqual(len(bindings), 24)
        self.assertEqual(stats['unqueried_requests'], 72)
        self.assertEqual(bindings['n000'], bindings['n002'])
        for query in queries:
            self.assertEqual(query['tokens'], nodes[query['query_id']]['tokens'][:257])
            self.assertEqual((len(query['tokens']) - 1) // 128 * 128, 256)
            self.assertEqual(query['salt'], 'scope')
        nodes['n000'].update(share=128, tokens=[3] * 128)
        query = prefix_queries(nodes, cfg, 'new-salt')[0][0]
        self.assertEqual(len(query['tokens']), 128)  # never invent a guard token
        self.assertEqual(query['salt'], 'new-salt')

    async def test_unqueried_expired_and_light_refresh_stay_unknown(self):
        cfg, nodes = backlog()
        native = NativeClient(None, ['r0', 'r1'])
        native.capabilities = [{'resources': observation()['resources']} for _ in range(2)]
        async def call(r, op, **payload):
            row = observation(); now = time.monotonic_ns()
            row.update(captured_ns=now, expires_ns=now + (120_000_000 if r == 0 else -1))
            row['cache'] = [{'query_id': q['query_id'], 'state': 'hit', 'reusable_tokens': 256}
                            for q in payload.get('queries', [])]
            return row
        native.call = AsyncMock(side_effect=call)
        s = await native.observe(nodes, cfg, 1, model())
        self.assertEqual(len(s['ready']), 96)  # candidates are not cropped
        known = [e for e in s['cache_evidence'] if e['known']]
        self.assertEqual(len(known), 24)
        self.assertTrue(all(e['replica'] == 0 and e['tokens'] == 256 for e in known))
        self.assertTrue(all(e['tokens'] is None for e in s['cache_evidence'] if not e['known']))
        light = await native.observe(nodes, cfg, 2, model(), cache_queries=False)
        self.assertEqual(light['native_query_budget']['unique_queries'], 0)
        self.assertTrue(all(not e['known'] and e['tokens'] is None for e in light['cache_evidence']))

    async def test_queue_delay_before_capture_does_not_expire_fresh_engine_snapshot(self):
        cfg, nodes = backlog()
        native = NativeClient(None, ['r0', 'r1'])
        native.capabilities = [{'resources': observation()['resources']} for _ in range(2)]
        async def delayed(r, op, **payload):
            await asyncio.sleep(.14)
            row = observation(); now = time.monotonic_ns()
            row.update(captured_ns=now, expires_ns=now + 120_000_000)
            return row
        native.call = delayed
        s = await native.observe(nodes, cfg, 1, model(), cache_queries=False)
        self.assertGreater(s['native_observe_ms'], 120)
        self.assertNotIn('snapshot_expired_in_transit', s['native_observation_errors'])
        self.assertTrue(all(age < .12 for age in s['state_ages_s']))
