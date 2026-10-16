"""CPU regression for the field's 96-ready observation overload; no NPU claim."""
import asyncio
import copy
import time
import unittest
from unittest.mock import AsyncMock

from dabs_ascend.native_client import NativeClient
from dabs_ascend.native_observation import prefix_queries
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
