"""CPU regressions for the Ascend timing contract and lifecycle boundaries."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from dabs_native import timing_policy as policy
from dabs_native.contracts import check_action, require_control_semantics
from dabs_ascend.backend import Backend
from dabs_ascend.common import ROOT, configuration, read_json
from dabs_ascend.native_client import NativeClient
from dabs_ascend.native_planning import planning_window, record_planner_return
from test_native import action, fake_engine, model, resources, state
from test_native_setup import setup
from test_native_simplification import capability, observation


def binding():
    return dict(node_id='node', client_request_id='client', server_request_id='dabs-client',
                engine_request_id='engine-1', action_id='permit-action')


def node():
    return dict(id='node', client_request_id='client', native_action_id='permit-action',
                cache_salt='scope', tokens=[1]*128, max_output_tokens=16,
                fixed_workload_replay=True, state='submitted')


def sse(value):
    return ('data: ' + json.dumps(value) + '\n\n').encode()


class TimingContractTests(unittest.TestCase):
    def test_every_runtime_timing_field_and_reference_file_agree(self):
        expected = policy.configuration()
        self.assertEqual((expected['epoch_s'], expected['snapshot_max_age_s'],
                          expected['planning_soft_limit_s'], expected['commit_reserve_s']),
                         (1.5, 1.5, 1.2, .025))
        self.assertEqual(expected['continuation_period_ms'], 1500)
        for cfg in (configuration(4), configuration(4, resources()),
                    read_json(ROOT/'profiles/reference/configuration.json')):
            for key, value in expected.items():
                self.assertEqual(cfg[key], value, key)

    def test_service_snapshot_and_default_action_both_use_1500ms(self):
        engine = fake_engine()
        with patch('dabs_native.engine.time.monotonic_ns', return_value=1_000_000_000):
            snapshot = engine.snapshot({})
        self.assertEqual(snapshot['expires_ns'] - snapshot['captured_ns'], 1_500_000_000)
        native = NativeClient(None, [])
        with patch('dabs_ascend.native_client.time.monotonic_ns', return_value=1_100_000_000):
            cmd = native.action(snapshot)
        self.assertEqual(cmd['action_expires_ns'], 2_500_000_000)
        self.assertIsNone(check_action(cmd, snapshot, engine.identity, 2_300_000_000, engine.step))
        bad = {**cmd, 'action_expires_ns': 2_500_000_001}
        self.assertEqual(check_action(bad, snapshot, engine.identity, 1_100_000_000, engine.step),
                         'invalid_fixed_action_window')
        self.assertIsNotNone(check_action(cmd, snapshot, engine.identity, 2_500_000_001, engine.step))

    def test_longer_lifetime_does_not_relax_queue_or_resource_checks(self):
        engine = fake_engine()
        snapshot = engine.snapshot({})
        cmd = action(engine, snapshot, ['y', 'x'])
        cmd['action_expires_ns'] = snapshot['captured_ns'] + policy.ACTION_WINDOW_NS
        engine.s.requests['y'].num_computed_tokens = 128
        receipt = engine.apply(cmd)
        self.assertEqual(receipt['reason'], 'request_started_since_snapshot')
        self.assertEqual([r.request_id for r in engine.s.waiting], ['x', 'y'])

    def test_old_live_service_requires_install_without_hash_gate(self):
        cap = capability()
        require_control_semantics(cap)
        cap['adapter_sha256'] = {'arbitrary.py': 'not-a-gate'}
        require_control_semantics(cap)
        for field in ('timing_policy',):
            old = copy.deepcopy(cap)
            old.pop(field)
            with self.assertRaisesRegex(ValueError, 'run bash run_native.sh install'):
                require_control_semantics(old)
        installer = setup.Installer(SimpleNamespace())
        def http(replica, route, payload=None):
            if route == '/health':
                return ''
            return json.dumps(old if payload['op'] == 'capabilities' else observation())
        with patch.object(installer, 'http', side_effect=http):
            self.assertIsNone(installer.ready(0))
            self.assertIn('1500/1500/1200', installer.last_ready_error)

    def test_window_reserves_commit_and_does_not_reset_snapshot_or_tick(self):
        cfg, snapshot = state()
        snapshot['now'] = 100.1
        snapshot['native_snapshots'] = [{'captured_ns': 100_000_000_000, 'expires_ns': 101_500_000_000}]
        window = planning_window(snapshot, cfg, 101.5, now=100.1)
        self.assertAlmostEqual(window['budget_ms'], 1200)
        self.assertAlmostEqual(window['deadline'], 101.3)
        later = planning_window(snapshot, cfg, 101.5, now=100.4)
        self.assertAlmostEqual(later['deadline'], 101.475)
        self.assertAlmostEqual(later['budget_ms'], 1075)
        self.assertEqual(planning_window(snapshot, cfg, 101.5, now=101.5)['budget_ms'], 0)
        with patch('dabs_ascend.native_planning.time.monotonic', return_value=101.3):
            result = record_planner_return({'planning_prepared_at': 100.1, 'planning_finished_at': 101.2})
        self.assertAlmostEqual(result['planning_return_delay_ms'], 100)
        self.assertAlmostEqual(result['planning_from_ready_ms'], 1200)

    def test_snapshot_returns_bounded_explicit_terminal_bindings(self):
        engine = fake_engine()
        bound = {**binding(), 'engine_finished_ns': 123}
        engine.engine_ids_by_client['client'] = 'engine-1'
        engine.mappings['engine-1'] = bound
        result = engine.snapshot({'client_request_ids': ['client', 'unknown']})
        self.assertEqual(result['request_bindings'], [bound])
        self.assertEqual(engine.snapshot({})['request_bindings'], [])
        for bad in ('client', ['x']*257, [''], [42]):
            with self.assertRaises(ValueError):
                engine.snapshot({'client_request_ids': bad})


class LifecycleAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_collects_terminal_mapping_after_dag_without_delaying_release(self):
        from dabs_ascend.native_probe import completed_bindings
        requests = [{'replica': 0, 'mapping': binding()}]
        completed = {**binding(), 'engine_admitted_ns': 10, 'engine_finished_ns': 20}
        native = SimpleNamespace(call=AsyncMock(return_value={'mappings': [completed]}))
        await completed_bindings(native, requests)
        self.assertEqual(requests[0]['mapping']['engine_finished_ns'], 20)
        native.call.assert_awaited_once_with(0, 'mapping', client_request_ids=['client'])
        native.call.return_value = {'mappings': [binding()]}
        with self.assertRaisesRegex(ValueError, 'terminal engine binding'):
            await completed_bindings(native, requests)

    async def test_return_callback_is_not_retimed_by_later_diagnostic_consumption(self):
        from dabs_ascend.native_planning import record_planner_completion
        future = asyncio.get_running_loop().create_future()
        future.set_result({'planning_prepared_at': 10., 'planning_finished_at': 11.})
        with patch('dabs_ascend.native_planning.time.monotonic', return_value=11.1):
            record_planner_completion(future)
        with patch('dabs_ascend.native_planning.time.monotonic', return_value=12.5):
            result = record_planner_return(future.result())
        self.assertAlmostEqual(result['planning_return_delay_ms'], 100)
        self.assertAlmostEqual(result['planning_from_ready_ms'], 1100)

    async def test_snapshot_evidence_resolves_pending_query_without_polling(self):
        native = NativeClient(None, [])
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def slow(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        native.call = AsyncMock(side_effect=slow)
        pending = asyncio.create_task(native.binding(0, 'client'))
        await started.wait()
        native.remember_bindings(0, {'identity': capability()['identity'],
                                   'request_bindings': [{**binding(), 'engine_finished_ns': 7}]})
        result = await asyncio.wait_for(pending, 1)
        self.assertEqual(result['mappings'][0]['engine_finished_ns'], 7)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(await native.binding(0, 'client'), result)
        self.assertEqual(native.call.call_count, 1)
        self.assertFalse(native.binding_waiters)
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            native.remember_bindings(0, {'mappings': [{**binding(), 'action_id': 'wrong'}]})

    async def test_mapping_transport_error_is_not_hidden(self):
        native = NativeClient(None, [])
        native.call = AsyncMock(side_effect=ValueError('engine restart'))
        with self.assertRaisesRegex(ValueError, 'engine restart'):
            await native.binding(0, 'client')
        self.assertFalse(native.binding_waiters)

    async def test_early_mapping_overlaps_stream_and_empty_early_lookup_retries_once(self):
        for early_empty in (False, True):
            with self.subTest(early_empty=early_empty), tempfile.TemporaryDirectory() as directory:
                n = node()
                binding_seen = asyncio.Event()
                calls = []
                async def lookup(*args):
                    calls.append(True)
                    binding_seen.set()
                    return {'mappings': [] if early_empty and len(calls) == 1 else [binding()]}
                class Stream(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        yield sse({'id': 'dabs-client', 'choices': [{'index': 0, 'text': 'x'}]})
                        await asyncio.wait_for(binding_seen.wait(), 1)
                        yield sse({'choices': [{'index': 0, 'text': '', 'finish_reason': 'length'}],
                                   'usage': {'prompt_tokens': 128, 'completion_tokens': 16}})
                        yield b'data: [DONE]\n\n'
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, stream=Stream()))) as client:
                    backend = Backend(client, ['http://fixture'])
                    backend.native = SimpleNamespace(binding=lookup, capabilities=[{'resources': resources()}])
                    result = await backend.generate(0, n, 1, Path(directory)/'requests.jsonl', schema=None)
                self.assertEqual(n['binding_status'], 'verified')
                self.assertEqual(result['engine_action_acknowledged'], 'permit-action')
                self.assertEqual(len(calls), 2 if early_empty else 1)
                if not early_empty:
                    self.assertLessEqual(n['binding_verified_at'], n['http_done_at'])

    async def test_http_done_does_not_complete_while_binding_is_missing(self):
        n = node()
        release = asyncio.Event()
        async def lookup(*args):
            await release.wait()
            return {'mappings': []}
        response = {'id': 'dabs-client', 'choices': [{'index': 0, 'text': 'x', 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 128, 'completion_tokens': 16}}
        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=sse(response)+b'data: [DONE]\n\n'))) as client:
                backend = Backend(client, ['http://fixture'])
                backend.native = SimpleNamespace(binding=lookup, capabilities=[{'resources': resources()}])
                task = asyncio.create_task(backend.generate(0, n, 1, Path(directory)/'requests.jsonl', schema=None))
                for _ in range(10):
                    await asyncio.sleep(0)
                    if 'http_done_at' in n:
                        break
                self.assertIn('http_done_at', n)
                self.assertEqual(n['state'], 'submitted')
                self.assertFalse(task.done())
                release.set()
                with self.assertRaisesRegex(ValueError, 'binding failed'):
                    await task
                self.assertEqual(n['binding_status'], 'failed')
                self.assertEqual(n['state'], 'submitted')

    async def test_retry_timing_includes_both_attempts(self):
        cfg, s = state()
        initial = dict(native_observation_errors=['observation_lifecycle_changed'],
                       native_observe_ms=8., native_prepare_ms=1., native_rpc_ms=5., native_projection_ms=2.,
                       native_prepare_parts_ms={'identity_ms': .3, 'query_pack_ms': .7})
        final = dict(native_observation_errors=[], native_observe_ms=6.,
                     native_prepare_ms=1., native_rpc_ms=4., native_projection_ms=1.,
                     native_prepare_parts_ms={'identity_ms': .2, 'query_pack_ms': .8})
        native = NativeClient(None, ['fixture'])
        with patch.object(native, '_observe_once', side_effect=[initial, final]):
            observed = await native.observe(s['nodes'], cfg, 1, model())
        self.assertEqual(observed['native_rpc_ms'], 9.)
        self.assertEqual(observed['native_prepare_parts_ms'], {'identity_ms': .5, 'query_pack_ms': 1.5})
        self.assertEqual(observed['native_observation_retry']['attempts'], 2)

    async def test_completion_crossing_snapshot_gets_one_fresh_projection(self):
        cfg, s = state()
        nodes = s['nodes']
        for n in nodes.values():
            n['tokens'] = [7]*3584
        nodes['b']['state'] = 'blocked'
        native = NativeClient(None, ['r0', 'r1'])
        native.capabilities = [capability(), capability()]
        calls = []
        async def call(replica, op, **payload):
            calls.append(payload)
            if len(calls) == 1:
                nodes['b']['state'] = 'ready'
            row = observation()
            now = time.monotonic_ns()
            row.update(captured_ns=now, expires_ns=now+policy.SNAPSHOT_TTL_NS)
            return row
        native.call = call
        result = await native.observe(nodes, cfg, 1, model())
        self.assertEqual(len(calls), 4)  # 2 replicas, exactly 2 attempts.
        retry = result['native_observation_retry']
        self.assertTrue(retry['recovered'])
        self.assertIn('observation_lifecycle_changed', retry['initial_errors'])
        self.assertEqual(retry['initial_query_budget']['ready_requests'], 1)
        self.assertEqual(result['native_query_budget']['ready_requests'], 2)
        self.assertFalse(result['native_observation_errors'])

    async def test_terminal_evidence_never_infers_http_done_or_masks_persistent_transit(self):
        for budget in (True, False):
            cfg, s = state()
            nodes = s['nodes']
            for n in nodes.values():
                n['tokens'] = [7]*3584
            nodes['b'].update(state='submitted', replica=0, client_request_id='client',
                              start=time.monotonic(), binding_status='pending')
            if not budget:
                cfg['native_observation_deadline'] = time.monotonic()-1
            native = NativeClient(None, ['r0', 'r1'])
            native.capabilities = [capability(), capability()]
            calls = []
            async def call(replica, op, **payload):
                calls.append(payload)
                row = observation()
                now = time.monotonic_ns()
                row.update(captured_ns=now, expires_ns=now+policy.SNAPSHOT_TTL_NS,
                           request_bindings=[{**binding(), 'node_id': 'b', 'engine_finished_ns': now}])
                return row
            native.call = call
            result = await native.observe(nodes, cfg, 1, model())
            self.assertIn('request_in_API_EngineCore_transit', result['native_observation_errors'])
            self.assertEqual(nodes['b']['state'], 'submitted')
            self.assertNotIn('http_done_at', nodes['b'])
            self.assertEqual(len(calls), 4 if budget else 2)
            self.assertEqual(result['native_observation_retry']['retry_skipped_budget'], not budget)
