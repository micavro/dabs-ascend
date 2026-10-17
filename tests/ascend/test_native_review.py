"""Adversarial boundary checks on the actual adapter code, CPU only."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from dabs_ascend.backend import Backend

from test_native import fake_engine, action


def admission():
    return {'node_id': 'z', 'client_request_id': 'client-z', 'prompt_tokens': 3584,
            'max_tokens': 16, 'input_digest': 'cpu-digest'}


class AdmissionBoundaryReview(unittest.TestCase):
    def test_atomic_admission_preserves_existing_priority_and_does_not_rank_new_client(self):
        a = fake_engine()
        cmd = action(a, a.snapshot({}), ['y', 'x'])
        a.apply(cmd)
        control = copy.deepcopy(a.active_control)
        before = [r.request_id for r in a.s.waiting]
        receipt = a.dispatch({'op': 'admit', 'admissions': [admission()]})
        self.assertEqual(receipt['status'], 'applied')
        self.assertEqual(a.active_control, control)
        self.assertEqual([r.request_id for r in a.s.waiting], before)
        a.engine_ids_by_client['client-z'] = 'z'
        z = copy.copy(a.s.requests['x']); z.request_id = 'z'
        a.s.requests['z'] = z; a.s.waiting.append(z)
        a.accepted_ns['z'] = cmd['not_before_ns'] + 1
        a.reorder_boundary(cmd['not_before_ns'] + 2)
        self.assertEqual([r.request_id for r in a.s.waiting], ['y', 'x', 'z'])
        # Admission must not extend the original priority window.
        a.reorder_boundary(cmd['action_expires_ns'])
        self.assertEqual([r.request_id for r in a.s.waiting], ['x', 'y', 'z'])

    def test_validation_crossing_deadline_never_mutates_queue_or_grants_permits(self):
        a = fake_engine()
        clock = [1_000_000_000]
        with patch('dabs_native.engine.time.monotonic_ns', side_effect=lambda: clock[0]):
            snap = a.snapshot({})
            cmd = action(a, snap, ['y', 'x'])
            cmd.update(admissions=[admission()], action_expires_ns=clock[0] + 50_000_000)
            def slow_resources():
                clock[0] += 60_000_000
                return a.resources
            a.read_resources = slow_resources
            receipt = a.apply(cmd)
        self.assertEqual(receipt['status'], 'expired')
        self.assertEqual(receipt['reason'], 'expired_action_window')
        self.assertEqual([r.request_id for r in a.s.waiting], ['x', 'y'])
        self.assertFalse(a.permits)
        self.assertIsNone(a.active_control)


class NativeBindingReview(unittest.IsolatedAsyncioTestCase):
    async def test_response_must_bind_node_client_server_engine_and_granted_action(self):
        expected = {'node_id': 'node', 'client_request_id': 'client', 'server_request_id': 'dabs-client',
                    'engine_request_id': 'engine-1', 'action_id': 'permit-action'}
        response = {'id': 'dabs-client', 'choices': [{'index': 0, 'text': 'x', 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 128, 'completion_tokens': 16}}
        def handler(request):
            return httpx.Response(200, text='data: ' + json.dumps(response) + '\n\ndata: [DONE]\n\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = Backend(client, ['http://test'])
            backend.native = SimpleNamespace(binding=AsyncMock(), capabilities=[{'resources': {'context_tokens': 4096}}])
            with tempfile.TemporaryDirectory() as directory:
                for field in ('node_id', 'client_request_id', 'server_request_id', 'action_id', 'engine_request_id', None):
                    node = {'id': 'node', 'client_request_id': 'client', 'native_action_id': 'permit-action',
                            'cache_salt': 'scope', 'tokens': [1] * 128, 'max_tokens': 16,
                            'max_output_tokens': 16, 'fixed_workload_replay': True}
                    row = dict(expected)
                    if field:
                        row[field] = '' if field == 'engine_request_id' else 'wrong'
                    backend.native.binding.return_value = {'mappings': [row]}
                    if field:
                        with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binding'):
                            await backend.generate(0, node, 1, Path(directory) / 'requests.jsonl', schema=None)
                    else:
                        result = await backend.generate(0, node, 1, Path(directory) / 'requests.jsonl', schema=None)
                        self.assertEqual(result['engine_action_acknowledged'], 'permit-action')

    def test_future_window_is_not_reported_applied_before_start(self):
        a = fake_engine()
        with patch('dabs_native.engine.time.monotonic_ns', return_value=1_000_000_000):
            cmd = action(a, a.snapshot({}), ['y', 'x'])
            cmd['not_before_ns'] += 50_000_000
            receipt = a.apply(cmd)
        self.assertEqual(receipt['status'], 'rejected')
        self.assertEqual(receipt['reason'], 'action_not_started')
        self.assertIsNone(a.active_control)
