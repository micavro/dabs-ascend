"""CPU fault injection only. These tests cannot satisfy the live NPU gate."""
import asyncio
from collections import deque
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch

from dabs_native.contracts import DeviceMeasurements, Receipts, check_action, digest, prefix_key
from dabs_native.engine import EngineAdapter
from dabs_native.install import patch_source, check, rollback, PATCHED, sha
from dabs_ascend.common import configuration, source_paths, write_json
from dabs_ascend.native_algorithms import adapted, baseline, choose, transformation_manifest
from dabs_ascend.native_calibration import batch_features, fit_model
from dabs_ascend.native_gate import evaluate_local
from dabs_ascend.scheduler import snapshot
from dabs_ascend import r234

source_paths()


def resources(step=1024, block=128):
    return {'block_size': block, 'step_tokens': step, 'server_max_seqs': 10,
            'context_tokens': 4096, 'physical_blocks': 1000}


def model():
    return {'models': {str(r): {'phases': {p: {'coef': [1., 2., 1., .1, .1, .1]}
              for p in ('prefill_only', 'decode_only', 'mixed')}} for r in range(4)}}


def state():
    cfg = configuration(2, resources())
    nodes = {k: {'id': k, 'wid': k, 'parents': [], 'compiled_successors': [], 'compiled_groups': [['a', 'b']],
             'compiled_descendants': [], 'state': 'ready', 'arrival': 1., 'ready_at': 1., 'start': None,
             'replica': None, 'phase': 'prefill', 'computed': 0, 'output_done': 0, 'prompt': 3584,
             'output': 16, 'max_output_tokens': 16, 'share': 256, 'prefix_hash': k}
             for k in ('a', 'b')}
    s = snapshot(nodes, {}, 2, cfg, 1, 1.)
    s.update(native_mutable=[], native_queue_groups={str(r): {'running_prefill': [], 'waiting': []} for r in range(2)},
             physical_active_blocks=[0, 0], capacity_blocks=[1000, 1000], reserved_blocks=[0, 0],
             native_observation_errors=[])
    return cfg, s


def fake_engine():
    # Directly construct boundary state, never claim this emulates an NPU.
    a = EngineAdapter.__new__(EngineAdapter)
    a.identity = {'engine_instance': 'test-epoch'}
    a.step, a.serial = 1, 0
    a.resources = resources()
    a.snapshots, a.pending, a.batches, a.permits, a.mappings = {}, {}, [], {}, {}
    a.receipts, a.client_ids, a.receipt_for_step = Receipts(), set(), []
    a.active_control, a.host_call, a.host_call_serial = None, None, 0
    a.last_completion_boundary = None
    a.engine_ids_by_client, a.batch_index = {}, {}
    a.accepted_ns = {'x': time.monotonic_ns() - 2_000_000, 'y': time.monotonic_ns() - 1_000_000}
    def req(rid, phase='WAITING', computed=0):
        return SimpleNamespace(request_id=rid, status=SimpleNamespace(name=phase), num_computed_tokens=computed,
            num_prompt_tokens=3584, num_output_tokens=0, num_output_placeholders=0, max_tokens=128)
    x, y = req('x'), req('y')
    a.s = SimpleNamespace(waiting=deque([x, y]), running=[], skipped_waiting=deque(), requests={'x': x, 'y': y},
                          get_request_counts=lambda: (0, len(a.s.waiting)))
    a.read_resources = lambda: a.resources
    a.physical = lambda: {'available_for_new_reservations': 1000}
    return a


def action(a, snap, order=()):
    return {'action_id': 'action-test', 'snapshot_id': snap['snapshot_id'], 'identity': a.identity,
        'not_before_ns': snap['captured_ns'], 'action_expires_ns': snap['captured_ns'] + 200_000_000,
        'expires_ns': snap['expires_ns'], 'resource_digest': digest(a.resources), 'order': list(order), 'admissions': []}


class NativeContracts(unittest.TestCase):
    def test_priority_expires_at_fixed_boundary_and_fifo_restores_without_undo(self):
        a = fake_engine(); snap = a.snapshot({}); cmd = action(a, snap, ['y', 'x'])
        a.apply(cmd)
        self.assertEqual([r.request_id for r in a.s.waiting], ['y', 'x'])
        row = a.reorder_boundary(cmd['action_expires_ns'])
        self.assertEqual(row['phase'], 'common_continuation')
        self.assertEqual([r.request_id for r in a.s.waiting], ['x', 'y'])
        self.assertEqual(len(a.s.requests), 2)

    def test_five_second_age_guard_abandons_priority(self):
        a = fake_engine(); a.accepted_ns['x'] -= 5_000_000_000
        snap = a.snapshot({}); row = a.apply(action(a, snap, ['y', 'x']))
        self.assertEqual(row['status'], 'rejected')
        self.assertEqual(row['rejected']['x'], 'prefill_age_guard_5s')
        self.assertEqual([r.request_id for r in a.s.waiting], ['x', 'y'])
        self.assertIsNone(a.active_control)  # rejected rank cannot activate later

    def test_actual_queue_reorder_receipt_and_duplicate(self):
        a = fake_engine(); snap = a.snapshot({}); cmd = action(a, snap, ['y', 'x'])
        row = a.apply(cmd)
        self.assertEqual(row['effective_waiting_order'], ['y', 'x'])
        self.assertEqual(row['status'], 'applied')
        self.assertEqual(row['identity'], a.identity)
        self.assertEqual(a.apply(cmd), row)
        with self.assertRaisesRegex(ValueError, 'different_content'):
            a.apply({**cmd, 'order': ['x', 'y']})

    def test_late_started_target_is_not_reordered(self):
        a = fake_engine(); snap = a.snapshot({})
        a.s.requests['y'].num_computed_tokens = 128
        row = a.apply(action(a, snap, ['y', 'x']))
        self.assertEqual(row['status'], 'expired')
        self.assertEqual([r.request_id for r in a.s.waiting], ['x', 'y'])

    def test_unrelated_step_may_advance_but_target_state_still_checked(self):
        a = fake_engine(); snap = a.snapshot({}); a.step += 1
        self.assertEqual(a.apply(action(a, snap, ['y', 'x']))['status'], 'applied')

    def test_new_queue_member_prevents_snapshot_overwrite(self):
        a = fake_engine(); snap = a.snapshot({})
        a.s.waiting.append(copy.copy(a.s.requests['x']))
        row = a.apply(action(a, snap, ['y', 'x']))
        self.assertEqual(row['reason'], 'queue_changed_since_snapshot')
        self.assertEqual(len(a.s.waiting), 3)

    def test_restart_expiry_and_resource_mismatch(self):
        for kind in ('restart', 'expired', 'resources'):
            a = fake_engine(); snap = a.snapshot({}); cmd = action(a, snap)
            if kind == 'restart': cmd['identity'] = {'engine_instance': 'old'}
            if kind == 'expired': cmd['expires_ns'] = 0
            if kind == 'resources': cmd['resource_digest'] = 'wrong'
            self.assertIn(a.apply(cmd)['status'], ('rejected', 'expired'))

    def test_eviction_invalidates_action_evidence(self):
        a = fake_engine()
        a.query_prefix = lambda q: {'state': 'hit', 'reusable_tokens': 128}
        snap = a.snapshot({'queries': [{}]})
        a.query_prefix = lambda q: {'state': 'miss', 'reusable_tokens': 0}
        self.assertEqual(a.apply(action(a, snap))['reason'], 'cache_evidence_invalidated')

    def test_async_progress_excludes_inflight_work(self):
        a = fake_engine(); req = a.s.requests['x']; req.num_computed_tokens = 2048
        a.pending = {'b1': {'scheduled_tokens': {'x': 1024}}, 'b2': {'scheduled_tokens': {'x': 512}}}
        row = a.request_row(req)
        self.assertEqual(row['completed_computed_tokens'], 512)
        self.assertFalse(row['queue_mutable'])

    def test_inflight_and_decode_slots_cannot_be_reordered(self):
        a = fake_engine(); a.s.requests['x'].num_computed_tokens = 3584
        snap = a.snapshot({}); result = a.apply(action(a, snap, ['x', 'y']))
        self.assertEqual(result['status'], 'partially_applied')
        self.assertIn('x', result['rejected'])
        self.assertEqual(result['applied'], ['y'])

    def test_raw_record_has_identity_and_no_signing_dependency(self):
        a = fake_engine(); row = a.snapshot({})
        self.assertNotIn('signature', row)
        cmd = action(a, row)
        a.identity = {'engine_instance': 'restarted'}
        self.assertEqual(a.apply(cmd)['reason'], 'engine_restarted')

    def test_ids_and_namespace_are_distinct(self):
        self.assertNotEqual(prefix_key([1]*129, 'a'), prefix_key([1]*129, 'b'))
        self.assertNotEqual(prefix_key([1]*128, 'a'), prefix_key([1]*129, 'a'))

    def test_query_hit_miss_and_inflight_unknown_use_real_coordinator_return(self):
        a = fake_engine()
        class Request:
            def __init__(self, rid, tokens, sampling, pooling, cache_salt, block_hasher):
                self.block_hashes = (cache_salt, tuple(tokens))
        block = SimpleNamespace(block_id=7, block_hash='resident')
        def hit(hashes, maximum):
            if hashes[0] == 'cold' or maximum < 128:
                return ([],), 0
            return ([block],), 128
        a.s.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(find_longest_cache_hit=hit))
        a.core = SimpleNamespace(request_block_hasher=object())
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(SamplingParams=lambda **kw: kw),
                                     'vllm.v1.request': SimpleNamespace(Request=Request)}):
            query = {'tokens': [1]*129, 'salt': 'hot', 'query_id': 'q'}
            self.assertEqual(a.query_prefix(query)['reusable_tokens'], 128)
            self.assertEqual(a.query_prefix({**query, 'salt': 'cold'})['state'], 'miss')
            self.assertEqual(a.query_prefix({**query, 'tokens': [1]*128})['reusable_tokens'], 0)
            a.pending = {'inflight': {'writing_blocks': [7]}}
            unknown = a.query_prefix(query)
            self.assertEqual(unknown['state'], 'unknown'); self.assertIsNone(unknown['reusable_tokens'])

    def test_physical_capacity_is_from_pool_and_free_includes_reclaimable(self):
        a = fake_engine()
        a.s.requests = {}
        a.pool = SimpleNamespace(blocks=[SimpleNamespace(is_null=False, ref_cnt=ref, block_hash=h)
            for ref, h in [(1, 'a'), (1, None), (0, 'b'), (0, 'c'), (0, None), (0, None)]],
            get_num_free_blocks=lambda: 4)
        row = EngineAdapter.physical(a)
        self.assertEqual(row['total_usable_blocks'], 6)
        self.assertEqual((row['active_blocks'], row['allocator_free_blocks'], row['reclaimable_cached_blocks']), (2, 4, 2))
        self.assertEqual(row['available_for_new_reservations'], 4)

    def test_receipt_ledger_does_not_forget_duplicates(self):
        ledger = Receipts(limit=1); ledger.begin({'action_id': 'x'})
        self.assertFalse(ledger.begin({'action_id': 'x'})[1])
        with self.assertRaisesRegex(ValueError, 'capacity'):
            ledger.begin({'action_id': 'y'})


class NativeTiming(unittest.TestCase):
    def test_host_features_survive_json_evidence_round_trip(self):
        batch = {'batch_id': 'A', 'requests': [{'prefill_tokens': 128, 'decode_tokens': 0,
            'context_tokens': 0, 'logical_prefill_tokens': 256, 'cache_input_skipped_tokens': 128}],
            'host_calls': [{'call_id': 'host-A', 'scheduled_batch_ids': ['A'], 'completed_batch_ids': ['A'], 'engine_host_ms': 3.}],
            'engine_host_ms': 3., 'host_batch_span_ms': 3., 'host_step_equivalent': True,
            'device': {'batch_id': 'A', 'device_event_complete': True, 'npu_forward_ms': 1.}}
        sample = batch_features(batch)
        self.assertEqual(sample, json.loads(json.dumps(sample)))
        self.assertEqual(sample['y'], 3.)  # not the 1 ms auxiliary forward

    def test_host_step_associates_async_submission_and_completion_separately(self):
        a = fake_engine()
        def step_with_batch_queue():
            a.host_call['scheduled_batch_ids'].append('B')
            a.host_call['completed_batch_ids'].append('A')
            a.batches.append({'batch_id': 'A', 'host_begin_ns': time.monotonic_ns() - 10_000_000})
            a.pending['B'] = {'batch_id': 'B'}
            a.batch_index.update(A=a.batches[-1], B=a.pending['B'])
        a.host_step(step_with_batch_queue)
        self.assertFalse(a.batches[0]['host_step_equivalent'])
        self.assertGreater(a.batches[0]['host_batch_span_ms'], a.batches[0]['engine_host_ms'])
        self.assertNotIn('engine_host_ms', a.pending['B'])
        self.assertEqual(a.pending['B']['host_calls'][0]['completed_batch_ids'], ['A'])

    def test_no_elapsed_read_before_device_complete_and_out_of_order_identity(self):
        class Event:
            def __init__(self, ready): self.complete = ready
            def query(self): return self.complete
            def elapsed_time(self, end):
                if not end.complete: raise AssertionError('used dispatch as completion')
                return 2.5
        m = DeviceMeasurements(); first = Event(False)
        m.add('a', Event(True), first, .1, {'step': 1})
        m.add('b', Event(True), Event(True), .1, {'step': 2})
        self.assertEqual([r['batch_id'] for r in m.ready()], ['b'])
        first.complete = True
        self.assertEqual([r['batch_id'] for r in m.ready()], ['b', 'a'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            m.add('a', Event(True), Event(True), .1, {})

    def test_batch_mismatch_cannot_calibrate(self):
        b = {'batch_id': 'one', 'requests': [], 'device': {'device_event_complete': True, 'batch_id': 'two'}}
        with self.assertRaisesRegex(ValueError, 'misassociated'):
            batch_features(b)

    def test_missing_phase_coverage_does_not_freeze(self):
        result = fit_model([], [{'resources': resources(), 'identity': {}}])
        self.assertFalse(result['pass']); self.assertFalse(result['frozen'])


class NativeParameterization(unittest.TestCase):
    def test_resume_cannot_replace_frozen_calibration_or_acceptance(self):
        from dabs_ascend.r234_run import preserve_native_freeze
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            gate = {'artifacts': {'raw': 'original-hash'}, 'paths': {'native-calibrate': 'fixed'}}
            fitted = {'identities': [{'epoch': 'one'}], 'resources': [resources()], 'models': model()}
            preserve_native_freeze(out, gate, fitted, False)
            write_json(out / 'ascend_profile.json', fitted)
            preserve_native_freeze(out, gate, fitted, True)
            preserve_native_freeze(out, {**gate, 'artifacts': {'raw': 'new-hash'}, 'paths': {}}, fitted, True)
            with self.assertRaisesRegex(ValueError, 'changed since pause'):
                preserve_native_freeze(out, gate, {**fitted, 'models': {}}, True)

    def test_rollback_preflights_paths_and_recovers_partially_restored_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); backup = root / 'dabs_native_backup_test'; backup.mkdir()
            target = root / 'dabs_native'; target.mkdir()
            paths, originals, patched = {}, {}, {}
            for i, name in enumerate(PATCHED):
                path = root / ('vendor' + str(i) + '.py'); path.write_bytes(b'new' if i else b'old')
                copy = backup / name; copy.parent.mkdir(parents=True, exist_ok=True); copy.write_bytes(b'old')
                paths[name] = str(path); originals[name] = sha(b'old'); patched[name] = sha(b'new')
            receipt = dict(backup=str(backup), adapter_path=str(target), paths=paths, original_sha256=originals, patched_sha256=patched)
            write_json(backup / 'installation.json', receipt)
            with patch('dabs_native.install.installed_path', side_effect=lambda n: Path(paths[n])):
                write_json(backup / 'installation.json', {**receipt, 'adapter_path': str(root / 'wrong')})
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    rollback(backup)
                self.assertEqual(Path(paths[PATCHED[1]]).read_bytes(), b'new')
                write_json(backup / 'installation.json', receipt)
                self.assertTrue(rollback(backup)['rolled_back'])
                self.assertTrue((backup / 'retired_adapter').is_dir())
                self.assertTrue(all(Path(p).read_bytes() == b'old' for p in paths.values()))

    def test_out_of_support_model_produces_explicit_nonrankable_score(self):
        cfg, s = state(); m = model()
        m['validation'] = {str(r): {p: {'range': {k: [0, 0] for k in ('p', 'd', 'dc', 'pc')}}
            for p in ('prefill_only', 'decode_only', 'mixed')} for r in range(4)}
        s['cache_evidence'] = [{'prefix_hash': k, 'replica': r, 'tokens': 256 if r == 0 else 0,
            'known': True, 'state': 'active_computed' if r == 0 else 'absent'} for k in s['nodes'] for r in range(2)]
        s['pressure'] = [8, 0]
        row = choose(s, m, cfg, time.monotonic() + 2)
        self.assertFalse(row['model_selection'])
        self.assertTrue(any('out_of_range' in (x.get('reason') or '') for x in row['evaluations']) or row.get('native_fallback'))

    def test_ayo_retains_conservative_reserved_load_not_physical_occupancy(self):
        cfg, s = state(); s['reserved_blocks'] = [100, 0]; s['physical_active_blocks'] = [0, 500]
        row = baseline(s, cfg, 'Ayo')
        self.assertEqual(row['placement']['a'], 1)
        self.assertEqual(row['ayo_load_metric'], 'normalized_conservative_reserved_KV_blocks')

    def test_all_four_methods_use_1024_128_and_distinct_client_server_limits(self):
        cfg, s = state()
        self.assertEqual((cfg['base_cap'], cfg['block_size'], cfg['max_num_seqs'], cfg['client_max_inflight']), (1024, 128, 10, 2))
        for method in r234.METHODS:
            a = baseline(s, cfg, method)
            self.assertEqual(a['caps'], [1024, 1024])
            score = adapted('engine').evaluate(s, a, model(), cfg)
            self.assertTrue(score['rankable'], score)
        result = choose(s, model(), cfg, time.monotonic() + 1)
        self.assertEqual(result['action']['caps'], [1024, 1024])

    def test_known_cache_pressure_fixture_scores_reference_and_legal_route_candidate(self):
        cfg, s = state(); s['pressure'] = [8, 0]
        s['cache_evidence'] = [{'prefix_hash': k, 'replica': r, 'tokens': 256 if r == 0 else 0,
            'known': True, 'state': 'active_computed' if r == 0 else 'absent'} for k in s['nodes'] for r in range(2)]
        result = choose(s, model(), cfg, time.monotonic() + 2)
        self.assertTrue(any(e['cell'] == '00' and e['complete'] for e in result['evaluations']))
        self.assertTrue(any(e['cell'] != '00' and e['complete'] for e in result['evaluations']))

    def test_cache_skip_precedes_budget_and_mixed_chunk_is_not_block_quantized(self):
        from dabs_ascend.native_scoring import select_batch
        cfg, s = state()
        ns = copy.deepcopy(s['nodes']); ns['a'].update(state='submitted', replica=0, phase='decode', computed=3584)
        ns['b'].update(state='submitted', replica=0)
        ledger = SimpleNamespace(length=lambda r, h, at: 2048)
        chosen, dec, context, _, _ = select_batch(ns, ['a', 'b'], ['a'], {}, {'a': 0, 'b': 1}, ledger, 0, 0, cfg)
        self.assertEqual(dec, ['a'])
        self.assertEqual(chosen, [('b', 1023)])
        self.assertEqual(context, 0)  # frozen model uses context before cache skip
        self.assertEqual(ns['b']['computed'], 2048)

    def test_original_4096_16_behavior_remains_available(self):
        cfg = configuration(2)
        self.assertEqual((cfg['base_cap'], cfg['block_size']), (4096, 16))
        r234.verify_import()
        for name in transformation_manifest(): adapted(name)

    def test_unknown_state_is_explicit_fallback_not_zero_observation(self):
        cfg, s = state(); s['native_observation_errors'] = ['snapshot_expired_in_transit']
        result = choose(s, model(), cfg, time.monotonic() + 1)
        self.assertFalse(result['model_selection'])
        self.assertEqual(result['native_fallback'], s['native_observation_errors'])

    def test_two_real_pending_batches_supported_in_scoring(self):
        cfg, s = state()
        s['nodes']['a'].update(state='submitted', replica=0, start=.9, computed=0)
        s['ready'] = ['b']
        s['inflight_batches'] = [{'replica': 0, 'prefill': [('a', 1024)], 'decode': [], 'remaining_ms': 2},
                                {'replica': 0, 'prefill': [('a', 1024)], 'decode': [], 'remaining_ms': 4}]
        score = adapted('engine').evaluate(s, baseline(s, cfg, 'RR'), model(), cfg)
        self.assertTrue(score['rankable'], score)

    def test_hand_written_pass_cannot_open_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / 'acceptance.json', {'pass': True, 'live_verified': True})
            self.assertFalse(evaluate_local({'native-probe-one': directory})['pass'])

    def test_patch_is_additive_and_refuses_ambiguous_api_return(self):
        original = b'def build_app():\n    app = object()\n    return app\n'
        patched = patch_source('vllm/entrypoints/openai/api_server.py', original)
        self.assertIn(b'install_api(app)', patched)
        self.assertIn(b'return app', patched)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            patch_source('vllm/entrypoints/openai/api_server.py', b'def build_app():\n    return 1\n    return 2\n')

    def test_missing_actual_hook_fails_even_with_matching_version(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'different.py'; file.write_text('# different source')
            with patch('dabs_native.install.metadata.version', side_effect=lambda p: '0.23.0+empty' if p == 'vllm' else '0.23.0rc1'), \
                 patch('dabs_native.install.installed_path', return_value=file):
                result = check()
                self.assertFalse(result['compatible'])
                self.assertIn('EngineCore', str(result['errors']))


if __name__ == '__main__':
    unittest.main()
