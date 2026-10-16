"""Synthetic CPU records exercise acceptance logic; never site evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from dabs_ascend.common import write_json, source_paths
from dabs_ascend.native_calibration import batch_features, fit_model
from dabs_ascend.native_gate import evaluate_local, evaluate_probe
from test_native_simplification import capability

source_paths()


def probe_fixture(out, replicas):
    out.mkdir()
    roles, events = {}, []
    def emit(op, row, replica=0):
        events.append({'op': op, 'replica': replica, 'response': {**row, 'identity': capability()['identity']}})
    def snapshot(role, cache=None, replica=0, **kw):
        row = {'snapshot_id': role, 'cache': cache or [], **kw}
        emit('snapshot', row, replica); roles[role] = {'replica': replica, 'snapshot_id': role}
    def cache(state, tokens, salt='a'):
        return [{'state': state, 'reusable_tokens': tokens, 'salt_digest': salt}]
    snapshot('cold', cache('miss', 0)); snapshot('warm', cache('hit', 1024))
    snapshot('namespace', cache('miss', 0, 'b'))
    snapshot('boundary', [{'query_id': str(n), 'reusable_tokens': hit} for n, hit in
                          [(127, 0), (128, 0), (129, 128), (256, 128), (257, 256)]])
    for role, status in [('expired', 'expired'), ('duplicate', 'applied'), ('late', 'expired')]:
        roles[role] = {'replica': 0, 'action_id': role}
        for _ in range(2 if role == 'duplicate' else 1):
            emit('apply', {'action_id': role, 'status': status})
    snapshot('waiting', waiting_count=2, waiting=[{'engine_request_id': i} for i in ('F', 'E')])
    roles['queue'] = {'replica': 0, 'action_id': 'queue'}
    emit('receipt', {'action_id': 'queue', 'status': 'applied', 'applied': ['E', 'F'],
                     'effective_waiting_order': ['E', 'F'], 'applied_step': 51, 'scheduled_step': 54,
                     'scheduled_order': ['E', 'F'], 'requested': {'order': ['E', 'F']},
                     'not_before_ns': 100, 'action_expires_ns': 300})
    batches = [{'batch_id': 'inside', 'scheduled_tokens': {'E': 128, 'F': 128},
                'device': {'batch_id': 'inside', 'device_event_complete': True},
                'queue_application': {'action_id': 'queue', 'phase': 'current_action', 'at_ns': 200}},
               {'batch_id': 'after', 'scheduled_tokens': {},
                'queue_application': {'action_id': 'queue', 'phase': 'common_continuation', 'at_ns': 310,
                                      'queues': {'waiting': {'reason': 'fifo_acceptance'}}}}]
    emit('batches', {'batches': batches})
    snapshot('drain', running_count=0, waiting_count=0, inflight_batches=[])
    def node(name, start, finish):
        return {'start': start, 'finish': finish, 'mapping': {'engine_request_id': name, 'action_id': name,
                'engine_admitted_ns': start, 'engine_finished_ns': finish}}
    dag = {'root': node('root', 0, 10), 'branches': [node('left', 10, 20), node('right', 10, 20)],
           'join': node('join', 20, 30)}
    emit('mapping', {'mappings': [n['mapping'] for n in [dag['root'], *dag['branches'], dag['join']]]})
    if replicas == 4:
        for r in (1, 2, 3):
            snapshot('cross-' + str(r), cache('miss', 0), replica=r)
            snapshot('cross-hot-' + str(r), cache('hit', 1024), replica=r)
    write_json(out / 'probe.json', {'roles': roles, 'dag': dag, 'replicas': replicas, 'source_sha256': 'old'} )
    (out / 'native_events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    return events


def calibration_fixture(out):
    out.mkdir()
    caps = [capability() for _ in range(4)]
    plan = [{'split': split, 'seed': i} for i, split in enumerate(('train', 'holdout'))]
    samples, events = [], []
    for replica in range(4):
        for group in plan:
            for phase in ('prefill_only', 'decode_only', 'mixed'):
                for i in range(20):
                    bid = f'{replica}-{group["split"]}-{phase}-{i}'
                    mapping = {'node_id': 'cal-' + group['split'] + '-' + str(group['seed']) + '-cpu'}
                    pre = dict(prefill_tokens=1024, decode_tokens=0, context_tokens=3072,
                               logical_prefill_tokens=1024, cache_input_skipped_tokens=0, mapping=mapping)
                    dec = dict(prefill_tokens=0, decode_tokens=2, context_tokens=7168,
                               logical_prefill_tokens=0, cache_input_skipped_tokens=0, mapping=mapping)
                    if phase == 'mixed':
                        pre.update(prefill_tokens=1022, logical_prefill_tokens=1022)
                    rows = [pre] if phase == 'prefill_only' else [dec] if phase == 'decode_only' else [pre, dec]
                    batch = {'batch_id': bid, 'requests': rows, 'host_step_equivalent': True, 'engine_host_ms': 10.,
                             'host_batch_span_ms': 10., 'host_calls': [{'call_id': bid, 'engine_host_ms': 10.,
                                 'scheduled_batch_ids': [bid], 'completed_batch_ids': [bid]}],
                             'device': {'batch_id': bid, 'device_event_complete': True, 'npu_forward_ms': 5.}}
                    samples.append({**batch_features(batch), 'replica': replica, 'split': group['split'], 'group': group})
                    events.append({'op': 'batches', 'replica': replica,
                                   'response': {'identity': caps[replica]['identity'], 'batches': [batch]}})
    model = fit_model(samples, caps)
    assert model['pass'], model['failures']
    model.update(source_sha256='different-controller', raw_samples_sha256='different-whitespace', raw_native_sha256='different')
    write_json(out / 'native_profile.json', model); write_json(out / 'capabilities.json', caps)
    write_json(out / 'calibration_samples.json', samples); write_json(out / 'calibration_plan.json', {'plan': plan})
    (out / 'native_events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))


class EvidenceTests(unittest.TestCase):
    def test_unsigned_raw_records_recompute_without_source_hash_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe_fixture(root / 'one', 1); probe_fixture(root / 'four', 4)
            calibration_fixture(root / 'cal')
            paths = {'native-probe-one': str(root / 'one'), 'native-probe-four': str(root / 'four'),
                     'native-calibrate': str(root / 'cal')}
            result = evaluate_local(paths)
            self.assertTrue(result['pass'], result['missing'])
            self.assertEqual(result['not_run'], ['native-invalidate', 'native-restart-check'])
            # PASS files and source hashes cannot override changed mechanism evidence.
            write_json(root / 'one/result.json', {'pass': True})
            rows = [json.loads(line) for line in (root / 'one/native_events.jsonl').read_text().splitlines()]
            batch = next(e['response']['batches'] for e in rows if e['op'] == 'batches')
            batch[1]['scheduled_tokens'] = batch[0]['scheduled_tokens']; batch[0]['scheduled_tokens'] = {}
            (root / 'one/native_events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            result = evaluate_local(paths)
            self.assertFalse(result['pass'])
            self.assertIn('native-probe-one: priority_reached_execution_within_window', result['missing'])
            self.assertIn('native-probe-one: executed_priority_order', result['missing'])

    def test_window_outside_execution_and_wrong_priority_remain_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'probe'; rows = probe_fixture(out, 1)
            self.assertTrue(all(evaluate_probe(out, 1).values()))
            batches = next(e['response']['batches'] for e in rows if e['op'] == 'batches')
            batches[0]['scheduled_tokens'] = {'F': 128, 'E': 128}
            (out / 'native_events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertFalse(evaluate_probe(out, 1)['executed_priority_order'])
