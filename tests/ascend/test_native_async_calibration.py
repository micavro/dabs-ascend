"""Synthetic CPU async evidence. No fixture is a claimed field measurement."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from dabs_ascend.common import source_paths, write_json
from dabs_ascend.native_calibration import calibration_plan, fit_model
from dabs_ascend.native_timing import TARGET, FIELDS, audit, reconstruct, require_timing, collection_floors
from dabs_ascend.native_gate import evaluate_local
from test_native_evidence import probe_fixture
from test_native_simplification import capability

source_paths()


def pipeline(plan, replicas=1):
    """Each group: submit A; submit B/complete A; submit C/complete B; complete C.

    Host-call times are 1,9,10,10 ms. Completion charges are 10,10,10 ms.
    In particular A's 9 ms call must not be attributed to B's cost.
    """
    events = []
    for r in range(replicas):
        previous = None
        for index, group in enumerate(plan):
            prefix = f'r{r}/g{index}/'
            bids = [prefix + x for x in ('A', 'B', 'C')]
            origin = (index * 100 + 1) * 1_000_000
            calls = []
            for i, (begin, end, scheduled, completed, pending) in enumerate([
                (0, 1, [bids[0]], [], []), (1, 10, [bids[1]], [bids[0]], [bids[0]]),
                (10, 20, [bids[2]], [bids[1]], [bids[1]]), (20, 30, [], [bids[2]], [bids[2]])]):
                calls.append({'call_id': prefix + 'call' + str(i), 'begin_ns': origin + begin * 1_000_000,
                    'end_ns': origin + end * 1_000_000, 'engine_host_ms': float(end - begin),
                    'scheduled_batch_ids': scheduled, 'completed_batch_ids': completed,
                    'host_call_serial': index * 4 + i, 'pending_before': pending,
                    'pending_after': [b for b in pending + scheduled if b not in completed]})
            for i, bid in enumerate(bids):
                mapping = {'node_id': 'cal-' + group['split'] + '-' + str(group['seed']) + '-cpu'}
                pre = {'prefill_tokens': 1024 if i == 0 else 1022, 'decode_tokens': 0,
                       'context_tokens': 3072, 'logical_prefill_tokens': 1024 if i == 0 else 1022,
                       'cache_input_skipped_tokens': 0, 'mapping': mapping}
                dec = {'prefill_tokens': 0, 'decode_tokens': 2, 'context_tokens': 7168,
                       'logical_prefill_tokens': 0, 'cache_input_skipped_tokens': 0, 'mapping': mapping}
                if group.get('coverage_contract'):
                    # Explicit synthetic coverage endpoints. Timings remain 10ms;
                    # no fixture claims these are onsite measurements.
                    tiny = index % 2 == 0
                    pre['prefill_tokens'] = pre['logical_prefill_tokens'] = 1 if tiny else (1024 if i == 0 else 1022)
                    pre['context_tokens'] = 3584 if tiny else 0
                    dec['decode_tokens'] = 1 if tiny else 2
                    dec['context_tokens'] = 3584 if tiny else 7936
                complete = calls[i + 1]
                batch = {'batch_id': bid, 'step': index * 3 + i + 1,
                    'requests': [pre] if i == 0 else [dec] if i == 1 else [pre, dec],
                    'host_calls': copy.deepcopy([calls[i], complete]),
                    'host_begin_ns': calls[i]['begin_ns'], 'scheduled_ns': calls[i]['begin_ns'],
                    'completion_predecessor': copy.deepcopy(previous), 'engine_host_ms': complete['engine_host_ms'],
                    'host_batch_span_ms': (complete['end_ns'] - calls[i]['begin_ns']) / 1e6,
                    'host_step_equivalent': False,
                    'device': {'batch_id': bid, 'device_event_complete': True, 'npu_forward_ms': 777.}}
                events.append({'replica': r, 'op': 'batches',
                    'response': {'identity': capability()['identity'], 'batches': [batch]}})
                previous = {'call_id': complete['call_id'], 'end_ns': complete['end_ns'],
                            'batch_ids': complete['completed_batch_ids']}
    return events


class AsyncCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.plan = [{'split': 'train', 'seed': 100}, {'split': 'holdout', 'seed': 200}]
        self.rows = pipeline(self.plan)

    def test_submit_b_complete_a_keeps_target_distinct_and_deduplicates_calls(self):
        report = audit(self.rows + self.rows)
        r = report['replicas']['0']
        self.assertEqual(r['unique_host_calls'], 8)
        self.assertEqual(r['legacy_same_batch_eligible'], 0)
        self.assertEqual(r['combinations']['submit_B_complete_A'], 4)
        samples, excluded = reconstruct(self.rows + self.rows, self.plan)
        self.assertEqual(excluded, [])
        self.assertEqual(len(samples), 6)
        self.assertEqual([s['y'] for s in samples], [10.] * 6)
        self.assertTrue(all(s['host_step_equivalent'] is False for s in samples))
        self.assertEqual(samples[0]['regime'], 'startup')
        self.assertEqual(samples[1]['previous_batch_id'], samples[0]['batch_id'])
        self.assertEqual(samples[1]['interval_begin_ns'], samples[0]['interval_end_ns'])
        self.assertNotEqual(samples[1]['y'], self.rows[0]['response']['batches'][0]['engine_host_ms'])

    def test_conflicting_duplicate_call_and_restart_are_rejected(self):
        wrong = copy.deepcopy(self.rows)
        wrong[1]['response']['batches'][0]['host_calls'][0]['engine_host_ms'] = 4.
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate host call'):
            reconstruct(wrong, self.plan)
        wrong = copy.deepcopy(self.rows)
        wrong[-1]['response']['identity']['engine_instance'] = 'restarted'
        with self.assertRaisesRegex(ValueError, 'epoch changed'):
            audit(wrong)

    def test_missing_predecessor_pending_state_and_device_do_not_create_samples(self):
        for field in ('completion_predecessor', 'device'):
            wrong = copy.deepcopy(self.rows)
            wrong[1]['response']['batches'][0].pop(field)
            samples, excluded = reconstruct(wrong, self.plan)
            self.assertEqual(len(samples), 5)
            self.assertEqual(len(excluded), 1)
        wrong = copy.deepcopy(self.rows)
        for row in wrong:
            for c in row['response']['batches'][0]['host_calls']:
                if c['call_id'] == 'r0/g0/call1':
                    c['pending_after'] = []
        _, excluded = reconstruct(wrong, self.plan)
        self.assertIn('not pending', str(excluded))

    def test_old_logs_can_be_audited_but_cannot_be_certified_as_new_capture(self):
        for row in self.rows:
            for c in row['response']['batches'][0]['host_calls']:
                for k in FIELDS[:3]:
                    c.pop(k)
        self.assertEqual(audit(self.rows)['replicas']['0']['legacy_excluded'], 6)
        self.assertEqual(len(reconstruct(self.rows, self.plan, require_new_fields=False)[0]), 6)
        self.assertEqual(len(reconstruct(self.rows, self.plan)[1]), 6)
        with self.assertRaisesRegex(ValueError, 'pending_before'):
            require_timing(capability())

    def test_split_overlap_is_not_eligible(self):
        wrong = copy.deepcopy(self.rows)
        wrong[1]['response']['batches'][0]['requests'][0]['mapping']['node_id'] = 'cal-holdout-200-cpu'
        _, excluded = reconstruct(wrong, self.plan)
        self.assertIn('overlapping train/holdout', str(excluded))

    def test_small_diagnostic_does_not_meet_phase_thresholds(self):
        samples, _ = reconstruct(self.rows, self.plan)
        m = fit_model(samples, [capability()])
        self.assertFalse(m['pass'])
        self.assertFalse(m['frozen'])
        self.assertFalse(m['host_step_equivalence'])
        self.assertEqual(m['validation']['0']['prefill_only']['train_batches'], 1)

    def test_start_watermark_excludes_old_samples_and_requires_actual_drained_snapshot(self):
        ref = {'snapshot_id': 'drained-start', 'step': 3}
        record = {'collection_start': {'0': ref}}
        with self.assertRaisesRegex(ValueError, 'drained snapshot'):
            collection_floors(self.rows, record)
        event = {'replica': 0, 'op': 'snapshot', 'response': {
            'identity': capability()['identity'], **ref, 'running_count': 0, 'waiting_count': 0, 'inflight_batches': []}}
        rows = [event, *self.rows]
        floors = collection_floors(rows, record)
        samples, excluded = reconstruct(rows, self.plan, after_steps=floors)
        self.assertFalse(excluded)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(s['split'] == 'holdout' for s in samples))
        event['response']['running_count'] = 1
        with self.assertRaisesRegex(ValueError, 'drained snapshot'):
            collection_floors(rows, record)

    def test_new_gate_recomputes_all_intervals_and_rejects_edited_or_omitted_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe_fixture(root / 'one', 1); probe_fixture(root / 'four', 4)
            out = root / 'cal'; out.mkdir()
            plan = calibration_plan()
            # Two independent episodes per plan group are NOT silently invented:
            # each group here has one prefill/decode/mixed batch. 14 groups per
            # split is insufficient; use repeated steps in the same raw group.
            rows = pipeline(plan, replicas=4)
            # Each phase requires >=20: create additional REAL fixture batches
            # within those groups by a second drained synthetic episode.
            again = copy.deepcopy(rows)
            for e in again:
                b = e['response']['batches'][0]
                b['batch_id'] += '-second'; b['step'] += 100000
                b['device']['batch_id'] = b['batch_id']
                for k in ('host_begin_ns', 'scheduled_ns'):
                    b[k] += 100_000_000_000
                if b['completion_predecessor']:
                    b['completion_predecessor']['call_id'] += '-second'
                    b['completion_predecessor']['batch_ids'] = [x + '-second' for x in b['completion_predecessor']['batch_ids']]
                    b['completion_predecessor']['end_ns'] += 100_000_000_000
                for c in b['host_calls']:
                    c['call_id'] += '-second'
                    for k in ('begin_ns', 'end_ns'):
                        c[k] += 100_000_000_000
                    for k in ('scheduled_batch_ids', 'completed_batch_ids', 'pending_before', 'pending_after'):
                        c[k] = [x + '-second' for x in c[k]]
            # First second-episode predecessor is the last actual completion.
            for r in range(4):
                first = next(e['response']['batches'][0] for e in again if e['replica'] == r)
                last = [e['response']['batches'][0] for e in rows if e['replica'] == r][-1]['host_calls'][-1]
                first['completion_predecessor'] = {'call_id': last['call_id'], 'end_ns': last['end_ns'], 'batch_ids': last['completed_batch_ids']}
            rows += again
            samples, excluded = reconstruct(rows, plan)
            self.assertFalse(excluded, excluded)
            caps = [capability() for _ in range(4)]
            m = fit_model(samples, caps)
            self.assertTrue(m['pass'], m['failures'])
            self.assertEqual(m['timing_target'], TARGET)
            write_json(out / 'native_profile.json', m)
            write_json(out / 'calibration_samples.json', samples)
            write_json(out / 'capabilities.json', caps)
            starts = {str(r): {'snapshot_id': f'start-{r}', 'step': 0} for r in range(4)}
            for r in range(4):
                rows.insert(0, {'replica': r, 'op': 'snapshot', 'response': {
                    'identity': capability()['identity'], **starts[str(r)], 'running_count': 0,
                    'waiting_count': 0, 'inflight_batches': []}})
            write_json(out / 'calibration_plan.json', {'plan': plan, 'diagnostic_only': False, 'collection_start': starts})
            (out / 'native_events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            paths = {'native-probe-one': str(root / 'one'), 'native-probe-four': str(root / 'four'), 'native-calibrate': str(out)}
            result = evaluate_local(paths)
            self.assertTrue(result['pass'], result['missing'])
            m['original_gpu_timing_equivalent'] = True
            write_json(out / 'native_profile.json', m)
            self.assertFalse(evaluate_local(paths)['pass'])
            m['original_gpu_timing_equivalent'] = False
            write_json(out / 'native_profile.json', m)
            write_json(out / 'calibration_samples.json', samples[1:])
            self.assertFalse(evaluate_local(paths)['pass'])
            samples[0]['y'] = 0.01
            write_json(out / 'calibration_samples.json', samples)
            self.assertFalse(evaluate_local(paths)['pass'])

    def test_actual_inflight_two_is_in_plan_and_diagnostic_is_separate(self):
        self.assertIn(2, {g['concurrency'] for g in calibration_plan()})
        self.assertEqual(len(calibration_plan(True)), 4)
        self.assertLess(len(calibration_plan(True)), len(calibration_plan()))

    def test_holdout_error_and_missing_phase_still_fail(self):
        plan = [{'split': split, 'seed': base + i} for split, base in [('train', 0), ('holdout', 100)] for i in range(20)]
        samples, excluded = reconstruct(pipeline(plan), plan)
        self.assertEqual(excluded, [])
        self.assertTrue(fit_model(samples, [capability()])['pass'])
        for s in samples:
            if s['split'] == 'holdout':
                s['y'] *= 10
        failed = fit_model(samples, [capability()])
        self.assertFalse(failed['pass'])
        self.assertTrue(any('error thresholds' in x for x in failed['failures']))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            fit_model(samples + samples[:1], [capability()])
        samples = [s for s in samples if s['phase'] != 'mixed']
        failed = fit_model(samples, [capability()])
        self.assertEqual(failed['validation']['0']['mixed']['holdout_batches'], 0)
        self.assertFalse(failed['frozen'])
