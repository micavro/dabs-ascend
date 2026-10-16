"""Raw-event audit and nonoverlapping asynchronous host completion accounting.

This is a NEW throughput proxy, NOT the original same-batch host-call latency.
For FIFO completed batch i, charge [max(submit_call.begin, previous completion
call.end), completion_call.end]. A pipelined B never inherits A's host-call
duration. Startup is included; idle time before submission is excluded. Charges
do not overlap. Device events validate association only, never supply the target.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

TARGET = 'async_host_completion_accounting_v1'
FIELDS = ('host_call_serial', 'pending_before', 'pending_after', 'completion_predecessor')


def require_timing(capability):
    missing = sorted(set(FIELDS) - set(capability.get('timing_fields', [])))
    if missing:
        raise ValueError('missing native timing fields: ' + ', '.join(missing) +
                         '; update the selected adapter with run_native.sh install --replica N')


def read_events(path):
    with Path(path).open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'invalid raw JSON at line {number}') from exc


def index_events(events):
    """Deduplicate repeated polls and host calls attached to both A and B."""
    batches, calls, epochs = {}, {}, {}
    for event in events:
        r, response = event['replica'], event['response']
        identity = response.get('identity', {})
        epoch = tuple(identity.get(k) for k in ('engine_instance', 'boot_id', 'clock_namespace_inode'))
        if r in epochs and epochs[r] != epoch:
            raise ValueError(f'replica {r}: engine epoch changed inside raw trace')
        epochs[r] = epoch
        if event['op'] != 'batches':
            continue
        for b in response.get('batches', []):
            key = r, b['batch_id']
            previous = batches.get(key)
            if previous:
                # A later poll may add device timing, but cannot rewrite host history.
                for field in ('requests', 'step', 'host_begin_ns', 'scheduled_ns'):
                    if previous.get(field) != b.get(field):
                        raise ValueError('conflicting duplicate batch: ' + str(key) + '.' + field)
            if not previous or b.get('device') or not previous.get('device'):
                batches[key] = b
            for call in b.get('host_calls', []):
                ck = r, call['call_id']
                if ck in calls and calls[ck] != call:
                    raise ValueError('conflicting duplicate host call: ' + str(ck))
                calls[ck] = call
    return batches, calls


def combination(call):
    s, c = call.get('scheduled_batch_ids', []), call.get('completed_batch_ids', [])
    if len(s) > 1 or len(c) > 1:
        return 'multiple'
    if s and c:
        return 'same_batch' if s == c else 'submit_B_complete_A'
    return 'submit_only' if s else 'complete_only' if c else 'neither'


def audit(events):
    batches, calls = index_events(events)
    result = {}
    for replica in sorted({r for r, _ in batches}):
        selected = [c for (r, _), c in calls.items() if r == replica]
        counts = Counter(combination(c) for c in selected)
        examples = {}
        for c in selected:
            rows = examples.setdefault(combination(c), [])
            if len(rows) < 3:
                rows.append({k: c.get(k) for k in ('call_id', 'begin_ns', 'end_ns',
                    'scheduled_batch_ids', 'completed_batch_ids')})
        bs = [b for (r, _), b in batches.items() if r == replica and b.get('requests')]
        same = sum(any(c.get('scheduled_batch_ids') == [b['batch_id']] == c.get('completed_batch_ids')
                       for c in b.get('host_calls', [])) for b in bs)
        result[str(replica)] = {'unique_batches': len(bs), 'unique_host_calls': len(selected),
            'legacy_same_batch_eligible': same, 'legacy_excluded': len(bs) - same,
            'combinations': dict(counts), 'examples': examples}
    return {'replicas': result, 'raw_event_audit': True,
            'note': 'Unique call IDs, not repeated copies on batches. No PASS or model inferred.'}


def workload_features(batch):
    from core18v2.service import features, phase
    rows = batch['requests']
    p = sum(r['prefill_tokens'] for r in rows)
    d = sum(r['decode_tokens'] for r in rows)
    dc = sum(r['context_tokens'] for r in rows if r['decode_tokens'])
    pc = max((r['context_tokens'] for r in rows if r['prefill_tokens']), default=0)
    return {'phase': phase(p, d), 'p': p, 'd': d, 'dc': dc, 'pc': pc,
            'x': list(features(p, d, dc, pc)),
            'logical_prefill_tokens': sum(r['logical_prefill_tokens'] for r in rows),
            'cache_input_skipped_tokens': sum(r['cache_input_skipped_tokens'] for r in rows)}


def collection_floors(events, record):
    """Bind the start watermark to a real drained snapshot, not a file counter."""
    floors = {}
    starts = record.get('collection_start', {})
    if not starts:
        raise ValueError('missing calibration collection-start snapshots')
    for replica, ref in starts.items():
        matching = [e['response'] for e in events if e['op'] == 'snapshot' and e['replica'] == int(replica)
                    and e['response'].get('snapshot_id') == ref['snapshot_id']]
        if not matching or any(s.get('step') != ref['step'] or s.get('running_count') != 0 or
                s.get('waiting_count') != 0 or s.get('inflight_batches') != [] for s in matching):
            raise ValueError('calibration start is not a recorded drained snapshot')
        floors[int(replica)] = ref['step']
    return floors


def reconstruct(events, plan, *, require_new_fields=True, after_steps=None):
    """Rebuild ALL declared calibration batches. Never select just passing rows."""
    batches, calls = index_events(events)
    submissions, completions = defaultdict(list), defaultdict(list)
    predecessor = {}
    for replica in sorted({r for r, _ in calls}):
        previous, previous_end = None, -1
        for call in sorted((c for (r, _), c in calls.items() if r == replica), key=lambda c: c['begin_ns']):
            begin, end = call['begin_ns'], call['end_ns']
            if (type(begin) is not int or type(end) is not int or end <= begin or begin < previous_end
                    or 'error' in call or not math.isclose(call['engine_host_ms'], (end - begin) / 1e6, abs_tol=1e-6)):
                raise ValueError('invalid/overlapping host call: ' + call['call_id'])
            previous_end = end
            for bid in call['scheduled_batch_ids']:
                submissions[replica, bid].append(call)
            for bid in call['completed_batch_ids']:
                completions[replica, bid].append(call)
                predecessor[replica, bid] = previous
            if call['completed_batch_ids']:
                previous = call
    samples, excluded = [], []
    for (replica, bid), batch in batches.items():
        if after_steps is not None and batch.get('step', -1) <= after_steps.get(replica, -1):
            continue
        if not batch.get('requests'):
            continue
        ids = [q.get('mapping', {}).get('node_id', '') if q.get('mapping') else '' for q in batch['requests']]
        matches = [g for g in plan if all(i.startswith('cal-' + g['split'] + '-' + str(g['seed']) + '-') for i in ids)]
        if not matches:
            if any(i.startswith('cal-') for i in ids):
                raise ValueError('mixed/undeclared calibration group in batch ' + bid)
            continue  # Warmups and prior probe traces are not calibration samples.
        if len(matches) != 1:
            raise ValueError('ambiguous calibration plan identity')
        group = matches[0]
        reason = None
        try:
            ss, cc = submissions[replica, bid], completions[replica, bid]
            if len(ss) != 1 or len(cc) != 1 or cc[0]['completed_batch_ids'] != [bid]:
                raise ValueError('missing/nonunique submit or completion boundary')
            submit, complete = ss[0], cc[0]
            if submit['call_id'] not in {c['call_id'] for c in batch.get('host_calls', [])}:
                raise ValueError('batch missing its submit call')
            if require_new_fields and any(k not in complete for k in FIELDS[:3]):
                raise ValueError('missing pending/serial fields; update adapter')
            prev = predecessor[replica, bid]
            start = submit['begin_ns']
            previous_id = None
            if prev and prev['end_ns'] > start:
                if len(prev['completed_batch_ids']) != 1:
                    raise ValueError('multiple predecessor completions')
                previous_id = prev['completed_batch_ids'][0]
                prior = batches.get((replica, previous_id))
                if not prior or not all(q.get('mapping') and q['mapping']['node_id'].startswith(
                        'cal-' + group['split'] + '-' + str(group['seed']) + '-') for q in prior['requests']):
                    raise ValueError('overlapping train/holdout or foreign group')
                if require_new_fields and bid not in prev['pending_after']:
                    raise ValueError('batch not pending at previous completion boundary')
                start = prev['end_ns']
            if require_new_fields:
                recorded = batch.get('completion_predecessor')
                expected = ({'call_id': prev['call_id'], 'end_ns': prev['end_ns'],
                             'batch_ids': prev['completed_batch_ids']} if prev else None)
                # A predecessor before this group's submission may have been acked.
                if prev and recorded != expected:
                    raise ValueError('completion predecessor differs from raw call order')
                if not prev and recorded and recorded['end_ns'] > submit['begin_ns']:
                    raise ValueError('missing overlapping predecessor in raw trace')
                if bid not in complete['pending_before'] and bid not in complete['scheduled_batch_ids']:
                    raise ValueError('completed batch was neither pending nor submitted')
            end = complete['end_ns']
            if not submit['begin_ns'] <= start < end or submit['begin_ns'] > complete['begin_ns']:
                raise ValueError('invalid batch completion order')
            device = batch.get('device')
            if not device or device['batch_id'] != bid or device.get('device_event_complete') is not True:
                raise ValueError('missing/misassociated NPU completion evidence')
            samples.append({**workload_features(batch), 'replica': replica, 'batch_id': bid,
                'group': group, 'split': group['split'], 'timing_target': TARGET,
                'host_step_equivalent': False, 'accounting_valid': True,
                'submit_call_id': submit['call_id'], 'host_call_id': complete['call_id'],
                'previous_batch_id': previous_id, 'interval_begin_ns': start, 'interval_end_ns': end,
                'regime': 'pipeline' if previous_id else 'startup', 'y': (end - start) / 1e6})
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc)
        if reason:
            excluded.append({'replica': replica, 'batch_id': bid, 'split': group['split'],
                             'phase': workload_features(batch)['phase'], 'reason': reason})
    # Completion accounting must partition elapsed time, never double-charge it.
    for replica in sorted({s['replica'] for s in samples}):
        end = -1
        for sample in sorted((s for s in samples if s['replica'] == replica), key=lambda s: s['interval_begin_ns']):
            if sample['interval_begin_ns'] < end:
                raise ValueError('overlapping charged intervals')
            end = sample['interval_end_ns']
    return samples, excluded
