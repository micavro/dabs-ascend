"""Wire contracts shared by the real engine and CPU fault-injection tests."""
import hashlib
import json
import math
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def prefix_key(tokens, salt):
    return digest({'tokens': tokens, 'salt': salt})


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError('invalid ' + name)
    return value


def validate_resources(row):
    for name in ('block_size', 'step_tokens', 'server_max_seqs', 'context_tokens', 'physical_blocks'):
        positive(row[name], name)
    if row['step_tokens'] > row['context_tokens'] * row['server_max_seqs']:
        raise ValueError('invalid engine token envelope')
    return row


def fresh(snapshot, identity, now_ns=None):
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    return (snapshot.get('identity') == identity and
            type(snapshot.get('captured_ns')) is int and
            snapshot['captured_ns'] <= now_ns <= snapshot.get('expires_ns', -1))


def check_action(action, snapshot, identity, now_ns, current_step):
    from .timing_policy import ACTION_WINDOW_NS
    if not fresh(snapshot, identity, now_ns):
        return 'expired_snapshot'
    if action.get('snapshot_id') != snapshot['snapshot_id']:
        return 'snapshot_mismatch'
    if action.get('identity') != identity:
        return 'engine_restarted'
    if type(action.get('expires_ns')) is not int or now_ns > action['expires_ns']:
        return 'expired_action'
    if action['expires_ns'] > snapshot['expires_ns']:
        return 'action_outlives_snapshot'
    begin, end = action.get('not_before_ns'), action.get('action_expires_ns')
    if (type(begin) is not int or type(end) is not int or
            not snapshot['captured_ns'] <= begin < end <= snapshot['captured_ns'] + ACTION_WINDOW_NS):
        return 'invalid_fixed_action_window'
    if now_ns >= end:
        return 'expired_action_window'
    if now_ns < begin:
        return 'action_not_started'
    if current_step < snapshot['step']:
        return 'scheduler_step_regressed'
    if action.get('resource_digest') != digest(snapshot['resources']):
        return 'resources_changed'
    ids = action.get('order', [])
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        return 'duplicate_or_invalid_priority'
    if any(not isinstance(i, str) for i in ids):
        return 'invalid_request_id'
    return None


def prefill_order(items, priority, accepted_ns, now_ns, mutable):
    """Finite-window ranks, 5 s age protection, then original admission FIFO."""
    positions = [i for i, item in enumerate(items) if mutable(item)]
    selected = [items[i] for i in positions]
    rank = {rid: i for i, rid in enumerate(priority)}
    if priority and any(now_ns - accepted_ns[x.request_id] >= 5_000_000_000 for x in selected):
        return list(items), 'age_fifo'
    if priority:
        selected.sort(key=lambda x: rank.get(x.request_id, len(rank)))
        reason = 'rank'
    else:
        selected.sort(key=lambda x: (accepted_ns[x.request_id], x.request_id))
        reason = 'fifo_acceptance'
    out = list(items)
    for i, item in zip(positions, selected):
        out[i] = item
    return out, reason


class Receipts:
    """Idempotence by action ID and exact content; never evict IDs within a boot."""
    def __init__(self, limit=100000):
        self.rows, self.limit = {}, limit

    def begin(self, action):
        key = action.get('action_id')
        if not isinstance(key, str) or not 1 <= len(key) <= 160:
            raise ValueError('invalid action_id')
        fingerprint = digest(action)
        if key in self.rows:
            row = self.rows[key]
            if row['request_digest'] != fingerprint:
                raise ValueError('action_id_reused_with_different_content')
            return row, False
        if len(self.rows) >= self.limit:
            raise ValueError('action_id_capacity_exhausted; start a new controlled experiment window')
        row = {'action_id': key, 'snapshot_id': action.get('snapshot_id'),
               'request_digest': fingerprint, 'requested': action, 'status': 'requested',
               'transitions': [{'status': 'requested', 'at_ns': time.monotonic_ns()}],
               'accepted': [], 'applied': [], 'rejected': {}, 'applied_step': None,
               'effective_waiting_order': None, 'scheduled_order': None}
        self.rows[key] = row
        return row, True


class DeviceMeasurements:
    """Associate event pairs with explicit batch IDs, never with RPC return order."""
    def __init__(self):
        self.pending, self.rows = {}, []

    def add(self, batch_id, start, end, host_dispatch_ms, metadata):
        if batch_id in self.pending or any(r['batch_id'] == batch_id for r in self.rows):
            raise ValueError('duplicate batch timing')
        self.pending[batch_id] = (start, end, host_dispatch_ms, metadata)

    def ready(self):
        for key, (start, end, host_ms, meta) in list(self.pending.items()):
            if not end.query():
                continue
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('invalid device event measurement')
            self.rows.append({'batch_id': key, 'npu_forward_ms': elapsed,
                              'worker_host_dispatch_ms': host_ms, 'batch': meta,
                              'device_event_complete': True, 'timing': 'torch.npu.Event; forward stream interval',
                              'includes_sampling': False, 'completion_observed_ns': time.monotonic_ns()})
            del self.pending[key]
        return list(self.rows)


def require_members(obj, label, *, methods=(), fields=()):
    """Name the actual missing interface, independent of package/class versions."""
    missing = [label + '.' + name + '()' for name in methods if not callable(getattr(obj, name, None))]
    missing += [label + '.' + name for name in fields if not hasattr(obj, name)]
    if missing:
        raise ValueError('missing native capability: ' + ', '.join(missing))


def engine_epoch(identity):
    return identity.get('engine_instance'), identity.get('boot_id'), identity.get('clock_namespace_inode')


def resource_values(resources):
    # Class names, versions and source hashes are diagnostic metadata only.
    return {k: v for k, v in resources.items() if k not in ('scheduler', 'kv_manager')}


def validate_capabilities(row):
    missing = []
    for key in ('engine_instance', 'boot_id', 'clock_namespace_inode'):
        if row.get('identity', {}).get(key) is None:
            missing.append('identity.' + key)
    for key in ('step_tokens', 'block_size', 'server_max_seqs', 'context_tokens', 'physical_blocks'):
        value = row.get('resources', {}).get(key)
        if type(value) is not int or value <= 0:
            missing.append('resources.' + key + ' (positive integer)')
    if row.get('native_execution') is not True:
        missing.append('native_execution (actual NPU worker)')
    if not row.get('workers'):
        missing.append('workers (execution-process response)')
    for action in ('admission_permits', 'unsubmitted_route', 'safe_prefill_queue_order'):
        if action not in row.get('supported_actions', []):
            missing.append('supported_actions.' + action)
    # Older functional deployments do not advertise an operations list. The
    # client also exercises the read RPCs; do not reject merely a schema label.
    if 'operations' in row:
        for op in ('snapshot', 'apply', 'admit', 'receipt', 'mapping', 'batches', 'ack-trace'):
            if op not in row['operations']:
                missing.append('operation.' + op)
    if not row.get('host_step_boundary'):
        missing.append('host_step_boundary')
    if missing:
        raise ValueError('missing native capability: ' + ', '.join(missing))
    return row


def require_control_semantics(row):
    """Required behavior of the admission/window interfaces, not a code hash."""
    fields = ('admission_only_preserves_queue', 'mutation_time_window_check')
    missing = [name for name in fields if row.get('control_semantics', {}).get(name) is not True]
    if missing:
        raise ValueError('missing native control semantics: ' + ', '.join(missing) +
                         '; run bash run_native.sh install to load the boundary fixes')
    from .timing_policy import require_timing_policy
    require_timing_policy(row)


def validate_observation(row):
    missing = [k for k in ('snapshot_id', 'step', 'captured_ns', 'expires_ns', 'resources', 'physical',
                          'running', 'waiting', 'skipped_waiting', 'cache', 'inflight_batches',
                          'running_count', 'waiting_count') if k not in row]
    for key in ('total_usable_blocks', 'active_blocks', 'reclaimable_cached_blocks',
                'allocator_free_blocks', 'available_for_new_reservations'):
        if type(row.get('physical', {}).get(key)) is not int:
            missing.append('physical.' + key)
    if not row.get('identity', {}).get('engine_instance'):
        missing.append('identity.engine_instance')
    if missing:
        raise ValueError('missing native capability: snapshot.' + ', snapshot.'.join(missing))
    if row['expires_ns'] <= row['captured_ns']:
        raise ValueError('invalid snapshot freshness interval')
    return row
