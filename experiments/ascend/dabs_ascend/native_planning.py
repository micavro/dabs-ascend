"""Controller-local planning deadlines and a lossless scorer transport view.

Parent and spawned worker run on the same host/monotonic clock. The deadline
includes IPC; an old plan never gets a fresh duration in the worker.
"""
import time

CONTRACT = 'bounded_native_observation_prediction_v3'


def planning_window(state, cfg, tick, now=None):
    now = time.monotonic() if now is None else now
    expiry = min([state['now'] + cfg['snapshot_max_age_s']] +
                 [s['expires_ns'] / 1e9 for s in state.get('native_snapshots', []) if 'expires_ns' in s])
    reserve = cfg['commit_reserve_s']
    deadline = min(now + cfg['planning_soft_limit_s'], tick - reserve, expiry - reserve)
    limits = {'soft_limit': now + cfg['planning_soft_limit_s'],
              'fixed_tick': tick - reserve, 'snapshot_expiry': expiry - reserve}
    return {'contract': CONTRACT, 'prepared_at': now, 'deadline': deadline,
            'budget_ms': max(0., (deadline - now) * 1000),
            'fixed_action_deadline': tick, 'snapshot_expires_at': expiry,
            'commit_reserve_ms': reserve * 1000,
            'limit_remaining_ms': {k: (v - now) * 1000 for k, v in limits.items()},
            'binding_limits': [k for k, v in limits.items() if abs(v - deadline) < 1e-9]}


def scorer_view(state):
    # Raw transport receipts/queries stay in the parent's native_events.jsonl. Keep
    # EVERY workflow node, observation error, resource, queue and progress value.
    view = {k: v for k, v in state.items() if k != 'native_snapshots'}
    view['nodes'] = {k: {f: v for f, v in n.items() if f != 'native_request'}
                     for k, n in state['nodes'].items()}
    view['cache_evidence'] = [{k: v for k, v in e.items() if k != 'native'}
                              for e in state.get('cache_evidence', [])]
    return view


def run_planner(planner, state, model, cfg, window):
    started = time.monotonic()
    result = planner(state, model, cfg, window['deadline'])
    result['planning_finished_at'] = time.monotonic()
    result['planning_dispatch_delay_ms'] = (started - window['prepared_at']) * 1000
    result['planning_worker_budget_ms'] = (window['deadline'] - started) * 1000
    result['planning_deadline'] = window['deadline']
    result['worker_clock_contract'] = 'same-host monotonic absolute deadline; IPC consumes budget'
    return result


def decision_record(decision):
    """Raw snapshots already occur once in native_events; retain joinable IDs."""
    row = dict(decision)
    for field in ('snapshot', 'fallback_snapshot'):
        if field not in row:
            continue
        original = row[field]
        row[field] = scorer_view(original)
        row[field]['native_snapshot_refs'] = [
            {'replica': r, 'snapshot_id': s.get('snapshot_id'), 'identity': s.get('identity'),
             'captured_ns': s.get('captured_ns'), 'expires_ns': s.get('expires_ns')}
            for r, s in enumerate(original.get('native_snapshots', []))]
    return row


def aligned_start(origin, desired, now, epoch, tolerance):
    """Skip missed fixed windows; never grant an old action a later deadline."""
    import math
    if now <= desired + tolerance:
        return desired
    return origin + math.ceil((now - origin) / epoch) * epoch
