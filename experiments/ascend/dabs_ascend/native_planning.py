"""Controller-local planning deadlines and a lossless scorer transport view.

Parent and spawned worker run on the same host/monotonic clock. The deadline
includes IPC; an old plan never gets a fresh duration in the worker.
"""
import time

CONTRACT = 'post_observation_bounded_planning_v2'


def planning_window(state, cfg, tick, now=None):
    now = time.monotonic() if now is None else now
    expiry = min([state['now'] + cfg['snapshot_max_age_s']] +
                 [s['expires_ns'] / 1e9 for s in state.get('native_snapshots', []) if 'expires_ns' in s])
    reserve = cfg['commit_reserve_s']
    deadline = min(now + cfg['planning_soft_limit_s'], tick - reserve, expiry - reserve)
    return {'contract': CONTRACT, 'prepared_at': now, 'deadline': deadline,
            'budget_ms': max(0., (deadline - now) * 1000),
            'fixed_action_deadline': tick, 'snapshot_expires_at': expiry,
            'commit_reserve_ms': reserve * 1000}


def scorer_view(state):
    # Raw transport receipts/queries stay in the parent decisions.jsonl. Keep
    # EVERY workflow node, observation error, resource, queue and progress value.
    view = {k: v for k, v in state.items() if k != 'native_snapshots'}
    view['cache_evidence'] = [{k: v for k, v in e.items() if k != 'native'}
                              for e in state.get('cache_evidence', [])]
    return view


def run_planner(planner, state, model, cfg, window):
    started = time.monotonic()
    result = planner(state, model, cfg, window['deadline'])
    result['planning_finished_at'] = time.monotonic()
    result['planning_dispatch_delay_ms'] = (started - window['prepared_at']) * 1000
    result['planning_deadline'] = window['deadline']
    result['worker_clock_contract'] = 'same-host monotonic absolute deadline; IPC consumes budget'
    return result
