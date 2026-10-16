"""Repair contracts at the HTTP boundary. These do not implement native hooks."""
from collections import Counter
import time

from .common import source_paths


def admission_check(action, nodes, cfg, snapshot_at, now):
    """Validate the whole batch before submitting any request; ordering is separate.

    The accepted membership may be identical even if resources() sorts it. A
    permutation alone must not reject Ayo/Ours priority. Physical KV remains
    unobserved and cannot be certified by this client admission check.
    """
    source_paths()
    from joint18v2.actions import resources
    from .scheduler import snapshot
    if now - snapshot_at > cfg['snapshot_max_age_s'] or now < snapshot_at:
        return 'stale_snapshot'
    admitted = action['admit']
    if len(admitted) != len(set(admitted)):
        return 'duplicate_admission'
    if len(action['order']) != len(set(action['order'])):
        return 'duplicate_priority'
    if any(k not in nodes or nodes[k]['state'] != 'ready' for k in admitted):
        return 'request_not_ready'
    if any(any(nodes[p]['state'] != 'done' for p in nodes[k]['parents']) for k in admitted):
        return 'dependency_not_complete'
    if any(type(action['placement'].get(k)) is not int or
           not 0 <= action['placement'][k] < cfg['replicas'] for k in admitted):
        return 'invalid_replica'
    if action['caps'] != [cfg['base_cap']] * cfg['replicas']:
        return 'unsupported_step_budget_change'
    if cfg.get('execution_scope') == 'native':
        for replica in range(cfg['replicas']):
            active = sum(n['state'] == 'submitted' and n.get('replica') == replica for n in nodes.values())
            new = sum(action['placement'][k] == replica for k in admitted)
            if active + new > cfg['client_max_inflight']:
                return 'client_capacity_changed'
        # Physical capacity, engine state/version, evidence and queue safety are
        # checked atomically by EngineCore; never replace them with HTTP proxies.
        return None
    fresh = snapshot(nodes, {}, cfg['replicas'], cfg, 0, now)
    accepted = resources(action, fresh, cfg)
    if set(accepted['admit']) != set(admitted):
        return 'client_capacity_changed'
    return None


def effect_signature(action, nodes):
    """Only per-replica order matters; inter-replica submission permutations do not."""
    from .r234 import submission_order
    queues = {}
    for key in submission_order(action, nodes):
        queues.setdefault(action['placement'][key], []).append(key)
    return {replica: tuple(keys) for replica, keys in queues.items()}


def with_budget(planner, state, model, cfg, budget_s):
    # The parent measures IPC+search+return in its own clock and rejects late
    # replies. The worker receives a duration, never an absolute parent deadline.
    # All state timestamps still share the snapshot's clock and are only used
    # as differences within that snapshot by the surrogate.
    result = planner(state, model, cfg, time.monotonic() + max(0., budget_s))
    result['worker_clock_contract'] = 'local_duration; parent_enforces_end_to_end_deadline'
    return result


def score_diagnostics(result):
    rows = result.get('evaluations', [])
    reference = [r for r in rows if r.get('cell') == '00']
    candidate = [r for r in rows if r.get('cell') != '00']
    reasons = Counter(r.get('reason') or 'unspecified' for r in rows if not r.get('complete'))
    return {'reference_score_started': len(reference),
            'reference_score_rankable': sum(bool(r.get('complete')) for r in reference),
            'candidate_score_started': len(candidate),
            'candidate_score_rankable': sum(bool(r.get('complete')) for r in candidate),
            'score_failure_reasons': dict(reasons),
            'guard_rejected': result.get('decode_guard_rejections', 0),
            'selected': int(result.get('selected_cell', '00') != '00')}


def aggregate_diagnostics(decisions):
    counters = Counter()
    reasons, commits, fallbacks = Counter(), Counter(), Counter()
    for decision in decisions:
        values = score_diagnostics(decision.get('result', {}))
        reasons.update(values.pop('score_failure_reasons'))
        counters.update(values)
        if decision.get('commit_reject_reason'):
            commits[decision['commit_reject_reason']] += 1
        fallback = decision.get('native_fallback_reason')
        if fallback:
            fallbacks.update(fallback if isinstance(fallback, list) else [fallback])
            counters['native_atomic_fallback_decisions'] += 1
    return {**dict(counters), 'score_failure_reasons': dict(reasons),
            'native_fallback_reasons': dict(fallbacks),
            'commit_reject_reasons': dict(commits)}


MISSING_NATIVE = [
    'decision-time per-prefix/per-replica computed KV evidence with restart/eviction invalidation',
    'actual running+2*waiting pressure and physical KV capacity',
    'engine action acknowledgement and applied queue order',
    'independent native batch-step calibration and holdout validation',
]


def mechanism_gate(paths=None):
    from .native_gate import evaluate_local
    result = evaluate_local(paths or {})
    result.update(adapter='dabs-native-v023-1', inference_executed=False,
                  next_command='bash run_native.sh test')
    # Local evidence is necessary; a second LIVE record/config/epoch check
    # is mandatory inside r234_run before any native experiment requests.
    return result
