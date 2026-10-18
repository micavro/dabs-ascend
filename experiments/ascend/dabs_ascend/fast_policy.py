"""Ascend request-event approximation, NOT the frozen V5 batch rollout.

Only unsubmitted ready requests may change admission/placement. Incumbent
queues, decode order, engine budgets and physical resource checks are retained.
All candidate costs use one model, one snapshot and one common continuation.
"""
from __future__ import annotations

import copy
import heapq
import math
import time
from collections import Counter

REVISION = 'ascend-fast-request-event-v1'


class BudgetExpired(Exception):
    pass


class ModelUnsupported(ValueError):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired('fast_planning_deadline')


def need_blocks(node, cfg):
    return math.ceil((node['prompt'] + node['max_output_tokens']) / cfg['block_size'])


def action_key(action):
    # Client submission order is deliberately not an engine queue-order claim.
    return tuple(sorted((k, action['placement'][k]) for k in action['admit']))


def admission_only(action):
    result = copy.deepcopy(action)
    result['order'] = [k for k in result.get('order', []) if k in result['admit']]
    result['execution_semantics'] = 'atomic_new_admissions_only; no engine queue permutation'
    return result


def validate_admissions(action, state, cfg):
    """Local conservative envelope. EngineCore remains the final authority."""
    ns, R = state['nodes'], state['replicas']
    admitted = action['admit']
    if len(admitted) != len(set(admitted)):
        raise ValueError('duplicate admission')
    if action.get('caps') != [cfg['base_cap']] * R:
        raise ValueError('fast policy cannot change engine token budget')
    counts = [sum(n['state'] == 'submitted' and n.get('replica') == r for n in ns.values()) for r in range(R)]
    used = list(state['reserved_blocks'])
    for k in admitted:
        if k not in state['ready'] or ns[k]['state'] != 'ready':
            raise ValueError('fast policy may only admit ready requests')
        if any(p not in ns or ns[p]['state'] != 'done' for p in ns[k]['parents']):
            raise ValueError('DAG dependency not complete')
        r = action['placement'].get(k)
        if type(r) is not int or not 0 <= r < R:
            raise ValueError('invalid target replica')
        counts[r] += 1
        used[r] += need_blocks(ns[k], cfg)
        if counts[r] > cfg['client_max_inflight'] or used[r] > int(state['capacity_blocks'][r] * cfg['admission_reserve_fraction']):
            raise ValueError('conservative admission envelope exceeded')
    if set(action.get('order', [])) - set(admitted):
        raise ValueError('incumbent queue mutations not supported in fast-one')
    return True


class ServiceCosts:
    """Frozen native coefficients -> request service-cost approximation.

    Prefill keeps the real chunk limit. Decode uses an exact arithmetic sum
    of the frozen affine single-decode formula, checking both context ends.
    No fitted coefficient, validation range or cache observation is changed.
    This is NOT a claim that a request runs alone on the real engine.
    """
    def __init__(self, state, model, cfg, deadline):
        self.state, self.model, self.cfg, self.deadline = state, model, cfg, deadline
        self.memo, self.evidence, self.request_memo = {}, {}, {}
        self.stats = Counter()
        for e in state.get('cache_evidence', []):
            key = e.get('prefix_hash'), e.get('replica')
            age = max(0., (time.monotonic() - state['now']) * 1000)
            if e.get('known') and e.get('valid_until_ms', 0) > age:
                tokens = int(e.get('tokens') or 0) if e.get('state') in ('active_computed', 'reclaimable_computed') else 0
                self.evidence[key] = max(tokens, self.evidence.get(key, 0))

    def primitive(self, replica, p, d, dc, pc, startup=False):
        check_deadline(self.deadline)
        phase = 'mixed' if p and d else 'prefill_only' if p else 'decode_only'
        measured = self.model.get('prediction_validation_map', {}).get(str(replica), str(replica))
        try:
            ranges = self.model['validation'][measured][phase]['range']
            row = self.model['models'][str(replica)]['phases'][phase]
            coeff = tuple(float(v) for v in row['coef'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelUnsupported('missing frozen phase model/range: ' + phase) from exc
        for key, value in dict(p=p, d=d, dc=dc, pc=pc).items():
            lo, hi = ranges[key]
            if not lo <= value <= hi:
                raise ModelUnsupported(f'out_of_range:r{replica}:{phase}:{key}={value}:range={lo},{hi}')
        if len(coeff) != 6 or any(not math.isfinite(c) or c < 0 for c in coeff):
            raise ModelUnsupported('invalid frozen coefficients')
        x = (1., p / 1024., d / 16., dc / 65536., p * pc / 4194304., p * d / 16384.)
        value = sum(c * a for c, a in zip(coeff, x))
        if p and not d and self.model.get('predictor') == 'phase6_prefill_startup_v2':
            regime = 'startup' if startup else 'pipeline'
            if regime not in row.get('observed_regimes', []):
                raise ModelUnsupported('unobserved prefill regime: ' + regime)
            extra = row['startup_extra_ms']
            if not math.isfinite(extra) or extra < 0:
                raise ModelUnsupported('invalid startup coefficient')
            value += extra if startup else 0.
        self.stats['primitive_calls'] += 1
        return max(.1, value), value

    def cost(self, key, replica, *, continuation=False):
        request_key = key, replica, continuation
        if request_key in self.request_memo:
            self.stats['cost_memo_hits'] += 1
            return self.request_memo[request_key]
        n, cfg = self.state['nodes'][key], self.cfg
        if n['state'] == 'done':
            return 0.
        submitted = n['state'] == 'submitted'
        computed = int(n.get('computed', 0)) if submitted else 0
        output_done = int(n.get('output_done', 0)) if submitted else 0
        prompt, output = int(n['prompt']), int(n['max_output_tokens'])
        if not 0 <= computed <= prompt + output or not 0 <= output_done <= output:
            raise ModelUnsupported('invalid request progress')
        cached = 0
        if not submitted and not continuation:
            cached = self.evidence.get((n.get('prefix_hash'), replica), 0)
            self.stats['unknown_cache_cold_assumption'] += (n.get('prefix_hash'), replica) not in self.evidence
            cached = min(cached, (prompt - 1) // cfg['block_size'] * cfg['block_size'])
        identity = (replica, prompt, output, min(computed, prompt), output_done,
                    n.get('phase') if submitted else 'prefill', cached, submitted)
        if identity in self.memo:
            self.stats['cost_memo_hits'] += 1
            self.request_memo[request_key] = self.memo[identity]
            return self.memo[identity]
        check_deadline(self.deadline)
        value = 0.
        if not submitted or n.get('phase') != 'decode':
            progress = max(min(computed, prompt), cached)
            first = True
            while progress < prompt:
                chunk = min(cfg['hard_total'], prompt - progress)
                if cfg.get('long_prefill_token_threshold', 0) > 0:
                    chunk = min(chunk, cfg['long_prefill_token_threshold'])
                # The native pc feature is before APC skipping on its first batch.
                context = min(computed, prompt) if first and cached else progress
                value += self.primitive(replica, chunk, 0, 0, context,
                                        startup=not submitted and first)[0]
                progress += chunk
                first = False
        count = output - output_done
        if count:
            first_dc = prompt + output_done
            first, raw_first = self.primitive(replica, 0, 1, first_dc, 0)
            last, raw_last = self.primitive(replica, 0, 1, first_dc + count - 1, 0)
            # Nonnegative affine coefficients permit an exact sum incl. .1 floor.
            slope = (raw_last - raw_first) / (count - 1) if count > 1 else 0.
            floored = count if raw_last <= .1 else 0
            if raw_first < .1 < raw_last and slope > 0:
                floored = min(count, max(0, math.ceil((.1 - raw_first) / slope)))
            remaining = count - floored
            value += .1 * floored + (remaining * (raw_first + slope * floored + raw_last) / 2 if remaining else 0.)
        if not math.isfinite(value) or value < 0:
            raise ModelUnsupported('nonfinite request service estimate')
        self.memo[identity] = value
        self.request_memo[request_key] = value
        return value


def topology(state, costs, deadline):
    ns = state['nodes']
    children = {k: [] for k in ns}
    degree = {}
    for k, n in ns.items():
        degree[k] = len(n['parents'])
        for parent in n['parents']:
            if parent not in ns:
                raise ValueError('incomplete arrived DAG')
            children[parent].append(k)
    todo = sorted(k for k in ns if degree[k] == 0)
    ordered = []
    while todo:
        check_deadline(deadline)
        k = todo.pop()
        ordered.append(k)
        for child in children[k]:
            degree[child] -= 1
            if degree[child] == 0:
                todo.append(child)
    if len(ordered) != len(ns):
        raise ValueError('cyclic workflow graph')
    minimum = {k: min(costs.cost(k, r, continuation=True) for r in
                      ([ns[k]['replica']] if ns[k]['state'] == 'submitted' else range(state['replicas'])))
               for k in ordered}
    tail = {}
    for k in reversed(ordered):
        tail[k] = minimum[k] + max((tail[c] for c in children[k]), default=0.)
    return ordered, children, tail


def generate_candidates(state, cfg, base, costs, tail, deadline):
    """Two distinct executable alternatives; invalid/no-op attempts use no quota."""
    ns, R = state['nodes'], state['replicas']
    ready = sorted(state['ready'], key=lambda k: (ns[k]['ready_at'], k))
    wf_tail = {}
    for k in ready:
        wf_tail[ns[k]['wid']] = max(wf_tail.get(ns[k]['wid'], 0.), tail[k])
    horizon = cfg.get('fast_starvation_s', 5.)
    def aged(k):
        return state['now'] - ns[k]['ready_at'] >= horizon
    # Reserve priority for the oldest aged request, not every aged request.
    # Once it is admitted, the next event protects the next oldest request.
    protected = next((k for k in ready if aged(k)), None)
    def priority(k):
        return (0, ns[k]['ready_at'], k) if k == protected else (1, wf_tail[ns[k]['wid']], -tail[k], ns[k]['ready_at'], k)
    order = sorted(ready, key=priority)
    seen, actions = {action_key(base)}, []
    diagnostics = Counter()
    for mode in ('dag_admission', 'dag_admission_route'):
        check_deadline(deadline)
        counts = [sum(n['state'] == 'submitted' and n.get('replica') == r for n in ns.values()) for r in range(R)]
        used = list(state['reserved_blocks'])
        loads = [sum(costs.cost(k, r, continuation=True) for k, n in ns.items()
                     if n['state'] == 'submitted' and n.get('replica') == r) for r in range(R)]
        placement, admitted = dict(base['placement']), []
        for k in order:
            check_deadline(deadline)
            need = need_blocks(ns[k], cfg)
            options = [base['placement'].get(k, min(range(R), key=lambda r: (loads[r], r)))] if mode == 'dag_admission' else sorted(range(R), key=lambda r: (loads[r] + costs.cost(k, r), r))
            for r in options:
                diagnostics['target_attempts'] += 1
                if counts[r] >= cfg['client_max_inflight'] or used[r] + need > int(state['capacity_blocks'][r] * cfg['admission_reserve_fraction']):
                    diagnostics['target_rejected'] += 1
                    continue  # A rejection at this target never suppresses another.
                placement[k] = r
                admitted.append(k)
                counts[r] += 1
                used[r] += need
                loads[r] += costs.cost(k, r)
                break
        a = {'family': mode, 'placement': placement, 'admit': admitted,
             'order': list(admitted), 'caps': [cfg['base_cap']] * R}
        validate_admissions(a, state, cfg)
        # Do not "improve" JCT by withholding the baseline's feasible work count.
        if len(admitted) < len(base['admit']):
            diagnostics['reduced_admission_count_rejected'] += 1
            continue
        key = action_key(a)
        if key in seen:
            diagnostics['no_executable_difference'] += 1
            continue
        seen.add(key)
        actions.append(admission_only(a))
    diagnostics['distinct_candidates'] = len(actions)
    return actions, dict(diagnostics)


def score(state, action, cfg, costs, topo, deadline):
    """One request completion event per node; serial service proxy per replica.

    All arrived DAG nodes remain. Unarrived workflows are not read. Continuation
    is the SAME FIFO-ready / earliest-finish legal placement for every action.
    No future cache residency credit is assumed.
    """
    ns, R = state['nodes'], state['replicas']
    ordered, children, _ = topo
    done = {k for k, n in ns.items() if n['state'] == 'done'}
    active, events, completions = set(), [], {k: 0. for k in done}
    clocks, counts, used = [0.] * R, [0] * R, [0] * R
    ready_at = {k: 0. for k in state['ready']}
    pending_parents = {k: sum(p not in done for p in n['parents']) for k, n in ns.items()}
    seq, now = 0, 0.
    def fits(k, r):
        return counts[r] < cfg['client_max_inflight'] and used[r] + need_blocks(ns[k], cfg) <= int(state['capacity_blocks'][r] * cfg['admission_reserve_fraction'])
    def admit(k, r, at, continuation):
        nonlocal seq
        if k in active or k in done:
            raise ValueError('simulation duplicate admission')
        dt = costs.cost(k, r, continuation=continuation)
        end = max(at, clocks[r]) + dt
        clocks[r] = end
        counts[r] += 1
        used[r] += need_blocks(ns[k], cfg)
        active.add(k)
        ready_at.pop(k, None)
        seq += 1
        heapq.heappush(events, (end, seq, k, r))
    for r in range(R):
        incumbents = [k for k in state.get('queue_order', {}).get(str(r), [])
                      if ns[k]['state'] == 'submitted' and ns[k].get('replica') == r]
        incumbents += sorted((k for k, n in ns.items() if n['state'] == 'submitted' and n.get('replica') == r and k not in incumbents), key=lambda k: (ns[k].get('start') or 0., k))
        for k in incumbents:
            admit(k, r, 0., True)
    # Canonical order: never credit a priority order the atomic API does not
    # guarantee. Real engine order is only observed, not changed by this mode.
    new_order = sorted(action['admit'], key=lambda k: (ns[k]['ready_at'], k))
    for k in new_order:
        admit(k, action['placement'][k], 0., False)
    event_count = 0
    def continuation(at):
        for k in sorted(list(ready_at), key=lambda k: (ready_at[k], ns[k].get('ready_at') or state['now'], k)):
            check_deadline(deadline)
            legal = [r for r in range(R) if fits(k, r)]
            if not legal:
                continue
            r = min(legal, key=lambda r: (max(at, clocks[r]) + costs.cost(k, r, continuation=True), r))
            admit(k, r, at, True)
    # The first action is the current admission decision; wait until the next
    # real event/200ms watchdog to continue, rather than undoing it at time 0.
    if not events and ready_at:
        continuation(cfg.get('fast_poll_s', .2) * 1000)
    while events:
        check_deadline(deadline)
        now, _, k, r = heapq.heappop(events)
        active.remove(k); done.add(k); completions[k] = now
        counts[r] -= 1; used[r] -= need_blocks(ns[k], cfg)
        event_count += 1
        for child in children[k]:
            pending_parents[child] -= 1
            if pending_parents[child] == 0 and child not in active and child not in done:
                ready_at[child] = now
        continuation(now)
    if len(done) != len(ns):
        raise ModelUnsupported('request proxy cannot complete DAG within resource envelope')
    by_workflow = {}
    for k in ordered:
        by_workflow[ns[k]['wid']] = max(by_workflow.get(ns[k]['wid'], 0.), completions[k])
    return {'cost_ms': sum(by_workflow.values()), 'workflow_completion_ms': by_workflow,
            'request_completion_ms': completions, 'events': event_count, 'complete': True,
            'score_kind': REVISION, 'native_batch_rollout': False}


def choose(state, model, cfg, base, deadline, guard=None):
    start = time.monotonic()
    base = admission_only(base)
    validate_admissions(base, state, cfg)
    result = {'action': base, 'reference_action': base, 'selected_cell': '00',
              'evaluations': [], 'candidate_diagnostics': {}, 'model_selection': False,
              'selector_revision': REVISION, 'native_batch_rollout': False,
              'stop_reason': 'reference_only', 'guard_details': []}
    def finish():
        result['wall_ms'] = (time.monotonic() - start) * 1000
        return result
    if state.get('native_observation_errors') or state.get('native_prediction_errors'):
        result['stop_reason'] = 'native_input_unusable'
        return finish()
    try:
        check_deadline(deadline)
        costs = ServiceCosts(state, model, cfg, deadline)
        topo = topology(state, costs, deadline)
        candidates, diag = generate_candidates(state, cfg, base, costs, topo[2], deadline)
        result['candidate_diagnostics'] = diag
        if not candidates:
            result['stop_reason'] = 'no_distinct_admission_candidate'
            return finish()
        z0 = score(state, base, cfg, costs, topo, deadline)
        result['evaluations'].append({'label': 'reference', **z0})
        best, best_score = base, z0
        best_label = '00'
        for index, candidate in enumerate(candidates, 1):
            z = score(state, candidate, cfg, costs, topo, deadline)
            violations = []
            for k, n in state['nodes'].items():
                if n['state'] == 'submitted' and n.get('phase') == 'decode':
                    a, b = z0['request_completion_ms'][k], z['request_completion_ms'][k]
                    if b > a * (1 + cfg.get('remaining_decode_relative', .03)) + cfg.get('remaining_decode_absolute_ms', 2.):
                        violations.append({'kind': 'incumbent_decode', 'node': k})
            if guard is None:
                # Never silently disable the native next-batch guard.
                if any(n['state'] == 'submitted' for n in state['nodes'].values()):
                    violations.append({'kind': 'native_guard_unavailable'})
            else:
                violations.extend(guard(state, base, candidate, model, cfg))
            check_deadline(deadline)
            result['evaluations'].append({'label': candidate['family'], 'guard': violations, **z})
            result['guard_details'].extend(violations)
            if not violations and z['cost_ms'] < best_score['cost_ms']:
                best, best_score, best_label = candidate, z, str(index)
        check_deadline(deadline)
        elapsed_ms = (time.monotonic() - start) * 1000
        # Charge actual planning CPU wait to all unfinished workflows, not just
        # candidate execution. This is conservative and explicitly approximate.
        margin = cfg.get('minimum_predicted_gain_ms', 1.) + elapsed_ms * len(z0['workflow_completion_ms'])
        result.update(model_selection=True, predicted_gain_ms=z0['cost_ms'] - best_score['cost_ms'],
                      decision_cost_margin_ms=margin, cost_diagnostics=dict(costs.stats))
        if best_label != '00' and result['predicted_gain_ms'] > margin:
            result.update(action=best, selected_cell=best_label, stop_reason='nonreference_selected')
        else:
            result['stop_reason'] = 'reference_retained_after_fast_scoring'
    except (BudgetExpired, ModelUnsupported) as exc:
        # Partial comparisons never select an action, even if an earlier score
        # happened to be lower. Retain diagnostics and the precomputed reference.
        result.update(action=base, selected_cell='00', model_selection=False, stop_reason=str(exc))
    return finish()


def native_guard(state, base, action, model, cfg):
    """Retain existing mixed-step decode guard on the real native batch model."""
    from .native_scoring import next_batch
    violations = []
    for r in range(state['replicas']):
        if not any(n['state'] == 'submitted' and n.get('replica') == r and n.get('phase') == 'decode' for n in state['nodes'].values()):
            continue
        try:
            b = next_batch(state, {**action, 'order': []}, model, r, cfg)
            ref = next_batch(state, {**base, 'order': []}, model, r, cfg)
        except ValueError as exc:
            violations.append({'kind': 'native_guard_model_unusable', 'reason': str(exc)})
            continue
        fields = ('prefill', 'decode', 'decode_context', 'cache_credit')
        if b['decode'] and tuple(b[k] for k in fields) != tuple(ref[k] for k in fields):
            limit = max(ref['step_ms'], b['decode_only_ms'] * (1 + cfg.get('local_decode_step_relative', .1)) + cfg.get('local_decode_step_absolute_ms', 2.))
            if b['step_ms'] + cfg.get('local_model_margin_ms', 1.) > limit:
                violations.append({'kind': 'next_mixed_step', 'replica': r, 'limit_ms': limit, 'candidate_ms': b['step_ms']})
    return violations


def worker_choose(state, model, cfg, base, deadline):
    return choose(state, model, cfg, base, deadline, guard=native_guard)


def warm_worker():
    from .common import source_paths
    source_paths()
    from .native_scoring import next_batch  # noqa: F401
    from core18v2.engine import CacheModel  # noqa: F401; import only, no evaluate()
    return REVISION
