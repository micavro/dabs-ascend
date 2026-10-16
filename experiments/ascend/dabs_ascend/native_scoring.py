"""Full-attention V1 batch semantics used by the isolated frozen scorer adapter.

The research candidate generator and objective remain frozen. This replaces only
the vendor-dependent batch construction: FCFS running order, then waiting, APC
skip BEFORE token-budget accounting, and token-granular chunked prefill. KV block
alignment applies to reuse/allocation, not to the number of scheduled tokens.
"""


def select_batch(ns, active, running, rank, native_rank, ledger, replica, at, cfg, *, active_window=True, snapshot_now=0):
    running[:] = [k for k in running if k in active]
    def accepted(k):
        return ns[k].get('accepted_at') if ns[k].get('accepted_at') is not None else ns[k].get('ready_at', snapshot_now)
    def ordered(keys):
        if active_window and rank:
            if any(snapshot_now + at / 1000 - accepted(k) >= cfg.get('age_limit_s', 5.) for k in keys):
                return list(keys)
            return sorted(keys, key=lambda k: rank.get(k, len(rank)))
        return sorted(keys, key=lambda k: (accepted(k), native_rank.get(k, 10**9), k))
    # Same safe-slot permutation as EngineCore; decode slots stay fixed.
    positions = [i for i, k in enumerate(running) if ns[k]['phase'] != 'decode']
    preferred = ordered([running[i] for i in positions])
    for i, key in zip(positions, preferred):
        running[i] = key
    waiting = ordered(sorted((k for k in active if k not in running), key=lambda k: (native_rank.get(k, 10**9), k)))
    left = cfg['hard_total']
    chosen, dec, context = [], [], 0
    for key in list(running) + waiting:
        if left <= 0:
            break
        n = ns[key]
        if key not in running and len(running) >= cfg['max_num_seqs']:
            break
        if n['phase'] == 'decode':
            dec.append(key)
            left -= 1
            continue
        before_cache = n.get('computed', 0)
        if key not in running:
            cached = ledger.length(replica, n.get('prefix_hash'), at)
            reusable = min(cached, (n['prompt'] - 1) // cfg['block_size'] * cfg['block_size'])
            n['computed'] = max(n.get('computed', 0), reusable)
        remaining = n['prompt'] - n.get('computed', 0)
        take = min(left, remaining)
        threshold = cfg.get('long_prefill_token_threshold', 0)
        if threshold > 0:
            take = min(take, threshold)
        if take <= 0:
            raise ValueError('native prefill progress inconsistent')
        # Original host model pc is computed_before, before APC input skipping.
        context = max(context, before_cache)
        chosen.append((key, take))
        left -= take
        if key not in running:
            running.append(key)
    # part/waiting drive the original scorer's conservative decode coalescing.
    part = [k for k in active if ns[k]['phase'] != 'decode']
    return chosen, dec, context, part, waiting


def check_model_range(model, replica, p, d, dc, pc):
    """No silent extrapolation outside this frozen native calibration."""
    if 'validation' not in model:
        return  # CPU fixtures; real runs require the raw native model gate.
    from .native_timing import TARGET
    if model.get('timing_target') == TARGET and model.get('host_step_equivalence') is not False:
        raise ValueError('native_calibration_out_of_range:async accounting mislabeled as same-batch host step')
    from core18v2.service import phase
    measured = model.get('prediction_validation_map', {}).get(str(replica), str(replica))
    observed = model['validation'][measured][phase(p, d)]['range']
    for key, value in dict(p=p, d=d, dc=dc, pc=pc).items():
        low, high = observed[key]
        if not low <= value <= high:
            raise ValueError('native_calibration_out_of_range:' + key + ':' + str(value) + ':replica=' + str(replica) + ':phase=' + phase(p, d) + ':observed=' + str([low, high]))


def next_batch(state, action, model, replica, cfg):
    from core18v2.engine import CacheModel
    from .native_prediction import predict
    # select_batch only changes scalar node progress. DAG lists are read-only;
    # copying the whole 216-node nested graph for each guard wastes the window.
    ns = {k: dict(n) for k, n in state['nodes'].items()}
    for key in action['admit']:
        ns[key].update(state='submitted', replica=action['placement'][key], computed=0, phase='prefill', output_done=0, accepted_at=state['now'])
    active = [k for k, n in ns.items() if n['state'] == 'submitted' and n['replica'] == replica]
    running = list(state.get('native_running', {}).get(str(replica), []))
    rank = {k: i for i, k in enumerate(action['order'])}
    fifo = {k: i for i, k in enumerate(state['queue_order'].get(str(replica), []))}
    ledger = CacheModel(state, cfg)
    pre, dec, pc, _, _ = select_batch(ns, active, running, rank, fifo, ledger, replica, 0, cfg, snapshot_now=state['now'])
    p, d = sum(n for _, n in pre), len(dec)
    dc = sum(ns[k]['prompt'] + ns[k].get('output_done', 0) for k in dec)
    if p or d:
        check_model_range(model, replica, p, d, dc, pc)
    if d:
        check_model_range(model, replica, 0, d, dc, 0)
    startup = not any(b['replica'] == replica for b in state.get('inflight_batches', []))
    return {'prefill': p, 'decode': d, 'decode_context': dc, 'cache_credit': 0,
            'timing_target': model.get('timing_target'),
            'prefill_startup_predicted': startup,
            'max_prefill_context': pc, 'step_ms': predict(model, replica, p, d, dc, pc, startup=startup) if p or d else 0.,
            'decode_only_ms': predict(model, replica, 0, d, dc, 0, startup=False) if d else 0.,
            'new_decode_next_step': sum(ns[k]['computed'] + n == ns[k]['prompt'] for k, n in pre)}


def cache_model_class():
    from core18v2.engine import CacheModel
    import math

    class NativeCacheModel(CacheModel):
        def evict_for_reservations(self, r, active, at):
            reserve = sum(math.ceil((n['prompt'] + n['output']) / self.block) for n in active)
            allowance = max(0, int(self.capacity[r]) - reserve)
            pinned = {n.get('prefix_hash') for n in active if n.get('computed', 0) > 0}
            inactive = [(k, e) for k, e in self.entries.items() if k[0] == r and k[1] not in pinned]
            used = sum(math.ceil(e.tokens / self.block) for _, e in inactive)
            if used <= allowance:
                return  # Identical no-eviction result; avoid sorting all past prefixes.
            for key, entry in sorted(inactive, key=lambda kv: (kv[1].last_touch, kv[0][1])):
                if used <= allowance:
                    break
                self.entries.pop(key, None); self.known[key] = 'modeled_evicted'
                used -= math.ceil(entry.tokens / self.block)
                self.counts['modeled_conservative_evictions'] += 1
    return NativeCacheModel
