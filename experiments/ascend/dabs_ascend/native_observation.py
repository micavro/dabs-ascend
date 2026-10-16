"""Bounded decision-time prefix queries; unqueried evidence stays unknown."""


def prefix_identities(nodes, salt, cache=None):
    """Hash concrete fixed inputs before RPC; identical prefixes only once.

    This is an input identity, not cache residency evidence. Check the key from
    actual tokens on each observation so a changed namespace/input cannot
    inherit an old identity. Nodes becoming ready during RPC are included.
    """
    from dabs_native.contracts import prefix_key
    unique, identities, live = ({} if cache is None else cache), {}, set()
    if unique and next(iter(unique))[0] != salt:
        unique.clear()
    for key, node in nodes.items():
        tokens = tuple(node['tokens'][:node['share']])
        signature = salt, tokens
        live.add(signature)
        if signature not in unique:
            unique[signature] = prefix_key(tokens, salt)
        identities[key] = unique[signature]
    # Bound lifetime to current inputs; this caches identity, never residency.
    if cache is not None:
        for old in list(unique):
            if old not in live:
                del unique[old]
    return identities


def query_priority(nodes, cfg, model, hints):
    from .native_algorithms import adapted, reference
    from .scheduler import snapshot
    # Hints only order queries. They cannot certify cache/state, route a request,
    # or bypass a fresh engine snapshot. All methods use this identical policy.
    current = snapshot(nodes, {}, cfg['replicas'], cfg, 0)
    if len(hints.get('pressure', [])) == cfg['replicas']:
        current['pressure'] = hints['pressure']
    current.update(native_mutable=[], native_queue_groups={
        str(r): {'running_prefill': [], 'waiting': []} for r in range(cfg['replicas'])})
    base = reference(current, cfg)
    _, tails = adapted('semantic_selector').tail_values(current, model, base['placement'])
    return sorted(current['ready'], key=lambda k: (
        k not in base['admit'], -tails.get(k, 0.), nodes[k]['ready_at'], k))


def prefix_queries(nodes, cfg, salt, *, enabled=True, model=None, hints=None):
    max_queries = cfg.get('native_cache_query_limit', 8)
    max_tokens = cfg.get('native_cache_query_token_limit', 8192)
    queries, bindings, groups = [], {}, {}
    tokens_used = 0
    ready = sorted((k for k, n in nodes.items() if n['state'] == 'ready'),
                   key=lambda k: (nodes[k]['ready_at'], k))
    if enabled and model is not None and ready:
        focused = query_priority(nodes, cfg, model, hints or {})
        focused_keys = set(focused)
        ready = focused + [k for k in ready if k not in focused_keys]
    for key in ready:
        node = nodes[key]
        if not enabled:
            continue
        shared = node['share'] // cfg['block_size'] * cfg['block_size']
        # Even a sub-block shared prefix can be queried (one real token).
        # Engine returns zero reusable blocks, instead of manufacturing unknown
        # forever for nodes whose original 16-token prefix is below 128.
        # Scoring credits only shared complete blocks. The allocator's
        # prompt-minus-one rule needs one extra actual token at that boundary.
        prefix = node['tokens'][:min(len(node['tokens']), shared + 1)]
        signature = tuple(prefix)
        if signature in groups:
            bindings[key] = groups[signature]
            continue
        if len(queries) >= max_queries or tokens_used + len(prefix) > max_tokens:
            continue
        groups[signature] = key
        bindings[key] = key
        queries.append({'query_id': key, 'tokens': prefix, 'salt': salt})
        tokens_used += len(prefix)
    return queries, bindings, {'enabled': enabled, 'ready_requests': len(ready),
        'unique_queries': len(queries), 'query_tokens_per_replica': tokens_used,
        'covered_requests': len(bindings), 'unqueried_requests': len(ready) - len(bindings),
        'query_limit': max_queries, 'token_limit': max_tokens,
        'priority': 'provisional-admission then original DAG tail; fresh evidence only',
        'queried_nodes': [q['query_id'] for q in queries],
        'scope': 'concrete shared-prefix blocks only; no full-prompt hit claim; unqueried is unknown'}
