"""Bounded decision-time prefix queries; unqueried evidence stays unknown."""


def prefix_queries(nodes, cfg, salt, *, enabled=True):
    max_queries = cfg.get('native_cache_query_limit', 8)
    max_tokens = cfg.get('native_cache_query_token_limit', 8192)
    queries, bindings, groups = [], {}, {}
    tokens_used = 0
    ready = sorted((k for k, n in nodes.items() if n['state'] == 'ready'),
                   key=lambda k: (nodes[k]['ready_at'], k))
    for key in ready:
        node = nodes[key]
        if not enabled:
            continue
        shared = node['share'] // cfg['block_size'] * cfg['block_size']
        if not shared:
            continue
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
        'scope': 'concrete shared-prefix blocks only; no full-prompt hit claim; unqueried is unknown'}
