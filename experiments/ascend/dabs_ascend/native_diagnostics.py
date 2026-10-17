"""Read-only instrumentation; never enumerate extra candidates or alter budgets."""
import time

REVISION = 'native-diagnostics-v1'


def instrument(text, name):
    if name == 'engine':
        replacements = [
            ('    cpu_start = time_fn()',
             '    from time import perf_counter as diagnostic_clock\n'
             '    diagnostic_start = diagnostic_clock()\n    cpu_start = time_fn()'),
            ("    ns = {k: dict(n) for k,n in snapshot['nodes'].items()}",
             "    diagnostic_validated = diagnostic_clock()\n    ns = {k: dict(n) for k,n in snapshot['nodes'].items()}"),
            ('    coeff = [', '    diagnostic_cloned = diagnostic_clock()\n    coeff = ['),
            ('    while pending_wf:', '    diagnostic_setup = diagnostic_clock()\n    while pending_wf:'),
            ('    complete = not pending_wf and reason is None',
             '    diagnostic_rolled = diagnostic_clock()\n    complete = not pending_wf and reason is None'),
            ("        'completed_workflows_at_cutoff': wf_done,",
             "        'completed_workflows_at_cutoff': wf_done,\n"
             "        'diagnostic_stage_ms': {'validate': (diagnostic_validated-diagnostic_start)*1000,\n"
             "            'clone': (diagnostic_cloned-diagnostic_validated)*1000,\n"
             "            'setup': (diagnostic_setup-diagnostic_cloned)*1000,\n"
             "            'rollout': (diagnostic_rolled-diagnostic_setup)*1000},\n"
             "        'diagnostic_progress': {'nodes_done': len(node_done), 'nodes_total': len(ns),\n"
             "            'workflows_done': len(wf_done), 'workflows_total': len(groups)},"),
        ]
    elif name == 'v5_selector':
        replacements = [
            ("    if set(a['admit'])!=set(base['admit']) or set(a['resource_held'])!=set(base['resource_held']):\n        return None",
             "    if set(a['admit'])!=set(base['admit']) or set(a['resource_held'])!=set(base['resource_held']):\n        diagnostic_hook('admission_change', (base,a))\n        return None"),
            ('    routes.sort(key=lambda z:z[0]);seen=set()',
             "    routes.sort(key=lambda z:z[0]);seen=set()\n    diagnostic_hook('route_pool', routes)"),
            ('        if k in seen:continue',
             "        if k in seen:\n            diagnostic_hook('seen_skip', (k,src,dst))\n            continue"),
        ]
    else:
        return text
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('diagnostic anchor changed: ' + name + ': ' + old)
        text = text.replace(old, new)
    return text


def opportunity(state, base, cfg, cache_index):
    """Describe the actual reference, without re-projecting an alternative."""
    ready, admitted = state['ready'], set(base['admit'])
    known = lambda k: (state['nodes'][k].get('prefix_hash'), base['placement'][k]) in cache_index
    active = [0] * state['replicas']
    for node in state['nodes'].values():
        if node['state'] == 'submitted' and node.get('replica') is not None:
            active[node['replica']] += 1
    assigned = [sum(base['placement'][k] == r for k in admitted) for r in range(state['replicas'])]
    return {'nodes': len(state['nodes']), 'ready': len(ready), 'reference_admitted': len(admitted),
            'reference_held': len(base.get('resource_held', [])), 'mutable': len(state['native_mutable']),
            'client_limit': cfg['client_max_inflight'], 'submitted_by_replica': active,
            'admitted_by_replica': assigned,
            'client_slots_after_reference': [max(0, cfg['client_max_inflight'] - a - b) for a, b in zip(active, assigned)],
            'reserved_blocks_before_reference': state.get('reserved_blocks'),
            'capacity_blocks': state.get('capacity_blocks'), 'reserve_fraction': cfg['admission_reserve_fraction'],
            'source_known_ready': sum(known(k) for k in ready),
            'source_known_admitted': sum(known(k) for k in admitted),
            'source_known_held': sum(known(k) for k in base.get('resource_held', [])),
            'pressure': list(state['pressure']),
            'note': 'client slots only, not KV feasibility; skipped alternatives were not tested'}


def evaluation_sample(result, started, stop, deadline, elapsed_ms):
    finished = time.monotonic()
    return {'slice_budget_ms': None if stop is None else (stop - started) * 1000,
            'planner_remaining_at_score_ms': None if deadline is None else (deadline - started) * 1000,
            'evaluator_elapsed_ms': elapsed_ms,
            'slice_overrun_ms': None if stop is None else max(0., finished - stop) * 1000,
            'stages_ms': result.get('diagnostic_stage_ms', {}),
            'progress': result.get('diagnostic_progress', {}),
            **{k: result.get(k) for k in ('rollout_steps', 'expanded_model_iterations', 'coalesced_decode_batches')}}
