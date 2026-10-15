"""Ascend boundary for the byte-preserved 20261015 R234 source package."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import sys

from .common import ROOT, read_json, sha, source_paths

SUITE = 'dabs-r234-balanced-64'
FROZEN = ROOT / 'source/spra_workflow_fullgrid_r234_sll_v1/wfsched'
METHODS = ('RR', 'SLL', 'Ayo', 'Ours')
LIMITS = [
    'Same 96 condition specifications, balanced DAG selection, arrivals and fixed token budgets as the new source package.',
    'HTTP controller adapter: not a native NVIDIA/Ascend engine-internal scheduler reproduction.',
    'Native KV hits, eviction, physical occupancy and prefill/decode kernel times are unobserved (null, never zero).',
    'Ours uses the supplied DAG-H V5 with executable ready-request ordering only; submitted engine queues cannot be reordered.',
    'V5 route candidates require observed source cache residency. HTTP has no such evidence, so these candidates are unavailable.',
    'Ayo is the supplied Algorithm-2 topology proxy; load uses logical client token reservations, not physical KV occupancy.',
    'Fresh HTTP calibration replaces frozen NVIDIA timings; the model is an approximation, not a speedup guarantee.',
    'Nonsemantic fixed-token DAG replay: children wait for parents, but parent answer text is not substituted; EM/F1 is inapplicable.',
    'Per-replica client concurrency follows the declared Ascend deployment (default 2), rather than the original native max_num_seqs=64.',
]


def frozen(name):
    """Alias-package loading keeps old wfsched and its tests isolated."""
    source_paths()
    alias = 'dabs_r234_frozen'
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, FROZEN / '__init__.py',
                                                     submodule_search_locations=[str(FROZEN)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return importlib.import_module(alias + '.' + name)


def verify_import():
    manifest = read_json(ROOT / 'provenance/r234_import.json')
    for name, expected in manifest['files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Imported R234 source changed: ' + name)
    return {'passed': True, 'bundle_sha256': manifest['sha256'],
            'verified_files': len(manifest['files'])}


def plan():
    conditions = copy.deepcopy(frozen('design').plan(SUITE)['conditions'])
    by_key = {(s['replicas'], s['rate'], s['repeat'], s['method']): s for s in conditions}
    cases = list(dict.fromkeys((s['replicas'], s['rate']) for s in conditions[:48]))
    cases.remove((4, 8))
    cases.remove((2, 16))
    cases = [(4, 8), (2, 16)] + cases
    first = []
    for index, (replicas, rate) in enumerate(cases):
        rotation = index % len(METHODS)
        methods = METHODS[rotation:] + METHODS[:rotation]
        first.extend(by_key[replicas, rate, 1, method] for method in methods)
    # Predetermined before any outcome: first eight cover ordinary and loaded
    # cases; method positions rotate, and round 2 mirrors the entire first round.
    return first + [by_key[s['replicas'], s['rate'], 2, s['method']] for s in reversed(first)]


def ready_projection(action, state):
    action = copy.deepcopy(action)
    # Remove unsupported queued-prefill reorder BEFORE surrogate evaluation.
    action['order'] = [k for k in action['order'] if k in state['ready'] and k in action['admit']]
    action['ascend_action_projection'] = 'ready_request_submission_order_only'
    return action


def scoring_action(action, state):
    """Do not score a newly sent request jumping ahead of an existing one.

    Native queue ranks are unknown. The surrogate must retain the same observed
    submission FIFO for already submitted requests in every compared action.
    """
    score = copy.deepcopy(action)
    submitted = sorted((k for k, n in state['nodes'].items()
                        if n['state'] == 'submitted' and n.get('phase') != 'decode'),
                       key=lambda k: (state['nodes'][k]['start'], k))
    score['order'] = submitted + submission_order(action, state['nodes'])
    return score


def baseline(state, cfg, method):
    if method in ('RR', 'SLL'):
        return frozen('routing').reference(state, {**cfg, 'wf_routing_policy': method})
    if method == 'Ayo':
        return ready_projection(frozen('ayo_topology').reference(state, cfg), state)
    if method == 'Ours':
        from joint18v2.search import reference
        return ready_projection(reference(state, cfg), state)
    raise ValueError('Unknown R234 method: ' + str(method))


def planner_ready():
    frozen('v5_selector')
    frozen('ayo_topology')
    return True


def plan_ours(state, model, cfg, deadline):
    selector = frozen('v5_selector')
    original_reference, original_proposals, original_evaluate = selector.reference, selector.proposals, selector.evaluate

    def reference(s, c):
        return ready_projection(original_reference(s, c), s)

    def proposals(s, m, c, base):
        return [(label, ready_projection(action, s), origin, affected)
                for label, action, origin, affected in original_proposals(s, m, c, base)]

    def evaluate(s, action, m, c, **kwargs):
        return original_evaluate(s, scoring_action(action, s), m, c, **kwargs)

    selector.reference, selector.proposals, selector.evaluate = reference, proposals, evaluate
    try:
        result = selector.choose_heterogeneous(state, model, cfg, 'D', deadline)
    finally:
        selector.reference, selector.proposals, selector.evaluate = original_reference, original_proposals, original_evaluate
    result.update(method='Ours', native_dag_equivalent=False,
                  native_cache_observed=False, calibration_adopted=True,
                  calibration_source='fresh_ascend_http',
                  surrogate_submitted_order='fixed_client_submission_fifo',
                  route_candidate_cache_unknown=len(state['ready']),
                  action_projection='ready_request_submission_order_only')
    from .repair import score_diagnostics
    result['diagnostics'] = score_diagnostics(result)
    if set(result['action']['order']) - set(state['ready']):
        raise ValueError('Unsupported engine-internal ordering')
    return result


def submission_order(action, nodes):
    admitted = set(action['admit'])
    if len(admitted) != len(action['admit']) or len(action['order']) != len(set(action['order'])):
        raise ValueError('Duplicate request submission')
    ordered = [k for k in action['order'] if k in admitted]
    ordered += sorted(admitted - set(ordered), key=lambda k: (nodes[k]['ready_at'], k))
    if len(ordered) != len(admitted):
        raise ValueError('Duplicate request submission')
    return ordered


async def convert(nodes, tokenizer):
    pad = (await tokenizer.encode(' .'))[-1]
    fingerprint = frozen('fixed_work').convert_nodes(nodes, pad, heterogeneous=True)
    if len(nodes) != 216 or sum(n['max_output_tokens'] for n in nodes.values()) != 13184:
        raise ValueError('R234 must have 216 requests and 13184 declared output tokens')
    return {'fixed_workload_replay': True, 'external_work_fingerprint': fingerprint,
            'fixed_prompt_tokens': 3584, 'declared_output_tokens': 13184,
            'sampling': 'fixed 3584 prompt; output length by DAG family; min_tokens=max_tokens; ignore_eos=true; no JSON schema',
            'child_input_uses_parent_answer': False, 'semantic_quality_applicable': False}


def audit_work(nodes):
    return (len(nodes) == 216 and len({n['wid'] for n in nodes.values()}) == 64
            and sum(n['stream']['usage']['completion_tokens'] for n in nodes.values()) == 13184
            and all(n['stream']['usage']['prompt_tokens'] == 3584
                    and n['stream']['usage']['completion_tokens'] == n['fixed_output_budget_predeclared']
                    for n in nodes.values()))
