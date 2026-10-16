"""Auditable parameterization of frozen V5/Ayo/RR/SLL, in isolated namespaces.

No frozen file is edited. Required transformation anchors are checked; source fingerprints are recorded
for provenance and do not authorize or block deployment.
16/1024/65536 in service.features are regression scales, not KV block sizes.
"""
import copy
import types
import sys
from functools import lru_cache

from . import r234
from .common import ROOT, sha, verify_original_assets

TRANSFORMS = {
    'engine': [
        ("    R = snapshot['replicas']", "    R = snapshot['replicas']\n    from dabs_ascend.native_prediction import startup_extra\n    pipeline_warm = [any(b['replica'] == r for b in snapshot.get('inflight_batches', [])) for r in range(R)]"),
        ("        if not active:\n            return", "        if not active:\n            pipeline_warm[r] = False\n            return"),
        ("            last_finish[r] = now", "            if not active_keys[r] and busy[r] is None:\n                pipeline_warm[r] = False\n            last_finish[r] = now"),
        ("        dt=max(.1,c[0]+c[1]*eff/1024.+c[2]*d/16.+c[3]*dc/65536.+c[4]*eff*context/4194304.+c[5]*eff*d/16384.)",
         "        startup = not pipeline_warm[r]\n        extra = startup_extra(model, r, p, d, startup)\n        dt=max(.1,c[0]+c[1]*eff/1024.+c[2]*d/16.+c[3]*dc/65536.+c[4]*eff*context/4194304.+c[5]*eff*d/16384.+extra)\n        pipeline_warm[r] = True"),
        ("'current_action_window': active_window, 'decode_steps': decode_steps}",
         "'current_action_window': active_window, 'decode_steps': decode_steps, 'prefill_startup_predicted': startup}"),
        ("return len(active) < cfg['max_num_seqs']", "return len(active) < cfg['client_max_inflight']"),
        ("    for initial in snapshot.get('inflight_batches', []):", "    pending_native = [[] for _ in range(R)]\n    for initial in snapshot.get('inflight_batches', []):"),
        ("        if busy[r] is not None:raise ValueError('duplicate native inflight replica')\n", ""),
        ("        busy[r]=b;started_keys.update(dec+[k for k,_ in pre]);push(b['end_ms'],'complete',b)",
         "        pending_native[r].append(b)\n        started_keys.update(dec+[k for k,_ in pre])\n        if busy[r] is None:\n            busy[r]=b;push(b['end_ms'],'complete',b)"),
        ("            busy[r] = None\n", "            busy[r] = None\n            if b.get('native_inflight_observation'):\n                pending_native[r].pop(0)\n                if pending_native[r]:\n                    busy[r] = pending_native[r][0]\n                    push(busy[r]['end_ms'], 'complete', busy[r])\n"),
        ("        ns[k]['state'] = 'submitted'", "        ns[k]['accepted_at'] = snapshot['now'] + at / 1000\n        ns[k]['state'] = 'submitted'"),
    ],
    'semantic_selector': [
        ('4096-d', "s['resource_units']['surrogate_step_budget_tokens']-d"),
        ('(chunk//16)*16', "(chunk//s['resource_units']['block_size_tokens'])*s['resource_units']['block_size_tokens']"),
    ],
    'v5_selector': [
        ("cfg['base_cap']!=4096", "cfg['base_cap']!=s['current_caps'][0]"),
        ('fixed native 4096 budget', 'verified fixed engine step budget'),
    ],
    'ayo_topology': [
        ("cfg.get('base_cap')!=4096", "cfg.get('base_cap')!=s['current_caps'][0]"),
        ("'caps':[4096]*R", "'caps':[cfg['base_cap']]*R"),
        ("cfg['max_num_seqs']", "cfg['client_max_inflight']"),
    ],
    'routing': [
        ("'caps':[4096]*R", "'caps':[cfg['base_cap']]*R"),
        ('from joint18v2.actions import resources', 'from dabs_ascend.native_algorithms import resources'),
    ],
}


def original_path(name):
    if name == 'engine':
        return ROOT / 'source/spra_stage19_musique_joint_v1/core18v2/engine.py'
    if name == 'semantic_selector':
        return ROOT / 'source/spra_stage19_musique_joint_v1/semantic19/selector.py'
    return r234.FROZEN / (name + '.py')


def transformation_manifest():
    result = {name: {'frozen_sha256': sha(original_path(name)), 'replacements': replacements}
            for name, replacements in TRANSFORMS.items()}
    result['engine']['batch_selection'] = 'native_scoring.select_batch: V1 running/waiting, APC before budget, token-granular chunks'
    return result


@lru_cache(None)
def verify_base():
    return verify_original_assets()


@lru_cache(None)
def adapted(name):
    r234.verify_import()
    verify_base()
    path = original_path(name)
    text = path.read_text(encoding='utf-8')
    for old, new in TRANSFORMS[name]:
        count = text.count(old)
        expected = 1
        if count != expected:
            raise ValueError('frozen adapter anchor changed: %s %r (%s)' % (name, old, count))
        text = text.replace(old, new)
    if name == 'engine':
        start = text.index('        # Declared running/decode, partial-prefill, new-prefill boundaries.')
        end = text.index('        p = sum(t for _, t in chosen)', start)
        text = text[:start] + """        from dabs_ascend.native_scoring import select_batch
        if dirty[r]:
            ledger.evict_for_reservations(r, active_at(r), at);dirty[r]=False
        chosen, dec, context, part, waiting = select_batch(ns, active, native_running[r], rank,
            native_rank, ledger, r, at, cfg, active_window=active_window, snapshot_now=snapshot['now'])
        cache = 0
        cap = current_caps[r]
""" + text[end:]
        anchor = '    def schedule(r, at):'
        text = text.replace(anchor, "    native_running = [list(snapshot.get('native_running', {}).get(str(r), [])) for r in range(R)]\n" + anchor)
        anchor = "        eff=max(0,p-cache);c=coeff[r]['mixed' if p and d else 'prefill_only' if p else 'decode_only']"
        if text.count(anchor) != 1:
            raise ValueError('native model support anchor changed')
        text = text.replace(anchor, "        from dabs_ascend.native_scoring import check_model_range\n        check_model_range(model, r, p, d, dc, context)\n" + anchor)
    module = types.ModuleType('dabs_native_derived_' + name)
    module.__package__ = 'core18v2' if name == 'engine' else ''
    sys.modules[module.__name__] = module
    exec(compile(text, str(path) + ':native-derived', 'exec'), module.__dict__)
    return module


def resources(action, state, cfg):
    from joint18v2.actions import resources as original
    # Allocator capacity and current reservations come from EngineCore. Client
    # admission count is distinct from max sequences in the engine rollout.
    return original(action, state, {**cfg, 'max_num_seqs': cfg['client_max_inflight']})


def project(action, state):
    action = copy.deepcopy(action)
    allowed = set(state['native_mutable']) | set(action['admit'])
    rank = {k: i for i, k in enumerate(action['order']) if k in allowed}
    orders = []
    for replica in range(state['replicas']):
        groups = state['native_queue_groups'][str(replica)]
        for group in ('running_prefill', 'waiting'):
            current = list(groups[group])
            positions = [i for i, k in enumerate(current) if k in allowed]
            values = sorted((current[i] for i in positions), key=lambda k: rank.get(k, len(rank)))
            for i, key in zip(positions, values):
                current[i] = key
            orders.extend(current)
        new = [k for k in action['admit'] if action['placement'][k] == replica]
        orders.extend(sorted(new, key=lambda k: (rank.get(k, len(rank)), state['nodes'][k]['ready_at'], k)))
    action['order'] = orders
    action['native_projection'] = 'running-prefill slots, then actual waiting, then new admissions; decode/inflight immutable'
    return action


def reference(state, cfg):
    from joint18v2.actions import affinity
    action = resources({'family': 'reference', 'placement': affinity(state, state['ready']),
        'admit': list(state['ready']), 'order': [], 'caps': [cfg['base_cap']] * state['replicas']}, state, cfg)
    return project(action, state)


def baseline(state, cfg, method):
    if method in ('RR', 'SLL'):
        return project(adapted('routing').reference(state, {**cfg, 'wf_routing_policy': method}), state)
    if method == 'Ayo':
        return project(adapted('ayo_topology').reference(state, cfg), state)
    return reference(state, cfg)


def choose(state, model, cfg, deadline):
    if state.get('native_observation_errors'):
        return {'action': reference(state, cfg), 'evaluations': [], 'model_selection': False,
                'selected_cell': '00', 'native_fallback': state['native_observation_errors']}
    cfg = {**cfg, 'action_horizon_ms': state.get('action_horizon_ms', cfg['epoch_s'] * 1000)}
    selector = adapted('v5_selector')
    selector.dag = adapted('semantic_selector')
    from .native_scoring import next_batch
    selector.dag.next_batch = lambda s, a, m, r: next_batch(s, a, m, r, cfg)
    def supported_evaluate(*args, **kwargs):
        try:
            return adapted('engine').evaluate(*args, **kwargs)
        except ValueError as exc:
            if not str(exc).startswith('native_calibration_out_of_range:'):
                raise
            return {'rankable': False, 'complete': False, 'sum_remaining_jct_ms': None, 'reason': str(exc)}
    selector.evaluate = supported_evaluate
    original_proposals = selector.proposals
    selector.resources, selector.reference = resources, reference
    def proposals(s, m, c, base):
        return [(label, project(a, s), origin, affected)
                for label, a, origin, affected in original_proposals(s, m, c, base)]
    selector.proposals = proposals
    try:
        try:
            result = selector.choose_heterogeneous(state, model, cfg, 'D', deadline)
        except ValueError as exc:
            if not str(exc).startswith('native_calibration_out_of_range:'):
                raise
            result = {'action': reference(state, cfg), 'evaluations': [], 'model_selection': False,
                      'selected_cell': '00', 'native_fallback': [str(exc)]}
    finally:
        selector.proposals = original_proposals
    result.update(method='Ours', action_projection='native_safe_prefill_queues',
                  calibration_source=model.get('timing_target', 'frozen_native_engine_host_step'), native_cache_observed=True,
                  native_dag_equivalent=False, surrogate_is_measured_counterfactual=False)
    return result
