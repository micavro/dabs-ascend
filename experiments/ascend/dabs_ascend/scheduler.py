"""Original DAG proposal/scorer with an explicit HTTP-only action projection.

No native queue, cache residency, exact output progress or GPU model is invented.
All methods use the same client state and logical admission envelope.
"""
from __future__ import annotations

import copy
import math
import time

from .common import source_paths

source_paths()
from joint18v2.common import validate_action
from joint18v2.search import reference as affinity_reference
from wfsched.routing import reference as routing_reference

METHODS = ("RR", "P2C", "Affinity", "DAG-Ascend")


def snapshot(nodes, history, replicas, cfg, index, now=None):
    now = time.monotonic() if now is None else now
    arrived = {n["wid"] for n in nodes.values() if n.get("arrival") is not None}
    completed = {w for w in arrived if all(n["state"] == "done" for n in nodes.values() if n["wid"] == w)}
    active = {k: {field: copy.deepcopy(value) for field, value in n.items()
                  if field not in ("tokens", "stream", "parent_response_bindings", "resolved_question")}
              for k, n in nodes.items() if n["wid"] in arrived - completed}
    loads = [sum(n["state"] == "submitted" and n["replica"] == r for n in active.values()) for r in range(replicas)]
    block_size = cfg['block_size']
    reservation = [sum(math.ceil((n["prompt"] + n["max_output_tokens"]) / block_size)
                       for n in active.values() if n["state"] == "submitted" and n["replica"] == r)
                   for r in range(replicas)]
    capacity = math.ceil(math.ceil(cfg['max_model_len'] / block_size) * cfg["max_num_seqs"] / cfg["admission_reserve_fraction"])
    return {"nodes": active, "ready": [k for k, n in active.items() if n["state"] == "ready"],
            "replicas": replicas, "now": now, "pressure": loads,
            "reserved_blocks": reservation, "capacity_blocks": [capacity] * replicas,
            "capacity_source": "logical client in-flight token envelope; NOT physical KV capacity",
            'resource_units': {'block_size_tokens': block_size, 'context_limit_tokens': cfg['max_model_len'],
                               'surrogate_step_budget_tokens': cfg['base_cap'],
                               'client_max_inflight': cfg['max_num_seqs'],
                               'server_max_num_seqs': None, 'physical_kv_capacity_blocks': None},
            'pressure_source': 'client_inflight_proxy; native running+2*waiting is unavailable',
            'native_pressure': None, 'native_state_ages_s': None,
            "prefix_history": copy.deepcopy(history), "prefix_certificates": {}, "cache_evidence": [],
            "queue_order": {str(r): sorted((k for k, n in active.items() if n["state"] == "submitted" and n["replica"] == r),
                                           key=lambda k: active[k]["start"]) for r in range(replicas)},
            "current_caps": [cfg['base_cap']] * replicas, "inflight_batches": [], "decision_index": index,
            "state_ages_s": [0.] * replicas,
            "state_source": "client arrival/submission/first-text/completion observations; native phase/progress unknown",
            "cache_unknown_count": sum(bool(n.get("prefix_hash")) for n in active.values()),
            "router_submitted_count": sum(n.get("start") is not None for n in nodes.values()),
            "router_ongoing": loads}


def baseline(state, cfg, method):
    if method in ("RR", "P2C"):
        return routing_reference(state, {**cfg, "wf_routing_policy": method})
    return affinity_reference(state, cfg)


def plan_dag(state, model, cfg, deadline):
    # Executed in one isolated planning process. Original source files stay unchanged.
    import semantic19.selector as semantic
    from wfsched.admission import schedule_alternative

    original = semantic.schedule_alternative

    def portable_alternative(s, m, c, base):
        candidate = schedule_alternative(s, m, c, base)
        # The public HTTP API cannot reorder prefills already inside the engine.
        # Remove ordering BEFORE scoring, rather than scoring an unexecutable action.
        candidate["order"] = []
        candidate["ascend_action_projection"] = "routing_and_bounded_admission_only"
        validate_action(candidate, s, c)
        return candidate

    semantic.schedule_alternative = portable_alternative
    try:
        result = semantic.choose(state, model, cfg, "D", deadline)
    finally:
        semantic.schedule_alternative = original
    if result["action"]["order"] or result["action"]["caps"] != [4096] * state["replicas"]:
        raise ValueError("HTTP port produced an unsupported action")
    result["method"] = "DAG-Ascend"
    result["native_dag_equivalent"] = False
    return result


def planner_ready():
    # Import in the child BEFORE measured conditions, including the first DAG turn.
    import semantic19.selector
    import wfsched.admission
    return True
