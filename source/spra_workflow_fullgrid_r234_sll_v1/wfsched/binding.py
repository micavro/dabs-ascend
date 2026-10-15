"""Scoped in-process bindings; never edit the installed native source files."""
from __future__ import annotations
from contextlib import contextmanager
from .common import METHODS

@contextmanager
def bindings(method):
    if method not in METHODS and method not in ('WARMUP','DAG-V2','Preble'): raise ValueError('unsupported method')
    import joint18v2.client as live
    from typed19.transport import stream
    from .tail_check import compare_iteration
    from .worker import process_plan
    from .routing import reference
    saved = (live.reference, live.make_snapshot, live.process_plan, live.stream, live.compare_iteration)
    original_snapshot = saved[1]

    def snapshot(nodes, *args, **kwargs):
        s = original_snapshot(nodes, *args, **kwargs)
        # Use full condition nodes, not just the unfinished-workflow subset.
        s['router_submitted_count'] = sum(n.get('start') is not None for n in nodes.values())
        s['router_ongoing'] = [sum(n['state']=='submitted' and n.get('replica')==r
                                   for n in nodes.values()) for r in range(s['replicas'])]
        return s

    live.stream = stream
    live.compare_iteration = compare_iteration
    live.process_plan = process_plan
    if method in ('RR','P2C','SLL'):
        # Only real baselines replace the native affinity reference. The
        # baseline's routing.reference() intentionally rejects DAG-A.
        live.reference = reference
        live.make_snapshot = snapshot
    elif method == 'Ayo':
        # Ayo topology-aware batching ALGORITHM 2 controller-level proxy;
        # no online JCT search, no modified vLLM, no cache-affinity oracle.
        from .ayo_topology import reference as ayo_reference
        live.reference = ayo_reference
        live.make_snapshot = snapshot
    elif method == 'DAG-A':
        # DAG-A must retain the original native reference for BOTH controller
        # fallbacks and V1 scoring. It still requires observed router counters
        # in snapshots for its optional P2C pressure candidate.
        live.make_snapshot = snapshot
    try:
        yield
    finally:
        live.reference, live.make_snapshot, live.process_plan, live.stream, live.compare_iteration = saved
