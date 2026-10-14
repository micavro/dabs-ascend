"""Pure non-overlapping timing helpers for the integrated executor's audit.
Interval overlap is not removable causal overhead.
"""
from __future__ import annotations
import math


def next_absolute_tick(origin: float, period: float, now: float) -> float:
    if period <= 0 or now < origin:
        raise ValueError('absolute cadence parameters')
    return origin + (math.floor((now - origin) / period + 1e-10) + 1) * period


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError('negative interval')
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def queue_partition(ready: float, submit: float, decision_intervals) -> dict:
    if submit < ready:
        raise ValueError('submit precedes readiness')
    overlap = sum(max(0., min(submit, b) - max(ready, a)) for a, b in union(decision_intervals))
    return {'total': submit - ready, 'in_decision': overlap,
            'outside_decision': submit - ready - overlap,
            'interpretation': 'interval overlap, not pure solver time or removable causal effect'}
