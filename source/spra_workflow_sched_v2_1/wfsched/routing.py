"""Request-level baseline kernels on the shared native executor, not Ray/NGINX ports.

No DAG, prefix, token-length or future-output information is used by RR or P2C.
P2C load is submitted-but-not-returned HTTP requests, plus this decision's admitted
assignments. It is deliberately not the native D0 pressure (running + 2*waiting).
Speculative reference calls never consume a global random or round-robin cursor.
"""
from __future__ import annotations
import hashlib
import random


def pair_for(seed: int, sequence: int, replicas: int):
    if replicas < 1 or sequence < 0: raise ValueError('invalid routing state')
    if replicas == 1: return (0,)
    h = hashlib.sha256(f'{seed}/{sequence}'.encode()).digest()
    return tuple(random.Random(int.from_bytes(h[:16], 'big')).sample(range(replicas), 2))


def reference(s, cfg):
    from joint18v2.actions import resources
    from joint18v2.common import validate_action
    policy = cfg['wf_routing_policy']
    if policy not in ('RR','P2C'): raise ValueError('unknown baseline')
    R = s['replicas']
    ready = sorted(s['ready'], key=lambda k:(s['nodes'][k]['ready_at'], k))
    count = s['router_submitted_count']
    loads = list(s['router_ongoing'])
    if type(count) is not int or count < 0 or len(loads) != R:
        raise ValueError('invalid observed routing counts')
    if any(type(x) is not int or x < 0 for x in loads): raise ValueError('invalid ongoing count')
    placement = {}
    samples = []
    for i, key in enumerate(ready):
        if policy == 'RR':
            candidates = ((count+i) % R,)
        else:
            candidates = pair_for(int(cfg['wf_router_seed']), count+i, R)
        # The sampled pair ordering breaks equal loads without a low-index bias.
        r = min(candidates, key=lambda r:loads[r])
        samples.append({'node':key, 'sequence':count+i, 'candidates':list(candidates),
                        'candidate_loads':[loads[v] for v in candidates], 'chosen':r})
        placement[key] = r
        loads[r] += 1
    a = resources({'family':policy+'_FIFO', 'placement':placement, 'admit':list(ready),
                   'order':[], 'caps':[4096]*R}, s, cfg)
    # Keep FIFO prefix on resource refusal. This is a shared experiment admission
    # envelope, not a reproduction of Ray rejection/backoff/autoscaling machinery.
    admitted_set = set(a['admit'])
    prefix = []
    for key in ready:
        if key not in admitted_set: break
        prefix.append(key)
    a['admit'] = prefix
    a['resource_held'] = [k for k in ready if k not in set(prefix)]
    a['routing_trace'] = samples
    validate_action(a, s, cfg)
    return a
