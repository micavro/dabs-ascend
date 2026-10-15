"""Ayo Algorithm 2 topology-aware batching *mechanism proxy* for native DABS.

Source: Tan et al., Ayo/Teola ASPLOS'25, arXiv:2407.00326v3, Algorithm 2.
It realizes per-query FIFO buckets and descending reverse-topological depth
WITHIN each bucket, and uses occupancy in estimated KV reservation blocks for
replica selection (paper Section 6 mentions occupied KV slots). The original
200ms engine controller applies a fixed 4096 budget and native resource guards.

NOT the entire Ayo framework: no Ray engine graph optimizer, dependency
pruning, partial prefilling, split decoding, exact engine-internal token-batch
formation, or proof of its published results. Do not label this full SOTA.
"""
from __future__ import annotations
import math
from collections import defaultdict

from joint18v2.common import validate_action

REVISION='Ayo-topology-algorithm2-controller-proxy-1'


def reverse_topological_depth(nodes):
    """Algorithm 2 Event 1; outputs depth zero, ancestors have larger depth.

    The snapshot can include completed-but-not-removed nodes from active DAGs.
    """
    children=defaultdict(list)
    for key,node in nodes.items():
        for parent in node['parents']:
            if parent in nodes:
                if nodes[parent]['wid']!=node['wid']:
                    raise ValueError('cross-query dependency in Ayo DAG')
                children[parent].append(key)
            else:
                raise ValueError('DAG parent absent from active workflow snapshot')
    depth={}; visiting=set()
    def calc(key):
        if key in depth:return depth[key]
        if key in visiting:raise ValueError('cycle in Ayo DAG')
        visiting.add(key)
        result=max((calc(k)+1 for k in children[key]),default=0)
        visiting.remove(key);depth[key]=result
        return result
    for key in nodes:calc(key)
    return depth


def topology_priority(s, keys):
    """Algorithm 2 Event 2: FIFO query buckets, descending depth per query."""
    ns=s['nodes'];keys=list(dict.fromkeys(keys))
    groups=defaultdict(list)
    for key in keys:
        if key not in ns:raise ValueError('unrecognized candidate node')
        groups[ns[key]['wid']].append(key)
    depth=reverse_topological_depth(ns)
    order=[]
    for wid in sorted(groups,key=lambda w:(min(ns[k].get('ready_at') or 0 for k in groups[w]),w)):
        order.extend(sorted(groups[wid],key=lambda k:(-depth[k],ns[k].get('ready_at') or 0,k)))
    return order


def reference(s,cfg):
    """Topology-aware controller order + KV-occupancy load proxy.

    All methods retain the SAME hard 4096/token and per-replica admission
    memory envelope. Ayo's engine batching rule is mapped to the available
    controller-level priority/admission surface, not injected into vLLM.
    """
    R=s['replicas'];ns=s['nodes']
    if cfg.get('base_cap')!=4096:raise ValueError('Ayo proxy requires native fixed 4096 cap')
    if not 1<=R<=4:raise ValueError('invalid replica count')
    candidates=topology_priority(s,s['ready'])
    reserved=list(s['reserved_blocks']);capacity=list(s['capacity_blocks'])
    pressure=list(s['pressure'])
    if not all(isinstance(x,int) and x>=0 for x in reserved):
        raise ValueError('invalid KV reservation estimates')
    if any(c<=0 for c in capacity):raise ValueError('invalid KV capacities')
    used=list(reserved)
    limit=[int(c*cfg['admission_reserve_fraction']) for c in capacity]
    counts=[sum(n['state']=='submitted' and n.get('replica')==r for n in ns.values()) for r in range(R)]
    placement={};need_blocks={};routing=[]
    for key in candidates:
        n=ns[key]
        need=math.ceil((n['prompt']+n.get('max_output_tokens',n['output']))/cfg['block_size'])
        if need<1:raise ValueError('invalid token work')
        # Ayo engine instances use occupied KV cache slots as the load metric.
        # Here reserved-block estimates are available instead of exact slots.
        # Do not credit unknown KV hits; do not use Preble cache-home affinity.
        dst=min(range(R),key=lambda r:(used[r]/capacity[r],pressure[r],r))
        placement[key]=dst;need_blocks[key]=need;used[dst]+=need
        routing.append({'node':key,'replica':dst,'kv_reserved_block_load_fraction':round((used[dst]-need)/capacity[dst],6)})
    # Prioritize *admission* under the EXACT same memory/sequence constraints
    # as native resources(), but in Algorithm 2 order, not FIFO-node order.
    remaining=list(reserved)
    active=[sum(n['state']=='submitted' and n.get('replica')==r for n in ns.values()) for r in range(R)]
    admitted=[];held=[]
    for key in candidates:
        dst=placement[key];need=need_blocks[key]
        if active[dst]>=cfg['max_num_seqs'] or remaining[dst]+need>limit[dst]:
            held.append(key)
        else:
            admitted.append(key);active[dst]+=1;remaining[dst]+=need
    order_candidates=[k for k,n in ns.items() if n['state'] in ('ready','submitted') and n.get('phase')!='decode']
    # Native queue priority is applied to ready / waiting-prefill nodes and
    # ongoing-prefill, not forcing reorder of running decode steps.
    order=topology_priority(s,order_candidates)
    action={'family':'Ayo_Alg2_TopologyProxy','placement':placement,
            'admit':admitted,'resource_held':held,'order':order,
            'caps':[4096]*R,
            'ayo_algorithm_2_proxy':True,
            'ayo_topological_admission':True,
            'ayo_engine_batch_slots_proxy':True,
            'ayo_exact_engine_batching':False,
            'ayo_actual_KV_occupancy_available':False,
            'ayo_load_metric':'normalized_conservative_reserved_KV_blocks',
            'ayo_depth_revision':REVISION,
            'ayo_routing_trace':routing}
    validate_action(action,s,cfg)
    return action
