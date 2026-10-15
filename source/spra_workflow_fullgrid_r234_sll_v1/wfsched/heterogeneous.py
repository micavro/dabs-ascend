"""Predeclared *nonsemantic* hetero-DAG and popularity-skew workload.

All input DAGs come from frozen 64 public MuSiQue packets; graph edges are not
edited. Group totals are identical in the balanced/skew variants, so node
counts, graph-shape mix and predeclared output budgets match exactly.
Only the packet popularity distribution changes. No measured runtime numbers,
quality labels, or future GPU states enter selection.
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict
from .common import digest

# Six naturally occurring graph families, all independently represented in
# the original 64-question public pool.  The per-family totals sum to 64.
# The budget is fixed per *node* of that graph family for all 5 policies.
# 3584 + max(output)=128 < 4096; exactly equal node work within each condition.
FAMILIES={
    '-;0':         {'count':8, 'output':16},   # 2-node serial
    '-;0;1':       {'count':12,'output':64},   # 3-node serial
    '-;-;0,1':     {'count':12,'output':32},   # two roots then join
    '-;0;1;2':     {'count':8, 'output':128},  # 4-node serial
    '-;0;-;1,2':   {'count':12,'output':48},   # branch and join
    '-;-;0,1;2':   {'count':12,'output':64},   # two roots, join then tail
}
N_WORKFLOWS=64
SELECTION_REVISION='six-original-DAG-families-v1-fixed-popularity-pair'


def output_budget(shape):
    if shape not in FAMILIES:raise ValueError('unknown original graph shape: '+str(shape))
    return FAMILIES[shape]['output']


def _weights(total,profile):
    if profile=='balanced':
        assert total%4==0
        return [total//4]*4
    if profile=='skew':
        return [5,1,1,1] if total==8 else [7,3,1,1]
    raise ValueError('unknown predeclared popularity profile')


def select_packets(original,profile,seed=510140):
    """Select four example prompts per graph family by lexicographic ID.

    A fixed shared DAG-family arrival order is then populated with either
    balanced or Zipf-like (not a fitted Zipf exponent) popularity counts.
    No post-hoc changes, reruns, or shuffled graph lengths are permitted.
    """
    if profile not in ('balanced','skew'):raise ValueError('profile')
    if len(original)!=64 or len({p['id'] for p in original})!=64:
        raise ValueError('original 64 questions required')
    grouped=defaultdict(list)
    for p in original:
        shape=p['graph']['shape']
        if shape not in FAMILIES:raise ValueError('unexpected graph family')
        if len(p['steps'])!=p['graph']['hop_count']:raise ValueError('graph inconsistent')
        grouped[shape].append(p)
    for shape in FAMILIES:
        if len(grouped[shape])<4:raise ValueError('not enough unselected source examples in '+shape)
    sources={shape:sorted(grouped[shape],key=lambda p:p['id'])[:4] for shape in FAMILIES}
    shapes=[shape for shape,spec in FAMILIES.items() for _ in range(spec['count'])]
    # SAME shape sequence for balanced/skew; only popularity changes.
    random.Random(int(seed)).shuffle(shapes)
    queues={}
    for si,(shape,spec) in enumerate(FAMILIES.items()):
        counts=_weights(spec['count'],profile)
        q=[src for src,num in zip(sources[shape],counts) for _ in range(num)]
        random.Random(int(seed)+131*si).shuffle(q)
        queues[shape]=q
    selected=[]
    for shape in shapes:selected.append(queues[shape].pop())
    if len(selected)!=N_WORKFLOWS or any(queues.values()):raise AssertionError('count mismatch')
    assert Counter(p['graph']['shape'] for p in selected)==Counter({k:v['count'] for k,v in FAMILIES.items()})
    src_counts=Counter(p['id'] for p in selected)
    shape_counts=Counter(p['graph']['shape'] for p in selected)
    token_work=sum(len(p['steps'])*output_budget(p['graph']['shape']) for p in selected)
    descriptor={'revision':SELECTION_REVISION,'profile':profile,'seed':seed,
         'shape_histogram':dict(sorted(shape_counts.items())),
         'per_shape_output_tokens':{k:v['output'] for k,v in FAMILIES.items()},
         'total_declared_output_tokens':token_work,
         'source_packet_counts':dict(sorted(src_counts.items())),
         'source_packet_sequence':[p['id'] for p in selected],
         'public_source_examples_only':True,'semantics':False,
         'graph_edges_edited':False,'dataset_test_labels_accessed':False}
    descriptor['manifest_digest']=digest(descriptor)
    return selected,descriptor
