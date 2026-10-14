"""Unified complete-action search and action-space ablations.
Candidates use a bounded complete-action neighborhood. Cost evaluation is in core18v2, with no optimistic terminal scalar.
It is not an exact physical simulator, an offline oracle, or a certified safety bound.
"""
from __future__ import annotations
import copy,time,math
from collections import defaultdict
from .common import action_key,validate_action,digest
from .model import predict

METHODS=("A","D","B","Decoupled","Joint")
def affinity(s,ready):
    R=s['replicas'];loads=list(s['pressure']);placement={}
    cache={}
    for e in s.get('cache_evidence',[]):
        if e.get('known',True) and e['state'] in ('active_computed','reclaimable_computed','resident_uncomputed','absent'):
            cache[e['prefix_hash'],int(e['replica'])]=int(e.get('tokens',0))
    for k in sorted(ready,key=lambda k:(s['nodes'][k]['ready_at'],k)):
        n=s['nodes'][k];h=n.get('prefix_hash');known=[cache.get((h,r)) for r in range(R)]
        if h and all(x is not None for x in known):
            best=max(known);options=[r for r in range(R) if known[r]==best]
        else:options=s.get('prefix_history',{}).get(h,[]) or list(range(R))
        r=min(options,key=lambda r:(loads[r],r));placement[k]=r;loads[r]+=1
    return placement

def remaining(n,r,model):
    if n["state"]=="done":return 0.
    if n.get("phase")=="decode":
        return max(0,n["output"]-n["output_done"])*predict(model,r,0,1,n["prompt"]+n["output_done"],0)
    pc=max(0,n["prompt"]-n.get("computed",0))
    return predict(model,r,pc,0,0,n.get("computed",0))+max(0,n["output"]-1)*predict(model,r,0,1,n["prompt"],0)

def structural(s,model,place):
    ns=s["nodes"];memo={}
    def finish(k):
        if k in memo:return memo[k]
        n=ns[k]
        if n["state"]=="done":v=0
        else:
            r=place.get(k,n.get("replica"))
            own=min(remaining(n,z,model) for z in range(s["replicas"])) if r is None else remaining(n,r,model)
            v=max((finish(p) for p in n["parents"]),default=0)+own
        memo[k]=v;return v
    for k in ns:finish(k)
    succ=defaultdict(list)
    for k,n in ns.items():
        for p in n["parents"]:succ[p].append(k)
    tail={}
    def t(k):
        if k not in tail:
            tail[k]=max((remaining(ns[c],place.get(c,ns[c].get("replica")) or 0,model)+t(c) for c in succ[k] if ns[c]["state"]!="done"),default=0)
        return tail[k]
    active=[k for k,n in ns.items() if n["state"] not in ("blocked","done")]
    heights={k:remaining(ns[k],place.get(k,ns[k].get("replica")) or 0,model)+t(k) for k in active}
    wfmax={}
    for k,v in heights.items():wfmax[ns[k]["wid"]]=max(wfmax.get(ns[k]["wid"],0),v)
    return memo,heights,{k:wfmax[ns[k]["wid"]]-v for k,v in heights.items()}

def resources(a,s,cfg):
    """Project onto a conservative reservation set before scoring and publication.
    No assumed cache saving in memory reservations. Native allocator remains the hard authority.
    """
    a=copy.deepcopy(a);R=s["replicas"];ns=s["nodes"];used=list(s.get("reserved_blocks",[0]*R))
    counts=[sum(n["state"]=="submitted" and n.get("replica")==r for n in ns.values()) for r in range(R)]
    limit=[int(v*cfg["admission_reserve_fraction"]) for v in s.get("capacity_blocks",[4096]*R)]
    a["resource_held"]=[]
    admitted=[]
    for k in sorted(a["admit"],key=lambda k:(ns[k]["ready_at"],k)):
        r=a["placement"][k];need=math.ceil((ns[k]["prompt"]+ns[k].get("max_output_tokens",ns[k]["output"]))/cfg["block_size"])
        if counts[r]>=cfg["max_num_seqs"] or used[r]+need>limit[r]:
            a["resource_held"].append(k)
        else:
            admitted.append(k);counts[r]+=1;used[r]+=need
    a["admit"]=admitted
    return a

def schedules(s,model,cfg,base):
    """Preserve the full reference, concentration, alternatives on all replicas, and admission variants."""
    ready=s["ready"];p=base["placement"];ns=s["nodes"]
    _,height,slack=structural(s,model,p)
    orderable=[k for k,n in ns.items() if n["state"] in ("ready","submitted") and n.get("phase")!="decode"]
    fifo=sorted(orderable,key=lambda k:(ns[k].get("ready_at") or 0,k))
    cp=sorted(orderable,key=lambda k:(slack.get(k,0),-height.get(k,0),ns[k].get("ready_at") or 0,k))
    out=[copy.deepcopy(base)]
    def add(label,pl,ad,order):
        out.append({"family":label,"placement":pl,"admit":ad,"order":order,"caps":list(base["caps"])})
    add("dag_order",dict(p),list(ready),cp)
    # Route the currently critical ready group, not a hardcoded branch label.
    if ready:
        focus=max(ready,key=lambda k:height.get(k,0));wid=ns[focus]["wid"]
        group=[k for k in ready if ns[k]["wid"]==wid and ns[k].get("prefix_hash")==ns[focus].get("prefix_hash")]
        for r in range(s["replicas"]):
            pl=dict(p);pl.update({k:r for k in group})
            add("group_to_"+str(r),pl,list(ready),cp)
        pl=dict(p);loads=list(s["pressure"])
        for k in sorted(ready,key=lambda k:-height.get(k,0)):
            r=min(range(s["replicas"]),key=lambda r:(loads[r],r))
            pl[k]=r;loads[r]+=1
        add("spread_frontier",pl,list(ready),cp)
        # Bounded, reconsidered admission: leave unsubmitted nodes in central ready state.
        aged={k for k in ready if s["now"]-ns[k]["ready_at"]>=cfg["max_admission_wait_s"]}
        ad=set(aged)
        for r in range(s["replicas"]):
            ks=[k for k in cp if k in ready and pl[k]==r]
            if ks:ad.add(ks[0])
        add("staged_spread",pl,sorted(ad),cp)
        ad=set(aged)
        for w in {ns[k]["wid"] for k in ready}:
            ks=[k for k in cp if k in ready and ns[k]["wid"]==w]
            if ks:ad.add(ks[0])
        add("staged_affinity",dict(p),sorted(ad),cp)
    unique={}
    for a in out:
        a=resources(a,s,cfg);validate_action(a,s,cfg);unique.setdefault(action_key(a),a)
    return list(unique.values())


# --- Selector repair: names below are internal mechanisms, not new paper methods. ---
_original_schedules = schedules


def join_groups(s, model, base):
    """Current-ready antichains sharing a real unfinished join and a prefix.

    The downstream tail ranks proposals only. It is NOT an independent objective:
    all decisions are evaluated using the common complete-workflow rollout.
    Neither branch labels, replica IDs nor measured future outcomes are features.
    """
    ns=s['nodes'];ready=set(s['ready'])
    _,height,slack=structural(s,model,base['placement'])
    predecessors={}
    def ancestors(k):
        if k not in predecessors:
            acc=set()
            for p in ns[k]['parents']:acc.add(p);acc.update(ancestors(p))
            predecessors[k]=acc
        return predecessors[k]
    result=[]
    for j,nj in ns.items():
        if nj['state']=='done' or len(nj['parents'])<2:continue
        groups=defaultdict(list)
        for k in ready & ancestors(j):
            n=ns[k]
            if n['wid']!=nj['wid']:continue
            groups[n.get('prefix_hash')].append(k)
        for h,ks in groups.items():
            if len(ks)<2:continue
            ks=sorted(ks,key=lambda k:(-height.get(k,0),slack.get(k,0),ns[k]['ready_at'],k))
            result.append({'join':j,'wid':nj['wid'],'prefix_hash':h,'nodes':ks,
                           'height_ms':{k:height.get(k,0) for k in ks},
                           'slack_ms':{k:slack.get(k,0) for k in ks},
                           'oldest_ready':min(ns[k]['ready_at'] for k in ks)})
    return sorted(result,key=lambda g:(-max(g['height_ms'].values()),g['oldest_ready'],g['join']))


def partial_join_candidates(s,model,cfg,base):
    """Move a proper subset of a ready join group together, keeping the rest at A.

    All nonempty nested critical subsets are proposed (not a fixed B3/B4 pair).
    A group of size m produces prefixes of sizes 1..m-1 in remaining-tail order.
    Every destination is possible. Proposal priority uses current load/cache;
    only the full objective can accept a move. Memory projection is unchanged.
    """
    _,height,slack=structural(s,model,base['placement'])
    orderable=[k for k,n in s['nodes'].items() if n['state'] in ('ready','submitted') and n.get('phase')!='decode']
    order=sorted(orderable,key=lambda k:(slack.get(k,0),-height.get(k,0),s['nodes'][k].get('ready_at') or 0,k))
    candidates=[]
    for g in join_groups(s,model,base)[:int(cfg.get('focus_join_limit',2))]:
        ks=g['nodes'];m=len(ks)
        # The size closest to half is tried first; remaining sizes are still available.
        sizes=sorted(range(1,m),key=lambda x:(abs(x-m/2),-x))
        destinations=sorted(range(s['replicas']),key=lambda r:(s['pressure'][r],r))
        for size in sizes:
            subset=ks[:size]
            for r in destinations:
                if all(base['placement'][k]==r for k in subset):continue
                a=copy.deepcopy(base);a['placement'].update({k:r for k in subset})
                a.update(family=f'partial_join/{g["join"]}/{size}/to_{r}',order=list(order),
                         proposal={'kind':'partial_join_group','join':g['join'],'moved_nodes':list(subset),
                         'retained_nodes':[k for k in ks if k not in subset], 'destination':r,
                         'prefix_hash':g['prefix_hash'],'remaining_tail_ms':g['height_ms'],
                         'is_prediction_not_future_measurement':True})
                a=resources(a,s,cfg);validate_action(a,s,cfg)
                candidates.append(a)
    return candidates


def schedules(s,model,cfg,base):
    """Same schedule library for D, Decoupled and Joint; all old families retained."""
    old=_original_schedules(s,model,cfg,base)
    partial=partial_join_candidates(s,model,cfg,base)
    combined=old[:1]+partial+old[1:]
    unique={}
    for a in combined:unique.setdefault(action_key(a),a)
    return list(unique.values())
