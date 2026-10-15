"""V4 reference-first DAG policy: isolate ordering from route changes.

This is an exploratory policy on the *current* real snapshot. It never uses
future outputs, realized JCTs, quality labels or performance from held-out runs.
The unchanged shared native Affinity (NOT published Preble) is the default.
It does not claim a physical non-regression guarantee.
"""
from __future__ import annotations
import copy
import time
from joint18v2.actions import resources
from joint18v2.common import action_key, validate_action
from joint18v2.search import reference
from core18v2.engine import evaluate
import semantic19.selector as dag

REVISION='reference-first-split-order-route-20261014'
# Fixed *exploratory* safety knobs. No selection of thresholds from this run.
MIN_SCORE_MARGIN_MS=70.0
MIN_SCORE_MARGIN_FRACTION=.015
MAX_NEW_CANDIDATES=3
MIN_SEARCH_REMAINING_S=.009
MIN_PRESSURE_DIFF=1.0


def _cache(s, k, r):
    """Computed native prefix blocks only; missing evidence remains UNKNOWN."""
    n=s['nodes'][k]
    h=n.get('prefix_hash')
    if not h:return None
    matches=[e for e in s.get('cache_evidence', [])
             if e.get('prefix_hash')==h and e.get('replica')==r and e.get('known')]
    if not matches:return None
    return max((int(e.get('tokens',0)) if e.get('state') in
               ('active_computed','reclaimable_computed') else 0) for e in matches)


def _ready_groups(s):
    ready=set(s['ready']);gs=set()
    for k in ready:
        for g in s['nodes'][k].get('compiled_groups',[]):
            present=tuple(x for x in g if x in ready)
            if len(present)>=2:gs.add(tuple(sorted(present)))
    return gs


def _action(base, s, cfg, *, place=None, order=None, family, proposal=None):
    """Isolate each intervention; keep FIFO and full admission unless testing ordering."""
    a=copy.deepcopy(base)
    if place is not None:a['placement']=dict(place)
    if order is not None:a['order']=list(order)
    else:a['order']=[]
    a['admit']=list(s['ready'])
    a['resource_held']=[]
    a['family']=family
    if proposal is not None:a['proposal']=proposal
    a=resources(a,s,cfg)
    validate_action(a,s,cfg)
    return a


def proposals(s,model,cfg,base):
    """Deterministic, bounded *online* options, no future/counterfactual readings.

    The previous V1 combined routing + order + sometimes bounded admission,
    making a cache miss or admission delay difficult to attribute.  V4 tests
    the mechanisms separately. Admission stays reference/FIFO; no intentional
    defer is added. Route alternatives only arise under observed imbalance.
    """
    if not s['ready'] or s['replicas']<2:return []
    P=s['pressure'];diff=max(P)-min(P)
    groups=_ready_groups(s)
    if diff<MIN_PRESSURE_DIFF and not groups:return []
    v1=dag.schedule_alternative(s,model,cfg,base)
    out=[];ready=s['ready']
    if groups and v1['order']:
        ordered=_action(base,s,cfg,order=v1['order'],family='v4_DAG_order_only',
                        proposal={'policy':'critical_path_order_without_relocation'})
        if action_key(ordered)!=action_key(base):
            out.append(('10',ordered,'order_only'))
    original=v1.get('proposal') or {}
    moved=list(original.get('moved_nodes') or [])
    if not moved or diff<MIN_PRESSURE_DIFF:return out[:MAX_NEW_CANDIDATES]
    if any(k not in ready for k in moved):return out[:MAX_NEW_CANDIDATES]
    srcs={base['placement'][k] for k in moved}
    if len(srcs)!=1:return out[:MAX_NEW_CANDIDATES]
    src=srcs.pop()
    # Use native load for BOTH trigger and destination, unlike V3's
    # P2C candidate which used a different HTTP outstanding count.
    cand=[]
    for r in range(s['replicas']):
        if r==src or P[r]>=P[src]:continue
        cache_pair=[(_cache(s,k,src),_cache(s,k,r)) for k in moved]
        # Do not invent zero-cache evidence; incomplete native certificate -> no move.
        if any(x is None or y is None for x,y in cache_pair):continue
        loss=sum(max(0,x-y) for x,y in cache_pair)
        gain=max(0.,P[src]-P[r])
        # A rank is a screening heuristic, NEVER the objective or a causal saving.
        rank=(gain,-loss,r)
        cand.append((rank,r,loss,gain))
    cand.sort(key=lambda x:(-x[0][0],-x[0][1],x[0][2]))
    for _rank,r,loss,gain in cand[:2]:
        place=dict(base['placement']);place.update({k:r for k in moved})
        a=_action(base,s,cfg,place=place,family='v4_DAG_route_only',
            proposal={'moved_nodes':moved,'source':src,'destination':r,
                'computed_cache_loss_tokens_proxy':loss,
                'observed_pressure_advantage_units':gain,
                'policy':'native_cache_and_current_pressure_no_future'})
        if action_key(a)!=action_key(base):
            label='20' if not any(x[2]=='route_only' for x in out) else '21'
            out.append((label,a,'route_only'))
    return out[:MAX_NEW_CANDIDATES]


def choose_workflow_robust(s,model,cfg,method,deadline=None):
    if method!='D' or cfg.get('base_cap')!=4096:
        raise ValueError('DAG-W is a fixed-4096 scheduling-only method')
    start=time.monotonic();deadline=deadline or start+cfg['planning_soft_limit_s']
    base=reference(s,cfg)
    result={'action':base,'evaluations':[],'selected_cell':'00',
        'model_selection':False,'decode_guard_rejections':0,
        'complete_distinct_evaluations':0,'deadline_reached':False,
        'v4_revision':REVISION,'v4_selected_deviation':False,
        'v4_reference_retained':True,'v4_candidate_generated':0,
        'v4_candidate_scored':0,'v4_predicted_gain_ms':0.,
        'nonregression_guaranteed':False}
    def finish():
        result['wall_ms']=1000*(time.monotonic()-start)
        result['deadline_reached']=time.monotonic()>=deadline
        validate_action(result['action'],s,cfg)
        return result
    if not s['ready'] or s['replicas']<2:return finish()
    # Fast gate: no eligible structural/pressure opportunity, identical Affinity action.
    options=proposals(s,model,cfg,base)
    unique=[];seen={action_key(base)}
    for label,a,origin in options:
        key=action_key(a)
        if key not in seen:
            seen.add(key);unique.append((label,a,origin))
    result['v4_candidate_generated']=len(unique)
    if not unique or time.monotonic()+MIN_SEARCH_REMAINING_S>=deadline:return finish()
    def do_score(label,a,origin,nleft):
        now=time.monotonic()
        if now+MIN_SEARCH_REMAINING_S>=deadline:return None
        slice_s=max(.002,(deadline-now)/max(1,nleft))
        stop=min(deadline,now+slice_s)
        z=evaluate(s,a,model,cfg,deadline=stop)
        rankable=bool(z.get('rankable'))
        result['evaluations'].append({'cell':label,'family':a['family'],
           'action':a,'complete':rankable,'cost_ms':z.get('sum_remaining_jct_ms') if rankable else None,
           'reason':z.get('reason'),'cpu_wall_ms':z.get('cpu_wall_ms'),
           'is_cross':False,'estimated_remaining_outputs':True,
           'score_is_real_counterfactual':False,'v4_origin':origin})
        if rankable:result['complete_distinct_evaluations']+=1
        return z if rankable else None
    z0=do_score('00',base,'native_affinity_reference',len(unique)+1)
    if z0 is None:
        result['v4_reference_unrankable']=True
        return finish()
    result['model_selection']=True
    initial=z0['sum_remaining_jct_ms'];best=initial
    # Reject small surrogate improvements relative to total unfinished-workflow JCT.
    margin=max(MIN_SCORE_MARGIN_MS,MIN_SCORE_MARGIN_FRACTION*initial)
    result['v4_required_gain_ms']=margin
    for idx,(label,a,origin) in enumerate(unique):
        z=do_score(label,a,origin,len(unique)-idx)
        if z is None:continue
        result['v4_candidate_scored']+=1
        bad=dag.guard(s,z0,z,base,a,model,cfg)
        if bad:
            result['decode_guard_rejections']+=1
            result.setdefault('v4_guard_details',{})[label]=bad
            continue
        cost=z['sum_remaining_jct_ms']
        if cost+margin<best:
            result.update(action=a,selected_cell=label,v4_reference_retained=False,
                          v4_selected_deviation=True)
            best=cost
    if result['v4_selected_deviation']:
        result['v4_predicted_gain_ms']=initial-best
    return finish()
