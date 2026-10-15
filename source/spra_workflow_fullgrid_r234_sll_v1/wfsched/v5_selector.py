"""V5 independent cache-pressure-DAG opportunity scheduler (non-adaptive batching).

Reference-first: original native Affinity/Preble-style policy is exactly the
fallback action. Candidates use only current cache evidence, replica pressure,
compiled DAG and known token budgets.  No measured JCT is used at runtime.
The full native-window surrogate remains a fallible MODEL; no GPU no-regression
certificate is implied.  Pure order and pure route candidates are asserted.
"""
from __future__ import annotations
import copy
import time
from joint18v2.actions import resources
from joint18v2.common import action_key, validate_action
from joint18v2.search import reference
from core18v2.engine import evaluate
import semantic19.selector as dag

REVISION='DAG-H-reference-first-pure-actions-local-margin-v1'
MAX_NEW_CANDIDATES=3
MIN_PRESSURE_GAP=.75
MIN_REMAINING_S=.009
MIN_GAIN_MS=35.0
RELATIVE_GAIN_FOR_AFFECTED_WF=.025
UNCERTAIN_TARGET_EXTRA_MS=35.0


def cache_tokens(snapshot,key,replica):
    """Returns None if unknown; zero ONLY for explicit known absent state."""
    h=snapshot['nodes'][key].get('prefix_hash')
    if not h:return None
    eligible=[e for e in snapshot.get('cache_evidence',[])
              if e.get('prefix_hash')==h and int(e.get('replica',-1))==replica
              and e.get('known')]
    if not eligible:return None
    return max((int(e.get('tokens',0)) if e.get('state') in
        ('active_computed','reclaimable_computed') else 0) for e in eligible)


def project(base,s,cfg,*,new_order=None,new_placement=None,family,proposal):
    """Reproject SAME full reference admission intent, never clear routing order.

    A route-only candidate that would implicitly change the accepted/held
    ready-node sets is REJECTED. The action must not smuggle in admission,
    ordering or batching changes. This fixes V4's coupled-action bug.
    """
    a=copy.deepcopy(base)
    a['admit']=list(s['ready'])
    a['resource_held']=[]
    if new_order is not None:a['order']=list(new_order)
    if new_placement is not None:a['placement']=dict(new_placement)
    a['family']=family;a['proposal']=proposal
    a=resources(a,s,cfg)
    if set(a['admit'])!=set(base['admit']) or set(a['resource_held'])!=set(base['resource_held']):
        return None
    if new_placement is not None and a['order']!=base['order']:
        raise ValueError('route-only unexpectedly modified FIFO/order')
    if new_order is not None and a['placement']!=base['placement']:
        raise ValueError('order-only unexpectedly modified placement')
    if a['caps']!=base['caps']:
        raise ValueError('candidate unexpectedly modified per-replica budget')
    validate_action(a,s,cfg)
    return a


def proposals(s,model,cfg,base):
    if not s['ready'] or s['replicas']<2:return []
    P=list(s['pressure']);ready=list(s['ready']);ns=s['nodes']
    ordered,tail=dag.tail_values(s,model,base['placement'])
    options=[]
    # Exactly isolate critical-path queue ordering from route/admission effects.
    groups=set()
    for k in ready:
        for group in ns[k].get('compiled_groups',[]):
            present=tuple(sorted(q for q in group if q in ready))
            if len(present)>1:groups.add(present)
    if groups:
        focus_order=sorted([k for k,n in ns.items() if n['state'] in ('ready','submitted') and n.get('phase')!='decode'],
            key=lambda k:(-tail.get(k,0.),ns[k].get('ready_at') or 0,k))
        ordered_action=project(base,s,cfg,new_order=focus_order,
            family='v5_critical_path_order_only',
            proposal={'kind':'order_only','compiled_DAG_used':True,'no_route_change':True})
        if ordered_action is not None and action_key(ordered_action)!=action_key(base):
            options.append(('10',ordered_action,'order_only',set(ns[k]['wid'] for k in ready)))
    if len(options)>=MAX_NEW_CANDIDATES:return options

    # A reference route on a busy cache-home is a potential global DAG conflict.
    # Unlike V4, a known cache miss is a legal target, and UNKNOWN remains
    # unknown (charged pessimistic 0 cache credit plus an uncertainty margin).
    focused=sorted(ready,key=lambda k:(-tail.get(k,0.),ns[k].get('ready_at') or 0,k))
    routes=[]
    for k in focused:
        src=base['placement'][k]
        cached_src=cache_tokens(s,k,src)
        if cached_src is None:continue # no invented source-cache certificate
        for dst in range(s['replicas']):
            if dst==src:continue
            gap=float(P[src])-float(P[dst])
            if gap<MIN_PRESSURE_GAP:continue
            observed_dst=cache_tokens(s,k,dst)
            # Conservative target assumption: UNKNOWN gets no cache credit.
            target_credit=0 if observed_dst is None else observed_dst
            cache_loss=max(0,cached_src-target_credit)
            gain_proxy=gap
            # Topological criticality breaks pressure ties; not an outcome label.
            rank=(-gain_proxy,-tail.get(k,0.),cache_loss,k,dst)
            routes.append((rank,k,src,dst,gap,cache_loss,observed_dst is None))
    routes.sort(key=lambda z:z[0]);seen=set()
    for _rank,k,src,dst,gap,loss,unknown in routes:
        if len(options)>=MAX_NEW_CANDIDATES:break
        if k in seen:continue
        seen.add(k)
        placement=dict(base['placement']);placement[k]=dst
        route_action=project(base,s,cfg,new_placement=placement,
            family='v5_critical_path_route_only',
            proposal={'kind':'route_only','moved_nodes':[k], 'source':src,'destination':dst,
                'known_source_cache_tokens':cache_tokens(s,k,src),
                'conservative_cache_loss_tokens':loss,
                'destination_cache_unknown':unknown,
                'observed_pressure_advantage_units':gap,'no_order_or_admission_change':True})
        if route_action is not None and action_key(route_action)!=action_key(base):
            label='20' if not any(o[2]=='route_only' for o in options) else '21'
            options.append((label,route_action,'route_only',{ns[k]['wid']}))
    return options[:MAX_NEW_CANDIDATES]


def choose_heterogeneous(s,model,cfg,method,deadline=None):
    if method!='D' or cfg['base_cap']!=4096:
        raise ValueError('DAG-H supports scheduling only with the fixed native 4096 budget')
    start=time.monotonic();deadline=deadline or start+cfg['planning_soft_limit_s']
    base=reference(s,cfg)
    out={'action':base,'evaluations':[],'selected_cell':'00','model_selection':False,
        'decode_guard_rejections':0,'complete_distinct_evaluations':0,
        'v5_revision':REVISION,'v5_reference_retained':True,
        'v5_selected_deviation':False,'v5_candidate_generated':0,
        'v5_candidate_scored':0,'v5_gain_model_ms':0.,
        'nonregression_guaranteed':False,'calibration_adopted':False}
    def finish():
        out['wall_ms']=(time.monotonic()-start)*1000
        out['deadline_reached']=time.monotonic()>=deadline
        validate_action(out['action'],s,cfg)
        return out
    if not s['ready'] or s['replicas']<=1 or time.monotonic()+MIN_REMAINING_S>=deadline:
        return finish()
    opts=proposals(s,model,cfg,base);unique=[];seen={action_key(base)}
    for item in opts:
        key=action_key(item[1])
        if key not in seen:seen.add(key);unique.append(item)
    out['v5_candidate_generated']=len(unique)
    if not unique or time.monotonic()+MIN_REMAINING_S>=deadline:return finish()
    def score(label,a,origin,nleft):
        now=time.monotonic()
        if now+MIN_REMAINING_S>=deadline:return None
        stop=min(deadline,now+max(.002,(deadline-now)/max(1,nleft)))
        z=evaluate(s,a,model,cfg,deadline=stop)
        good=bool(z.get('rankable'))
        out['evaluations'].append({'cell':label,'family':a['family'],'action':a,
            'complete':good,'cost_ms':z.get('sum_remaining_jct_ms') if good else None,
            'reason':z.get('reason'),'cpu_wall_ms':z.get('cpu_wall_ms'),
            'v5_origin':origin,'is_cross':False,'score_is_real_counterfactual':False,
            'estimated_remaining_outputs':True})
        if good:out['complete_distinct_evaluations']+=1
        return z if good else None
    z0=score('00',base,'native_reference',len(unique)+1)
    if z0 is None:
        out['v5_reference_unrankable']=True
        return finish()
    out['model_selection']=True
    baseline=z0['sum_remaining_jct_ms'];best=baseline
    per_wf=z0.get('workflow_prediction_ms') or {}
    for j,(label,a,origin,affected) in enumerate(unique):
        z=score(label,a,origin,len(unique)-j)
        if z is None:continue
        out['v5_candidate_scored']+=1
        violations=dag.guard(s,z0,z,base,a,model,cfg)
        if violations:
            out['decode_guard_rejections']+=1
            out.setdefault('v5_guard_details',{})[label]=violations
            continue
        impacted=max((float(per_wf.get(w,0.)) for w in affected),default=0.)
        threshold=max(MIN_GAIN_MS,RELATIVE_GAIN_FOR_AFFECTED_WF*impacted)
        if a.get('proposal',{}).get('destination_cache_unknown'):
            threshold+=UNCERTAIN_TARGET_EXTRA_MS
        # Objective is still complete global summed workflow JCT; the margin
        # reflects affected-WF uncertainty, not 1.5% of ALL remaining workflows.
        cost=z['sum_remaining_jct_ms']
        if cost+threshold<best:
            best=cost
            out.update(action=a,selected_cell=label,
                       v5_reference_retained=False,v5_selected_deviation=True,
                       v5_required_gain_ms=threshold,
                       v5_affected_workflows=sorted(affected),
                       v5_selected_cache_loss_tokens=a.get('proposal',{}).get('conservative_cache_loss_tokens'))
    if out['v5_selected_deviation']:
        out['v5_gain_model_ms']=baseline-best
    return finish()
