"""Bounded complete-action refinement with an explicit result-handoff reserve.

A/D/B/Decoupled/Joint share state, candidate library, duration model and validity
rules. No partial objective is ranked. Joint uses the SAME coordinate rule as Decoupled, interleaves a complete cross
early, then spends remaining time on alternative schedules and local caps.
No score is a measured counterfactual and no real non-regression is certified.
"""
from __future__ import annotations
import copy,time
from .actions import affinity,resources,schedules
from .common import action_key,validate_action
from core18v2.engine import evaluate
METHODS=('A','D','B','Decoupled','Joint')
REVISION='stage18v2-selector-repair-1'

def reference(s,cfg):
    a=resources({'family':'reference','placement':affinity(s,s['ready']),
      'admit':list(s['ready']),'order':[],'caps':[cfg['base_cap']]*s['replicas']},s,cfg)
    validate_action(a,s,cfg);return a

def post_budget(s,a,cfg):
    """Complete vectors based on state AFTER the candidate admission/placement.

    Keep-current and base remain explicit. The local mixed/prefill composition
    proposes low/base/high vectors; it never declares which vector will win.
    All five legal levels remain reachable. No changes are proposed on replicas
    with no current admitted/submitted prefill. Subsequent state evolution is
    charged by the complete-workflow evaluator.
    """
    R=s['replicas'];pre=[0]*R;dec=[0]*R;dc=[0]*R
    cert={(e['prefix_hash'],int(e['replica'])):int(e.get('tokens',0))
          for e in s.get('cache_evidence',[]) if e.get('known',True)
          and e.get('state') in ('active_computed','reclaimable_computed')}
    for k,n in s['nodes'].items():
        if n['state']=='submitted':r=n['replica']
        elif k in a['admit']:r=a['placement'][k]
        else:continue
        if n.get('phase')=='decode':
            dec[r]+=1;dc[r]+=n['prompt']+n.get('output_done',0)
        else:
            # Logical eligibility, with observed cached overlap as a proposal hint.
            computed=int(n.get('computed',0))
            pre[r]+=max(1,n['prompt']-max(computed,min(n.get('share',0),cert.get((n.get('prefix_hash'),r),0))))
    active=[r for r in range(R) if pre[r]>0]
    focal=min(active,key=lambda r:(-pre[r],r)) if active else None
    current=list(s.get('current_caps',a['caps']));base=[cfg['base_cap']]*R
    vectors=[tuple(a['caps']),tuple(current),tuple(base)]
    protective=list(current);aggressive=list(current)
    for r in active:
        protective[r]=min(cfg['caps']) if dec[r] else cfg['base_cap']
        aggressive[r]=max(cfg['caps'])
    vectors.extend((tuple(protective),tuple(aggressive)))
    if focal is not None:
        # Intermediate legal levels are rotated AFTER complete low/base/high packets.
        levels=list(cfg['caps']);i=int(s.get('decision_index',0))%len(levels)
        for cap in levels[i:]+levels[:i]:
            v=list(current);v[focal]=cap;vectors.append(tuple(v))
    return list(dict.fromkeys(vectors)),{'resulting_prefill_query_proxy':pre,
       'resulting_decode':dec,'resulting_decode_context':dc,'focal':focal,
       'active_prefill_replicas':active,'post_placement_state':True}

def route_packet(s,model,cfg,current):
    full=schedules(s,model,cfg,current);i=int(s.get('decision_index',0))
    base=reference(s,cfg);packet=[base]
    partial=[a for a in full if a['family'].startswith('partial_join/')]
    # Preferred proper group split is always in the bounded packet. Alternate
    # subset sizes/destinations are covered by a deterministic rotation thereafter.
    if partial:packet.append(partial[0])
    rest=[a for a in full if a['family'] not in ('reference',) and a not in packet]
    if rest:packet.append(rest[i%len(rest)])
    if partial and len(partial)>1 and len(packet)<int(cfg.get('route_neighborhood_limit',4)):
        packet.append(partial[1+(i%(len(partial)-1))])
    elif not partial:
        ordered=next((a for a in full if a['family']=='dag_order'),None)
        if ordered is not None:packet.append(ordered)
    out={}
    for a in packet:out.setdefault(action_key(a),a)
    return list(out.values()),{'full_schedule_count':len(full),'partial_group_candidates_generated':len(partial),
       'packet_families':[a['family'] for a in out.values()],
       'shared_with_D_Decoupled_Joint':True}

def choose(s,model,cfg,method,deadline=None):
    begin=time.monotonic();deadline=begin+cfg['planning_soft_limit_s'] if deadline is None else deadline
    base=reference(s,cfg);records=[];memo={};coordinate_values={};best=copy.deepcopy(base);bestcost=None
    result={'action':base,'evaluations':records,'model_selection':False,'deadline_reached':False,
        'joint_cross_candidates_evaluated':0,'incomplete_not_ranked':0,'selector_revision':REVISION,
        'candidate_neighborhood':'shared partial-join-group + old complete families; post-placement cap packets',
        'worker_evaluation_deadline':deadline,'hard_realtime_guaranteed':False}
    def finish():
        result.update(action=best,wall_ms=1000*(time.monotonic()-begin),
            deadline_reached=time.monotonic()>=deadline,
            joint_cross_candidates_evaluated=sum(x['is_cross'] and x['complete'] for x in records),
            complete_distinct_evaluations=sum(x['complete'] for x in records),
            incomplete_not_ranked=sum(not x['complete'] for x in records),
            partial_group_candidates_evaluated=sum(x['family'].find('partial_join/')>=0 for x in records),
            partial_group_complete_evaluations=sum(x['family'].find('partial_join/')>=0 and x['complete'] for x in records),
            selected_partial_group='partial_join/' in best['family'],
            selected_cap_changes=sum(x!=y for x,y in zip(best['caps'],s.get('current_caps',base['caps']))),
            real_nonregression_guaranteed=False)
        validate_action(best,s,cfg);return result
    if method=='A':return finish()
    if method not in METHODS:raise ValueError('unknown method '+method)
    if not s['ready'] and not any(n['state']=='submitted' and n.get('phase')!='decode' for n in s['nodes'].values()):
        result['policy_equivalent_no_prefill']=True;return finish()
    def score(a,is_cross=False):
        nonlocal best,bestcost
        key=action_key(a)
        if key in memo:
            if not is_cross and memo[key] is not None:coordinate_values[key]=(memo[key],copy.deepcopy(a))
            return memo[key]
        if time.monotonic()>=deadline or len(records)>=cfg['candidate_limit']:return None
        validate_action(a,s,cfg)
        # A slow candidate cannot consume the entire packet's remaining CPU budget.
        # Incomplete evaluation gets no scalar score and cannot improve the incumbent.
        cap=min(deadline,time.monotonic()+float(cfg.get('per_candidate_slice_s',.018)))
        z=evaluate(s,a,model,cfg,deadline=cap)
        val=z['sum_remaining_jct_ms'] if z['rankable'] else None;memo[key]=val
        records.append({'family':a['family'],'action':copy.deepcopy(a),'cost_ms':val,
          'complete':z['complete'],'reason':z['reason'],'cpu_wall_ms':z['cpu_wall_ms'],
          'rollout_steps':z['rollout_steps'],'current_window_cost_ms':z['unfinished_area_to_horizon_ms'],
          'continuation_cost_ms':z['common_continuation_value_ms'],'cache_counts':z['cache_counters'],
          'workflow_completion_ms':z.get('workflow_prediction_ms'),'is_cross':bool(is_cross),
          'objective':'sum remaining completion times of all arrived unfinished workflows'})
        if val is not None:
            if not is_cross:coordinate_values[key]=(val,copy.deepcopy(a))
            tie=float(cfg.get('score_tie_ms',1e-6))
            better=bestcost is None or val<bestcost-tie
            if bestcost is not None and abs(val-bestcost)<=tie:
                current=s.get('current_caps',base['caps'])
                better=sum(c!=p for c,p in zip(a['caps'],current))<sum(c!=p for c,p in zip(best['caps'],current))
            if better:best,bestcost=copy.deepcopy(a),val
        return val
    refcost=score(base)
    if refcost is None:
        result['reference_unrankable']=True;return finish()
    result['model_selection']=True
    current=copy.deepcopy(base);current['family']='keep_current';current['caps']=list(s.get('current_caps',base['caps']))
    if method in ('B','Decoupled','Joint'):score(current)
    if time.monotonic()>=deadline:return finish()
    routes,coverage=route_packet(s,model,cfg,current);result.update(coverage)
    # Joint gets an early complete crossed packet rather than waiting until the
    # entire coordinate pass consumes the budget. Decoupled has the same FULL
    # deadline and candidates; it uses that budget for its coordinate sequence.
    # Memoized scores are shared within this decision only, never across states.
    if method=='Joint':
        primary=next((a for a in routes if a['family'].startswith('partial_join/')),None)
        if primary is not None:
            vectors,_=post_budget(s,primary,cfg)
            crossed=[v for v in vectors if list(v)!=current['caps']]
            if crossed:
                v=crossed[int(s.get('decision_index',0))%min(2,len(crossed))]
                a=copy.deepcopy(primary);a.update(family='joint/'+primary['family'],caps=list(v))
                score(a,True)
    if method=='D':
        for a0 in routes[1:]:
            a=copy.deepcopy(a0);a['caps']=list(base['caps']);score(a)
            if time.monotonic()>=deadline:break
    elif method=='B':
        vectors,feat=post_budget(s,base,cfg);result['budget_proposal_state']=feat
        # Common complete vectors first, then one intermediate budget in later epochs.
        for v in vectors:
            if action_key(dict(base,caps=list(v))) in memo:continue
            a=copy.deepcopy(base);a.update(family='budget_only',caps=list(v));score(a)
            if time.monotonic()>=deadline or len(records)>=6:break
    else:
        # Identical, finite coordinate pass for Decoupled and Joint: same route
        # packet, then novel cap vectors in the same order. Both receive the SAME full
        # deadline; Decoupled is not cut short to reserve time for another method.
        coordinate_stop=deadline
        coordinate=current;coordinate_cost=memo.get(action_key(current),refcost)
        for a0 in routes:
            if time.monotonic()>=coordinate_stop:break
            a=copy.deepcopy(a0);a['caps']=list(current['caps']);val=score(a)
            if val is not None and (coordinate_cost is None or val<coordinate_cost):coordinate,coordinate_cost=a,val
        vectors,feat=post_budget(s,coordinate,cfg);result['coordinate_budget_state']=feat
        novel=[v for v in vectors if action_key(dict(coordinate,caps=list(v))) not in memo]
        for v in novel:
            if time.monotonic()>=coordinate_stop:break
            a=copy.deepcopy(coordinate);a.update(family='sequential_budget/'+coordinate['family'],caps=list(v))
            score(a)
        if coordinate_values:
            cv,ca=min(coordinate_values.values(),key=lambda pair:pair[0])
            result['decoupled_action']=copy.deepcopy(ca);result['decoupled_cost_ms']=cv
        else:
            result['decoupled_action']=copy.deepcopy(base);result['decoupled_cost_ms']=refcost
        result['coordinate_scope']='same coordinate rule; completed prefix under this decision deadline, not guaranteed whole independent Decoupled run'
        if method=='Joint':
            # Evaluate crossed partial-group packets BEFORE spending time on extra
            # whole-group/queue variants. No candidate is chosen merely by its tag.
            alternatives=sorted(routes[1:],key=lambda a:not a['family'].startswith('partial_join/'))
            for a0 in alternatives:
                vectors,feat=post_budget(s,a0,cfg)
                vv=[v for v in vectors if list(v)!=current['caps'] and action_key(dict(a0,caps=list(v))) not in memo]
                for v in vv[:2]:
                    a=copy.deepcopy(a0);a.update(family='joint/'+a0['family'],caps=list(v))
                    score(a,True)
                    if time.monotonic()>=deadline:break
                if time.monotonic()>=deadline:break
    result.update(model_selection=True,predicted_reference_cost_ms=refcost,predicted_best_cost_ms=bestcost)
    return finish()
