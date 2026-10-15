"""Conservative multi-policy DAG scheduler, V1-first by construction.

Exploratory policy, NOT guaranteed to improve real GPU JCT. All candidate
scoring is on current snapshot only, with original decode guards and resource
checks. Zero modifications to the native model-serving containers.
"""
from __future__ import annotations
import time
import semantic19.selector as semantic
from core18v2.engine import evaluate
from joint18v2.common import action_key, validate_action
from joint18v2.search import reference
from .ours_v2_selector import second_destination
from .routing import reference as load_policy_action

# Frozen exploratory constants, never adapt using current experiment outcomes.
MIN_SCORE_GAIN_MS = 40.0
MIN_SCORE_GAIN_FRACTION = 0.008
MIN_REMAINING_SEARCH_S = 0.008
PRESSURE_SKEW_TRIGGER = 0.25

def choose_adaptive(snapshot, model, cfg, method, deadline=None):
    if method!='D' or cfg.get('base_cap')!=4096:
        raise ValueError('DAG-A needs the original fixed 4096 D planner')
    start=time.monotonic()
    deadline = deadline if deadline is not None else start+cfg['planning_soft_limit_s']
    # Invariant: V1 gets its full, unchanged original decision BEFORE exploring.
    result=semantic.choose(snapshot,model,cfg,'D',deadline)
    incumbent=result['action']
    result.update(v3_revision='v1_first_two_opportunistic_candidates-1',
        v3_v1_fallback_kept=True,v3_selected_extra_candidate=False,
        v3_additional_generated=0,v3_additional_scored=0,v3_policy='V1',
        v3_min_score_gain_ms=MIN_SCORE_GAIN_MS,
        v3_min_score_gain_fraction=MIN_SCORE_GAIN_FRACTION)
    def done():
        result['wall_ms']=(time.monotonic()-start)*1000.
        result['deadline_reached']=time.monotonic()>=deadline
        validate_action(result['action'],snapshot,cfg)
        return result
    if (snapshot['replicas']<2 or not snapshot['ready'] or
            not result.get('model_selection') or result.get('reference_unrankable') or
            time.monotonic()+MIN_REMAINING_SEARCH_S >= deadline):
        return done()
    candidates=[]
    base=reference(snapshot,cfg)
    # P2C-style action gives an independently constructed load-balancing route.
    # Do not use performance measured at the end of this workload.
    pressures=snapshot['pressure']
    scale=max(1.0,sum(abs(x) for x in pressures)/len(pressures))
    if (max(pressures)-min(pressures))/scale >= PRESSURE_SKEW_TRIGGER:
        candidate=load_policy_action(snapshot,dict(cfg,wf_routing_policy='P2C'))
        candidates.append(('30',candidate,'current_snapshot_p2c_load'))
    if snapshot['replicas']>=3:
        original=semantic.schedule_alternative(snapshot,model,cfg,base)
        extra=second_destination(snapshot,model,cfg,base,original)
        if extra is not None:
            candidates.append(('20',extra,'second_destination'))
    dedup=[];seen={action_key(incumbent)}
    for label,action,description in candidates:
        key=action_key(action)
        if key not in seen:
            seen.add(key);dedup.append((label,action,description))
    result['v3_additional_generated']=len(dedup)
    if not dedup or time.monotonic()+MIN_REMAINING_SEARCH_S>=deadline:
        return done()
    # Re-evaluate the FULL selected V1 action. No score from the prior V1
    # evaluation is silently reused after a changed time budget.
    per=max(.001,(deadline-time.monotonic())/(len(dedup)+1))
    z0=evaluate(snapshot,incumbent,model,cfg,deadline=min(deadline,time.monotonic()+per))
    if not z0.get('rankable'):
        result['v3_reference_unrankable']=True
        return done()
    baseline=z0['sum_remaining_jct_ms']
    result['v3_incumbent_estimated_jct_sum_ms']=baseline
    margin=max(MIN_SCORE_GAIN_MS,MIN_SCORE_GAIN_FRACTION*baseline)
    result['v3_required_improvement_ms']=margin
    chosen='V1';best=baseline;chosen_action=incumbent
    for label,action,description in dedup:
        if time.monotonic()+MIN_REMAINING_SEARCH_S>=deadline:break
        z=evaluate(snapshot,action,model,cfg,deadline=deadline)
        valid=bool(z.get('rankable'))
        cost=z.get('sum_remaining_jct_ms') if valid else None
        entry={'cell':label,'family':action['family'],'action':action,
               'complete':valid,'cost_ms':cost,'reason':z.get('reason'),
               'cpu_wall_ms':z.get('cpu_wall_ms'),'is_cross':False,
               'estimated_remaining_outputs':True,'score_is_real_counterfactual':False,
               'v3_action_origin':description}
        result['evaluations'].append(entry)
        if not valid:continue
        result['v3_additional_scored']+=1
        result['complete_distinct_evaluations']+=1
        violations=semantic.guard(snapshot,z0,z,incumbent,action,model,cfg)
        if violations:
            result['decode_guard_rejections']+=1
            result.setdefault('v3_guard_details',{})[label]=violations
            continue
        if cost + margin < best:
            chosen=label;chosen_action=action;best=cost
    if chosen!='V1':
        result.update(action=chosen_action,selected_cell=chosen,
            v3_selected_extra_candidate=True,
            v3_v1_fallback_kept=False,v3_policy=chosen,
            v3_predicted_gain_vs_v1_ms=baseline-best)
    return done()
