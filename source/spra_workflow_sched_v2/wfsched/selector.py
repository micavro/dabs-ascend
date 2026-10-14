"""D0 fixed-budget fast path; no new scheduling heuristic.

Derived from the uploaded semantic19/selector.py choose(). Only budget candidate
construction is removed. No source file in the installed native project is edited.
Under real deadlines action traces need not be identical to the old version.
"""
from __future__ import annotations
import copy
import time
from semantic19.selector import schedule_alternative, guard
from joint18v2.common import action_key, validate_action
from joint18v2.search import reference
from core18v2.engine import evaluate

def choose_lite(s,model,cfg,method,deadline=None):
    if method != 'D' or cfg.get('base_cap') != 4096:
        raise ValueError('DAG-Lite accepts fixed-4096 D only')
    begin=time.monotonic();deadline=deadline or begin+cfg['planning_soft_limit_s'];base=reference(s,cfg)
    result={'action':base,'evaluations':[],'model_selection':False,'wall_ms':0.,'joint_cross_candidates_evaluated':0,
        'complete_distinct_evaluations':0,'paired_packet_attempted':False,'paired_packet_complete':False,
        'decode_guard_rejections':0,'deadline_reached':False,'knowledge':'known native DAG; unknown future content/output length/cache',
        'tpot_nonregression_guaranteed':False,'hard_realtime_guaranteed':False}
    def done():
        result['wall_ms']=(time.monotonic()-begin)*1000;result['deadline_reached']=time.monotonic()>=deadline
        validate_action(result['action'],s,cfg);return result
    if method=='A':return done()
    if not s['ready'] and not any(n['state']=='submitted' and n.get('phase')!='decode' for n in s['nodes'].values()):
        result.update(no_state_opportunity=True,policy_equivalent_no_prefill=True);return done()
    schedule=base if method=='B' else schedule_alternative(s,model,cfg,base)
    result['known_dag_group_proposed']=bool(schedule.get('proposal',{}).get('compiled_native_group'))
    result['known_dag_alternative_generated']=action_key(schedule)!=action_key(base)
    # D never scores or executes 01/11. Omit only this unreachable budget work.
    # Keep proposal, scorer, guards, deadline splitting and 00/10 tie rule intact.
    capvec,local=list(base['caps']),[]
    cells={'00':copy.deepcopy(base),'10':copy.deepcopy(schedule)}
    result['budget_candidate_generation_skipped']=True
    result['local_budget_proposals']=local;memo={};scores={};guards={};counted=set()
    result['paired_distinct_actions']=len({action_key(a) for a in cells.values()})
    result['paired_has_distinct_placement_and_budget']=action_key(cells['00'])!=action_key(cells['10']) and capvec!=base['caps']
    def score(label,remaining):
        a=cells[label];key=action_key(a)
        if key in memo:z=memo[key]
        else:
            now=time.monotonic()
            if now>=deadline:return None
            # Fixed overall deadline; no optimistic incomplete scalar is ranked.
            stop=min(deadline,now+max(.001,(deadline-now)/max(1,remaining)))
            z=evaluate(s,a,model,cfg,deadline=stop);memo[key]=z
        valid=z.get('rankable',False);val=z.get('sum_remaining_jct_ms') if valid else None
        scores[label]=val
        result['evaluations'].append({'cell':label,'family':a['family'],'action':a,'complete':valid,'cost_ms':val,
            'cpu_wall_ms':z.get('cpu_wall_ms'),'reason':z.get('reason'),'is_cross':label=='11',
            'estimated_remaining_outputs':True,'score_is_real_counterfactual':False})
        if valid:
            if key not in counted:result['complete_distinct_evaluations']+=1;counted.add(key)
            if label=='11':result['joint_cross_candidates_evaluated']+=1
        return z if valid else None
    wanted=['00','10'] if method=='D' else ['00','01'] if method=='B' else ['00','10','01','11']
    z0=score('00',len(wanted))
    if z0 is None:result['reference_unrankable']=True;return done()
    if method=='Decoupled':
        z1=score('10',2)
        first='10' if z1 is not None and scores['10']<scores['00'] and not guard(s,z0,z1,base,cells['10'],model,cfg) else '00'
        nxt='11' if first=='10' else '01';z2=score(nxt,1)
        selected=first
        if z2 is not None:
            violations=guard(s,z0,z2,base,cells[nxt],model,cfg);result['decode_guard_rejections']+=bool(violations)
            if not violations and scores[nxt]<scores[first]:selected=nxt
        result.update(action=cells[selected],model_selection=True,selected_cell=selected,sequential_coordinate_choice=True)
        return done()
    result['paired_packet_attempted']=method=='Joint'
    complete=True;valid_labels=['00']
    for index,label in enumerate(wanted[1:],1):
        z=score(label,len(wanted)-index)
        if z is None:complete=False;continue
        violations=guard(s,z0,z,base,cells[label],model,cfg);guards[label]=violations
        if violations:result['decode_guard_rejections']+=1
        else:valid_labels.append(label)
    result['guard_details']=guards
    if method=='Joint' and not complete:
        result['pair_incomplete_reference_retained']=True;return done()
    result['paired_packet_complete']=method=='Joint' and complete
    best=min(valid_labels,key=lambda label:(scores[label],label!='00'))
    if scores['00']-scores[best]<cfg['minimum_predicted_gain_ms']:best='00'
    result.update(action=cells[best],model_selection=True,selected_cell=best)
    result['paired_packet_distinct_actions']=len({action_key(a) for a in cells.values()})
    return done()
