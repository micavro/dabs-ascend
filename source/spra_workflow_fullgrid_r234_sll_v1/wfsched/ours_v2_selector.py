"""Ours-v2: one additional legal destination, full-workflow scoring, unchanged guards.

This code runs in the existing spawn-safe planning worker, not in GPU service
containers. No change to native KV allocator, routing for baselines, resource
limits, 200 ms loop, output contract or source data. No future real output is read.

Only 3 actions may be fully scored: 00 reference; 10 original Ours candidate;
20 alternative destination of the *same* selected workflow-ready group.
"""
from __future__ import annotations

import copy
import time
from core18v2.engine import evaluate
from joint18v2.actions import resources
from joint18v2.common import action_key, validate_action
from joint18v2.search import reference
import semantic19.selector as semantic


def _cache_evidence_tokens(snapshot, moved, replica):
    """Use only current, known, actually computed native cache evidence."""
    maximum = {}
    for record in snapshot.get('cache_evidence', []):
        if (record.get('replica') != replica or not record.get('known', False)
                or record.get('state') not in ('active_computed', 'reclaimable_computed')):
            continue
        h = record.get('prefix_hash')
        maximum[h] = max(maximum.get(h, 0), int(record.get('tokens', 0)))
    return sum(min(max(0,int(snapshot['nodes'][k].get('share',0))),
                   maximum.get(snapshot['nodes'][k].get('prefix_hash'),0)) for k in moved)


def _bounded_admission_intent(snapshot, cfg, action):
    """Regenerate the original bounded-admission template AFTER rerouting.

    The original schedule projector may have excluded nodes for resources.  A
    different target can release those exclusions; recompute from the SAME
    active/age rule and re-project all normal resource limits.  The old age
    validator and decode guards are never disabled.
    """
    ready = set(snapshot['ready'])
    if 'bounded_admission' not in action['family']:
        return list(snapshot['ready'])
    ns = snapshot['nodes'];ad=[]
    for r in range(snapshot['replicas']):
        ks=[k for k in action['order'] if k in ready and action['placement'][k] == r]
        active=sum(n['state']=='submitted' and n.get('replica')==r for n in ns.values())
        if active >= cfg['admission_soft_active'] and len(ks)>1:
            ad += ks[:1] + [k for k in ks[1:]
                if snapshot['now'] - ns[k]['ready_at'] >= cfg['max_admission_wait_s']]
        else:
            ad += ks
    return ad


def second_destination(snapshot, model, cfg, base, original):
    """Choose one new destination by cache evidence, then resource pressure.

    The heuristic only proposes an action. The complete workflow evaluator
    and original safety guard decide whether it is admissible and preferable.
    """
    prop = original.get('proposal') or {}
    moved = list(prop.get('moved_nodes') or [])
    if not moved or snapshot['replicas'] < 3:
        return None
    if any(k not in snapshot['ready'] or k not in base['placement'] for k in moved):
        return None
    src = base['placement'][moved[0]]
    if any(base['placement'][k] != src for k in moved):
        return None
    first = prop.get('destination')
    choices = [r for r in range(snapshot['replicas']) if r not in (src, first)]
    if not choices:
        return None
    loads = list(snapshot['pressure'])
    for k,r in base['placement'].items():
        loads[r] += max(1., snapshot['nodes'][k]['prompt']/1024.)
    observed = {r: _cache_evidence_tokens(snapshot,moved,r) for r in choices}
    # If no usable native evidence, prefer the next distinct lightly loaded
    # destination.  Cache amounts do NOT become an artificial performance gain.
    second = min(choices, key=lambda r: (-observed[r], loads[r], r))
    action = copy.deepcopy(original)
    for k in moved:
        action['placement'][k] = second
    action['family'] = original['family'].replace('known_dag_partial_group','known_dag_partial_group_second',1)
    action['proposal'] = dict(prop, destination=second, first_destination=first,
        policy='one_extra_native_cache_or_pressure_destination',
        cache_evidence_tokens_on_new_destination=observed[second])
    action['admit'] = _bounded_admission_intent(snapshot,cfg,action)
    action['resource_held'] = []
    action = resources(action,snapshot,cfg)
    validate_action(action,snapshot,cfg)
    if action_key(action) in (action_key(base), action_key(original)):
        return None
    return action


def choose_v2(snapshot, model, cfg, method, deadline=None):
    if method != 'D' or cfg.get('base_cap') != 4096:
        raise ValueError('Ours-v2 requires original fixed-4096 D scheduler')
    start = time.monotonic()
    deadline = deadline if deadline is not None else start+cfg['planning_soft_limit_s']
    base = reference(snapshot,cfg)
    answer = dict(action=base,evaluations=[],model_selection=False,wall_ms=0.,
        joint_cross_candidates_evaluated=0,complete_distinct_evaluations=0,
        paired_packet_attempted=False,paired_packet_complete=False,
        decode_guard_rejections=0,deadline_reached=False,
        knowledge='observed DAG, native cache evidence, estimated future output only',
        tpot_nonregression_guaranteed=False, hard_realtime_guaranteed=False,
        v2_candidate_limit=3,v2_extra_destination_generated=False,
        v2_selected_extra_destination=False,v2_decision_revision='one_extra_destination-1')
    def done():
        answer['wall_ms']=(time.monotonic()-start)*1000.
        answer['deadline_reached']=time.monotonic()>=deadline
        validate_action(answer['action'],snapshot,cfg)
        return answer
    if not snapshot['ready'] and not any(n['state']=='submitted' and n.get('phase')!='decode'
        for n in snapshot['nodes'].values()):
        answer.update(no_state_opportunity=True,policy_equivalent_no_prefill=True)
        return done()
    # Called under the same admission binding as v1 inside worker.process_plan.
    original = semantic.schedule_alternative(snapshot,model,cfg,base)
    answer['known_dag_group_proposed']=bool(original.get('proposal',{}).get('compiled_native_group'))
    answer['known_dag_alternative_generated']=action_key(original)!=action_key(base)
    extra = second_destination(snapshot,model,cfg,base,original)
    cells={'00':base,'10':original}
    if extra is not None:
        cells['20']=extra
        answer['v2_extra_destination_generated']=True
        answer['v2_extra_destination']=extra['proposal']['destination']
    answer['paired_distinct_actions']=len({action_key(a) for a in cells.values()})
    answer['paired_has_distinct_placement_and_budget']=False
    memo={};valid={};guards={};tested=set()
    def score(label, remaining):
        a=cells[label];key=action_key(a)
        if key in memo:
            z=memo[key]
        else:
            now=time.monotonic()
            if now>=deadline:
                answer['evaluations'].append(dict(cell=label,family=a['family'],action=a,
                    complete=False,cost_ms=None,reason='deadline_before_candidate',
                    is_cross=False,estimated_remaining_outputs=True,
                    score_is_real_counterfactual=False))
                return None
            stop=min(deadline, now+max(.001,(deadline-now)/max(1,remaining)))
            z=evaluate(snapshot,a,model,cfg,deadline=stop)
            memo[key]=z
        rankable=bool(z.get('rankable',False))
        cost=z.get('sum_remaining_jct_ms') if rankable else None
        answer['evaluations'].append(dict(cell=label,family=a['family'],action=a,
            complete=rankable,cost_ms=cost,cpu_wall_ms=z.get('cpu_wall_ms'),
            reason=z.get('reason'),is_cross=False,estimated_remaining_outputs=True,
            score_is_real_counterfactual=False))
        if rankable:
            valid[label]=cost
            if key not in tested:tested.add(key);answer['complete_distinct_evaluations']+=1
        return z if rankable else None
    labels=list(cells)
    reference_eval=score('00',len(labels))
    if reference_eval is None:
        answer['reference_unrankable']=True
        return done()
    allowed=['00']
    for index,label in enumerate(labels[1:],1):
        evaluated=score(label,len(labels)-index)
        if evaluated is None:continue
        problems=semantic.guard(snapshot,reference_eval,evaluated,base,cells[label],model,cfg)
        guards[label]=problems
        if problems:answer['decode_guard_rejections']+=1
        else:allowed.append(label)
    answer['guard_details']=guards
    best=min(allowed,key=lambda key:(valid[key],key!='00',key))
    if valid['00']-valid[best]<cfg['minimum_predicted_gain_ms']:
        best='00'
    answer.update(action=cells[best],model_selection=True,selected_cell=best,
        v2_selected_extra_destination=(best=='20'),
        v2_model_predicted_gain_ms=max(0.,valid['00']-valid[best]))
    return done()
