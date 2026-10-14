"""Known-DAG complete-action selector with paired comparisons and decode guards.

Only a bounded 2x2 neighborhood is scored per event. The model is still a
surrogate, not an optimum certificate. The 200ms physical-action contract and
shared continuation are inherited unchanged from the repaired native runtime.
"""
from __future__ import annotations
import copy,math,time,statistics
from joint18v2.actions import affinity,resources
from joint18v2.common import action_key,validate_action,quantile
from joint18v2.search import reference
from joint18v2.model import predict
from core18v2.engine import evaluate


def tail_values(s,model,placement):
    ns=s['nodes'];own={};tail={}
    for k,n in ns.items():
        if n['state']=='done':own[k]=0.;continue
        r=placement.get(k,n.get('replica'))
        if r is None:r=0
        d=max(0,n['output']-n.get('output_done',0))
        remp=max(0,n['prompt']-n.get('computed',0)) if n.get('phase')!='decode' else 0
        own[k]=(predict(model,r,remp,0,0,n.get('computed',0)) if remp else 0)+d*predict(model,r,0,1,n['prompt']+n.get('output_done',0),0)
    # Traversal comes from the compiled source graph; no gold/output-future access.
    def visit(k):
        if k not in tail:
            tail[k]=own[k]+max((visit(j) for j in ns[k].get('compiled_successors',[]) if j in ns),default=0.)
        return tail[k]
    for k in ns:visit(k)
    return own,tail


def schedule_alternative(s,model,cfg,base):
    ns=s['nodes'];ready=s['ready'];a=copy.deepcopy(base)
    if not ready:return a
    own,tails=tail_values(s,model,base['placement'])
    wfmax={}
    for k,n in ns.items():
        if n['state'] in ('submitted','ready'):wfmax[n['wid']]=max(wfmax.get(n['wid'],0.),tails[k])
    orderable=[k for k,n in ns.items() if n['state'] in ('submitted','ready') and n.get('phase')!='decode']
    a['order']=sorted(orderable,key=lambda k:((wfmax.get(ns[k]['wid'],tails[k])-tails[k])/max(1.,wfmax.get(ns[k]['wid'],tails[k])),wfmax.get(ns[k]['wid'],tails[k]),ns[k].get('ready_at') or 0,k))
    # Prefer an actual current ready antichain from the precompiled native graph.
    groups=set()
    for k in ready:
        for g in ns[k].get('compiled_groups',[]):
            actual=tuple(x for x in g if x in ready)
            if len(actual)>1:groups.add(actual)
    if groups:
        group=max(groups,key=lambda g:(len(g),max(tails[k] for k in g)))
        group=sorted(group,key=lambda k:(-tails[k],ns[k]['ready_at'],k))
        moved=group[:max(1,len(group)//2)]
    else:moved=[max(ready,key=lambda k:(tails[k],-ns[k]['ready_at']))]
    source=base['placement'][moved[0]]
    # Post-assignment pressure includes already submitted work and planned new work.
    loads=list(s['pressure'])
    for k,r in base['placement'].items():loads[r]+=max(1.,ns[k]['prompt']/1024.)
    choices=[r for r in range(s['replicas']) if r!=source]
    if choices:
        destination=min(choices,key=lambda r:(loads[r],r))
        for k in moved:a['placement'][k]=destination
        a['family']='known_dag_partial_group'
        a['proposal']={'moved_nodes':moved,'destination':destination,'compiled_native_group':bool(groups),'not_hardcoded_by_node_name':True}
    else:a['family']='known_dag_order'
    # Bounded admission template is periodically considered, never infinite holding.
    if s.get('decision_index',0)%3==0:
        ad=[]
        for r in range(s['replicas']):
            ks=[k for k in a['order'] if k in ready and a['placement'][k]==r]
            active=sum(n['state']=='submitted' and n.get('replica')==r for n in ns.values())
            if active>=cfg['admission_soft_active'] and len(ks)>1:
                ad+=ks[:1]+[k for k in ks[1:] if s['now']-ns[k]['ready_at']>=cfg['max_admission_wait_s']]
            else:ad+=ks
        a['admit']=ad;a['family']+='/bounded_admission'
    a=resources(a,s,cfg);validate_action(a,s,cfg);return a


def next_batch(s,a,model,r):
    ns=s['nodes'];dec=[];pre=[]
    for k,n in ns.items():
        where=n.get('replica') if n['state']=='submitted' else a['placement'].get(k) if k in a['admit'] else None
        if where!=r:continue
        if n.get('phase')=='decode':dec.append((k,n))
        else:pre.append((k,n))
    rank={k:i for i,k in enumerate(a['order'])}
    pre.sort(key=lambda kn:(rank.get(kn[0],len(rank)),kn[1].get('ready_at') or 0,kn[0]))
    dc=sum(n['prompt']+n.get('output_done',0) for k,n in dec);d=len(dec)
    left=max(0,min(a['caps'][r],4096-d));p=pc=credit=0;admitted_to_decode=0
    cert={(e['prefix_hash'],e['replica']):e['tokens'] for e in s.get('cache_evidence',[]) if e.get('known') and e['state'] in ('active_computed','reclaimable_computed')}
    for k,n in pre:
        rem=max(0,n['prompt']-n.get('computed',0));chunk=min(rem,left)
        if chunk<rem:chunk=(chunk//16)*16
        if not chunk:continue
        p+=chunk;left-=chunk;pc=max(pc,n.get('computed',0))
        credit+=min(max(0,chunk-1),max(0,cert.get((n.get('prefix_hash'),r),0)-n.get('computed',0)))
        admitted_to_decode+=int(chunk==rem)
        if left<=0:break
    step=predict(model,r,p,d,dc,pc,credit) if p or d else 0.
    return {'prefill':p,'decode':d,'decode_context':dc,'cache_credit':credit,'max_prefill_context':pc,
            'step_ms':step,'decode_only_ms':predict(model,r,0,d,dc,0) if d else 0.,
            'new_decode_next_step':admitted_to_decode}


def budget_alternative(s,model,cfg,schedule):
    """Largest legal proposal under a local host-step guard, not a TPOT theorem."""
    a=copy.deepcopy(schedule);caps=list(a['caps']);evidence=[]
    for r in range(s['replicas']):
        current=next_batch(s,a,model,r)
        if current['prefill']==0:continue
        options=[]
        for cap in cfg['caps']:
            q=copy.deepcopy(a);q['caps'][r]=cap;b=next_batch(s,q,model,r)
            tau=b['decode_only_ms']*(1+cfg['local_decode_step_relative'])+cfg['local_decode_step_absolute_ms']
            uncertainty=cfg['local_model_margin_ms']
            allowed=not b['decode'] or b['step_ms']+uncertainty<=tau
            options.append({'cap':cap,'allowed':allowed,**b,'tau_ms':tau,'uncertainty_margin_ms':uncertainty})
        feasible=[x for x in options if x['allowed']]
        caps[r]=max((x['cap'] for x in feasible),default=cfg['base_cap'])
        evidence.append({'replica':r,'options':options,'none_feasible_keeps_reference':not feasible})
    # If the preferred vector equals base, include a lower legal protective alternative
    # when it changes a real mixed chunk. It is still scored, not automatically chosen.
    if caps==schedule['caps']:
        mixed=[r for r in range(s['replicas']) if next_batch(s,a,model,r)['decode'] and next_batch(s,a,model,r)['prefill']]
        if mixed:
            r=min(mixed,key=lambda r:-next_batch(s,a,model,r)['step_ms']);i=cfg['caps'].index(caps[r]);caps[r]=cfg['caps'][max(0,i-1)]
    return caps,evidence


def guard(s,base_eval,new_eval,base_action,new_action,model,cfg):
    """Local step and incumbent-decode completion proxies. All are MODEL guards."""
    incumbent=[k for k,n in s['nodes'].items() if n['state']=='submitted' and n.get('phase')=='decode']
    violations=[]
    for k in incumbent:
        a=base_eval.get('node_completion_ms',{}).get(k);b=new_eval.get('node_completion_ms',{}).get(k)
        if a is not None and b is not None and b>a*(1+cfg['remaining_decode_relative'])+cfg['remaining_decode_absolute_ms']:
            violations.append({'kind':'remaining_incumbent_decode','node':k,'reference_ms':a,'candidate_ms':b})
    for r in range(s['replicas']):
        b=next_batch(s,new_action,model,r);ref=next_batch(s,base_action,model,r)
        # Do not veto unchanged next-step work using its own nonzero uncertainty margin.
        if not b['decode'] or (b['prefill'],b['decode'],b['decode_context'],b['cache_credit'])==(ref['prefill'],ref['decode'],ref['decode_context'],ref['cache_credit']):continue
        tau=max(ref['step_ms'],b['decode_only_ms']*(1+cfg['local_decode_step_relative'])+cfg['local_decode_step_absolute_ms'])
        if b['step_ms']+cfg['local_model_margin_ms']>tau:
            violations.append({'kind':'next_mixed_step','replica':r,'candidate_ms':b['step_ms'],'reference_ms':ref['step_ms'],'limit_ms':tau})
    return violations


def choose(s,model,cfg,method,deadline=None):
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
    capvec,local=budget_alternative(s,model,cfg,schedule)
    cells={'00':copy.deepcopy(base),'10':copy.deepcopy(schedule),'01':copy.deepcopy(base),'11':copy.deepcopy(schedule)}
    cells['01']['caps']=list(capvec);cells['11']['caps']=list(capvec)
    cells['01']['family']='known_dag_budget_only';cells['11']['family']='joint/'+schedule['family']
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
