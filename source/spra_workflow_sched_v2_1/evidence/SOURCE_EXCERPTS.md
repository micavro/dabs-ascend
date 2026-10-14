# 原实现证据摘录（只读，原文件行号）

完整原文件由 freeze 命令打包；这些摘录不替代运行前哈希校验。

## semantic19/selector.py

```text
0001: """Known-DAG complete-action selector with paired comparisons and decode guards.
0002: 
0003: Only a bounded 2x2 neighborhood is scored per event. The model is still a
0004: surrogate, not an optimum certificate. The 200ms physical-action contract and
0005: shared continuation are inherited unchanged from the repaired native runtime.
0006: """
0007: from __future__ import annotations
0008: import copy,math,time,statistics
0009: from joint18v2.actions import affinity,resources
0010: from joint18v2.common import action_key,validate_action,quantile
0011: from joint18v2.search import reference
0012: from joint18v2.model import predict
0013: from core18v2.engine import evaluate
0014: 
0015: 
0016: def tail_values(s,model,placement):
0017:     ns=s['nodes'];own={};tail={}
0018:     for k,n in ns.items():
0019:         if n['state']=='done':own[k]=0.;continue
0020:         r=placement.get(k,n.get('replica'))
0021:         if r is None:r=0
0022:         d=max(0,n['output']-n.get('output_done',0))
0023:         remp=max(0,n['prompt']-n.get('computed',0)) if n.get('phase')!='decode' else 0
0024:         own[k]=(predict(model,r,remp,0,0,n.get('computed',0)) if remp else 0)+d*predict(model,r,0,1,n['prompt']+n.get('output_done',0),0)
0025:     # Traversal comes from the compiled source graph; no gold/output-future access.
0026:     def visit(k):
0027:         if k not in tail:
0028:             tail[k]=own[k]+max((visit(j) for j in ns[k].get('compiled_successors',[]) if j in ns),default=0.)
0029:         return tail[k]
0030:     for k in ns:visit(k)
0031:     return own,tail
0032: 
0033: 
0034: def schedule_alternative(s,model,cfg,base):
0035:     ns=s['nodes'];ready=s['ready'];a=copy.deepcopy(base)
0036:     if not ready:return a
0037:     own,tails=tail_values(s,model,base['placement'])
0038:     wfmax={}
0039:     for k,n in ns.items():
0040:         if n['state'] in ('submitted','ready'):wfmax[n['wid']]=max(wfmax.get(n['wid'],0.),tails[k])
0041:     orderable=[k for k,n in ns.items() if n['state'] in ('submitted','ready') and n.get('phase')!='decode']
0042:     a['order']=sorted(orderable,key=lambda k:((wfmax.get(ns[k]['wid'],tails[k])-tails[k])/max(1.,wfmax.get(ns[k]['wid'],tails[k])),wfmax.get(ns[k]['wid'],tails[k]),ns[k].get('ready_at') or 0,k))
0043:     # Prefer an actual current ready antichain from the precompiled native graph.
0044:     groups=set()
0045:     for k in ready:
0046:         for g in ns[k].get('compiled_groups',[]):
0047:             actual=tuple(x for x in g if x in ready)
0048:             if len(actual)>1:groups.add(actual)
0049:     if groups:
0050:         group=max(groups,key=lambda g:(len(g),max(tails[k] for k in g)))
0051:         group=sorted(group,key=lambda k:(-tails[k],ns[k]['ready_at'],k))
0052:         moved=group[:max(1,len(group)//2)]
0053:     else:moved=[max(ready,key=lambda k:(tails[k],-ns[k]['ready_at']))]
0054:     source=base['placement'][moved[0]]
0055:     # Post-assignment pressure includes already submitted work and planned new work.
0056:     loads=list(s['pressure'])
0057:     for k,r in base['placement'].items():loads[r]+=max(1.,ns[k]['prompt']/1024.)
0058:     choices=[r for r in range(s['replicas']) if r!=source]
0059:     if choices:
0060:         destination=min(choices,key=lambda r:(loads[r],r))
0061:         for k in moved:a['placement'][k]=destination
0062:         a['family']='known_dag_partial_group'
0063:         a['proposal']={'moved_nodes':moved,'destination':destination,'compiled_native_group':bool(groups),'not_hardcoded_by_node_name':True}
0064:     else:a['family']='known_dag_order'
0065:     # Bounded admission template is periodically considered, never infinite holding.
0066:     if s.get('decision_index',0)%3==0:
0067:         ad=[]
0068:         for r in range(s['replicas']):
0069:             ks=[k for k in a['order'] if k in ready and a['placement'][k]==r]
0070:             active=sum(n['state']=='submitted' and n.get('replica')==r for n in ns.values())
0071:             if active>=cfg['admission_soft_active'] and len(ks)>1:
0072:                 ad+=ks[:1]+[k for k in ks[1:] if s['now']-ns[k]['ready_at']>=cfg['max_admission_wait_s']]
0073:             else:ad+=ks
0074:         a['admit']=ad;a['family']+='/bounded_admission'
0075:     a=resources(a,s,cfg);validate_action(a,s,cfg);return a
0076: 
0077: 
0078: def next_batch(s,a,model,r):
0079:     ns=s['nodes'];dec=[];pre=[]
0080:     for k,n in ns.items():
0081:         where=n.get('replica') if n['state']=='submitted' else a['placement'].get(k) if k in a['admit'] else None
0082:         if where!=r:continue
0083:         if n.get('phase')=='decode':dec.append((k,n))
0084:         else:pre.append((k,n))
0085:     rank={k:i for i,k in enumerate(a['order'])}
0086:     pre.sort(key=lambda kn:(rank.get(kn[0],len(rank)),kn[1].get('ready_at') or 0,kn[0]))
0087:     dc=sum(n['prompt']+n.get('output_done',0) for k,n in dec);d=len(dec)
0088:     left=max(0,min(a['caps'][r],4096-d));p=pc=credit=0;admitted_to_decode=0
0089:     cert={(e['prefix_hash'],e['replica']):e['tokens'] for e in s.get('cache_evidence',[]) if e.get('known') and e['state'] in ('active_computed','reclaimable_computed')}
0090:     for k,n in pre:
0091:         rem=max(0,n['prompt']-n.get('computed',0));chunk=min(rem,left)
0092:         if chunk<rem:chunk=(chunk//16)*16
0093:         if not chunk:continue
0094:         p+=chunk;left-=chunk;pc=max(pc,n.get('computed',0))
0095:         credit+=min(max(0,chunk-1),max(0,cert.get((n.get('prefix_hash'),r),0)-n.get('computed',0)))
0096:         admitted_to_decode+=int(chunk==rem)
0097:         if left<=0:break
0098:     step=predict(model,r,p,d,dc,pc,credit) if p or d else 0.
0099:     return {'prefill':p,'decode':d,'decode_context':dc,'cache_credit':credit,'max_prefill_context':pc,
0100:             'step_ms':step,'decode_only_ms':predict(model,r,0,d,dc,0) if d else 0.,
0101:             'new_decode_next_step':admitted_to_decode}
0102: 
0103: 
0104: def budget_alternative(s,model,cfg,schedule):
0105:     """Largest legal proposal under a local host-step guard, not a TPOT theorem."""
0106:     a=copy.deepcopy(schedule);caps=list(a['caps']);evidence=[]
0107:     for r in range(s['replicas']):
0108:         current=next_batch(s,a,model,r)
0109:         if current['prefill']==0:continue
0110:         options=[]
0111:         for cap in cfg['caps']:
0112:             q=copy.deepcopy(a);q['caps'][r]=cap;b=next_batch(s,q,model,r)
0113:             tau=b['decode_only_ms']*(1+cfg['local_decode_step_relative'])+cfg['local_decode_step_absolute_ms']
0114:             uncertainty=cfg['local_model_margin_ms']
0115:             allowed=not b['decode'] or b['step_ms']+uncertainty<=tau
0116:             options.append({'cap':cap,'allowed':allowed,**b,'tau_ms':tau,'uncertainty_margin_ms':uncertainty})
0117:         feasible=[x for x in options if x['allowed']]
0118:         caps[r]=max((x['cap'] for x in feasible),default=cfg['base_cap'])
0119:         evidence.append({'replica':r,'options':options,'none_feasible_keeps_reference':not feasible})
0120:     # If the preferred vector equals base, include a lower legal protective alternative
0121:     # when it changes a real mixed chunk. It is still scored, not automatically chosen.
0122:     if caps==schedule['caps']:
0123:         mixed=[r for r in range(s['replicas']) if next_batch(s,a,model,r)['decode'] and next_batch(s,a,model,r)['prefill']]
0124:         if mixed:
0125:             r=min(mixed,key=lambda r:-next_batch(s,a,model,r)['step_ms']);i=cfg['caps'].index(caps[r]);caps[r]=cfg['caps'][max(0,i-1)]
0126:     return caps,evidence
0127: 
0128: 
0129: def guard(s,base_eval,new_eval,base_action,new_action,model,cfg):
0130:     """Local step and incumbent-decode completion proxies. All are MODEL guards."""
0131:     incumbent=[k for k,n in s['nodes'].items() if n['state']=='submitted' and n.get('phase')=='decode']
0132:     violations=[]
0133:     for k in incumbent:
0134:         a=base_eval.get('node_completion_ms',{}).get(k);b=new_eval.get('node_completion_ms',{}).get(k)
0135:         if a is not None and b is not None and b>a*(1+cfg['remaining_decode_relative'])+cfg['remaining_decode_absolute_ms']:
0136:             violations.append({'kind':'remaining_incumbent_decode','node':k,'reference_ms':a,'candidate_ms':b})
0137:     for r in range(s['replicas']):
0138:         b=next_batch(s,new_action,model,r);ref=next_batch(s,base_action,model,r)
0139:         # Do not veto unchanged next-step work using its own nonzero uncertainty margin.
0140:         if not b['decode'] or (b['prefill'],b['decode'],b['decode_context'],b['cache_credit'])==(ref['prefill'],ref['decode'],ref['decode_context'],ref['cache_credit']):continue
0141:         tau=max(ref['step_ms'],b['decode_only_ms']*(1+cfg['local_decode_step_relative'])+cfg['local_decode_step_absolute_ms'])
0142:         if b['step_ms']+cfg['local_model_margin_ms']>tau:
0143:             violations.append({'kind':'next_mixed_step','replica':r,'candidate_ms':b['step_ms'],'reference_ms':ref['step_ms'],'limit_ms':tau})
0144:     return violations
0145: 
0146: 
0147: def choose(s,model,cfg,method,deadline=None):
0148:     begin=time.monotonic();deadline=deadline or begin+cfg['planning_soft_limit_s'];base=reference(s,cfg)
0149:     result={'action':base,'evaluations':[],'model_selection':False,'wall_ms':0.,'joint_cross_candidates_evaluated':0,
0150:         'complete_distinct_evaluations':0,'paired_packet_attempted':False,'paired_packet_complete':False,
0151:         'decode_guard_rejections':0,'deadline_reached':False,'knowledge':'known native DAG; unknown future content/output length/cache',
0152:         'tpot_nonregression_guaranteed':False,'hard_realtime_guaranteed':False}
0153:     def done():
0154:         result['wall_ms']=(time.monotonic()-begin)*1000;result['deadline_reached']=time.monotonic()>=deadline
0155:         validate_action(result['action'],s,cfg);return result
0156:     if method=='A':return done()
0157:     if not s['ready'] and not any(n['state']=='submitted' and n.get('phase')!='decode' for n in s['nodes'].values()):
0158:         result.update(no_state_opportunity=True,policy_equivalent_no_prefill=True);return done()
0159:     schedule=base if method=='B' else schedule_alternative(s,model,cfg,base)
0160:     result['known_dag_group_proposed']=bool(schedule.get('proposal',{}).get('compiled_native_group'))
0161:     result['known_dag_alternative_generated']=action_key(schedule)!=action_key(base)
0162:     capvec,local=budget_alternative(s,model,cfg,schedule)
0163:     cells={'00':copy.deepcopy(base),'10':copy.deepcopy(schedule),'01':copy.deepcopy(base),'11':copy.deepcopy(schedule)}
0164:     cells['01']['caps']=list(capvec);cells['11']['caps']=list(capvec)
0165:     cells['01']['family']='known_dag_budget_only';cells['11']['family']='joint/'+schedule['family']
0166:     result['local_budget_proposals']=local;memo={};scores={};guards={};counted=set()
0167:     result['paired_distinct_actions']=len({action_key(a) for a in cells.values()})
0168:     result['paired_has_distinct_placement_and_budget']=action_key(cells['00'])!=action_key(cells['10']) and capvec!=base['caps']
0169:     def score(label,remaining):
0170:         a=cells[label];key=action_key(a)
0171:         if key in memo:z=memo[key]
0172:         else:
0173:             now=time.monotonic()
0174:             if now>=deadline:return None
0175:             # Fixed overall deadline; no optimistic incomplete scalar is ranked.
0176:             stop=min(deadline,now+max(.001,(deadline-now)/max(1,remaining)))
0177:             z=evaluate(s,a,model,cfg,deadline=stop);memo[key]=z
0178:         valid=z.get('rankable',False);val=z.get('sum_remaining_jct_ms') if valid else None
0179:         scores[label]=val
0180:         result['evaluations'].append({'cell':label,'family':a['family'],'action':a,'complete':valid,'cost_ms':val,
0181:             'cpu_wall_ms':z.get('cpu_wall_ms'),'reason':z.get('reason'),'is_cross':label=='11',
0182:             'estimated_remaining_outputs':True,'score_is_real_counterfactual':False})
0183:         if valid:
0184:             if key not in counted:result['complete_distinct_evaluations']+=1;counted.add(key)
0185:             if label=='11':result['joint_cross_candidates_evaluated']+=1
0186:         return z if valid else None
0187:     wanted=['00','10'] if method=='D' else ['00','01'] if method=='B' else ['00','10','01','11']
0188:     z0=score('00',len(wanted))
0189:     if z0 is None:result['reference_unrankable']=True;return done()
0190:     if method=='Decoupled':
0191:         z1=score('10',2)
0192:         first='10' if z1 is not None and scores['10']<scores['00'] and not guard(s,z0,z1,base,cells['10'],model,cfg) else '00'
0193:         nxt='11' if first=='10' else '01';z2=score(nxt,1)
0194:         selected=first
0195:         if z2 is not None:
0196:             violations=guard(s,z0,z2,base,cells[nxt],model,cfg);result['decode_guard_rejections']+=bool(violations)
0197:             if not violations and scores[nxt]<scores[first]:selected=nxt
0198:         result.update(action=cells[selected],model_selection=True,selected_cell=selected,sequential_coordinate_choice=True)
0199:         return done()
0200:     result['paired_packet_attempted']=method=='Joint'
0201:     complete=True;valid_labels=['00']
0202:     for index,label in enumerate(wanted[1:],1):
0203:         z=score(label,len(wanted)-index)
0204:         if z is None:complete=False;continue
0205:         violations=guard(s,z0,z,base,cells[label],model,cfg);guards[label]=violations
0206:         if violations:result['decode_guard_rejections']+=1
0207:         else:valid_labels.append(label)
0208:     result['guard_details']=guards
0209:     if method=='Joint' and not complete:
0210:         result['pair_incomplete_reference_retained']=True;return done()
0211:     result['paired_packet_complete']=method=='Joint' and complete
0212:     best=min(valid_labels,key=lambda label:(scores[label],label!='00'))
0213:     if scores['00']-scores[best]<cfg['minimum_predicted_gain_ms']:best='00'
0214:     result.update(action=cells[best],model_selection=True,selected_cell=best)
0215:     result['paired_packet_distinct_actions']=len({action_key(a) for a in cells.values()})
0216:     return done()
```

## core18v2/service.py

```text
0001: """Shared phase-aware HOST-duration contract (not GPU kernel time)."""
0002: from __future__ import annotations
0003: import math
0004: PHASES=('prefill_only','decode_only','mixed')
0005: 
0006: def phase(p,d):
0007:     return 'mixed' if p>0 and d>0 else 'prefill_only' if p>0 else 'decode_only'
0008: 
0009: def features(p,d,dc,pc,cache=0):
0010:     eff=max(0.,float(p)-float(cache))
0011:     return (1.,eff/1024.,d/16.,dc/65536.,eff*pc/4194304.,eff*d/16384.)
0012: 
0013: def coefficients(model,r,which):
0014:     m=model['models'][str(r)]
0015:     record=m.get('phases',{}).get(which,m)
0016:     c=tuple(float(x) for x in record['coef'])
0017:     if len(c)!=6 or any(not math.isfinite(v) or v<0 for v in c):
0018:         raise ValueError('service model requires finite nonnegative six-feature coefficients')
0019:     return c
0020: 
0021: def predict(model,r,p,d,dc,pc,cache=0):
0022:     c=coefficients(model,r,phase(p,d));x=features(p,d,dc,pc,cache)
0023:     return max(.1,sum(a*b for a,b in zip(c,x)))
```

## core18v2/engine.py

```text
0550:                     release_candidates.update(succ[k])
0551:                     unfinished_counts[n['wid']]-=1
0552:                     if unfinished_counts[n['wid']]==0:finished_workflow_events.add(n['wid'])
0553:                     if n.get('prefix_hash'):
0554:                         history.setdefault(n['prefix_hash'], set()).add(r)
0555:             busy[r] = None
0556:             last_finish[r] = now
0557:             batch_completions += 1
0558:             expanded_iterations += b.get('decode_steps',1)
0559:             if b.get('decode_steps',1)>1:coalesced_decode_batches+=1
0560:         for w in finished_workflow_events:
0561:             wf_done[w] = max(node_done[k] for k in groups[w])
0562:             pending_wf.remove(w)
0563:         finished_workflow_events.clear()
0564:         release_successors(now)
0565:         for _, _, kind, payload in at_events:
0566:             if kind == 'cache_event':
0567:                 if payload['state'] == 'evicted':
0568:                     key=(int(payload['replica']),payload['prefix_hash'])
0569:                     ledger.entries.pop(key,None);ledger.known[key]='explicit_evicted'
0570:                     ledger.counts['explicit_evictions'] += 1
0571:                 else:
0572:                     raise ValueError('unsupported explicit cache event')
0573:             elif kind == 'apply':
0574:                 active_window = True
0575:                 current_caps = list(action['caps'])
0576:                 for k in action['admit']:
0577:                     if not admit(k, action['placement'][k], now, 'current_action'):
0578:                         # Invalid complete candidates are rejected, not silently improved.
0579:                         reason = 'candidate_admission_exceeds_common_resources'
0580:                         break
0581:             elif kind == 'expire':
0582:                 expired = True
0583:                 active_window = False
0584:                 current_caps = [cfg['base_cap']] * R
0585:                 for k in original_ready:
0586:                     if ns[k]['state'] == 'ready':
0587:                         ready_due[k] = now
0588:             elif kind == 'wake':
0589:                 lastwake = None
0590:         if reason:
0591:             break
0592:         common_admit(now)
0593:         for r in range(R):
0594:             if check_cpu():
0595:                 reason = 'cpu_deadline'
0596:                 break
0597:             schedule(r, now)
0598:         if reason:
0599:             break
0600:     complete = not pending_wf and reason is None
0601:     total = sum(wf_done.values()) if complete else None
0602:     if complete and abs(area - total) > max(.001, abs(total) * 1e-9):
0603:         raise RuntimeError('unfinished-workflow area identity violated')
0604:     return {
0605:         'complete': complete, 'rankable': complete,
0606:         'reason': 'complete_surrogate' if complete else reason,
0607:         'sum_remaining_jct_ms': total,
0608:         'workflow_prediction_ms': wf_done if complete else None,
0609:         'completed_workflows_at_cutoff': wf_done,
0610:         'node_completion_ms': node_done,
0611:         'unfinished_area_to_horizon_ms': area_h,
0612:         'common_continuation_value_ms': (total - area_h) if complete else None,
0613:         'partial_unfinished_area_ms': area,
0614:         'rollout_steps': batch_completions,
0615:         'expanded_model_iterations': expanded_iterations,
0616:         'coalesced_decode_batches': coalesced_decode_batches,
0617:         'terminal_estimate_used': False,
0618:         'physical_horizon_ms': horizon,
0619:         'apply_delay_ms': apply_at,
0620:         'continuation_cap': cfg['base_cap'],
0621:         'continuation_period_ms': period,
0622:         'cache_scenario': cfg['cache_scenario'],
0623:         'cache_counters': dict(ledger.counts),
0624:         'admissions': seen_admits,
0625:         'scheduled_prefill_peak': max_p,
0626:         'scheduled_total_peak': max_total,
0627:         'event_trace': trace if record_events else None,
0628:         'cpu_wall_ms': 1000 * (time_fn() - cpu_start),
0629:         'physical_gpu_time': False,
0630:         'native_cache_residency_proven': False,
0631:         'initial_boundary_assumption': 'native batch membership/progress preserved when supplied; residual service duration is a model estimate',
0632:         'scope': 'corrected causal surrogate only; no counterfactual ground truth or online-runtime release',
0633:     }
```

## joint18v2/actions.py

```text
0001: """Unified complete-action search and action-space ablations.
0002: Candidates use a bounded complete-action neighborhood. Cost evaluation is in core18v2, with no optimistic terminal scalar.
0003: It is not an exact physical simulator, an offline oracle, or a certified safety bound.
0004: """
0005: from __future__ import annotations
0006: import copy,time,math
0007: from collections import defaultdict
0008: from .common import action_key,validate_action,digest
0009: from .model import predict
0010: 
0011: METHODS=("A","D","B","Decoupled","Joint")
0012: def affinity(s,ready):
0013:     R=s['replicas'];loads=list(s['pressure']);placement={}
0014:     cache={}
0015:     for e in s.get('cache_evidence',[]):
0016:         if e.get('known',True) and e['state'] in ('active_computed','reclaimable_computed','resident_uncomputed','absent'):
0017:             cache[e['prefix_hash'],int(e['replica'])]=int(e.get('tokens',0))
0018:     for k in sorted(ready,key=lambda k:(s['nodes'][k]['ready_at'],k)):
0019:         n=s['nodes'][k];h=n.get('prefix_hash');known=[cache.get((h,r)) for r in range(R)]
0020:         if h and all(x is not None for x in known):
0021:             best=max(known);options=[r for r in range(R) if known[r]==best]
0022:         else:options=s.get('prefix_history',{}).get(h,[]) or list(range(R))
0023:         r=min(options,key=lambda r:(loads[r],r));placement[k]=r;loads[r]+=1
0024:     return placement
0025: 
0026: def remaining(n,r,model):
0027:     if n["state"]=="done":return 0.
0028:     if n.get("phase")=="decode":
0029:         return max(0,n["output"]-n["output_done"])*predict(model,r,0,1,n["prompt"]+n["output_done"],0)
0030:     pc=max(0,n["prompt"]-n.get("computed",0))
0031:     return predict(model,r,pc,0,0,n.get("computed",0))+max(0,n["output"]-1)*predict(model,r,0,1,n["prompt"],0)
0032: 
0033: def structural(s,model,place):
0034:     ns=s["nodes"];memo={}
0035:     def finish(k):
0036:         if k in memo:return memo[k]
0037:         n=ns[k]
0038:         if n["state"]=="done":v=0
0039:         else:
0040:             r=place.get(k,n.get("replica"))
0041:             own=min(remaining(n,z,model) for z in range(s["replicas"])) if r is None else remaining(n,r,model)
0042:             v=max((finish(p) for p in n["parents"]),default=0)+own
0043:         memo[k]=v;return v
0044:     for k in ns:finish(k)
0045:     succ=defaultdict(list)
0046:     for k,n in ns.items():
0047:         for p in n["parents"]:succ[p].append(k)
0048:     tail={}
0049:     def t(k):
0050:         if k not in tail:
0051:             tail[k]=max((remaining(ns[c],place.get(c,ns[c].get("replica")) or 0,model)+t(c) for c in succ[k] if ns[c]["state"]!="done"),default=0)
0052:         return tail[k]
0053:     active=[k for k,n in ns.items() if n["state"] not in ("blocked","done")]
0054:     heights={k:remaining(ns[k],place.get(k,ns[k].get("replica")) or 0,model)+t(k) for k in active}
0055:     wfmax={}
0056:     for k,v in heights.items():wfmax[ns[k]["wid"]]=max(wfmax.get(ns[k]["wid"],0),v)
0057:     return memo,heights,{k:wfmax[ns[k]["wid"]]-v for k,v in heights.items()}
0058: 
0059: def resources(a,s,cfg):
0060:     """Project onto a conservative reservation set before scoring and publication.
0061:     No assumed cache saving in memory reservations. Native allocator remains the hard authority.
0062:     """
0063:     a=copy.deepcopy(a);R=s["replicas"];ns=s["nodes"];used=list(s.get("reserved_blocks",[0]*R))
0064:     counts=[sum(n["state"]=="submitted" and n.get("replica")==r for n in ns.values()) for r in range(R)]
0065:     limit=[int(v*cfg["admission_reserve_fraction"]) for v in s.get("capacity_blocks",[4096]*R)]
0066:     a["resource_held"]=[]
0067:     admitted=[]
0068:     for k in sorted(a["admit"],key=lambda k:(ns[k]["ready_at"],k)):
0069:         r=a["placement"][k];need=math.ceil((ns[k]["prompt"]+ns[k].get("max_output_tokens",ns[k]["output"]))/cfg["block_size"])
0070:         if counts[r]>=cfg["max_num_seqs"] or used[r]+need>limit[r]:
0071:             a["resource_held"].append(k)
0072:         else:
0073:             admitted.append(k);counts[r]+=1;used[r]+=need
0074:     a["admit"]=admitted
0075:     return a
0076: 
0077: def schedules(s,model,cfg,base):
0078:     """Preserve the full reference, concentration, alternatives on all replicas, and admission variants."""
0079:     ready=s["ready"];p=base["placement"];ns=s["nodes"]
0080:     _,height,slack=structural(s,model,p)
0081:     orderable=[k for k,n in ns.items() if n["state"] in ("ready","submitted") and n.get("phase")!="decode"]
0082:     fifo=sorted(orderable,key=lambda k:(ns[k].get("ready_at") or 0,k))
0083:     cp=sorted(orderable,key=lambda k:(slack.get(k,0),-height.get(k,0),ns[k].get("ready_at") or 0,k))
0084:     out=[copy.deepcopy(base)]
0085:     def add(label,pl,ad,order):
0086:         out.append({"family":label,"placement":pl,"admit":ad,"order":order,"caps":list(base["caps"])})
0087:     add("dag_order",dict(p),list(ready),cp)
0088:     # Route the currently critical ready group, not a hardcoded branch label.
0089:     if ready:
0090:         focus=max(ready,key=lambda k:height.get(k,0));wid=ns[focus]["wid"]
0091:         group=[k for k in ready if ns[k]["wid"]==wid and ns[k].get("prefix_hash")==ns[focus].get("prefix_hash")]
0092:         for r in range(s["replicas"]):
0093:             pl=dict(p);pl.update({k:r for k in group})
0094:             add("group_to_"+str(r),pl,list(ready),cp)
0095:         pl=dict(p);loads=list(s["pressure"])
0096:         for k in sorted(ready,key=lambda k:-height.get(k,0)):
0097:             r=min(range(s["replicas"]),key=lambda r:(loads[r],r))
0098:             pl[k]=r;loads[r]+=1
0099:         add("spread_frontier",pl,list(ready),cp)
0100:         # Bounded, reconsidered admission: leave unsubmitted nodes in central ready state.
0101:         aged={k for k in ready if s["now"]-ns[k]["ready_at"]>=cfg["max_admission_wait_s"]}
0102:         ad=set(aged)
0103:         for r in range(s["replicas"]):
0104:             ks=[k for k in cp if k in ready and pl[k]==r]
0105:             if ks:ad.add(ks[0])
0106:         add("staged_spread",pl,sorted(ad),cp)
0107:         ad=set(aged)
0108:         for w in {ns[k]["wid"] for k in ready}:
0109:             ks=[k for k in cp if k in ready and ns[k]["wid"]==w]
0110:             if ks:ad.add(ks[0])
0111:         add("staged_affinity",dict(p),sorted(ad),cp)
0112:     unique={}
0113:     for a in out:
0114:         a=resources(a,s,cfg);validate_action(a,s,cfg);unique.setdefault(action_key(a),a)
0115:     return list(unique.values())
0116: 
0117: 
0118: # --- Selector repair: names below are internal mechanisms, not new paper methods. ---
0119: _original_schedules = schedules
0120: 
0121: 
0122: def join_groups(s, model, base):
0123:     """Current-ready antichains sharing a real unfinished join and a prefix.
0124: 
0125:     The downstream tail ranks proposals only. It is NOT an independent objective:
0126:     all decisions are evaluated using the common complete-workflow rollout.
0127:     Neither branch labels, replica IDs nor measured future outcomes are features.
0128:     """
0129:     ns=s['nodes'];ready=set(s['ready'])
0130:     _,height,slack=structural(s,model,base['placement'])
0131:     predecessors={}
0132:     def ancestors(k):
0133:         if k not in predecessors:
0134:             acc=set()
0135:             for p in ns[k]['parents']:acc.add(p);acc.update(ancestors(p))
0136:             predecessors[k]=acc
0137:         return predecessors[k]
0138:     result=[]
0139:     for j,nj in ns.items():
0140:         if nj['state']=='done' or len(nj['parents'])<2:continue
0141:         groups=defaultdict(list)
0142:         for k in ready & ancestors(j):
0143:             n=ns[k]
0144:             if n['wid']!=nj['wid']:continue
0145:             groups[n.get('prefix_hash')].append(k)
0146:         for h,ks in groups.items():
0147:             if len(ks)<2:continue
0148:             ks=sorted(ks,key=lambda k:(-height.get(k,0),slack.get(k,0),ns[k]['ready_at'],k))
0149:             result.append({'join':j,'wid':nj['wid'],'prefix_hash':h,'nodes':ks,
0150:                            'height_ms':{k:height.get(k,0) for k in ks},
```
