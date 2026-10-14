# age_admission 源码证据

以下摘录来自上传的冻结D0快照，未修改。

## semantic19/selector.py

```text
33: 
34: def schedule_alternative(s,model,cfg,base):
35:     ns=s['nodes'];ready=s['ready'];a=copy.deepcopy(base)
36:     if not ready:return a
37:     own,tails=tail_values(s,model,base['placement'])
38:     wfmax={}
39:     for k,n in ns.items():
40:         if n['state'] in ('submitted','ready'):wfmax[n['wid']]=max(wfmax.get(n['wid'],0.),tails[k])
41:     orderable=[k for k,n in ns.items() if n['state'] in ('submitted','ready') and n.get('phase')!='decode']
42:     a['order']=sorted(orderable,key=lambda k:((wfmax.get(ns[k]['wid'],tails[k])-tails[k])/max(1.,wfmax.get(ns[k]['wid'],tails[k])),wfmax.get(ns[k]['wid'],tails[k]),ns[k].get('ready_at') or 0,k))
43:     # Prefer an actual current ready antichain from the precompiled native graph.
44:     groups=set()
45:     for k in ready:
46:         for g in ns[k].get('compiled_groups',[]):
47:             actual=tuple(x for x in g if x in ready)
48:             if len(actual)>1:groups.add(actual)
49:     if groups:
50:         group=max(groups,key=lambda g:(len(g),max(tails[k] for k in g)))
51:         group=sorted(group,key=lambda k:(-tails[k],ns[k]['ready_at'],k))
52:         moved=group[:max(1,len(group)//2)]
53:     else:moved=[max(ready,key=lambda k:(tails[k],-ns[k]['ready_at']))]
54:     source=base['placement'][moved[0]]
55:     # Post-assignment pressure includes already submitted work and planned new work.
56:     loads=list(s['pressure'])
57:     for k,r in base['placement'].items():loads[r]+=max(1.,ns[k]['prompt']/1024.)
58:     choices=[r for r in range(s['replicas']) if r!=source]
59:     if choices:
60:         destination=min(choices,key=lambda r:(loads[r],r))
61:         for k in moved:a['placement'][k]=destination
62:         a['family']='known_dag_partial_group'
63:         a['proposal']={'moved_nodes':moved,'destination':destination,'compiled_native_group':bool(groups),'not_hardcoded_by_node_name':True}
64:     else:a['family']='known_dag_order'
65:     # Bounded admission template is periodically considered, never infinite holding.
66:     if s.get('decision_index',0)%3==0:
67:         ad=[]
68:         for r in range(s['replicas']):
69:             ks=[k for k in a['order'] if k in ready and a['placement'][k]==r]
70:             active=sum(n['state']=='submitted' and n.get('replica')==r for n in ns.values())
71:             if active>=cfg['admission_soft_active'] and len(ks)>1:
72:                 ad+=ks[:1]+[k for k in ks[1:] if s['now']-ns[k]['ready_at']>=cfg['max_admission_wait_s']]
73:             else:ad+=ks
74:         a['admit']=ad;a['family']+='/bounded_admission'
75:     a=resources(a,s,cfg);validate_action(a,s,cfg);return a
76: 
77: 
```

## joint18v2/actions.py

```text
60:     """Project onto a conservative reservation set before scoring and publication.
61:     No assumed cache saving in memory reservations. Native allocator remains the hard authority.
62:     """
63:     a=copy.deepcopy(a);R=s["replicas"];ns=s["nodes"];used=list(s.get("reserved_blocks",[0]*R))
64:     counts=[sum(n["state"]=="submitted" and n.get("replica")==r for n in ns.values()) for r in range(R)]
65:     limit=[int(v*cfg["admission_reserve_fraction"]) for v in s.get("capacity_blocks",[4096]*R)]
66:     a["resource_held"]=[]
67:     admitted=[]
68:     for k in sorted(a["admit"],key=lambda k:(ns[k]["ready_at"],k)):
69:         r=a["placement"][k];need=math.ceil((ns[k]["prompt"]+ns[k].get("max_output_tokens",ns[k]["output"]))/cfg["block_size"])
70:         if counts[r]>=cfg["max_num_seqs"] or used[r]+need>limit[r]:
71:             a["resource_held"].append(k)
72:         else:
73:             admitted.append(k);counts[r]+=1;used[r]+=need
74:     a["admit"]=admitted
75:     return a
76: 
77: def schedules(s,model,cfg,base):
78:     """Preserve the full reference, concentration, alternatives on all replicas, and admission variants."""
```

## joint18v2/common.py

```text
103: 
104: def validate_action(a,snapshot,config):
105:     nodes=snapshot["nodes"];R=snapshot["replicas"];ready=set(snapshot["ready"])
106:     if set(a["placement"])!=ready:raise ValueError("placement must cover exactly unsubmitted ready set")
107:     if not set(a["admit"])<=ready or len(a["admit"])!=len(set(a["admit"])):raise ValueError("admit")
108:     for k,r in a["placement"].items():
109:         integer(r,0,R-1,"replica")
110:         if nodes[k]["state"]!="ready":raise ValueError("rewrite_submitted")
111:         if any(nodes[p]["state"]!="done" for p in nodes[k]["parents"]):raise ValueError("dependency")
112:     if len(a["caps"])!=R or any(type(x) is not int or x not in config["caps"] for x in a["caps"]):raise ValueError("caps")
113:     if len(a["order"])!=len(set(a["order"])):raise ValueError("duplicate_priority")
114:     for k in a["order"]:
115:         if k not in nodes:raise ValueError("unknown_priority")
116:     aged={k for k in ready if snapshot["now"]-nodes[k]["ready_at"]>=config["max_admission_wait_s"]}
117:     if not aged<=set(a["admit"])|set(a.get("resource_held",[])):raise ValueError("age_admission")
118:     return True
```
