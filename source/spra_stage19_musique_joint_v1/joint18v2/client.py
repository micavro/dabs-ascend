"""Event-driven DAG runner and finite-window control for the dedicated Stage18 runtime.

No model results are executed after their deadline. Ready discovery occurs in arrival
and HTTP-completion callbacks even while the planner runs in a separate process.
Logging is asynchronous; measured JCT includes all actual waiting and publication.
"""
from __future__ import annotations
import asyncio,copy,json,math,time,uuid,statistics
from pathlib import Path
from .common import VERSION,atomic,load,boot_id,clock_namespace,digest,token_hash,validate_action,quantile
from .transport import stream,metrics
from .search import choose,reference,METHODS
from .actions import resources,affinity
from .model import predict
from .io import JsonWriter
from .batchcheck import compare_iteration,binding_evidence
from .latency_metrics import backend_progress_metrics
from .bootstrap import REVISION
from .lifetime_probe import finite_window_evidence, REVISION as LIFETIME_REVISION
ENDPOINTS=[f'http://127.0.0.1:{8000+i}' for i in range(4)]

def identity(s):return {k:s[k] for k in ('pod','pid','boot_id')}
class BackendNotReady(RuntimeError):pass

def statuses(runtime,R):
    paths=[Path(runtime)/'joint18v2'/f'spra-vllm-{r}.status.json' for r in range(R)]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise BackendNotReady('native runtime status missing: '+', '.join(missing)+
            '; no generation request sent by gate; inspect Pod readiness, current/previous container logs and joint18v2 boot records; do not create a fake status file')
    ss=[load(Path(runtime)/'joint18v2'/f'spra-vllm-{r}.status.json') for r in range(R)]
    if any(s.get('version')!=VERSION for s in ss):raise RuntimeError('wrong hook version; do not read legacy status')
    if any(s.get('fatal_engine_error') for s in ss):raise RuntimeError('native engine failure; new cache evidence invalid')
    return ss

def same_engines(now,initial):
    if [identity(s) for s in now]!=[identity(s) for s in initial]:raise RuntimeError('engine identity changed during condition')

async def gate(client,runtime,R=4):
    ss=await asyncio.to_thread(statuses,runtime,R)
    for r,s in enumerate(ss):
        if s.get('recovery_revision')!=REVISION:raise BackendNotReady('expected boot/metric recovery hook on '+s.get('pod','unknown'))
        if s['boot_id']!=boot_id() or s.get('clock_namespace_inode')!=clock_namespace():raise RuntimeError('clock domains differ; no subtraction of unrelated monotonic clocks')
        for k,v in [('hard_total',4096),('max_model_len',4096),('block_size',16),('max_num_seqs',64),('num_scheduler_steps',1)]:
            if s.get(k)!=v:raise RuntimeError(f'{s["pod"]}: expected {k}={v}')
        if any(s[k] for k in ('running','waiting','swapped')):raise RuntimeError('dedicated engines must be idle')
        if time.monotonic_ns()-s['mono_ns']>2_000_000_000:raise RuntimeError('idle observation heartbeat stale')
        if s.get('native_cache_allocator') not in ('v1','v2'):raise RuntimeError('native cache observer absent')
        z=await client.get(ENDPOINTS[r]+'/health');z.raise_for_status()
        z=await client.get(ENDPOINTS[r]+'/v1/models');z.raise_for_status()
        if not any(m.get('id')=='Qwen2.5-1.5B-Instruct' for m in z.json().get('data',[])):raise RuntimeError('model mismatch')
    gpus=[s.get('nvidia_visible_devices') for s in ss]
    if None in gpus or len(set(gpus))!=R:raise RuntimeError('GPU identities not distinct')
    return ss

async def token_pool(client,model):
    z=await client.post(ENDPOINTS[0]+'/tokenize',json={'model':model,'prompt':'The evidence supports careful analysis. Explain the result using the provided context and report assumptions.','add_special_tokens':False})
    z.raise_for_status();pool=list(dict.fromkeys(z.json()['tokens']))
    if len(pool)<12:raise RuntimeError('token pool too small')
    return pool

async def wait_drained(runtime,initial,timeout=120):
    end=time.monotonic()+timeout;count=0
    while time.monotonic()<end:
        ss=await asyncio.to_thread(statuses,runtime,len(initial));same_engines(ss,initial)
        count=count+1 if all(not any(s[k] for k in ('running','waiting','swapped')) for s in ss) else 0
        if count>=2:return True
        await asyncio.sleep(.05)
    return False

class Publisher:
    def __init__(self,runtime,initial,case,cfg,base_caps=None):
        self.path=Path(runtime)/'joint18v2/control.json';self.initial=initial;self.case=case;self.cfg=cfg
        self.base_caps=list(base_caps or [cfg['base_cap']]*len(initial));self.record=None;self.records=[];self.task=None;self.lock=asyncio.Lock()
        self.capture_flushes=False;self.flush_records=[]
    def caps_at(self,at_ns=None):
        at_ns=time.monotonic_ns() if at_ns is None else at_ns
        if self.record is None or at_ns<self.record['not_before_ns'] or at_ns>=self.record['action_expires_ns']:return list(self.base_caps)
        return list(self.record['caps'])
    async def set(self,caps,orders,snapshot_ns,expire_ns,plan_id=None,total=None):
        now=time.monotonic_ns()
        if now>=expire_ns:raise TimeoutError('action expired before control publication')
        self.record={'plan_id':plan_id or uuid.uuid4().hex,'caps':list(caps),'orders':copy.deepcopy(orders),
              'snapshot_ns':int(snapshot_ns),'not_before_ns':now,'action_expires_ns':int(expire_ns),
              'totals':list(total or [self.cfg['hard_total']]*len(self.initial))}
        self.records.append(copy.deepcopy(self.record));await self.flush();return self.record['plan_id']
    async def flush(self, cause="publication"):
        async with self.lock:
            if self.record is None:return
            now=time.monotonic_ns();r=self.record
            replicas={s['pod']:{'enabled':True,**identity(s),'case':self.case,'plan_id':r['plan_id'],
                'issued_ns':now,'expires_ns':now+int(self.cfg['lease_s']*1e9),
                'snapshot_ns':r['snapshot_ns'],'not_before_ns':r['not_before_ns'],'action_expires_ns':r['action_expires_ns'],
                'total':r['totals'][i],'prefill_cap':r['caps'][i],'order':r['orders'][i],
                'continuation':{'total':r['totals'][i],'prefill_cap':self.base_caps[i],'order':[]}}
                for i,s in enumerate(self.initial)}
            await asyncio.to_thread(atomic,self.path,{'enabled':True,'replicas':replicas})
            if self.capture_flushes:
                self.flush_records.append({'cause':cause,'plan_id':r['plan_id'],'issued_ns':now,
                    'written_ns':time.monotonic_ns(),'transport_expires_ns':now+int(self.cfg['lease_s']*1e9),
                    **{k:r[k] for k in ('snapshot_ns','not_before_ns','action_expires_ns')}})
    async def heartbeat(self):
        while True:await asyncio.sleep(self.cfg['lease_s']/3);await self.flush(cause='heartbeat')
    async def start(self):
        now=time.monotonic_ns();await self.set(self.base_caps,[[] for _ in self.initial],now,now+200_000_000)
        self.task=asyncio.create_task(self.heartbeat())
    async def close(self):
        if self.task:self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        await asyncio.to_thread(atomic,self.path,{'enabled':False})

async def register_prefixes(runtime,initial,nodes,case):
    prefixes={n['prefix_hash']:n['tokens'][:n['share']] for n in nodes.values() if n.get('prefix_hash')}
    q={'schema':'joint18v2-prefix-query-1','query_id':uuid.uuid4().hex,'boot_id':boot_id(),'case':case,
       'prefixes':[{'prefix_hash':h,'tokens':t} for h,t in prefixes.items()]}
    await asyncio.to_thread(atomic,Path(runtime)/'joint18v2/prefix_query.json',q)
    end=time.monotonic()+5
    while time.monotonic()<end:
        ss=await asyncio.to_thread(statuses,runtime,len(initial));same_engines(ss,initial)
        if all(s.get('cache',{}).get('query_id')==q['query_id'] for s in ss):return q,ss
        await asyncio.sleep(.025)
    raise RuntimeError('native cache query registration not acknowledged; no model calls sent')

def make_snapshot(nodes,ss,current,history,now,model,cfg,decision_index):
    now_ns=int(now*1e9);live={};queue={};cache=[];inflight=[]
    arrived={n['wid'] for n in nodes.values() if n.get('arrival') is not None}
    completed={w for w in arrived if all(n['state']=='done' for n in nodes.values() if n['wid']==w)}
    ns={k:{kk:copy.deepcopy(v) for kk,v in n.items() if kk not in ('tokens','stream','parent_response_bindings','resolved_question')} for k,n in nodes.items() if n['wid'] in arrived-completed}
    byhash={n['hash']:k for k,n in ns.items()}
    for r,s in enumerate(ss):
        queue[str(r)]=[]
        for q in ('running','waiting','swapped'):
            for v in s[q]:
                k=byhash.get(v['prompt_hash'])
                if k is None:
                    # Completed client requests can remain visible in an older frame; foreign requests cannot.
                    if not any(n['hash']==v['prompt_hash'] for n in nodes.values()):raise RuntimeError('foreign request in dedicated runtime')
                    continue
                live[k]=(r,v);queue[str(r)].append(k)
        c=s.get('cache',{});age=(now_ns-c.get('observed_ns',0))/1e9
        for e in c.get('records',[]):
            if age<=cfg['cache_evidence_max_age_s']:
                cache.append({**e,'replica':r,'observed_age_s':age,'valid_until_ms':None})
    for k,n in ns.items():
        if n['state']=='done':continue
        if k in live:
            r,v=live[k]
            if n.get('replica') is not None and n['replica']!=r:raise RuntimeError('actual route differs from submitted route')
            n.update(computed=v['computed'],output_done=v['output_done'],accepted_at=v['accepted_s'],phase='prefill' if v['prefill'] else 'decode')
            if n.get('semantic_mode') and n.get('output_is_estimate'):
                n['output']=min(n['max_output_tokens'],max(n['output'],v['output_done']+8))
            nodes[k]['native_seen']=True
        elif n['state']=='submitted' and nodes[k].get('native_seen'):
            # Native no longer owns this sequence. Keep its HTTP-tail cost in real JCT, not simulated GPU work.
            n.update(state='done',computed=n['prompt'],output_done=n['output'],phase='done',native_output_finished=True)
    if model is not None:
        for r,s in enumerate(ss):
            it=s.get('inflight')
            if not it or not s.get('engine_step_inflight'):continue
            pre=[];dec=[];p=d=dc=pc=credit=0
            for x in it.get('scheduled',[]):
                k=byhash.get(x['prompt_hash'])
                if k is None or ns[k]['state']!='submitted':continue
                n=ns[k];n.update(computed=x['computed_before'],output_done=x['output_before'],phase='prefill' if x['prefill'] else 'decode')
                if x['prefill']:
                    pre.append((k,x['tokens']));p+=x['tokens'];pc=max(pc,x['computed_before'])
                    credit+=min(max(0,x['tokens']-1),max(0,x.get('cached_blocks_metadata',0)*16-x['computed_before']))
                else:dec.append(k);d+=x['tokens'];dc+=x['context_before']
            elapsed=max(0,(now_ns-it['begin_ns'])/1e6)
            rem=max(.1,predict(model,r,p,d,dc,pc,credit)-elapsed)
            if pre or dec:inflight.append({'replica':r,'prefill':pre,'decode':dec,'elapsed_ms':elapsed,'remaining_ms':rem,
                                         'cap':it['prefill_cap'],'cache_overlap_proxy':credit,'duration_is_estimate':True})
    reserve=[sum(math.ceil((n['prompt']+n.get('max_output_tokens',n['output']))/cfg['block_size']) for n in ns.values() if n['state']=='submitted' and n.get('replica')==r) for r in range(len(ss))]
    return {'nodes':ns,'ready':[k for k,n in ns.items() if n['state']=='ready'],'replicas':len(ss),'now':now,
      'pressure':[len(s['running'])+2*len(s['waiting']) for s in ss],'reserved_blocks':reserve,
      'capacity_blocks':[s['num_gpu_blocks'] for s in ss],'prefix_history':copy.deepcopy(history),'prefix_certificates':{},
      'cache_evidence':cache,'queue_order':queue,'current_caps':list(current),'inflight_batches':inflight,
      'decision_index':decision_index,'state_ages_s':[(now_ns-s['mono_ns'])/1e9 for s in ss],
      'cache_unknown_count':sum(1 for n in ns.values() if n.get('prefix_hash') and not any(e['prefix_hash']==n['prefix_hash'] for e in cache))}

def process_plan(snapshot,model,cfg,method,deadline):
    if cfg.get('semantic_selector'):
        from semantic19.selector import choose as semantic_choose
        return semantic_choose(snapshot,model,cfg,method,deadline)
    return choose(snapshot,model,cfg,method,deadline)

class ForcedMechanism:
    """Only the first target fork uses a forced complete action, then common A continues."""
    def __init__(self,placement,vector):self.placement=placement;self.vector=list(vector);self.triggered=False;self.record=None
    def apply(self,snap,base,ss,cfg):
        ready=[k for k in snap['ready'] if k.startswith('w0000/B')]
        if self.triggered or len(ready)<4:return base,None
        root=snap['nodes']['w0000/root'];h=root['prefix_hash']
        known={e['replica']:e for e in snap['cache_evidence'] if e['prefix_hash']==h}
        ndec=sum(not x['prefill'] for x in ss[0]['running'])
        satisfied=ndec>=2 and known.get(0,{}).get('tokens',0)>=root['share'] and known.get(1,{}).get('state')=='absent' and not ss[1]['running'] and not ss[1]['waiting']
        a=copy.deepcopy(base)
        for k in ready:a['placement'][k]=1 if self.placement=='split' and k.rsplit('/',1)[-1] in ('B3','B4') else 0
        a['caps']=list(self.vector);a['order']=[];a['admit']=list(snap['ready']);a['family']='forced/'+self.placement
        a=resources(a,snap,cfg);self.triggered=True
        self.record={'trigger_satisfied':satisfied,'decode_on_affinity':ndec,'affinity_prefix_tokens':known.get(0,{}).get('tokens'),
                     'remote_prefix_state':known.get(1,{}).get('state'),'snapshot':copy.deepcopy(snap),'action':copy.deepcopy(a)}
        return a,self.record

async def execute(client,runtime,initial,cfg,nodes,folder,method,model=None,executor=None,force_caps=None,force_total=None,mechanism=None,lifetime_probe=False,materializer=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);case=folder.parent.name+'/'+folder.name;R=len(initial)
    frozen=copy.deepcopy(nodes)
    atomic(folder/'manifest.json',{'case':case,'method':method,'config':cfg,'input_nodes':frozen,
          'model_id':None if model is None else model.get('model_id'),'identities':[identity(s) for s in initial]})
    offsets=[]
    for r in range(R):
        path=Path(runtime)/'joint18v2'/f'spra-vllm-{r}.events.jsonl';offsets.append(path.stat().st_size if path.exists() else 0)
    writer=JsonWriter(cfg['log_queue_limit']);publisher=Publisher(runtime,initial,case,cfg,force_caps)
    publisher.capture_flushes=bool(lifetime_probe);lifetime_trigger=None
    if lifetime_probe and (method!='A' or mechanism is not None or force_caps is not None or force_total is not None or model is not None or executor is not None):
        raise ValueError('finite canary must be model-free A without other forced modes')
    wake=asyncio.Event();tasks=[];arrivals_tasks=[];actions=[];timeline=[];history={};errors=[];pending=None;failure=None
    ss_latest=initial;sampling_task=None;decision_index=0;latencies=[];current=list(force_caps or [cfg['base_cap']]*R)
    queue_intervals=[];start=finish=None;query=None;collected=[]
    successors={k:[] for k in nodes}
    for k,n in nodes.items():
        n['arrival']=None
        for parent in n['parents']:successors[parent].append(k)
    async def prepare_node(k):
        n=nodes[k]
        try:
            await materializer(k,nodes)
            n.update(state='ready',input_prepare_end=time.monotonic(),prompt_materialized=True)
            timeline.append({'event':'input_prepared','node':k,'at':n['input_prepare_end']})
            wake.set()
        except BaseException as ex:
            if not isinstance(ex,asyncio.CancelledError):errors.append(ex);wake.set()
            raise
    def release(keys):
        now=time.monotonic()
        for k in keys:
            n=nodes[k]
            if n['state']=='blocked' and not n.get('preparing') and n['arrival'] is not None and all(nodes[p]['state']=='done' for p in n['parents']):
                true_ready=max([n['arrival']]+[nodes[p]['finish'] for p in n['parents']])
                n.update(ready_at=true_ready,ready_discovered_at=now)
                timeline.append({'event':'ready','node':k,'eligible_at':true_ready,'discovered_at':now})
                if materializer:
                    n.update(preparing=True,input_prepare_start=now)
                    tasks.append(asyncio.create_task(prepare_node(k)))
                else:n.update(state='ready')
        wake.set()
    async def arrive(keys):
        at=start+nodes[keys[0]]['arrival_offset'];await asyncio.sleep(max(0,at-time.monotonic()));now=time.monotonic()
        for k in keys:nodes[k]['arrival']=now
        release(keys)
    async def do(k,r):
        n=nodes[k]
        try:
            await stream(client,ENDPOINTS[r],n,cfg['model'])
            if n.get('semantic_mode'):n.update(output=n['stream']['usage']['completion_tokens'],output_is_estimate=False)
            n.update(finish=time.monotonic(),state='done',phase='done')
            if n['prefix_hash']:
                hist=history.setdefault(n['prefix_hash'],[])
                if r not in hist:hist.append(r)
            timeline.append({'event':'complete','node':k,'at':n['finish']});release(successors[k])
        except BaseException as ex:
            if not isinstance(ex,asyncio.CancelledError):errors.append(ex);wake.set()
            raise
    async def sampler():
        nonlocal ss_latest
        while True:
            ss=await asyncio.to_thread(statuses,runtime,R);same_engines(ss,initial);ss_latest=ss
            await asyncio.sleep(cfg['snapshot_poll_s'])
    try:
        query,ss_latest=await register_prefixes(runtime,initial,nodes,case)
        atomic(folder/'initial_cache.json',{'query_id':query['query_id'],'status':ss_latest})
        if any(e['tokens'] for s in ss_latest for e in s['cache']['records']):raise RuntimeError('new condition namespace unexpectedly warm; no hidden cache reuse')
        await publisher.start()
        if force_total is not None:
            ns=time.monotonic_ns();await publisher.set(current,[[] for _ in initial],ns,ns+200_000_000,total=force_total)
        sampling_task=asyncio.create_task(sampler());start=time.monotonic();anchor=start;next_tick=start
        groups={}
        for k,n in nodes.items():groups.setdefault(n['wid'],[]).append(k)
        arrivals_tasks=[asyncio.create_task(arrive(keys)) for keys in groups.values()]
        # No pulse at condition setup: steady 1/s has its first arrival at +0.5s.
        # Arm the one-shot pulse only with the first validated nonempty admission.
        async with asyncio.timeout(cfg['condition_wall_s']):
            while not all(n['state']=='done' for n in nodes.values()):
                if errors:raise errors[0]
                if sampling_task.done():await sampling_task
                writer.check()
                now=time.monotonic();timeout=max(0,next_tick-now)
                # Dependency callbacks continue during this sleep; the plan clock never shifts by solver time.
                await asyncio.sleep(timeout)
                wake.clear();await asyncio.sleep(cfg['ready_coalesce_s'])
                if all(n['state']=='done' for n in nodes.values()):break
                begin=time.monotonic();next_tick=anchor+(math.floor((begin-anchor)/cfg['epoch_s'])+1)*cfg['epoch_s']
                if next_tick-begin<cfg['commit_reserve_s']:continue
                # The current horizon ends at the existing absolute tick, never plan_end+200ms.
                expire_ns=int(next_tick*1e9);deadline=min(begin+cfg['planning_soft_limit_s'],next_tick-cfg['commit_reserve_s'])
                decision_index+=1
                # Use a just-read frame rather than only the periodic sampler's cached frame.
                # The bounded read is common infrastructure; dependency callbacks keep running.
                ss_latest=await asyncio.to_thread(statuses,runtime,R);same_engines(ss_latest,initial)
                scfg=dict(cfg)
                horizon=(next_tick-begin)*1000;expected=min(max(1.,quantile(latencies,.9) if latencies else cfg['initial_expected_apply_delay_ms']),max(1.,horizon-5.))
                scfg.update(action_horizon_ms=horizon,apply_delay_ms=expected)
                current=publisher.caps_at(int(begin*1e9))
                snap=make_snapshot(nodes,ss_latest,current,history,begin,model,cfg,decision_index)
                if not snap['nodes']:continue
                base=reference(snap,cfg);ans={'action':base,'evaluations':[],'model_selection':False,'wall_ms':0.0}
                busy=False
                if force_caps is not None:ans['action']['caps']=list(force_caps)
                elif mechanism is not None:
                    # Background and root setup are predeclared and identical across the four cells.
                    for k in base['placement']:
                        if k.startswith('bg/') or k=='w0000/root' or k=='w0000/sink':base['placement'][k]=0
                    base=resources(base,snap,cfg);a,trigger=mechanism.apply(snap,base,ss_latest,cfg);ans['action']=a
                    if trigger:
                        writer.submit(folder/'mechanism_trigger.json',trigger,replace=True)
                    elif mechanism.triggered and publisher.record and publisher.record['action_expires_ns']>time.monotonic_ns():
                        # Do not overwrite the fixed 200ms mechanism window when a sibling completes.
                        ans['action']['caps']=list(publisher.record['caps'])
                elif method!='A' and snap['cache_unknown_count']:
                    ans['cache_unknown_fallback']=True
                elif method!='A':
                    if pending is not None and not pending.done():busy=True;ans['worker_busy_fallback']=True
                    else:
                        if pending is not None:
                            try:pending.result()
                            except BaseException:pass
                            pending=None
                        if executor is None or model is None:raise RuntimeError('optimized method requires isolated worker/model')
                        worker_snapshot=dict(snap)
                        worker_snapshot['cache_evidence']=[{k:v for k,v in e.items() if k not in ('blocks','block_ids')} for e in snap['cache_evidence']]
                        handoff=min(float(cfg.get('result_handoff_reserve_s',.020)),max(0.,deadline-time.monotonic())*.25)
                        evaluation_deadline=deadline-handoff
                        pending=asyncio.get_running_loop().run_in_executor(executor,process_plan,worker_snapshot,model,scfg,method,evaluation_deadline)
                        try:
                            ans=await asyncio.wait_for(asyncio.shield(pending),timeout=max(.0001,deadline-time.monotonic()))
                            pending=None
                            ans.update(result_receive_deadline=deadline,worker_evaluation_deadline=evaluation_deadline,
                                reserved_handoff_ms=handoff*1000,result_received_s=time.monotonic(),
                                received_before_total_deadline=time.monotonic()<deadline)
                        except asyncio.TimeoutError:ans.update(worker_timeout_fallback=True,
                            result_receive_deadline=deadline,worker_evaluation_deadline=evaluation_deadline,reserved_handoff_ms=handoff*1000)
                plan_done=time.monotonic();fresh=make_snapshot(nodes,ss_latest,publisher.caps_at(int(plan_done*1e9)),history,plan_done,model,cfg,decision_index)
                a=ans['action'];reason=None
                if any(k not in fresh['nodes'] or fresh['nodes'][k]['state']!='ready' for k in a['placement']):reason='touched_request_changed'
                subset=copy.deepcopy(fresh);subset['ready']=list(a['placement'])
                if not reason and resources(a,subset,cfg)['admit']!=a['admit']:reason='resource_state_changed'
                if not reason and any(age>cfg['snapshot_max_age_s'] for age in fresh['state_ages_s']):reason='native_state_stale'
                # An observed loss of a scored prefix is not allowed to masquerade as the old state.
                before={(e['prefix_hash'],e['replica']):e['tokens'] for e in snap['cache_evidence']}
                after={(e['prefix_hash'],e['replica']):e['tokens'] for e in fresh['cache_evidence']}
                if not reason and any(before.get((snap['nodes'][k].get('prefix_hash'),a['placement'][k]),0)>after.get((snap['nodes'][k].get('prefix_hash'),a['placement'][k]),0) for k in a['admit']):reason='scored_cache_prefix_lost'
                if time.monotonic_ns()>=expire_ns:reason='commit_missed_action_window'
                if reason:
                    a=reference(fresh,cfg);ans['commit_fallback']=reason;snap=fresh
                    if force_caps is not None:a['caps']=list(force_caps)
                validate_action(a,snap if not reason else fresh,cfg)
                publish_begin=time.monotonic();orders=[[] for _ in initial]
                for k in a['order']:
                    r=a['placement'].get(k,nodes[k].get('replica'))
                    if r is not None and nodes[k]['state']!='done':orders[r].append(nodes[k]['hash'])
                valid_time=time.monotonic_ns()<expire_ns
                publication_performed=False
                if valid_time and not lifetime_probe:
                    pid=await publisher.set(a['caps'],orders,int(begin*1e9),expire_ns,total=force_total)
                    current=list(a['caps']);publication_performed=True
                elif lifetime_probe:
                    if lifetime_trigger is None and a['admit'] and valid_time:
                        a['caps']=[256]*R
                        pid=await publisher.set(a['caps'],orders,int(begin*1e9),expire_ns)
                        publication_performed=True
                        lifetime_trigger={'kind':'first_validated_nonempty_admission',
                            'admit':list(a['admit']),'decision_index':decision_index,
                            'ready_at_s':{k:nodes[k]['ready_at'] for k in a['admit']},
                            'continuation_cap':cfg['base_cap'],
                            'control':copy.deepcopy(publisher.record)}
                    pid=publisher.record['plan_id']
                    # Do not reissue or slide the pulse. All later caps follow its
                    # immutable deadline, including plans that only admit children.
                    a['caps']=publisher.caps_at()
                    expire_ns=publisher.record['action_expires_ns']
                    current=list(a['caps'])
                else:
                    pid=None;a=reference(fresh,cfg);snap=fresh;ans['commit_fallback']='expired_before_publish'
                    # Native common continuation is already defined in the outer lease; no extension.
                publication=time.monotonic();latencies.append(1000*(publication-begin))
                # Create request tasks before enqueuing detailed plan records. Logging itself is nonblocking.
                for k in a['admit']:
                    n=nodes[k]
                    if n['state']!='ready':raise RuntimeError('double admission')
                    n.update(replica=a['placement'][k],state='submitted',phase='prefill',start=time.monotonic(),commit_plan_id=pid)
                    tasks.append(asyncio.create_task(do(k,n['replica'])))
                policy_held=list(set(snap['ready'])-set(a['admit'])-set(a.get('resource_held',[])))
                log={'method':method,'decision_index':decision_index,'snapshot':snap,'action':copy.deepcopy(a),
                    'begin_s':begin,'plan_done_s':plan_done,'publish_begin_s':publish_begin,'published_s':publication,
                    'snapshot_ns':int(begin*1e9),'action_expires_ns':expire_ns,'published_plan_id':pid,
                    'publication_performed':publication_performed,
                    'control_snapshot_ns':publisher.record['snapshot_ns'] if lifetime_probe else int(begin*1e9),
                    'estimated_apply_delay_ms':expected,'actual_publish_delay_ms':1000*(publication-begin),
                    'policy_deferred':policy_held,'new_ready_discovered_during_plan':[k for k in fresh['ready'] if k not in snap['ready']],
                    **{k:v for k,v in ans.items() if k!='action'}}
                actions.append(log);writer.submit(folder/'plans.jsonl',log)
                for k in snap['ready']:queue_intervals.append({'node':k,'begin':begin,'end':publication,'phase':'planning_and_publish'})
                for k in policy_held:queue_intervals.append({'node':k,'begin':publication,'end':next_tick,'phase':'policy_deferred'})
            await asyncio.gather(*tasks,*arrivals_tasks);finish=time.monotonic()
            if not await wait_drained(runtime,initial):raise RuntimeError('backend did not drain')
            await asyncio.sleep(.06)
    except BaseException as ex:
        failure=repr(ex)
        for t in tasks+arrivals_tasks:
            if not t.done():t.cancel()
        await asyncio.gather(*tasks,*arrivals_tasks,return_exceptions=True)
        raise
    finally:
        if sampling_task:sampling_task.cancel();await asyncio.gather(sampling_task,return_exceptions=True)
        await publisher.close()
        if pending is not None:
            def discard(f):
                try:f.result()
                except BaseException:pass
            pending.add_done_callback(discard)
        await asyncio.to_thread(writer.close)
        atomic(folder/'nodes.json',nodes);atomic(folder/'controls.json',publisher.records);atomic(folder/'ready_events.json',timeline)
        if lifetime_probe:
            atomic(folder/'lifetime_trigger.json',lifetime_trigger)
            atomic(folder/'lifetime_control_flushes.json',publisher.flush_records)
        atomic(folder/'queue_intervals.json',queue_intervals);atomic(folder/'status.json',{'execution':'failed' if failure else 'complete','error':failure})
        partial=0
        for r,offset in enumerate(offsets):
            path=Path(runtime)/'joint18v2'/f'spra-vllm-{r}.events.jsonl'
            if not path.exists():continue
            with path.open('rb') as f:f.seek(offset);data=f.read()
            for line in data.splitlines():
                try:x=json.loads(line)
                except (ValueError,UnicodeDecodeError):partial+=1;continue
                if x.get('case')==case:x['replica']=r;collected.append(x)
        with (folder/'backend.jsonl').open('w') as f:
            for x in collected:f.write(json.dumps(x,separators=(',',':'))+'\n')
        atomic(folder/'trace_capture.json',{'partial_lines':partial,'records':len(collected),'condition_failed':bool(failure)})
    summary=metrics(nodes,start,finish,actions)
    atomic(folder/'request_latency.json',summary['request_latency_rows'])
    atomic(folder/'request_latency_summary.json',summary['request_latency_metrics'])
    progress=backend_progress_metrics(nodes,collected)
    atomic(folder/'backend_output_progress_metrics.json',progress)
    summary['backend_output_progress_metrics']={k:v for k,v in progress.items() if k!='rows'}
    summary['measurement_revision']=REVISION
    summary['online_search_calls']=sum(bool(x.get('model_selection')) or bool(x.get('worker_timeout_fallback')) for x in actions)
    summary['execution_mode']='preset_complete_action_no_online_search' if mechanism else ('fixed_canary' if lifetime_probe else 'online_or_fixed_reference')
    if mechanism and summary['online_search_calls']:
        raise RuntimeError('preset mechanism unexpectedly invoked model selection')
    accepts=[x for x in collected if x.get('event')=='accept'];match={}
    for x in accepts:match.setdefault(x['prompt_hash'],[]).append(x)
    if any(len(match.get(n['hash'],[]))!=1 or match[n['hash']][0]['replica']!=n['replica'] for n in nodes.values()):raise RuntimeError('request/native routing correlation failed')
    its=[x for x in collected if x.get('event')=='iteration' and x['scheduled']]
    inputs=[x for x in collected if x.get('event')=='model_input']
    if not its or not inputs:raise RuntimeError('missing native batch or model-input evidence')
    checks=[compare_iteration(x) for x in its];bind=[binding_evidence(x) for x in its]
    summary.update(request_correlation=f'{len(nodes)}/{len(nodes)}',budget_invariants=all(x['prefill_logical_tokens']+x['decode_tokens']<=x['total'] and (x['prefill_cap'] is None or x['prefill_logical_tokens']<=x['prefill_cap']) for x in its),
        native_next_batch_matches=sum(x['match'] for x in checks),native_next_batch_checked=len(checks),
        model_input_records=len(inputs),native_skipped_query_tokens=sum(x['skipped_query_tokens'] for x in inputs),
        native_cap_binding_candidates=sum(x['binding_candidate'] for x in bind),cache_transition_events=sum(x.get('event')=='cache_transition' for x in collected),
        queue_reorder_iterations=sum(any(a['changed'] for a in x['order'].values()) for x in its),
        prefill_caps_applied=sorted({x['prefill_cap'] for x in its if x['prefill_cap'] is not None}),
        preemptions=sum(x['preemptions'] for x in its),
        actual_commit_route_mismatches=0,policy_deferred_ready_decisions=sum(len(x['policy_deferred']) for x in actions),
        worker_timeout_fallback=sum(bool(x.get('worker_timeout_fallback')) for x in actions),
        commit_fallbacks=sum(bool(x.get('commit_fallback')) for x in actions),
        cache_unknown_fallback=sum(bool(x.get('cache_unknown_fallback')) for x in actions),
        worker_busy_fallback=sum(bool(x.get('worker_busy_fallback')) for x in actions),
        search_deadline_reached=sum(bool(x.get('deadline_reached')) for x in actions),
        nontrivial_joint_decisions=sum(method=='Joint' and not x.get('policy_equivalent_no_prefill') for x in actions),
        joint_decisions_with_complete_cross=sum(x.get('joint_cross_candidates_evaluated',0)>0 for x in actions),
        complete_candidate_evaluations=sum(x.get('complete_distinct_evaluations',0) for x in actions),
        joint_cross_candidates_evaluated=sum(x.get('joint_cross_candidates_evaluated',0) for x in actions),
        decision_wall_ms={'mean':statistics.mean(latencies) if latencies else 0,'p95':quantile(latencies,.95),'max':max(latencies,default=0)},
        ready_discovery_ms={'mean':statistics.mean(1000*(x['discovered_at']-x['eligible_at']) for x in timeline if x['event']=='ready')},
        algorithm_window_renewed_by_heartbeat=False,common_continuation_preserves_state=True,real_nonregression_guaranteed=False)
    summary['routing_deviations_from_snapshot_affinity']=sum(p['action']['placement'][k]!=affinity(p['snapshot'],p['snapshot']['ready'])[k] for p in actions for k in p['action']['admit'])
    summary['per_replica_execution']={}
    for r in range(R):
        seq=[x for x in its if x['replica']==r];vals=[x['prefill_cap'] for x in seq]
        summary['per_replica_execution'][str(r)]={'iterations':len(seq),'budget_switches':sum(a!=b for a,b in zip(vals,vals[1:])),
            'current_action_iterations':sum(x['execution_phase']=='current_action' for x in seq),
            'common_continuation_iterations':sum(x['execution_phase']=='common_continuation' for x in seq),
            'native_fallback_iterations':sum(x['plan_reason']!='lease_ok' for x in seq)}
    summary['budget_switches_total']=sum(x['budget_switches'] for x in summary['per_replica_execution'].values())
    summary['selector_revision']=cfg.get('selector_revision')
    summary['selector_coverage']={
        'partial_group_generated':sum(x.get('partial_group_candidates_generated',0) for x in actions),
        'partial_group_evaluated':sum(x.get('partial_group_candidates_evaluated',0) for x in actions),
        'partial_group_complete':sum(x.get('partial_group_complete_evaluations',0) for x in actions),
        'partial_group_selected_by_worker':sum(bool(x.get('selected_partial_group')) for x in actions),
        'partial_group_committed':sum('partial_join/' in x['action']['family'] and not x.get('commit_fallback') for x in actions),
        'received_before_total_deadline':sum(bool(x.get('received_before_total_deadline')) for x in actions),
        'reference_unrankable':sum(bool(x.get('reference_unrankable')) for x in actions),
        'retained_backend_revision':REVISION}
    summary['semantic_selector_coverage']={
        'paired_packets_complete':sum(bool(x.get('paired_packet_complete')) for x in actions),
        'paired_packets_attempted':sum(bool(x.get('paired_packet_attempted')) for x in actions),
        'decisions_with_distinct_placement_and_budget':sum(bool(x.get('paired_has_distinct_placement_and_budget')) for x in actions),
        'distinct_paired_packets_complete':sum(bool(x.get('paired_has_distinct_placement_and_budget')) and bool(x.get('paired_packet_complete')) for x in actions),
        'decode_guard_rejections':sum(x.get('decode_guard_rejections',0) for x in actions),
        'no_state_opportunity':sum(bool(x.get('no_state_opportunity')) for x in actions),
        'known_dag_group_proposals':sum(bool(x.get('known_dag_group_proposed')) for x in actions),
        'known_dag_alternatives_generated':sum(bool(x.get('known_dag_alternative_generated')) for x in actions),
        'known_dag_partial_group_committed':sum('known_dag_partial_group' in x['action']['family'] and not x.get('commit_fallback') for x in actions),
        'full_scores_not_token_ratio':True,
        'compiled_DAG_used':bool(cfg.get('semantic_selector'))}
    planned={x['plan_id']:x for x in publisher.records}
    def matches(x):
        if x['plan_id'] not in planned:return False
        p=planned[x['plan_id']];at=x.get('effective_ns',x['begin_ns']);active=p['not_before_ns']<=at<p['action_expires_ns']
        expected_cap=p['caps'][x['replica']] if active else publisher.base_caps[x['replica']]
        expected_total=p['totals'][x['replica']] if active else cfg['hard_total']
        return (x['execution_phase']=='current_action')==active and x['prefill_cap']==expected_cap and x['total']==expected_total
    summary['control_time_match']=all(matches(x) for x in its if x['plan_reason']=='lease_ok')
    effective=[]
    for p in actions:
        if lifetime_probe and not p.get('publication_performed'):continue
        matches_for_plan=[x for x in its if x['plan_id']==p['published_plan_id'] and x['execution_phase']=='current_action']
        for r in range(R):
            xs=[x for x in matches_for_plan if x['replica']==r]
            if xs:
                first=min(xs,key=lambda x:x.get('effective_ns',x['begin_ns']));dt=(first.get('effective_ns',first['begin_ns'])-p['snapshot_ns'])/1e6
                effective.append({'decision_index':p['decision_index'],'replica':r,'actual_effective_delay_ms':dt,'estimated_apply_delay_ms':p['estimated_apply_delay_ms'],
                                  'prediction_error_ms':dt-p['estimated_apply_delay_ms'],'snapshot_ns':p['snapshot_ns'],'plan_done_s':p['plan_done_s'],
                                  'published_s':p['published_s'],'effective_ns':first.get('effective_ns',first['begin_ns']),'expire_ns':p['action_expires_ns']})
    summary['backend_effective_delay_ms']={'n':len(effective),'mean':statistics.mean(x['actual_effective_delay_ms'] for x in effective) if effective else None,'p95':quantile([x['actual_effective_delay_ms'] for x in effective],.95)}
    atomic(folder/'action_timing.json',effective)
    # Explicit cache evidence -> actual first model-input token skip correspondence.
    chains=[]
    for k,n in nodes.items():
        if not n.get('prefix_hash') or not n.get('parents'):continue
        plan=next((p for p in actions if k in p['action']['admit']),None)
        if plan is None:continue
        evidence=next((e for e in plan['snapshot']['cache_evidence'] if e['replica']==n['replica'] and e['prefix_hash']==n['prefix_hash']),None)
        first=next((x for x in inputs if x['prompt_hash']==n['hash']),None)
        shared_parent=any(nodes[pp].get('prefix_hash')==n['prefix_hash'] for pp in n['parents'])
        record={'node':k,'shared_parent':shared_parent,'replica':n['replica'],
                'parent_finished_before_ready':all(nodes[pp]['finish']<=n['ready_at']+.00001 for pp in n['parents']),
                'observed_tokens':None if evidence is None else evidence['tokens'],
                'first_input_skipped':None if first is None else first['skipped_query_tokens'],
                'first_input_query':None if first is None else first['query_tokens'],
                'native_cache_evidence':evidence}
        lower=None if not evidence or not first else min(max(0,first['query_tokens_before_cache']-1),max(0,evidence['tokens']-first['context_before_cache']))
        record['expected_min_skip_from_snapshot']=lower
        record['skip_below_observed_prefix_bound']=bool(lower is not None and first['skipped_query_tokens']<lower)
        record['root_to_branch_evidence_pass']=bool(shared_parent and evidence and evidence['tokens']>=n['share'] and first and first['skipped_query_tokens']>0 and not record['skip_below_observed_prefix_bound'] and record['parent_finished_before_ready'])
        chains.append(record)
    summary['root_to_branch_cache_links_verified']=sum(x['root_to_branch_evidence_pass'] for x in chains)
    atomic(folder/'cache_chain_evidence.json',chains)
    atomic(folder/'one_step_check.json',checks);atomic(folder/'cap_activity.json',bind)
    if model is not None:
        from .validation import service_diagnostics
        diag=service_diagnostics(collected,model);atomic(folder/'service_prediction_diagnostics.json',diag)
        summary['service_prediction_by_phase']=diag['by_phase']
    if mechanism and mechanism.record is not None:
        forced_ids=[p['published_plan_id'] for p in actions if p['action']['family'].startswith('forced/')]
        forced_iterations=[x for x in its if x['plan_id'] in forced_ids and x['execution_phase']=='current_action']
        summary['forced_window_activity']={'iterations':len(forced_iterations),'mixed_iterations':sum(x['prefill_logical_tokens']>0 and x['decode_tokens']>0 for x in forced_iterations),
             'prefill_logical_tokens':sum(x['prefill_logical_tokens'] for x in forced_iterations),'binding_candidates':sum(binding_evidence(x)['binding_candidate'] for x in forced_iterations),
             'caps_seen':sorted({x['prefill_cap'] for x in forced_iterations if x['prefill_cap'] is not None})}
    if mechanism:summary['mechanism_trigger']=None if mechanism.record is None else {k:v for k,v in mechanism.record.items() if k not in ('snapshot','action')}
    if lifetime_probe:
        life=finite_window_evidence(lifetime_trigger,publisher.records,publisher.flush_records,its,nodes)
        summary['finite_lifetime_evidence']=life
        summary['finite_lifetime_probe_pass']=life['passed']
        summary['lifetime_probe_revision']=LIFETIME_REVISION
        atomic(folder/'finite_lifetime_evidence.json',life)
    summary['condition_valid']=bool(summary.get('condition_valid',True) and summary['budget_invariants'] and summary['control_time_match'])
    atomic(folder/'summary.json',summary)
    if not summary['condition_valid']:raise RuntimeError('condition control/budget invariant failed; records preserved')
    print('CONDITION_DONE='+json.dumps({'method':method,'folder':str(folder),**{k:v for k,v in summary.items() if k not in ('workflow_rows','unfinished_workflow_curve','request_latency_rows')}},ensure_ascii=False),flush=True)
    return summary,collected
