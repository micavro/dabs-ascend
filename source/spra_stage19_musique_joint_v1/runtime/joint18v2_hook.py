"""Pinned vLLM 0.6.3 hook: native cache evidence, finite action windows and input-token audit.

Install only through this package's launcher in an idle experiment deployment.
No vLLM source file is edited. No model inference occurs during --precheck.
"""
from __future__ import annotations
import contextvars,inspect,json,os,threading,time
from collections import deque
from pathlib import Path
from joint18v2.common import VERSION,PREFIX_SIZES,atomic,load,boot_id,clock_namespace,token_hash,make_budget_class,resolve_control,stable_prefill_reorder
from joint18v2.native_contract import capabilities as prior_capabilities,native_facts
from joint18v2.cache import NativeCacheObserver,cache_contract_precheck
from joint18v2.io import JsonWriter
from joint18v2.bootstrap import marker, precheck_bootstrap, REVISION
_CTX=contextvars.ContextVar('joint18v2-budget',default={})
_INSTALLED=False;_SCHEDULERS=[]

def root():return Path(os.environ.get('SPRA_E1_ROOT','/spra-e1'))/'joint18v2'
def capabilities():
    c=prior_capabilities()
    from vllm.worker.model_runner import ModelInputForGPUBuilder
    sig=list(inspect.signature(ModelInputForGPUBuilder._compute_for_prefix_cache_hit).parameters)
    if sig!=['self','inter_data','seq_idx','seq_group_metadata']:raise RuntimeError('model input prefix interface mismatch: '+str(sig))
    c['cache_metadata_cpu_contract']=cache_contract_precheck();c['runner_prefix_signature']=sig
    c['integration_version']=VERSION
    c['bootstrap_contract']=precheck_bootstrap()
    return c

def event(s,obj):
    record={'version':VERSION,**s._v2_identity,'mono_ns':time.monotonic_ns(),'iteration':s._v2_iter,
            'case':s._v2_plan.get('case',s._v2_query_case or 'bootstrap'),**obj}
    s._v2_writer.submit(root()/(s._v2_identity['pod']+'.events.jsonl'),record)

def meta(s,g):
    if g.request_id not in s._v2_meta:
        seqs=g.get_seqs()
        if len(seqs)!=1:raise RuntimeError('only n=1 requests supported')
        t=list(seqs[0].get_prompt_token_ids())
        s._v2_meta[g.request_id]={'request_id':g.request_id,'prompt_hash':token_hash(t),'prompt':len(t),
            'prefixes':{str(k):token_hash(t[:k]) for k in PREFIX_SIZES if k<=len(t)},'accepted_s':time.monotonic()}
    return s._v2_meta[g.request_id]

def entry(s,g):
    seq=g.get_seqs()[0]
    try:allocated=len(s.block_manager.get_block_table(seq))
    except (KeyError,AttributeError,AssertionError):allocated=0
    return {**meta(s,g),'prefill':bool(g.is_prefill()),'computed':int(seq.data.get_num_computed_tokens()),
            'output_done':int(seq.get_output_len()),'allocated_blocks':allocated,'context':int(seq.get_len())}

def snapshot(s):return {q:[entry(s,g) for g in getattr(s,q)] for q in ('running','waiting','swapped')}

def refresh_query(s):
    path=root()/'prefix_query.json'
    try:
        stamp=path.stat().st_mtime_ns
        if stamp==s._v2_query_stamp:return
        q=load(path)
        if q.get('boot_id')!=s._v2_identity['boot_id']:raise RuntimeError('cache query clock mismatch')
        s._v2_observer.register(q);s._v2_query_stamp=stamp;s._v2_query_case=q.get('case')
    except FileNotFoundError:return

def capture(s,completed=False,idle=False):
    # Caller holds the scheduler's re-entrant lock. Native changes are suspended here.
    refresh_query(s);now=time.monotonic_ns();ss=snapshot(s)
    if not s._v2_inflight and s._v2_fatal is None:
        cache=s._v2_observer.sample(now);s._v2_cache=cache
        for change in cache['changes']:
            detail=next(e for e in cache['records'] if e['prefix_hash']==change['prefix_hash'])
            event(s,{'event':'cache_transition','cache_change':change,'native_record':detail,'query_id':cache['query_id']})
    else:cache=s._v2_cache
    # Full block records are retained on observed transitions, not copied into every heartbeat.
    cache_status={k:v for k,v in cache.items() if k not in ('records','changes')}
    cache_status['records']=[{**{k:v for k,v in e.items() if k!='blocks'},'block_ids':[b['block_id'] for b in e['blocks']]} for e in cache['records']]
    free=s.block_manager.get_num_free_gpu_blocks()
    obj={'version':VERSION,**s._v2_identity,**s._v2_env,'iteration':s._v2_iter,'mono_ns':now,
         'snapshot_after_engine_step':completed,'idle_observation':idle,'engine_step_inflight':bool(s._v2_inflight),
         'case':s._v2_plan.get('case',s._v2_query_case or 'bootstrap'),
         'plan_id':s._v2_plan.get('plan_id'),'plan_reason':s._v2_reason,
         'execution_phase':s._v2_plan.get('execution_phase','native_fallback'),
         'total':s._v2_plan.get('total',s._v2_env['hard_total']),'prefill_cap':s._v2_plan.get('prefill_cap'),
         'kv_used_blocks':s._v2_env['num_gpu_blocks']-free,'cache':cache_status,
         'inflight':s._v2_latest if s._v2_inflight and s._v2_latest else None,
         'writer_high_water':s._v2_writer.high_water,'fatal_engine_error':s._v2_fatal,**ss}
    s._v2_writer.submit(root()/(s._v2_identity['pod']+'.status.json'),obj,replace=True)
    return obj

def read_plan(s):
    try:
        b=load(root()/'control.json');p=b.get('replicas',{}).get(s._v2_identity['pod'],{}) if b.get('enabled') else {}
        now=time.monotonic_ns();out=resolve_control(p,s._v2_identity,s._v2_env,now)
        out['effective_ns']=now
        return out,'lease_ok'
    except (OSError,ValueError,TypeError) as ex:
        return {'enabled':False,'case':s._v2_plan.get('case',s._v2_query_case or 'bootstrap'),
                'plan_id':None,'total':s._v2_env['hard_total'],'prefill_cap':None,'order':[],
                'execution_phase':'native_fallback'},str(ex)

def reorder(s,priority):
    result={}
    for name in ('running','waiting'):
        gs=list(getattr(s,name));items=[entry(s,g) for g in gs]
        if not priority:
            # Common continuation is FIFO by original acceptance for prefills; decodes remain native.
            pos=[i for i,x in enumerate(items) if x['prefill']]
            order=list(items)
            for i,x in zip(pos,sorted([items[i] for i in pos],key=lambda x:(x['accepted_s'],x['request_id']))):order[i]=x
            changed=[x['request_id'] for x in order]!=[x['request_id'] for x in items];why='fifo_acceptance'
        else:order,changed,why=stable_prefill_reorder(items,priority,time.monotonic())
        mapping={g.request_id:g for g in gs};setattr(s,name,deque(mapping[x['request_id']] for x in order))
        result[name]={'changed':changed,'reason':why}
    return result

def install():
    global _INSTALLED
    if _INSTALLED:return
    cap=capabilities()
    import vllm.core.scheduler as sm
    from vllm.engine.async_llm_engine import _AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.worker.model_runner import ModelInputForGPUBuilder
    ni,na,ng,ns=sm.Scheduler.__init__,sm.Scheduler.add_seq_group,sm.Scheduler._get_num_new_tokens,sm.Scheduler.schedule
    n_async,n_sync=_AsyncLLMEngine.step_async,LLMEngine.step
    n_prefix=ModelInputForGPUBuilder._compute_for_prefix_cache_hit
    sm.SchedulingBudget=make_budget_class(sm.SchedulingBudget,_CTX.get)
    def init_impl(s,*args,**kwargs):
        ni(s,*args,**kwargs);sc,cc=s.scheduler_config,s.cache_config
        if not sc.chunked_prefill_enabled or sc.is_multi_step or not cc.enable_prefix_caching:raise RuntimeError('requires single-step APC/chunked-prefill')
        if sc.num_lookahead_slots or s.lora_enabled or getattr(cc,'sliding_window',None):raise RuntimeError('no lookahead, LoRA or sliding window in this envelope')
        s._v2_identity={'pod':os.environ.get('POD_NAME','unknown'),'pid':os.getpid(),'boot_id':boot_id()}
        s._v2_env={'recovery_revision':REVISION,'launch_id':os.environ.get('SPRA_J18V2_LAUNCH_ID'),'hard_total':int(sc.max_num_batched_tokens),'max_num_seqs':int(sc.max_num_seqs),'max_model_len':int(sc.max_model_len),
          'block_size':int(cc.block_size),'num_gpu_blocks':int(cc.num_gpu_blocks),'num_scheduler_steps':int(sc.num_scheduler_steps),
          'prefix_caching':True,'chunked_prefill':True,'nvidia_visible_devices':os.environ.get('NVIDIA_VISIBLE_DEVICES'),
          'clock_namespace_inode':clock_namespace(),'scheduler_source_sha256':cap['scheduler_source_sha256'],'vllm_imported_version':cap['imported_version'],
          'vllm_distribution_version':cap['distribution_version'],'hook_source_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}
        if s._v2_env['hard_total']!=4096 or s._v2_env['block_size']!=16:raise RuntimeError('requires hard4096/block16')
        s._v2_iter=0;s._v2_meta={};s._v2_output_counts={};s._v2_step_groups=[];s._v2_plan={};s._v2_reason='bootstrap';s._v2_latest=None;s._v2_inflight=0
        s._v2_stop=threading.Event();s._v2_fatal=None
        s._v2_lock=threading.RLock();s._v2_writer=JsonWriter();s._v2_query_stamp=None;s._v2_query_case=None
        s._v2_observer=NativeCacheObserver(s.block_manager,16);s._v2_env['native_cache_allocator']=s._v2_observer.kind
        s._v2_cache={'records':[],'query_id':None,'observed_ns':0};_SCHEDULERS.append(s)
        with s._v2_lock:capture(s,completed=True)
        s._v2_writer.drain()
        marker("scheduler_status_written",status_file=str(root()/(s._v2_identity["pod"]+".status.json")),native_cache_allocator=s._v2_observer.kind)
        def idle_watch():
            while not s._v2_stop.is_set():
                try:
                    with s._v2_lock:
                        if not s._v2_inflight and s._v2_fatal is None:capture(s,idle=True)
                    time.sleep(.03)
                except BaseException as ex:
                    s._v2_writer.error='idle_cache_observer: '+repr(ex);marker('idle_observer_failed',error=repr(ex));return
        threading.Thread(target=idle_watch,daemon=True,name='joint18v2-idle-cache').start()
    def init(s,*args,**kwargs):
        marker('scheduler_init_enter')
        try:return init_impl(s,*args,**kwargs)
        except BaseException as exc:
            import traceback
            marker('scheduler_init_failed',error=repr(exc),traceback=traceback.format_exc())
            raise
    def add(s,g):
        with s._v2_lock:
            if s._v2_fatal is not None:raise RuntimeError('engine failure: refuse new requests')
            s._v2_writer.check();na(s,g);s._v2_plan,s._v2_reason=read_plan(s)
            event(s,{'event':'accept',**entry(s,g)});capture(s)
    def get(s,seq_group,status,enable_chunking,budget):
        if hasattr(budget,'j18_phases'):
            pre=bool(seq_group.is_prefill());budget.j18_current_prefill=pre;budget.j18_phases[seq_group.request_id]=pre
        return ng(s,seq_group,status,enable_chunking,budget)
    def schedule(s,*args,**kwargs):
        with s._v2_lock:
            if s._v2_fatal is not None:raise RuntimeError('engine failure: refuse scheduling')
            s._v2_writer.check();s._v2_iter+=1;s._v2_plan,s._v2_reason=read_plan(s);plan=s._v2_plan
            refresh_query(s);before=snapshot(s);order=reorder(s,plan.get('order',[]));ordered=snapshot(s)
            old=s.scheduler_config.max_num_batched_tokens;s.scheduler_config.max_num_batched_tokens=plan['total']
            token=_CTX.set({'prefill_cap':plan['prefill_cap']});begin=time.monotonic_ns()
            try:result=ns(s,*args,**kwargs)
            finally:s.scheduler_config.max_num_batched_tokens=old;_CTX.reset(token)
            md,out,_=result
            s._v2_step_groups=[x.seq_group for x in out.scheduled_seq_groups]
            bymd={m.request_id:m for m in md};prior={v['request_id']:v for q in ('running','waiting') for v in before[q]}
            scheduled=[]
            for x in out.scheduled_seq_groups:
                g=x.seq_group;m=bymd[g.request_id];v=prior[g.request_id];cache=getattr(m,'computed_block_nums',None)
                scheduled.append({'request_id':g.request_id,'prompt_hash':meta(s,g)['prompt_hash'],
                    'prefill':bool(m.is_prompt),'tokens':int(x.token_chunk_size),
                    'computed_before':v['computed'],'output_before':v['output_done'],'context_before':v['context'],
                    'cached_blocks_metadata':len(cache) if cache is not None else 0})
            p=sum(x['tokens'] for x in scheduled if x['prefill']);d=sum(x['tokens'] for x in scheduled if not x['prefill'])
            if p+d>plan['total'] or (plan['prefill_cap'] is not None and p>plan['prefill_cap']):raise RuntimeError('native budget invariant')
            s._v2_latest={'begin_ns':begin,'scheduler_host_ms':(time.monotonic_ns()-begin)/1e6,'scheduled':scheduled,
                'total':plan['total'],'prefill_cap':plan['prefill_cap'],'plan_id':plan.get('plan_id'),'plan_reason':s._v2_reason,
                'execution_phase':plan.get('execution_phase'),'action_expires_ns':plan.get('action_expires_ns'),
                'snapshot_ns':plan.get('snapshot_ns'),'not_before_ns':plan.get('not_before_ns'),'effective_ns':plan.get('effective_ns',begin),
                'before':before,'ordered_before':ordered,'order':order,'prefill_logical_tokens':p,'decode_tokens':d,
                'preemptions':int(out.preempted),'remaining_prefill_in_before':sum(max(0,v['prompt']-v['computed']) for q in ('running','waiting') for v in before[q] if v['prefill']),
                'free_blocks_before':s.block_manager.get_num_free_gpu_blocks()}
            event(s,{'event':'iteration',**s._v2_latest});capture(s)
            return result
    def runner_prefix(builder,inter_data,seq_idx,metadata):
        before_count=len(inter_data.input_tokens[seq_idx]);before_context=int(inter_data.context_lens[seq_idx])
        z=n_prefix(builder,inter_data,seq_idx,metadata)
        if inter_data.is_prompt:
            for s in _SCHEDULERS:
                if metadata.request_id in s._v2_meta:
                    event(s,{'event':'model_input','request_id':metadata.request_id,'prompt_hash':s._v2_meta[metadata.request_id]['prompt_hash'],
                      'logical_chunk':int(metadata.token_chunk_size),'context_before_cache':before_context,
                      'query_tokens_before_cache':before_count,'query_tokens':int(inter_data.query_lens[seq_idx]),
                      'input_token_count':len(inter_data.input_tokens[seq_idx]),'skipped_query_tokens':before_count-len(inter_data.input_tokens[seq_idx]),
                      'context_after_cache':int(inter_data.context_lens[seq_idx]),'prefix_cache_hit':bool(inter_data.prefix_cache_hit)})
                    break
        return z
    def observe_outputs(s,result,observed_ns):
        # Sequence counters survive removal from the active queue. RequestOutput
        # is also inspected because asynchronous postprocessing may release an
        # output for a request scheduled in an earlier iteration.
        counts={}
        for g in s._v2_step_groups:
            seqs=g.get_seqs()
            if len(seqs)==1:counts[g.request_id]=int(seqs[0].get_output_len())
        if isinstance(result,(list,tuple)):
            for obj in result:
                rid=getattr(obj,'request_id',None);outs=getattr(obj,'outputs',None)
                if rid in s._v2_meta and isinstance(outs,(list,tuple)) and len(outs)==1:
                    tids=getattr(outs[0],'token_ids',None)
                    if tids is not None:counts[rid]=max(counts.get(rid,0),len(tids))
        for rid,count in counts.items():
            previous=s._v2_output_counts.get(rid,0)
            if count>previous:
                event(s,{'event':'output_progress','request_id':rid,
                    'prompt_hash':s._v2_meta[rid]['prompt_hash'],
                    'observed_ns':observed_ns,'output_tokens':count,
                    'previous_output_tokens':previous,'delta_tokens':count-previous,
                    'boundary':'host_after_engine_step','exact_gpu_token_time':False})
                s._v2_output_counts[rid]=count
            elif count<previous:
                event(s,{'event':'output_progress_anomaly','request_id':rid,
                    'previous_output_tokens':previous,'output_tokens':count})
    def begin_step(engine):
        scheds=[s for s in engine.scheduler if hasattr(s,'_v2_lock')]
        for s in scheds:
            with s._v2_lock:s._v2_inflight+=1;s._v2_latest=None
        return scheds,time.monotonic_ns()
    def end_step(scheds,begin,error=None,result=None):
        for s in scheds:
            with s._v2_lock:
                s._v2_inflight=max(0,s._v2_inflight-1)
                if error is not None:s._v2_fatal=repr(error)
                else:observe_outputs(s,result,time.monotonic_ns())
                event(s,{'event':'engine_step','engine_host_ms':(time.monotonic_ns()-begin)/1e6,'gpu_kernel_ms':None,'engine_error':None if error is None else repr(error)})
                # A failed forward cannot certify newly marked native blocks as computed.
                capture(s,completed=error is None)
    async def step_async(engine,*args,**kwargs):
        scheds,begin=begin_step(engine)
        try:result=await n_async(engine,*args,**kwargs)
        except BaseException as ex:end_step(scheds,begin,ex);raise
        end_step(scheds,begin,result=result);return result
    def step(engine,*args,**kwargs):
        scheds,begin=begin_step(engine)
        try:result=n_sync(engine,*args,**kwargs)
        except BaseException as ex:end_step(scheds,begin,ex);raise
        end_step(scheds,begin,result=result);return result
    sm.Scheduler.__init__=init;sm.Scheduler.add_seq_group=add;sm.Scheduler._get_num_new_tokens=get;sm.Scheduler.schedule=schedule
    _AsyncLLMEngine.step_async=step_async;LLMEngine.step=step;ModelInputForGPUBuilder._compute_for_prefix_cache_hit=runner_prefix
    _INSTALLED=True

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--precheck',action='store_true');p.add_argument('--diagnose',action='store_true');a=p.parse_args()
    if not (a.precheck or a.diagnose):raise SystemExit('Use --precheck; not a live install CLI')
    try:
        c=capabilities() if a.precheck else native_facts();print('NATIVE_CONTRACT='+json.dumps(c,sort_keys=True),flush=True)
        print('PRECHECK_PASS=True GPU_INTEGRATION_CHECK_REQUIRED=True',flush=True)
    except Exception as ex:
        print('NATIVE_DIAGNOSTIC='+json.dumps({**getattr(ex,'facts',{}),'error':repr(ex)}),flush=True)
        print('PRECHECK_PASS=False DO_NOT_RESTART=True',flush=True);raise SystemExit(2)
