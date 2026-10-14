"""CPU-only API-shaped native engine. Not a vLLM/GPU fidelity claim."""
from __future__ import annotations
import asyncio,types
from dataclasses import dataclass,field
from collections import deque
from types import SimpleNamespace as NS

@dataclass
class NativeBudget:
    token_budget:int
    max_num_seqs:int
    _request_ids_num_batched_tokens:set=field(default_factory=set)
    _request_ids_num_curr_seqs:set=field(default_factory=set)
    _num_batched_tokens:int=0
    _num_curr_seqs:int=0
    @property
    def num_batched_tokens(self):return self._num_batched_tokens
    @property
    def num_curr_seqs(self):return self._num_curr_seqs
    def remaining_token_budget(self):return self.token_budget-self._num_batched_tokens
    def can_schedule(self,num_new_tokens,num_new_seqs):return num_new_tokens>0 and self._num_batched_tokens+num_new_tokens<=self.token_budget and self._num_curr_seqs+num_new_seqs<=self.max_num_seqs
    def add_num_batched_tokens(self,req_id,num_batched_tokens):
        if req_id not in self._request_ids_num_batched_tokens:self._request_ids_num_batched_tokens.add(req_id);self._num_batched_tokens+=num_batched_tokens
    def subtract_num_batched_tokens(self,req_id,num_batched_tokens):
        if req_id in self._request_ids_num_batched_tokens:self._request_ids_num_batched_tokens.remove(req_id);self._num_batched_tokens-=num_batched_tokens
    def add_num_seqs(self,req_id,num_curr_seqs):
        if req_id not in self._request_ids_num_curr_seqs:self._request_ids_num_curr_seqs.add(req_id);self._num_curr_seqs+=num_curr_seqs
    def subtract_num_seqs(self,req_id,num_curr_seqs):
        if req_id in self._request_ids_num_curr_seqs:self._request_ids_num_curr_seqs.remove(req_id);self._num_curr_seqs-=num_curr_seqs

class FakeGroup:
    def __init__(self,rid,tokens,output):
        self.request_id=rid;self.tokens=list(tokens);self.computed=0;self.done=0;self.output=output
        self.data=NS(get_num_computed_tokens=lambda:self.computed)
    def get_seqs(self,status=None):return [self]
    def is_prefill(self):return self.computed<len(self.tokens)
    def get_prompt_token_ids(self):return self.tokens
    def get_output_len(self):return self.done
    def get_len(self):return len(self.tokens)+self.done
    def get_num_new_tokens(self):return max(1,len(self.tokens)-self.computed)

def hashes(t,b,kind='v1'):return [hash(tuple(t[:i])) for i in range(b,len(t)+1,b)]

def modules():
    sm=types.ModuleType('vllm.core.scheduler');sm.SchedulingBudget=NativeBudget
    class Builder:
        def _compute_for_prefix_cache_hit(self,inter_data,seq_idx,seq_group_metadata):
            cached=len(seq_group_metadata.computed_block_nums)*16
            before=inter_data.context_lens[seq_idx];inputs=inter_data.input_tokens[seq_idx]
            skip=min(max(0,len(inputs)-1),max(0,cached-before))
            inter_data.input_tokens[seq_idx]=inputs[skip:];inter_data.query_lens[seq_idx]=len(inputs)-skip
            inter_data.context_lens[seq_idx]+=skip;inter_data.prefix_cache_hit=bool(skip)
    class Scheduler:
        def __init__(self):
            self.scheduler_config=NS(chunked_prefill_enabled=True,is_multi_step=False,num_lookahead_slots=0,max_num_batched_tokens=4096,max_num_seqs=64,max_model_len=4096,num_scheduler_steps=1)
            self.cache_config=NS(enable_prefix_caching=True,block_size=16,num_gpu_blocks=4096,sliding_window=None)
            self.lora_enabled=False;self.running=deque();self.waiting=deque();self.swapped=deque()
            alloc=NS(cached_blocks={},evictor=NS(free_table={}))
            self.block_manager=NS(gpu_allocator=alloc,get_block_table=lambda q:list(range((q.computed+q.done+15)//16)),get_num_free_gpu_blocks=lambda:4096-sum((g.computed+g.done+15)//16 for g in self.running))
        def add_seq_group(self,g):self.waiting.append(g)
        def _get_num_new_tokens(self,seq_group,status,enable_chunking,budget):
            left=seq_group.get_num_new_tokens()
            if left>budget.remaining_token_budget():left=budget.remaining_token_budget()//16*16
            return left
        def schedule(self):
            budget=sm.SchedulingBudget(self.scheduler_config.max_num_batched_tokens,64);out=[];md=[];newrunning=[]
            active=deque(self.running);self.running.clear()
            while active or self.waiting:
                source=active if active else self.waiting;g=source[0]
                take=self._get_num_new_tokens(g,None,True,budget)
                if not take or not budget.can_schedule(num_new_tokens=take,num_new_seqs=1):newrunning+=list(active);active.clear();break
                source.popleft();pre=g.is_prefill();budget.add_num_batched_tokens(g.request_id,take);budget.add_num_seqs(g.request_id,1)
                a=self.block_manager.gpu_allocator;hit=[]
                for hh in hashes(g.tokens,16)[:-1]:
                    x=a.cached_blocks.get(hh) or a.evictor.free_table.get(hh)
                    if x is None or not x.computed:break
                    hit.append(x.block_number)
                out.append(NS(seq_group=g,token_chunk_size=take));md.append(NS(request_id=g.request_id,is_prompt=pre,computed_block_nums=hit,token_chunk_size=take));newrunning.append(g)
            self.running.extend(newrunning);return md,NS(scheduled_seq_groups=out,preempted=0),False
    class Engine:
        def __init__(self,s):self.scheduler=[s];self.fail=False
        def do_step(self):
            s=self.scheduler[0];md,out,_=s.schedule()
            if self.fail:raise RuntimeError('fixture forward failure')
            alloc=s.block_manager.gpu_allocator
            for m,x in zip(md,out.scheduled_seq_groups):
                g=x.seq_group
                if m.is_prompt:
                    inter=NS(is_prompt=True,input_tokens=[g.tokens[g.computed:g.computed+x.token_chunk_size]],query_lens=[x.token_chunk_size],context_lens=[g.computed],prefix_cache_hit=False)
                    Builder()._compute_for_prefix_cache_hit(inter,0,m)
                    g.computed+=x.token_chunk_size
                    for j,hh in enumerate(hashes(g.tokens[:min(len(g.tokens),g.computed)],16)):
                        old=alloc.evictor.free_table.pop(hh,None)
                        alloc.cached_blocks[hh]=old or NS(block_hash=hh,computed=True,block_number=abs(hh)%100000,ref_count=1)
                    if g.computed>=len(g.tokens):g.done=max(1,g.done)
                else:g.done+=1;g.computed+=1
            keep=[]
            for g in s.running:
                if g.done<g.output:keep.append(g)
                else:
                    for hh in hashes(g.tokens,16):
                        block=alloc.cached_blocks.pop(hh,None)
                        if block:block.ref_count=0;alloc.evictor.free_table[hh]=block
            s.running=deque([g for g in keep if not g.is_prefill()]+[g for g in keep if g.is_prefill()]);return []
        async def step_async(self,virtual_engine=0):z=self.do_step();await asyncio.sleep(0);return z
        def step(self):return self.do_step()
    sm.Scheduler=Scheduler
    v=types.ModuleType('vllm');v.__version__='0.6.3';core=types.ModuleType('vllm.core');core.scheduler=sm;v.core=core
    eng=types.ModuleType('vllm.engine');ae=types.ModuleType('vllm.engine.async_llm_engine');le=types.ModuleType('vllm.engine.llm_engine');ae._AsyncLLMEngine=Engine;le.LLMEngine=Engine
    worker=types.ModuleType('vllm.worker');mr=types.ModuleType('vllm.worker.model_runner');mr.ModelInputForGPUBuilder=Builder;worker.model_runner=mr;v.worker=worker
    mods={'vllm':v,'vllm.core':core,'vllm.core.scheduler':sm,'vllm.engine':eng,'vllm.engine.async_llm_engine':ae,'vllm.engine.llm_engine':le,'vllm.worker':worker,'vllm.worker.model_runner':mr}
    return mods,sm,Engine
