"""Read-only native APC observer, scoped to registered token prefixes.

The adapter observes allocator-owned computed blocks, including refcount-zero blocks.
It never allocates a block, changes a reference count, touches LRU or uses route history
as residency. Calls must run at a completed-engine boundary or under the hook idle lock.
"""
from __future__ import annotations
import inspect,time
from types import SimpleNamespace
from .common import token_hash

class UnsupportedCache(RuntimeError):pass


def identify_allocator(manager):
    allocator=getattr(manager,'gpu_allocator',None)
    if allocator is not None and hasattr(allocator,'cached_blocks') and hasattr(allocator,'evictor'):
        ev=allocator.evictor
        if not isinstance(getattr(ev,'free_table',None),dict):
            raise UnsupportedCache('v1 evictor.free_table not available; refuse history substitution')
        return 'v1',allocator
    combined=getattr(manager,'block_allocator',None)
    if combined is not None:
        candidates=[]
        for attr in ('_gpu_allocator','gpu_allocator'):
            a=getattr(combined,attr,None)
            if a is not None:candidates.append(a)
        for attr in ('_allocators','allocators'):
            for key,a in getattr(combined,attr,{}).items():
                if str(getattr(key,'name',key)).upper().endswith('GPU'):candidates.append(a)
        for a in candidates:
            if hasattr(a,'_cached_blocks') and callable(getattr(a,'block_is_computed',None)):
                return 'v2',a
    raise UnsupportedCache('unrecognized native APC allocator: '+type(manager).__name__)


def native_hashes(tokens,block_size,kind):
    """Use the installed backend's own hash functions, not a cross-process hash guess."""
    length=len(tokens)//block_size*block_size
    if kind=='v1':
        from vllm.sequence import Sequence,SequenceData
        obj=SimpleNamespace(data=SequenceData.from_seqs(tokens),lora_int_id=0,
                            num_hashed_tokens_of_block=lambda i:(i+1)*block_size)
        return [Sequence.hash_of_block(obj,i) for i in range(length//block_size)]
    from vllm.core.block.prefix_caching_block import PrefixCachingBlock
    result=[];prev=None
    for i in range(0,length,block_size):
        prev=PrefixCachingBlock.hash_block_tokens(i==0,prev,list(tokens[i:i+block_size]));result.append(prev)
    return result


class NativeCacheObserver:
    def __init__(self,manager,block_size,hash_fn=native_hashes):
        self.kind,self.allocator=identify_allocator(manager);self.block=int(block_size);self.hash_fn=hash_fn
        self.queries={};self.previous={};self.generation=0;self.query_id=None
    def register(self,query):
        if query.get('schema')!='joint18v2-prefix-query-1':raise ValueError('cache query schema')
        if query.get('query_id')==self.query_id:return
        entries={}
        for x in query.get('prefixes',[]):
            toks=x['tokens'];h=x['prefix_hash']
            if not toks or len(toks)>4096 or any(type(t) is not int or t<0 for t in toks):raise ValueError('cache query tokens')
            if token_hash(toks)!=h:raise ValueError('cache query content hash mismatch')
            if h in entries and entries[h]['tokens']!=toks:raise ValueError('prefix hash collision')
            entries[h]={'tokens':list(toks),'native_hashes':self.hash_fn(toks,self.block,self.kind)}
        if len(entries)>512:raise ValueError('too many watched prefixes for this envelope')
        self.queries=entries;self.query_id=query['query_id'];self.previous={};self.generation+=1
    def _block(self,h):
        a=self.allocator
        if self.kind=='v1':
            b=a.cached_blocks.get(h)
            state='active_computed'
            if b is None:b=a.evictor.free_table.get(h);state='reclaimable_computed'
            if b is None:return {'state':'absent','computed':False,'block_id':None}
            if b.block_hash!=h:raise UnsupportedCache('v1 allocator hash identity mismatch')
            computed=bool(b.computed)
            return {'state':state if computed else 'resident_uncomputed','computed':computed,
                    'block_id':int(b.block_number),'ref_count':int(b.ref_count)}
        bid=a._cached_blocks.get(h)
        if bid is None:return {'state':'absent','computed':False,'block_id':None}
        comp=bool(a.block_is_computed(bid));reclaimable=bid in a.evictor
        return {'state':('reclaimable_computed' if reclaimable else 'active_computed') if comp else 'resident_uncomputed',
                'computed':comp,'block_id':int(bid),'ref_count':int(a._refcounter.get(bid))}
    def sample(self,at_ns=None):
        at_ns=time.monotonic_ns() if at_ns is None else int(at_ns);records=[];changes=[]
        for h,q in self.queries.items():
            blocks=[self._block(nh) for nh in q['native_hashes']]
            contiguous=0
            for b in blocks:
                if not b['computed']:break
                contiguous+=self.block
            signature=tuple((b['block_id'],b['state']) for b in blocks)
            previous=self.previous.get(h)
            if previous is None or previous!=signature:
                self.generation+=1
                dropped=(previous is not None and any(old[1].endswith('computed') and old[1]!='resident_uncomputed' and (i>=len(signature) or signature[i][0]!=old[0] or signature[i][1]=='absent') for i,old in enumerate(previous)))
                changes.append({'prefix_hash':h,'change':'eviction_or_replacement' if dropped else 'observed_state_change',
                                'generation':self.generation,'at_ns':at_ns,'previous':previous,'current':signature})
                self.previous[h]=signature
            state=('reclaimable_computed' if contiguous and all(b['state']=='reclaimable_computed' for b in blocks[:contiguous//self.block]) else
                   'active_computed' if contiguous else 'resident_uncomputed' if any(b['state']=='resident_uncomputed' for b in blocks) else 'absent')
            records.append({'prefix_hash':h,'query_tokens':len(q['tokens']),'tokens':contiguous,
                'state':state,'known':True,'blocks':blocks,'observed_ns':at_ns,
                'generation':self.generation,'allocator':self.kind,'evidence':'native_allocator_after_completed_step_or_idle'})
        return {'query_id':self.query_id,'observed_ns':at_ns,'generation':self.generation,
                'allocator':self.kind,'records':records,'changes':changes}


def cache_contract_precheck():
    """Allocate CPU metadata only; no model or GPU tensors are constructed."""
    from vllm.core.block_manager_v1 import CachedBlockAllocator
    from vllm.utils import Device
    a=CachedBlockAllocator(Device.GPU,16,4)
    manager=SimpleNamespace(gpu_allocator=a);observer=NativeCacheObserver(manager,16)
    tokens=list(range(32));h=token_hash(tokens)
    observer.register({'schema':'joint18v2-prefix-query-1','query_id':'native-precheck','prefixes':[{'prefix_hash':h,'tokens':tokens}]})
    assert observer.sample()['records'][0]['tokens']==0
    hs=observer.queries[h]['native_hashes'];blocks=[]
    for i,hh in enumerate(hs):
        b=a.allocate(hh,(i+1)*16);b.computed=True;blocks.append(b)
    assert observer.sample()['records'][0]['tokens']==32
    for b in blocks:a.free(b)
    record=observer.sample()['records'][0]
    assert record['tokens']==32 and record['state']=='reclaimable_computed'
    # Native precheck does not install this adapter into a running engine.
    return {'v1_free_prefix_readback':True,'tokens':32,'native_hash_used':True,'model_requests':0,'gpu_forward_passes':0}
