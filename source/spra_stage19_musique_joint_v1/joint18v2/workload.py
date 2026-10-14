from __future__ import annotations
import random
from .common import token_hash
SPECS={
 "original6": [(2048,16),(2816,48),(3328,64),(2176,192),(2304,224),(256,32)],
 # New <=4096 heterogeneous profile, not the long-context proposal and not historical data.
 "contrast4k": [(1536,64),(3328,48),(3072,64),(768,512),(1024,384),(1536,96)]
}
NAMES=("root","B1","B2","B3","B4","sink")
def edges(topology):
    if topology=="ParallelRAG":return [[],[0],[0],[0],[0],[1,2,3,4]]
    if topology=="AsymmetricAgent":return [[],[0],[0],[0],[1],[2,3,4]]
    if topology=="Sequential":return [[],[0],[1],[2],[3],[4]]
    raise ValueError("unsupported topology")
def arrivals(n,rate,shape,seed):
    if n<1 or rate<=0:raise ValueError("arrival parameters")
    rng=random.Random(seed);out=[]
    if shape=="steady":out=[(i+.5)/rate for i in range(n)]
    elif shape=="wave":
        count=max(1,round(2*rate))
        out=[2*(i//count)+(i%count+.5)/count for i in range(n)]
    elif shape=="poisson":
        now=0
        for _ in range(n):now+=rng.expovariate(rate);out.append(now)
    else:raise ValueError(shape)
    return out
def build(cfg,pool,seed,salt):
    """Synthetic token IDs from valid tokenizer outputs; no hidden call or output-token reduction.
    Dependencies gate execution, but parent text is intentionally not rewritten into child inputs.
    """
    specs=SPECS[cfg["input_profile"]];parents=edges(cfg["topology"])
    R=random.Random(seed);result={}
    offsets=arrivals(cfg["workflow_count"],cfg["arrival_rate"],cfg["arrival_shape"],seed)
    for w in range(cfg["workflow_count"]):
        fam=w%cfg["prefix_families"];pseed=seed*7919+fam
        prng=random.Random(pseed)
        share=min(cfg["share_tokens"],min(x[0]-16 for x in specs[:5]))
        prefix=(salt+[prng.choice(pool) for _ in range(share)])[:share]
        # All methods get matching token work and prefix-family relation; salt is a fresh namespace.
        for j,name in enumerate(NAMES):
            total,output=specs[j]
            if total+output>cfg["max_model_len"]:raise ValueError("prompt+output exceeds engine context; no silent clipping")
            rng=random.Random(seed*1000003+w*29+j)
            pref=prefix if name!="sink" else []
            private=salt+[pool[(w*7+j)%len(pool)]]+[rng.choice(pool) for _ in range(total)]
            tokens=(pref+private[:total-len(pref)])[:total]
            key=f"w{w:04d}/{name}";h=token_hash(tokens)
            if any(x["hash"]==h for x in result.values()):raise ValueError("duplicate request token hash")
            sh=len(pref)
            result[key]={"id":key,"wid":f"w{w:04d}","name":name,"parents":[f"w{w:04d}/{NAMES[p]}" for p in parents[j]],
              "prompt":total,"output":output,"tokens":tokens,"hash":h,
              "prefix_hash":token_hash(tokens[:sh]) if sh else None,"share":sh,
              "arrival_offset":offsets[w],"state":"blocked","replica":None,
              "computed":0,"output_done":0,"ready_at":None,"start":None,"finish":None}
    return result
