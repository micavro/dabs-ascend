"""Shared phase-aware HOST-duration contract (not GPU kernel time)."""
from __future__ import annotations
import math
PHASES=('prefill_only','decode_only','mixed')

def phase(p,d):
    return 'mixed' if p>0 and d>0 else 'prefill_only' if p>0 else 'decode_only'

def features(p,d,dc,pc,cache=0):
    eff=max(0.,float(p)-float(cache))
    return (1.,eff/1024.,d/16.,dc/65536.,eff*pc/4194304.,eff*d/16384.)

def coefficients(model,r,which):
    m=model['models'][str(r)]
    record=m.get('phases',{}).get(which,m)
    c=tuple(float(x) for x in record['coef'])
    if len(c)!=6 or any(not math.isfinite(v) or v<0 for v in c):
        raise ValueError('service model requires finite nonnegative six-feature coefficients')
    return c

def predict(model,r,p,d,dc,pc,cache=0):
    c=coefficients(model,r,phase(p,d));x=features(p,d,dc,pc,cache)
    return max(.1,sum(a*b for a,b in zip(c,x)))
