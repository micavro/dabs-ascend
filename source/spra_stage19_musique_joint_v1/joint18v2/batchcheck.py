"""One-step logical batch predictor for the common low-memory, n=1 envelope.

Mismatch is retained and classified; it is never silently called a model hit.
Model-runner input evidence separately measures skipped query tokens.
"""
def predict_next(ordered_before,total,cap,block=16,max_seqs=64):
    active=list(ordered_before['running']);wait=list(ordered_before['waiting'])
    dec=[x for x in active if not x['prefill']][:max_seqs]
    pre=[x for x in active if x['prefill']]+wait
    left=total-len(dec);room=min(left,cap if cap is not None else left)
    out={x['request_id']:1 for x in dec}
    for x in pre:
        if len(out)>=max_seqs or room<=0:break
        remaining=x['prompt']-x['computed']
        # Native APC path rounds budget down to block granularity.
        take=min(remaining,room//block*block)
        if take<=0:break
        out[x['request_id']]=take;room-=take
    return out

def compare_iteration(x):
    predicted=predict_next(x['ordered_before'],x['total'],x['prefill_cap'])
    actual={a['request_id']:a['tokens'] for a in x['scheduled']}
    return {'iteration':x['iteration'],'replica':x['replica'],'predicted':predicted,'actual':actual,
            'match':predicted==actual,'preemptions':x.get('preemptions'),
            'mismatch_scope':'native queue, memory/allocation, or predictive boundary requires inspection' if predicted!=actual else None}

def binding_evidence(x):
    p=x['prefill_logical_tokens'];d=x['decode_tokens'];cap=x.get('prefill_cap')
    remaining=x.get('remaining_prefill_in_before',0)-p
    # This is strong logical candidate evidence, not a native solver counterfactual or GPU time.
    strict=cap is not None and cap<x['total']-d
    near=cap is not None and p>=cap-16
    return {'has_prefill':p>0,'decode_only':d>0 and p==0,'mixed':d>0 and p>0,
            'cap_stricter_than_total':strict,'prefill_remaining_after_logical_schedule':max(0,remaining),
            'binding_candidate':bool(p>0 and strict and near and remaining>0),
            'not_physical_utilization':True}
