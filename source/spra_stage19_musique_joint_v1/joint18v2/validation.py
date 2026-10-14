"""Diagnostics for executed batches; never counterfactual labels or kernel timings."""
from collections import defaultdict
import statistics
from .model import predict
from .common import quantile

def service_diagnostics(events,model):
    eng={};inputs=defaultdict(list)
    for x in events:
        key=x.get('replica'),x.get('iteration')
        if x.get('event')=='engine_step' and not x.get('engine_error'):eng[key]=x
        elif x.get('event')=='model_input':inputs[key].append(x)
    rows=[];group=defaultdict(list)
    for x in events:
        if x.get('event')!='iteration' or not x.get('scheduled'):continue
        key=x['replica'],x['iteration'];step=eng.get(key)
        if step is None:continue
        p=x['prefill_logical_tokens'];d=x['decode_tokens'];phase='mixed' if p and d else 'prefill_only' if p else 'decode_only'
        dc=sum(v['context_before'] for v in x['scheduled'] if not v['prefill']);pc=max([v['computed_before'] for v in x['scheduled'] if v['prefill']]+[0])
        skip=sum(v['skipped_query_tokens'] for v in inputs[key]);guess=predict(model,x['replica'],p,d,dc,pc,skip);actual=step['engine_host_ms']
        row={'replica':x['replica'],'iteration':x['iteration'],'phase':phase,'predicted_host_ms':guess,'actual_host_ms':actual,'native_model_input_skipped_tokens':skip,
             'relative_absolute_error':abs(guess-actual)/max(.001,actual),'host_not_GPU':True}
        rows.append(row);group[phase].append(row)
    stats={k:{'n':len(v),'mean_relative_absolute_error':statistics.mean(x['relative_absolute_error'] for x in v),'p95_relative_absolute_error':quantile([x['relative_absolute_error'] for x in v],.95)} for k,v in group.items()}
    return {'rows':rows,'by_phase':stats,'scope':'same executed batches; retained service model with actual model-input skips; no counterfactual inference'}

_legacy_service_diagnostics=service_diagnostics

def service_diagnostics(events,model):
    z=_legacy_service_diagnostics(events,model)
    groups=defaultdict(list)
    for row in z['rows']:
        err=row['predicted_host_ms']-row['actual_host_ms']
        row['signed_error_ms']=err;row['absolute_error_ms']=abs(err)
        groups[row['phase']].append(row)
    for ph,rows in groups.items():
        z['by_phase'][ph].update(mean_signed_error_ms=statistics.mean(r['signed_error_ms'] for r in rows),
          mean_absolute_error_ms=statistics.mean(r['absolute_error_ms'] for r in rows),
          p95_absolute_error_ms=quantile([r['absolute_error_ms'] for r in rows],.95),
          mean_actual_host_ms=statistics.mean(r['actual_host_ms'] for r in rows))
    z['counterfactual_rank_certified']=False
    z['frozen_phase_model_id']=model.get('model_id')
    return z
