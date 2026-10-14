"""Offline phase-aware HOST-step fit from archived native v2 batches.

No GPU or sockets. No JCT labels, action winner labels, or future experiment
outcomes enter fitting. Cross validation holds out a WHOLE old pilot block,
including all methods in it. Phase replacement requires lower held-block MAE
in milliseconds AND non-worse P95 absolute error than the old transferred curve.
Raw samples and rejection counts are retained; short/slow batches are not dropped
just because their latency is unfavorable.
"""
from __future__ import annotations
import json,math,re,statistics,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from core18v2.service import features,phase,predict,coefficients
from .common import quantile,digest
PHASES=('prefill_only','mixed','decode_only')

def read_rows(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    if not summary.get('condition_valid'):raise ValueError('archived condition invalid: '+str(folder))
    path=folder/'backend.jsonl'
    events=[]
    with path.open() as f:
        for number,line in enumerate(f,1):
            try:events.append(json.loads(line))
            except ValueError as ex:raise ValueError(f'{path}:{number}: malformed archived event') from ex
    case=manifest['case'];it={};eng={};inp=defaultdict(list);rej=Counter()
    for e in events:
        if e.get('case')!=case:continue
        key=(e.get('replica'),e.get('pid'),e.get('case'),e.get('iteration'))
        typ=e.get('event')
        if typ=='iteration':
            if key in it:raise ValueError('duplicate native iteration '+str(key))
            it[key]=e
        elif typ=='engine_step':
            if key in eng:raise ValueError('duplicate engine step '+str(key))
            eng[key]=e
        elif typ=='model_input':inp[key].append(e)
    m=re.match(r'^B(\d+)_',folder.name)
    if not m:raise ValueError('expected old pilot B<block>_<position>_<method> condition')
    rows=[]
    for key,e in it.items():
        scheduled=e.get('scheduled',[])
        if not scheduled:rej['empty_iteration']+=1;continue
        step=eng.get(key)
        if step is None or step.get('engine_error'):rej['missing_or_failed_engine_step']+=1;continue
        p=int(e['prefill_logical_tokens']);d=int(e['decode_tokens'])
        if sum(x['tokens'] for x in scheduled if x['prefill'])!=p or sum(x['tokens'] for x in scheduled if not x['prefill'])!=d:
            raise ValueError('native token accounting mismatch')
        pre=[x for x in scheduled if x['prefill']]
        mi=inp.get(key,[])
        # Do not interpret a missing model-input observation as zero cache saving.
        ph=phase(p,d)
        if pre and (not mi or {x['prompt_hash'] for x in pre}!={x.get('prompt_hash') for x in mi}):
            rej['missing_prefill_model_input']+=1;continue
        skip=sum(int(x['skipped_query_tokens']) for x in mi)
        if skip<0 or skip>p:raise ValueError('invalid skipped query accounting')
        dc=sum(x['context_before'] for x in scheduled if not x['prefill'])
        pc=max([x['computed_before'] for x in pre]+[0])
        y=float(step['engine_host_ms'])
        if not math.isfinite(y) or y<=0:rej['nonpositive_or_nonfinite_host_step']+=1;continue
        rows.append({'replica':int(e['replica']),'case':case,'block':int(m.group(1)),
             'iteration':e['iteration'],'phase':ph,'p':p,'d':d,'dc':dc,'pc':pc,'skip':skip,
             'x':list(features(p,d,dc,pc,skip)),'y':y})
    return rows,dict(rej)

def nnls(rows):
    """Six-dimensional nonnegative ridge coordinate descent on sufficient stats.

    Each source condition has equal total weight, preventing its long decode
    stretch alone from dominating the phase fit. No added numerical dependency.
    """
    counts=Counter(x['case'] for x in rows);G=[[0.]*6 for _ in range(6)];h=[0.]*6
    for r in rows:
        w=1./counts[r['case']];x=r['x'];y=r['y']
        for i in range(6):
            h[i]+=w*x[i]*y
            for j in range(6):G[i][j]+=w*x[i]*x[j]
    b=[0.]*6
    for _ in range(350):
        old=b[:]
        for i in range(6):b[i]=max(0.,(h[i]-sum(G[i][j]*b[j] for j in range(6) if i!=j))/(G[i][i]+1e-7))
        if max(abs(x-y) for x,y in zip(b,old))<1e-8:break
    return b

def predicted(b,r):return max(.1,sum(x*y for x,y in zip(b,r['x'])))

def errors(rows,estimates):
    signed=[p-r['y'] for r,p in zip(rows,estimates)];absolute=[abs(x) for x in signed]
    return {'n':len(rows),'mean_signed_ms':statistics.mean(signed) if signed else None,
      'mae_ms':statistics.mean(absolute) if absolute else None,'p95_abs_ms':quantile(absolute,.95),
      'mean_actual_ms':statistics.mean(r['y'] for r in rows) if rows else None,
      'mean_absolute_relative_error':statistics.mean(abs(p-r['y'])/r['y'] for r,p in zip(rows,estimates)) if rows else None}

def fit_phase_model(rows,legacy,replicas=4):
    output=json.loads(json.dumps(legacy));report={};blocks=sorted({r['block'] for r in rows})
    if len(blocks)!=4:raise ValueError('require all 4 archived pilot blocks; do not select favorable blocks')
    for r in range(replicas):
        phases={};report[str(r)]={}
        for ph in PHASES:
            rr=[x for x in rows if x['replica']==r and x['phase']==ph];coef=coefficients(legacy,r,ph)
            folds=[];allnew=[];allold=[];allrows=[]
            enough=len(rr)>=24 and set(x['block'] for x in rr)==set(blocks)
            for held in blocks:
                tr=[x for x in rr if x['block']!=held];te=[x for x in rr if x['block']==held]
                if len(tr)<12 or len(te)<2:enough=False;continue
                b=nnls(tr);new=[predicted(b,x) for x in te];old=[predict(legacy,r,x['p'],x['d'],x['dc'],x['pc'],x['skip']) for x in te]
                folds.append({'held_block':held,'train_blocks':sorted({x['block'] for x in tr}),
                     'training_conditions':sorted({x['case'] for x in tr}),'test_conditions':sorted({x['case'] for x in te}),
                     'candidate':errors(te,new),'legacy':errors(te,old)})
                allnew+=new;allold+=old;allrows+=te
            newstats=errors(allrows,allnew);oldstats=errors(allrows,allold)
            # Absolute ms, not relative % on tiny samples, controls model replacement.
            candidate_mae=statistics.mean(f['candidate']['mae_ms'] for f in folds) if folds else float('inf')
            old_mae=statistics.mean(f['legacy']['mae_ms'] for f in folds) if folds else float('inf')
            use=bool(enough and len(folds)==4 and candidate_mae<old_mae and newstats['p95_abs_ms']<=oldstats['p95_abs_ms'])
            fitted=nnls(rr) if enough else list(coef)
            chosen=fitted if use else list(coef)
            phases[ph]={'coef':chosen,'origin':'archived_block_cv_phase_fit' if use else 'legacy_retained_after_block_cv',
                'count':len(rr),'feature_min':[min((x['x'][i] for x in rr),default=0.) for i in range(6)],
                'feature_max':[max((x['x'][i] for x in rr),default=0.) for i in range(6)]}
            report[str(r)][ph]={'count':len(rr),'support_ok':enough,'phase_fit_selected':use,
                'folds':folds,'held_all_phase':newstats,'legacy_all_phase':oldstats,
                'candidate_block_equal_mae_ms':None if not folds else candidate_mae,
                'legacy_block_equal_mae_ms':None if not folds else old_mae,
                'counterfactual_ranking_certified':False}
        output['models'][str(r)]['phases']=phases
    output.update(schema='stage18_phase_host_v1',base_model_id=legacy.get('model_id'),
         trained_from='previous four-block online pilot; no fresh-suite data',counterfactual_rank_certified=False,
         selection='per-replica per-phase 4-block out-of-fold host-duration MAE/P95; no JCT labels',
         fitting_continues_during_measured_run=False)
    output.pop('model_id',None);output['model_id']=digest(output)
    return output,report
