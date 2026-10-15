"""Fixed-Block confirmation design. No scheduling policy is changed."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import math
from .common import HERE, load, sha, write_new, digest

REGISTRATION = HERE/'plans/selected_cases.json'
METHODS = ['RR', 'P2C', 'Affinity', 'DAG']
ROUNDS = 8

def parse_case(text: str) -> dict:
    parts=text.split(':')
    if len(parts)!=3:raise ValueError('case must be R:rate:block')
    r=int(parts[0]);rate=float(parts[1]);b=int(parts[2])
    if r not in (1,2,3,4) or b not in (1,2,3,4) or rate not in (1.,2.,4.,8.):
        raise ValueError('only existing R=1..4, rate=1/2/4/8, Block=1..4 supported')
    return {'replicas':r,'rate':rate,'block':b}

def validate_cases(cases: list[dict]) -> list[dict]:
    if not 1<=len(cases)<=3:raise ValueError('register 1 to 3 cases, before execution')
    seen=set()
    for c in cases:
        if set(c)!={'replicas','rate','block'}:raise ValueError('unexpected case fields')
        if type(c['replicas']) is not int or type(c['block']) is not int or type(c['rate']) not in (int,float):
            raise ValueError('case fields must be numeric; Boolean values are not valid')
        if c!=parse_case(f"{c['replicas']}:{c['rate']}:{c['block']}"):raise ValueError('invalid case')
        key=(c['replicas'],c['rate'])
        if key in seen:raise ValueError('use exactly one fixed Block per replica/rate case')
        seen.add(key)
    return cases

def register(texts: list[str], basis: Path, reason: str, path: Path | None=None) -> dict:
    path=path or REGISTRATION
    cases=validate_cases([parse_case(x) for x in texts])
    if not reason.strip():raise ValueError('selection reason is required')
    basis=Path(basis).resolve(strict=True)
    if not basis.is_file() or basis.stat().st_size==0:raise ValueError('selection evidence file is empty')
    if (HERE/'runs').exists() and any((HERE/'runs').glob('*/protocol.json')):
        raise ValueError('register before any run in this application; do not rewrite selection')
    value={'schema':'workflow-fixed-case-registration-1','cases':cases,
           'methods':METHODS,'rounds':ROUNDS,
           'selection_basis':{'path':str(basis),'sha256':sha(basis),'bytes':basis.stat().st_size},
           'reason':reason.strip(),'created_utc':datetime.now(timezone.utc).isoformat(),
           'selected_after_observing_historical_results':True,
           'scope':'new executions of selected historical 16-question Blocks; not unseen-question holdout',
           'no_result_driven_stopping':True,'automatic_retry':False}
    write_new(path,value)
    return value

def case_plan(path: Path | None=None) -> dict:
    path=path or REGISTRATION
    reg=load(path)
    if reg.get('schema')!='workflow-fixed-case-registration-1' or reg.get('methods')!=METHODS or reg.get('rounds')!=ROUNDS:
        raise ValueError('registration contract changed')
    cases=validate_cases(reg['cases'])
    if reg.get('selected_after_observing_historical_results') is not True:
        raise ValueError('case selection provenance must remain explicit')
    from .design import orders, plan
    # Inherit the actual input/cache/arrival/engine/clock contract, not results.
    value=plan('grid')
    seq=orders(METHODS,4)*2
    rows=[]
    for t,order in enumerate(seq,1):
        for c in (cases if t%2 else list(reversed(cases))):
            r,rate,b=c['replicas'],float(c['rate']),c['block']
            for pos,m in enumerate(order,1):
                rows.append({'label':f'R{r}_L{rate:g}_T{t}_B{b}_{m}',
                    'replicas':r,'rate':rate,'repeat':t,'block':b,'position':pos,'method':m,
                    'seed':190500+b,'router_seed':940000+b+100*t,
                    'packet_slice':[(b-1)*16,b*16],'workflow_count':16,'base_cap':4096})
    value.update(suite='case-confirm',conditions=rows,expected_conditions=len(rows),methods=METHODS,
        scope=reg['scope'],selection=reg,selection_registration_sha256=sha(path),
        ordering='two cycles of four-method Williams rows; each position and within-row ordered adjacent pair twice per case',
        primary='within fixed (R,rate,Block), mean of all 8 paired-repeat percentage reductions; no average over different Blocks required',
        cross_repeat_block_selection=False,held_out_questions=False,
        independent_confirmation_execution=True,source_model_or_algorithm_changed=False)
    return value
