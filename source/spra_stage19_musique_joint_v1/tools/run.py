#!/usr/bin/env python3
from __future__ import annotations
import argparse,asyncio,copy,fcntl,json,os,random,sys,time,uuid,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint18v2.common import VERSION,atomic,load,digest,token_hash
from joint18v2.client import gate,token_pool,execute,wait_drained,ForcedMechanism,BackendNotReady
from joint18v2.workload import build
from joint18v2.lifetime_probe import reuse_cache_chain, REVISION as LIFETIME_REVISION
from joint18v2.search import METHODS
from joint18v2.report import comparison,mechanism_report
from core18v2.engine import evaluate

def compact(s):return {k:v for k,v in s.items() if k not in ('workflow_rows','unfinished_workflow_curve','request_latency_rows')}
def warm_worker():return os.getpid()
def salt(pool):
    n=uuid.uuid4().int;out=[]
    for _ in range(12):n,i=divmod(n,len(pool));out.append(pool[i])
    return out

def mechanism_nodes(cfg,pool,seed,nsalt):
    c=dict(cfg,workflow_count=1,prefix_families=1);nodes=build(c,pool,seed,nsalt)
    for n in nodes.values():n['arrival_offset']=.05
    for i in range(cfg['mechanism_background_count']):
        rng=random.Random(seed*999+i);t=nsalt+[rng.choice(pool) for _ in range(512-len(nsalt))];k=f'bg/{i}'
        nodes[k]={'id':k,'wid':k,'name':'background_decode','parents':[],'prompt':512,'output':cfg['mechanism_bg_output'],
           'tokens':t,'hash':token_hash(t),'prefix_hash':None,'share':0,'arrival_offset':0.0,'state':'blocked','replica':None,
           'computed':0,'output_done':0,'ready_at':None,'start':None,'finish':None}
    return nodes

def model_path(arg):
    if arg:return arg
    p=ROOT.parent/'spra_stage18_joint_core_v1/runs/CALIBRATE_20261008T152619Z_edfd9e63/host_model.json'
    if not p.is_file():raise RuntimeError('Model absent: provide --model <existing Stage18 host_model.json>; no refit is automatic')
    return p

def validate_model(m,R):
    if set(m.get('models',{}))!={str(i) for i in range(R)}:raise ValueError('four-replica service model required')
    if any(len(v.get('coef',[]))!=6 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in v.get('coef',[])) for v in m['models'].values()):raise ValueError('finite nonnegative six-feature model contract')

def require_integrated_canary(initial):
    # Correctness prerequisite only. No CPU_REPLAY or singleton-gain prerequisite.
    valid=[]
    for f in sorted((ROOT/'runs').glob('CANARY_*/canary_result.json')):
        try:
            z=load(f);cleanup=load(f.parent/'cleanup.json');status=load(f.parent/'suite_status.json')
            if z.get('pass') and z.get('lifetime_revision')==LIFETIME_REVISION and z.get('version')==VERSION and z.get('hook_hash')==initial[0].get('hook_source_sha256') and cleanup.get('backend_drained') and status.get('status')=='complete':valid.append(f)
        except (OSError,ValueError):continue
    if not valid:raise RuntimeError('No completed native cache/lifetime canary for this hook; run tools/run.py canary first')
    return str(valid[-1])

def provenance():
    import hashlib
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('joint18v2','core18v2','runtime','tools') for p in sorted((ROOT/d).glob('*.py'))}

async def live(a):
    import httpx
    c=load(a.config)
    if a.blocks is not None:c['blocks']=a.blocks
    if a.command=='canary':c.update(replicas=1,workflow_count=1,prefix_families=1,arrival_rate=1,arrival_shape='steady')
    with (a.runtime/'experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180,connect=5,pool=10),trust_env=False,
             limits=httpx.Limits(max_connections=192,max_keepalive_connections=64,keepalive_expiry=2)) as client:
            try:
                initial=await gate(client,a.runtime,c['replicas'])
            except BackendNotReady as exc:
                print('STARTUP_GATE_READY=False MODEL_GENERATION_REQUESTS=0',flush=True)
                raise SystemExit(str(exc)) from None
            if a.command=='gate':print('GATE_READY=True MODEL_GENERATION_REQUESTS=0\n'+json.dumps(initial),flush=True);return
            canary_source=None if a.command=='canary' else require_integrated_canary(initial)
            out=ROOT/'runs'/(a.command.upper()+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
            print('RESULT='+str(out),flush=True);atomic(out/'environment.json',initial);atomic(out/'code_identity.json',provenance())
            atomic(out/'prerequisites.json',{'native_canary':canary_source,'CPU_REPLAY_required':False,'singleton_gain_gate':False})
            pool=await token_pool(client,c['model']);rows=[];executor=None;model=None
            try:
                if a.command=='canary':
                    reused_cache=None
                    if a.reuse_cache_chain is not None:
                        cached,reused_cache=reuse_cache_chain(a.reuse_cache_chain,initial,c)
                        rows.append(compact(cached))
                        atomic(out/'reused_cache_chain.json',reused_cache)
                        print('CACHE_CHAIN_REUSED='+json.dumps(reused_cache),flush=True)
                    for i,probe in enumerate((False,True)):
                        if not probe and reused_cache is not None:continue
                        nodes=build(c,pool,182001+i,salt(pool))
                        s,e=await execute(client,a.runtime,initial,c,nodes,out/('CACHE_CHAIN' if not probe else 'FINITE_LIFETIME'),'A',lifetime_probe=probe)
                        rows.append(compact(s))
                    ok=all(s['budget_invariants'] and s['control_time_match'] and s['root_to_branch_cache_links_verified']>=4 and s['native_next_batch_matches']==s['native_next_batch_checked'] and s['request_correlation']=='6/6' for s in rows)
                    lifetime_caps=set(rows[1]['prefill_caps_applied']);ok=ok and {256,1024}<=lifetime_caps
                    ok=ok and rows[1].get('finite_lifetime_probe_pass') is True
                    for s in rows:
                        ok=ok and s.get('condition_valid') is True and s['output_tokens']==576 and s['native_next_batch_checked']>0
                    atomic(out/'canary_result.json',{'pass':ok,'rows':rows,'version':VERSION,'hook_hash':initial[0]['hook_source_sha256'],'policy_gain_evaluated':False,'lifetime_revision':LIFETIME_REVISION,
                        'reused_cache_chain':reused_cache,
                        'new_generation_requests':6 if reused_cache else 12})
                    print('INTEGRATION_CANARY_PASS='+str(ok)+' POLICY_GAIN_NOT_EVALUATED=True',flush=True)
                    if not ok:raise RuntimeError('cache chain, finite lifetime or token-work evidence failed: '+json.dumps(rows[-1].get('finite_lifetime_evidence',{}).get('failure_reasons',[])))
                else:
                    if a.command!='mechanism' or a.score_model:
                        mp=model_path(a.model);model=load(mp);validate_model(model,c['replicas'])
                        atomic(out/'model_provenance.json',{'path':str(mp),'model_sha256':__import__('hashlib').sha256(mp.read_bytes()).hexdigest(),
                              'transferred_service_model':True,'new_executor_service_validation_complete':False,'counterfactual_rank_certified':False,
                              'used_for_online_selection':a.command!='mechanism'})
                    else:
                        atomic(out/'model_provenance.json',{'model_required':False,'model_loaded':False,'online_search_calls':0,
                            'selection':'predeclared complete action triggered by actual ready state; not offline optimum'})
                    if a.command!='mechanism':
                        # Online methods only. Preset actions do not start a solver worker.
                        executor=ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'))
                        await asyncio.get_running_loop().run_in_executor(executor,warm_worker)
                    if a.command=='mechanism':
                        vectors={'protect':[256,1024,1024,1024],'aggressive':[4096,4096,1024,1024]}
                        cells=[('concentrate','protect'),('split','protect'),('split','aggressive'),('concentrate','aggressive')]
                        for b in range(1,c['mechanism_blocks']+1):
                            order=cells if b==1 else list(reversed(cells));atomic(out/f'block{b}_order.json',order)
                            for pos,(placement,budget) in enumerate(order,1):
                                ns=mechanism_nodes(c,pool,182100+b,salt(pool));forced=ForcedMechanism(placement,vectors[budget])
                                folder=out/f'M{b}_{pos}_{placement}_{budget}'
                                s,e=await execute(client,a.runtime,initial,c,ns,folder,'Forced',None,executor=None,mechanism=forced)
                                row={'block':b,'position':pos,'placement':placement,'budget':budget,'summary':compact(s)};rows.append(row)
                                if forced.record and model is not None:
                                    # After the GPU condition completes, score ALL FOUR complete actions on the same observed state.
                                    snap=forced.record['snapshot'];scores=[]
                                    recorded=[json.loads(line) for line in (folder/'plans.jsonl').read_text().splitlines()]
                                    fp=next((x for x in recorded if x['action']['family'].startswith('forced/')),None)
                                    horizon=200. if fp is None else (fp['action_expires_ns']-fp['snapshot_ns'])/1e6
                                    timing=load(folder/'action_timing.json')
                                    linked=[] if fp is None else [x for x in timing if x['decision_index']==fp['decision_index']]
                                    lag=max([x['actual_effective_delay_ms'] for x in linked],default=(5. if fp is None else fp['estimated_apply_delay_ms']))
                                    for pl in ('concentrate','split'):
                                        for bp,v in vectors.items():
                                            action=copy.deepcopy(forced.record['action']);action['caps']=list(v)
                                            for k in action['placement']:
                                                if k.startswith('w0000/B'):action['placement'][k]=1 if pl=='split' and k.rsplit('/',1)[-1] in ('B3','B4') else 0
                                            z=evaluate(snap,action,model,dict(c,action_horizon_ms=horizon,apply_delay_ms=min(horizon,lag)),deadline=time.monotonic()+2)
                                            scores.append({'placement':pl,'budget':bp,'sum_surrogate_jct_ms':z['sum_remaining_jct_ms'],'rankable':z['rankable'],'cpu_wall_ms':z['cpu_wall_ms'],'physical_counterfactual':False,'common_matched_horizon_ms':horizon,'common_apply_delay_ms':lag,'apply_delay_has_native_sample':bool(linked)})
                                    atomic(folder/'same_state_four_action_predictions.json',scores)
                                atomic(out/'progress.json',rows)
                        report=mechanism_report(rows);atomic(out/'mechanism_report.json',report);print('MECHANISM_REPORT='+json.dumps(report),flush=True)
                    elif a.command in ('pilot','fixed-check'):
                        if a.command=='fixed-check':
                            for i,cap in enumerate(c['caps']):
                                cc=dict(c,base_cap=cap);ns=build(cc,pool,182200,salt(pool))
                                s,e=await execute(client,a.runtime,initial,cc,ns,out/f'FIXED_{cap}','A',model,executor=executor)
                                rows.append({'cap':cap,'summary':compact(s)});atomic(out/'progress.json',rows)
                            best=min(rows,key=lambda x:x['summary']['mean_jct_ms'])['cap']
                            atomic(out/'fixed_selection.json',{'fixed_cap':best,'rows':rows,'development_only':True})
                            print('FIXED_SELECTION='+json.dumps({'fixed_cap':best,'run':str(out)}),flush=True)
                        else:
                            if a.fixed_selection:
                                c['base_cap']=load(a.fixed_selection)['fixed_cap'];ref_label='new_executor_development_selected'
                            else:ref_label='1024_v1_regression_reference_NOT_v2_optimum'
                            orderings=[['A','D','B','Decoupled','Joint'],['D','B','Decoupled','Joint','A'],
                                       ['Joint','Decoupled','A','B','D'],['B','A','Joint','D','Decoupled']]
                            atomic(out/'frozen.json',{'config':c,'reference_scope':ref_label,'orders':orderings,'model_digest':digest(model),
                                  'code':provenance(),'no_singleton_efficacy_gate':True,'same_arrival_shape':True,'block_seeds_change_tokens_not_wave_offsets':True})
                            for b in range(1,c['blocks']+1):
                                order=orderings[(b-1)%len(orderings)];atomic(out/f'block{b}_order.json',order)
                                for pos,m in enumerate(order,1):
                                    ns=build(c,pool,182300+b,salt(pool))
                                    s,e=await execute(client,a.runtime,initial,c,ns,out/f'B{b}_{pos}_{m}',m,model,executor=executor)
                                    rows.append({'block':b,'position':pos,'method':m,'summary':compact(s)});atomic(out/'progress.json',rows)
                            report=comparison(rows);atomic(out/'joint_result.json',report);print('JOINT_RESULT='+json.dumps(report,ensure_ascii=False),flush=True)
                atomic(out/'suite_status.json',{'status':'complete','command':a.command,'conditions':len(rows)})
            except BaseException as ex:
                atomic(out/'suite_status.json',{'status':'failed','command':a.command,'error':repr(ex),'partial_conditions':len(rows),'retry':False});raise
            finally:
                if executor:executor.shutdown(wait=True,cancel_futures=True)
                atomic(a.runtime/'joint18v2/control.json',{'enabled':False})
                try:drained=await wait_drained(a.runtime,initial)
                except Exception:drained=False
                atomic(out/'cleanup.json',{'backend_drained':drained,'control_disabled':True,'legacy_results_unchanged':True})
                if not drained:
                    print('STOP: backend not drained; do not start another condition',flush=True)
                    raise RuntimeError('cleanup did not drain dedicated backend; preserve failed run')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','gate','canary','mechanism','pilot','fixed-check','report'])
    p.add_argument('--runtime',type=Path,default=Path('/home/wmy/sci/spra_core_exp/joint_e1'))
    p.add_argument('--reuse-cache-chain',type=Path,help='canary only: reuse a passed cache subcondition on the SAME current pod0; no source modification')
    p.add_argument('--config',type=Path,default=ROOT/'configs/development.json');p.add_argument('--model',type=Path)
    p.add_argument('--score-model',action='store_true',help='optional offline scoring AFTER each preset mechanism condition; never online search');p.add_argument('--blocks',type=int);p.add_argument('--fixed-selection',type=Path);p.add_argument('--run',type=Path);a=p.parse_args()
    if a.reuse_cache_chain is not None and a.command!='canary':raise SystemExit('--reuse-cache-chain is for canary only')
    if a.command=='plan':
        print(json.dumps({'next':'explicit stage -> pod0 native precheck/restart -> integrated cache/lifetime canary -> all4 -> mechanism -> same-version 4-block pilot',
            'CPU_REPLAY_upload_required':False,'repeat_old_audit':False,'singletons_gate_joint':False,
            'primary_comparisons':['Joint vs A','Joint vs Decoupled'],'four_actions':['placement','admission','local order','per-replica prefill cap'],
            'mechanism':'model-free preset complete actions; no worker; 2 placements x 2 fixed vectors x 2 blocks; all background counted',
            'measurement':'request TTFT, node-eligible-to-first-content, client TPOT proxy, full workflow JCT; no cost subtraction',
            'model_scope':'old measured service curve is a transferred approximation; native next-batch and model-input observations separately validated',
            'config':load(a.config)},ensure_ascii=False,indent=2));return
    if a.command=='report':
        if a.run is None:raise SystemExit('--run required')
        for name in ('joint_result.json','mechanism_report.json','canary_result.json'):
            if (a.run/name).is_file():print((a.run/name).read_text())
        return
    asyncio.run(live(a))
if __name__=='__main__':main()
