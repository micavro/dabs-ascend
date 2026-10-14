#!/usr/bin/env python3
"""One real source dataset, known DAG, shared native runtime, five complete controls."""
from __future__ import annotations
import argparse,asyncio,copy,fcntl,hashlib,json,multiprocessing,os,statistics,sys,time,uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from typed19.dependency import activate,identity,SOURCE
activate()
from typed19 import REVISION
from semantic19.dataset import load_public,sha
from typed19.workflow import ServerTokenizer,compile_packets,instantiate
from typed19.quality import evaluate as quality_evaluate
from typed19.transport import stream as typed_stream
import joint18v2.client as native_client
native_client.stream=typed_stream
from joint18v2.common import atomic,load,digest,quantile
from joint18v2.client import gate,execute,wait_drained,token_pool
HOOK='78ca1d716c5701a9f83ba7dc0b22669b6b50a38bd54626a56642130cd1684418'
METHODS=['A','D','B','Decoupled','Joint']
ORDERS=[['A','D','B','Decoupled','Joint'],['D','Decoupled','A','Joint','B'],
        ['B','Joint','D','A','Decoupled'],['Decoupled','B','Joint','D','A']]

def code_hashes():
    inherited={"source/"+k:v for k,v in identity().items()}
    own={"adapter/"+str(p.relative_to(ROOT)):sha(p) for d in ('typed19','configs','tools')
         for p in sorted((ROOT/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.json')}
    return {**inherited,**own}

def compact(s):
    return {k:v for k,v in s.items() if k not in ('workflow_rows','unfinished_workflow_curve','request_latency_rows','quality_rows','typed_response_rows')}

def worker_ready():return os.getpid()

def progress_summary(rows):
    out={'schema':'stage19-semantic-paired-report-1','paired':[],'scope':'known-plan MuSiQue subset; natural outputs; not production or full leaderboard',
         'same_external_questions_per_block':True,'exact_physical_snapshot_restore':False,'posthoc_search_cost_subtraction':False}
    fields=['mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','p95_request_ttft_ms',
            'mean_node_time_to_first_content_ms','mean_client_tpot_proxy_ms','p95_client_tpot_proxy_ms']
    for block in sorted({r['block'] for r in rows}):
        g={r['method']:r['summary'] for r in rows if r['block']==block}
        if set(g)!=set(METHODS):continue
        def gain(x,y):return 100*(x-y)/x if isinstance(x,(float,int)) and x>0 and isinstance(y,(float,int)) else None
        pair={'block':block,'methods':{},'joint_vs_decoupled_jct_gain_pct':gain(g['Decoupled']['mean_jct_ms'],g['Joint']['mean_jct_ms']),
              'output_tokens_by_method':{m:z['output_tokens'] for m,z in g.items()},
              'actual_work_identical':len({z['output_tokens'] for z in g.values()})==1}
        for m,z in g.items():
            q=z['semantic_quality'];aq=g['A']['semantic_quality']
            pair['methods'][m]={'gain_vs_A_pct':{f:gain(g['A'][f],z[f]) for f in fields},
             'cohort_tokens_s_gain_pct':100*(z['cohort_tokens_s']/g['A']['cohort_tokens_s']-1) if g['A']['cohort_tokens_s'] else None,
             'final_em':q['final_em'],'final_f1':q['final_f1'],'quality_delta_pp':100*(q['final_f1']-aq['final_f1']),
             'metric_guard_pass':bool(q['final_em']>=aq['final_em']-.02 and q['final_f1']>=aq['final_f1']-.02 and
              isinstance(z.get('mean_client_tpot_proxy_ms'),(float,int)) and isinstance(g['A'].get('mean_client_tpot_proxy_ms'),(float,int)) and
              z['mean_client_tpot_proxy_ms']<=g['A']['mean_client_tpot_proxy_ms']*1.03 and
              z['p95_client_tpot_proxy_ms']<=g['A']['p95_client_tpot_proxy_ms']*1.03)}
        out['paired'].append(pair)
    for m in METHODS:
        vals=[p['methods'][m]['gain_vs_A_pct']['mean_jct_ms'] for p in out['paired']]
        if vals:out['mean_'+m+'_JCT_gain_pct']=statistics.mean(vals)
    out['mean_Joint_vs_Decoupled_JCT_gain_pct']=statistics.mean(p['joint_vs_decoupled_jct_gain_pct'] for p in out['paired']) if out['paired'] else None
    out['quality_or_TPOT_failure_never_removed']=True;out['formal_nonregression_established']=False
    return out


def prepared(a):
    path=a.prepared or Path(load(ROOT/'prepared/latest.json')['directory'])
    m=load(path/'manifest.json')
    if m.get('application_revision')!=REVISION or m.get('application_source_hashes')!=code_hashes():raise ValueError('profile must use this exact typed application version; old schema/profile cannot be reused')
    for f,h in m['files'].items():
        if sha(path/f)!=h:raise ValueError('frozen artifact hash changed '+f)
    return path,load(path/'model.json'),load(path/'priors.json'),m


def freeze_profile(run,legacy,dataset):
    from joint18v2.service_fit import read_rows,fit_phase_model
    run=Path(run);idx=load(run/'index.json');samples=[];outputs={};reject={};stats={};files={}
    if len(idx)!=20 or {(x['block'],x['cap']) for x in idx}!={(b,c) for b in range(1,5) for c in (256,512,1024,2048,4096)}:
        raise ValueError('profile freeze requires all 20 unique block/cap conditions')
    if any(not x['summary'].get('condition_valid',False) for x in idx):raise ValueError('invalid profile condition')
    for record in idx:
        folder=Path(record['folder']);rr,rejs=read_rows(folder);samples+=rr;reject[folder.name]=rejs
        ns=load(folder/'nodes.json')
        for n in ns.values():outputs.setdefault(str(n['hop_count']),[]).append(n['stream']['usage']['completion_tokens'])
        for f in ('backend.jsonl','summary.json','nodes.json','manifest.json'):files[str(folder/f)]=sha(folder/f)
        stats.setdefault(record['cap'],[]).append(record['summary'])
    old=load(legacy);model,report=fit_phase_model(samples,old,4)
    priors={k:{'n':len(v),'p75':quantile(v,.75),'mean':statistics.mean(v),'future_exact_length_known':False} for k,v in outputs.items()}
    choices=[]
    reference_em=statistics.mean(s['semantic_quality']['final_em'] for s in stats[1024]);reference_f1=statistics.mean(s['semantic_quality']['final_f1'] for s in stats[1024])
    for cap,ss in stats.items():
        em=statistics.mean(s['semantic_quality']['final_em'] for s in ss);f1=statistics.mean(s['semantic_quality']['final_f1'] for s in ss)
        choices.append({'cap':cap,'mean_jct_ms':statistics.mean(s['mean_jct_ms'] for s in ss),'em':em,'f1':f1,
                        'eligible':em>=reference_em-.02 and f1>=reference_f1-.02})
    best=min((c for c in choices if c['eligible']),key=lambda x:x['mean_jct_ms'])
    out=ROOT/'prepared'/('MUSIQUE_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    atomic(out/'model.json',model);atomic(out/'priors.json',priors);atomic(out/'fit_report.json',report)
    manifest={'schema':'stage19-frozen-profile-1','dataset_manifest_sha256':sha(Path(dataset)/'manifest.json'),
       'profile_run':str(run.resolve()),'legacy_model':str(Path(legacy).resolve()),'legacy_sha256':sha(legacy),
       'source_hashes':files,'best_fixed_cap':best['cap'],'fixed_candidates':choices,
       'selection_scope':'same development questions; not per-test oracle','labels_used_to_fit_service_times':False,
       'output_lengths':'estimated from completed profile generations, not gold text',
       'profile_configuration':load(run/'frozen_protocol.json')['config'],
       'application_revision':REVISION,'application_source_hashes':code_hashes(),
       'files':{f:sha(out/f) for f in ('model.json','priors.json','fit_report.json')}}
    if any(sha(p)!=h for p,h in files.items()):raise RuntimeError('profile source changed during fit')
    atomic(out/'manifest.json',manifest);atomic(ROOT/'prepared/latest.json',{'directory':str(out.resolve())})
    print('REAL_PROFILE_FROZEN='+json.dumps({'directory':str(out),'best_fixed_cap':best['cap'],'fixed_candidates':choices,'no_singleton_gain_gate':True}),flush=True)
    return out


async def warm_schema(client,runtime,initial,cfg,out,tokenizer,pool):
    from tools.experiment import warm_nodes,FixedWarmup
    from tools.experiment import salt
    from joint18v2.common import token_hash
    ns=warm_nodes(cfg,pool,(512,32),salt(pool))
    system='Run '+uuid.uuid4().hex+'. Return only a JSON object with string key answer.'
    for n in ns.values():
        tokens=await tokenizer.chat_tokens(system+' Replica '+str(n['warm_replica'])+'.','Return {"answer":"READY"}.')
        n.update(tokens=list(tokens),prompt=len(tokens),hash=token_hash(tokens),prefix_hash=None,share=0,
                 output=8,output_is_estimate=True,max_output_tokens=cfg['max_output_tokens'],
                 semantic_mode=True,typed_answer_mode=True)
    summary,_=await execute(client,runtime,initial,cfg,ns,out/'SERVICE_WARMUP'/'GUIDED_JSON',
                          'Forced',None,mechanism=FixedWarmup())
    counts={str(r):sum(n.get('replica')==r for n in ns.values()) for r in range(len(initial))}
    if list(counts.values())!=[1]*len(initial):raise RuntimeError('guided warmup must cover every replica once')
    errors=sum(n['answer_contract']['status']=='format_error' for n in ns.values())
    if errors:raise RuntimeError('native guided JSON warmup violated schema; no fallback or installation performed')
    atomic(out/'SERVICE_WARMUP'/'guided_receipt.json',{'replica_counts':counts,'requests':len(ns),
           'fixed_count':True,'measured':False,'schema_errors':errors,'actual_output_tokens':summary['output_tokens']})
    print('GUIDED_SERVICE_WARMUP_COMPLETE=True REQUESTS=4 MEASURED=False',flush=True)


async def run(a):
    import httpx
    cfg=load(ROOT/'configs/semantic.json');cfg['arrival_rate']=a.rate
    if a.command=='pilot':
        artifact,model,priors,mf=prepared(a)
        if sha(a.dataset/'manifest.json')!=mf['dataset_manifest_sha256']:raise ValueError('dataset changed after development freeze')
        if mf.get('profile_configuration',{}).get('arrival_rate')!=cfg['arrival_rate']:
            raise ValueError('pilot rate differs from frozen development profile; keep the predeclared rate')
        cfg['base_cap']=mf['best_fixed_cap']
    else:model=None;priors=None
    split={'smoke':'smoke','profile':'profile','pilot':'pilot'}.get(a.command)
    records,dmanifest=load_public(a.dataset,split) if split else (None,None)
    if a.command=='profile' and not a.legacy_model.is_file():raise FileNotFoundError('legacy model missing: '+str(a.legacy_model))
    initial_code=code_hashes()
    with (a.runtime/'experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180,connect=5,pool=10),trust_env=False,
                limits=httpx.Limits(max_connections=192,max_keepalive_connections=64,keepalive_expiry=2)) as client:
            initial=await gate(client,a.runtime,4)
            if any(s.get('hook_source_sha256')!=HOOK for s in initial):raise ValueError('unexpected native hook; no restart or replacement performed')
            if a.command=='gate':print('GATE_READY=True MODEL_GENERATION_REQUESTS=0 BACKEND_CHANGED=False');return
            out=ROOT/'runs'/(a.command.upper()+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
            print('RESULT='+str(out),flush=True)
            atomic(out/'environment.json',initial);atomic(out/'dataset_manifest.json',dmanifest);atomic(out/'code_identity.json',initial_code)
            atomic(out/'frozen_protocol.json',{'revision':REVISION,'command':a.command,'config':cfg,
                'known_plan':True,'unknown_future_outputs':True,'source_gold_steps_are_privileged_plan_annotations':True,
                'same_plan_executor_for_all_methods':True,'quality_guard_pp':2.0,'mean_p95_TPOT_noninferiority_pct':3.0,
                'all_waits_in_JCT':True,'no_posthoc_subtraction':True,'replicas_or_backend_changed':False,
                'no_automatic_retries':True,'output_schema':'one JSON string answer, native guided_json',
                'raw_prose_never_substituted_into_child':True,'canonical_chat_tokens_sent_unchanged':True,
                'overall_question_sent':False,'no_new_dependencies_or_backend_changes':True,'orders':ORDERS,'profile_input_tuning_disjoint_from_pilot':True})
            rows=[];executor=None;tok=ServerTokenizer(client,cfg['model']);packets=None
            try:
                packets,info=await compile_packets(records,tok,cfg);atomic(out/'packet_preparation.json',info)
                atomic(out/'compiled_topologies.json',[{'id':p['id'],'graph':p['graph'],'coverage':p['paragraph_coverage']} for p in packets])
                # Fixed-count service warmup is separate from semantic measured cohorts.
                if a.command!='smoke':
                    from tools.experiment import warm_nodes,FixedWarmup
                    from tools.experiment import salt
                    pool=await token_pool(client,cfg['model'])
                    for wi,profile in enumerate(((512,32),(3328,64)),1):
                        ns=warm_nodes(cfg,pool,profile,salt(pool))
                        await execute(client,a.runtime,initial,cfg,ns,out/'SERVICE_WARMUP'/f'P{wi}','Forced',None,mechanism=FixedWarmup())
                    print('SERVICE_WARMUP_COMPLETE=True REQUESTS=8 MEASURED=False')
                    await warm_schema(client,a.runtime,initial,cfg,out,tok,pool)
                if a.command=='smoke':design=[(0,1,'A',cfg['base_cap'],packets,190301)]
                elif a.command=='profile':
                    design=[]
                    for block in range(1,5):
                        caps=cfg['caps'];caps=caps[block-1:]+caps[:block-1]
                        for pos,cap in enumerate(caps,1):design.append((block,pos,'A',cap,packets[(block-1)*8:block*8],190400+block))
                else:
                    executor=ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'))
                    await asyncio.get_running_loop().run_in_executor(executor,worker_ready)
                    design=[(b,pos,m,cfg['base_cap'],packets[(b-1)*16:b*16],190500+b) for b,order in enumerate(ORDERS,1) for pos,m in enumerate(order,1)]
                for block,pos,method,cap,ps,seed in design:
                    if code_hashes()!=initial_code:raise RuntimeError('source changed mid-run')
                    cc=copy.deepcopy(cfg);cc['base_cap']=cap;cc['workflow_count']=len(ps)
                    ns,materialize,proof=await instantiate(ps,tok,cc,seed,priors)
                    label=f'B{block}_{pos}_{method}'+(f'_F{cap}' if a.command=='profile' else '')
                    folder=out/label
                    summary,events=await execute(client,a.runtime,initial,cc,ns,folder,method,model,
                         executor=executor,materializer=materialize)
                    # Scoring labels are opened only after every request in this condition returned.
                    q=quality_evaluate(ns,a.dataset,summary['cohort_duration_s']);summary.update(q)
                    paired_bindings=sum(len(n.get('parent_response_bindings',{})) for n in ns.values())
                    expected_edges=sum(len(n['parents']) for n in ns.values())
                    if paired_bindings!=expected_edges:raise RuntimeError('semantic edge binding count mismatch')
                    for n in ns.values():
                        for pp in n['parents']:
                            entry=n['parent_response_bindings'][str(ns[pp]['step_position']+1)]
                            if entry['response']!=ns[pp]['stream']['text']:raise RuntimeError('child did not consume actual parent output')
                    if summary['native_next_batch_matches']!=summary['native_next_batch_checked']:raise RuntimeError('native next batch mismatch')
                    summary.update(dataset='MuSiQue-Ans-v1.0',known_DAG=True,semantic_edges_verified=paired_bindings,
                        output_work_forced_equal=False,source_questions=[p['id'] for p in ps],
                        subgraph_width_counts={str(w):sum(p['graph']['max_parallel_width']==w for p in ps) for w in (1,2,3,4)},
                        model_future_outputs_are_estimates=True)
                    atomic(folder/'semantic_proof.json',proof);atomic(folder/'quality.json',q)
                    atomic(folder/'typed_responses.json',q['typed_response_rows'])
                    atomic(folder/'summary.json',summary)
                    rows.append({'block':block,'position':pos,'method':method,'cap':cap,'folder':str(folder.resolve()),'summary':compact(summary)})
                    atomic(out/'index.json',rows)
                    print('TYPED_SEMANTIC_CONDITION_DONE='+json.dumps({'block':block,'position':pos,'method':method,'cap':cap,**compact(summary)},ensure_ascii=False),flush=True)
                if a.command=='profile':
                    freeze_profile(out,a.legacy_model,a.dataset)
                elif a.command=='pilot':
                    report=progress_summary(rows);atomic(out/'joint_result.json',report);print('TYPED_MUSIQUE_JOINT_RESULT='+json.dumps(report),flush=True)
                else:
                    receipt={'transport_pass':True,'answer_contract_pass':rows[0]['summary']['semantic_quality']['answer_contract_errors']==0,
                      'semantic_efficacy_established':False,'format_pass_is_not_quality_pass':True,
                      'semantic_edges_verified':rows[0]['summary']['semantic_edges_verified'],
                      'quality':rows[0]['summary']['semantic_quality'],'quality_success_not_required_for_transport_check':True,
                      'predictive_gain_claimed':False,'actual_questions':len(packets),'actual_LLM_requests':rows[0]['summary']['requests']}
                    atomic(out/'semantic_smoke.json',receipt);print('TYPED_HANDOFF_RESULT='+json.dumps(receipt),flush=True)
                atomic(out/'suite_status.json',{'status':'complete','measured_conditions':len(rows)})
            except BaseException as ex:
                atomic(out/'suite_status.json',{'status':'failed','completed_conditions':len(rows),'error':repr(ex),'retry':False});raise
            finally:
                if executor:executor.shutdown(wait=True,cancel_futures=True)
                atomic(a.runtime/'joint18v2/control.json',{'enabled':False})
                try:dr=await wait_drained(a.runtime,initial)
                except Exception:dr=False
                cleanup={'backend_drained':dr,'backend_changed':False,'source_files_unchanged':code_hashes()==initial_code,'old_runs_overwritten':False}
                atomic(out/'cleanup.json',cleanup);print('CLEANUP='+json.dumps(cleanup),flush=True)
                if not dr:raise RuntimeError('backend not drained; do not start next suite')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','gate','smoke','profile','pilot','report'])
    p.add_argument('--runtime',type=Path,default=Path('/home/wmy/sci/spra_core_exp/joint_e1'))
    p.add_argument('--dataset',type=Path,default=SOURCE/'dataset/compiled');p.add_argument('--prepared',type=Path)
    p.add_argument('--legacy-model',type=Path,default=ROOT.parent/'spra_stage18_joint_core_v1/runs/CALIBRATE_20261008T152619Z_edfd9e63/host_model.json')
    p.add_argument('--rate',type=float,default=2.0);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.rate<.25:raise ValueError('this declared wave protocol requires rate>=0.25')
    if a.command=='plan':
        print(json.dumps({'dataset':'official MuSiQue-Ans v1.0','mode':'known annotation DAG, real parent outputs',
            'smoke':'same 2 existing train-smoke questions, 7 requests; canonical tokens and typed actual parents; no four-cell repeat','profile':'32 disjoint train questions, 4blocks x5 fixed caps; 20conditions',
            'pilot':'64 dev questions,4blocks x5 methods; eachblock16 questions','outputs':'guided JSON answer-only; natural EOS max96; no min_tokens/ignore_eos; format alone is not semantic success',
            'source_gold':'post-run quality only; annotated question DAG allowed','context':'all supplied paragraphs, gold-blind equal prefix budget2700; entire messages canonical tokenized; no overall question in prompt',
            'backend_restart':False,'new_dependencies':False,'dataset_download_only_on_CPU':True,'no_D_or_B_gain_gate':True},indent=2,ensure_ascii=False));return
    if a.command=='report':
        if not a.run_dir:p.error('--run-dir required')
        print((a.run_dir/'joint_result.json').read_text());return
    asyncio.run(run(a))
if __name__=='__main__':main()
