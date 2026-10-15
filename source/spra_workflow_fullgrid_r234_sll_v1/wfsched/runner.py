"""Real typed19 application + original native executor; no placeholder backend."""
from __future__ import annotations
import asyncio
import copy
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor
from . import REVISION
from .common import HERE, PROFILE, atomic, write_new, load, sha, digest, own_identity, condition_files, verify_files
from .legacy import inspect, load_original, load_profile, freeze, reproduction_module
from .design import plan, config_for
from .binding import bindings
from .worker import ready as worker_ready
from .audit import validate_condition, external_work, inactive_offsets, inactive_proof
from . import report
from .fixed_work import convert_nodes,INPUT_TOKENS,OUTPUT_TOKENS

HARDWARE_FIELDS=('pod','hook_source_sha256','scheduler_source_sha256','nvidia_visible_devices',
  'hard_total','max_model_len','max_num_seqs','block_size','num_gpu_blocks','num_scheduler_steps',
  'prefix_caching','chunked_prefill','native_cache_allocator')

def environment_contract(ss):
    if len(ss)!=4:raise ValueError('all four existing services must be checked before selecting active subset')
    if any(k not in s for s in ss for k in HARDWARE_FIELDS):raise ValueError('missing runtime hardware/config identity')
    versions={}
    for name in ('httpx','numpy','torch','vllm','transformers'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'services':[{k:s[k] for k in HARDWARE_FIELDS} for s in ss],
            'python':sys.version,'python_executable':sys.executable,'package_metadata':versions}

async def all_gate(original, client, runtime):
    ss=await original.gate(client,runtime,4)
    if any(s.get('hook_source_sha256')!=original.HOOK for s in ss):
        raise ValueError('unexpected native hook; no deployment changed')
    return ss

def read_run(out, design=None):
    out=Path(out);p=load(out/'protocol.json')
    if sha(out/'protocol.json')!=load(out/'protocol.sha256.json')['sha256']:
        raise ValueError('frozen run protocol changed')
    if p['extension_identity']!=own_identity():raise ValueError('extension changed after run was prepared')
    if p['plan']!=json.loads(json.dumps(plan(p['plan']['suite']),ensure_ascii=False,allow_nan=False)):raise ValueError('declared plan differs from executable plan')
    if design is not None and p['plan']['conditions']!=design:raise ValueError('suite differs from saved plan')
    status=load(out/'suite_status.json')['status'];rows=load(out/'index.json')
    # validate exact row specs and external hashes even on resume/report.
    report.summarize(rows,p['plan']['conditions'])
    for row in rows:
        folder=(out/row['folder']).resolve()
        if not folder.is_relative_to(out.resolve()):raise ValueError('condition folder escaped run directory')
        verify_files(folder,row['files'])
        summary=load(folder/'summary.json')
        if digest(summary)!=row['summary_sha256']:raise ValueError('saved summary differs from accepted summary')
        if original_compact(summary)!=row['summary']:raise ValueError('index summary differs from raw summary')
        validate_condition(folder,summary,load(folder/'nodes.json'),row)
    return p,rows,status

def original_compact(summary):
    return {k:v for k,v in summary.items() if k not in
            ('workflow_rows','unfinished_workflow_curve','request_latency_rows','quality_rows','typed_response_rows')}

async def warmup(original, client, runtime, initial, cfg, out, tokenizer):
    from tools.experiment import warm_nodes, FixedWarmup, salt
    from joint18v2.common import token_hash
    import joint18v2.client as live
    R=len(initial);pool=await live.token_pool(client,cfg['model'])
    with bindings('WARMUP'):
        for i,shape in enumerate(((512,32),(3328,64)),1):
            ns={k:n for k,n in warm_nodes(cfg,pool,shape,salt(pool)).items() if n['warm_replica']<R}
            summary,_=await live.execute(client,runtime,initial,cfg,ns,out/f'P{i}',
                                        'Forced',None,mechanism=FixedWarmup())
            reproduction_module().validate_fixed_summary(summary,R)
        ns={k:n for k,n in warm_nodes(cfg,pool,(512,32),salt(pool)).items() if n['warm_replica']<R}
        system='Run '+uuid.uuid4().hex+'. Return only a JSON object with string key answer.'
        for n in ns.values():
            ts=await tokenizer.chat_tokens(system+' Replica '+str(n['warm_replica'])+'.','Return {"answer":"READY"}.')
            n.update(tokens=list(ts),prompt=len(ts),hash=token_hash(ts),prefix_hash=None,share=0,
                     output=8,output_is_estimate=True,max_output_tokens=cfg['max_output_tokens'],
                     semantic_mode=True,typed_answer_mode=True)
        summary,_=await live.execute(client,runtime,initial,cfg,ns,out/'GUIDED_JSON',
                                    'Forced',None,mechanism=FixedWarmup())
        reproduction_module().validate_fixed_summary(summary,R)
        if sorted(n['replica'] for n in ns.values())!=list(range(R)):
            raise ValueError('guided warmup did not cover each active service exactly once')
        if any(n['answer_contract']['status']=='format_error' for n in ns.values()):
            raise ValueError('guided warmup schema failure')
    atomic(out/'warmup_receipt.json',{'requests':3*R,'replicas':R,'measured':False,
                                     'fixed_count':True,'repeated_until_fast':False})

async def gate_only(args):
    import httpx
    inspection=inspect(args.repo,require_data=False)
    original=load_original(args.repo)
    load_profile(original,args.repo)
    with (args.runtime/'experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        async with httpx.AsyncClient(timeout=15,trust_env=False) as client:
            ss=await all_gate(original,client,args.runtime)
            print('WF_GATE='+json.dumps({'ready':True,'new_generation_requests':0,
                  'services':environment_contract(ss)['services']},ensure_ascii=False),flush=True)
    return inspection

async def run(args):
    import httpx
    root=Path(args.repo).resolve();runtime=Path(args.runtime).resolve();out=Path(args.out).resolve()
    frozen=freeze(root,HERE/'frozen/D0')
    expected=inspect(root,require_data=True)
    original=load_original(root)
    _,model,priors,mf=load_profile(original,root)
    from semantic19.dataset import load_public
    from typed19.workflow import ServerTokenizer,compile_packets,instantiate
    import joint18v2.client as live
    base=copy.deepcopy(mf['profile_configuration']);base['base_cap']=4096
    dataset=root/'spra_stage19_musique_joint_v1/dataset/compiled'
    records,dm=load_public(dataset,'pilot')
    if len(records)!=64 or len({r['id'] for r in records})!=64:raise ValueError('not original64 distinct questions')
    declared=plan(args.suite);design=declared['conditions'];identity=own_identity()
    if out.exists():
        if not args.resume:raise ValueError('output exists; use explicit --resume, never overwrite old results')
        p,rows,status=read_run(out,design)
        if status not in ('paused','prepared','complete'):
            raise ValueError('failed/interrupted session is preserved; no automatic retry or skip')
        if p['checkpoint']!=expected or p['runtime']!=str(runtime):raise ValueError('source/profile/runtime changed')
        if status=='complete':
            print('WF_ALREADY_COMPLETE='+str(out),flush=True);return
    else:
        if args.resume:raise ValueError('cannot resume missing run')
        out.mkdir(parents=True,exist_ok=False);rows=[]
        p={'schema':'workflow-real-native-suite-2','revision':REVISION,'plan':declared,
           'run_uuid':uuid.uuid4().hex,'runtime':str(runtime),'checkpoint':expected,
           'extension_identity':identity,'frozen_D0_archive_sha256':frozen['archive_sha256'],
           'selected_source_ids':[r['id'] for r in records],'selected_public_sha256':digest(records),
           'profile_configuration':base,'model_refitted':False,
           'model_coefficients_mapping':'unchanged physical replica coefficients, first R models used',
           'created_utc':time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())}
        write_new(out/'protocol.json',p)
        write_new(out/'protocol.sha256.json',{'sha256':sha(out/'protocol.json')})
        atomic(out/'index.json',rows);atomic(out/'suite_status.json',{'status':'prepared'})
        with zipfile.ZipFile(out/'extension_source.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in identity:z.write(HERE/name,name)
    if p['selected_public_sha256']!=digest(records):raise ValueError('public workload drift')
    accepted={r['label'] for r in rows}
    pending=[cell for cell in design if cell['label'] not in accepted and
             (not args.replicas or cell['replicas'] in args.replicas)]
    if args.max_conditions is not None:pending=pending[:args.max_conditions]
    if not pending:
        print('WF_NO_PENDING_CONDITIONS_FOR_FILTER='+str(out),flush=True);return
    # Do not auto-accept interrupted conditions even when a raw summary exists.
    for cell in design:
        marker=out/'attempts'/cell['label']/'status.json'
        if cell['label'] not in accepted and marker.exists():
            raise ValueError('unaccepted attempt exists; preserve it: '+str(marker))
    session=out/'sessions'/('SESSION_'+uuid.uuid4().hex[:12]);session.mkdir(parents=True)
    touched=[];pool=None;drained=False;initial_all=None;success=False
    with (runtime/'experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180,connect=5,pool=10),trust_env=False,
                 limits=httpx.Limits(max_connections=192,max_keepalive_connections=64,keepalive_expiry=2)) as client:
            initial_all=await all_gate(original,client,runtime)
            ec=environment_contract(initial_all)
            if (out/'environment_contract.json').exists():
                if load(out/'environment_contract.json')!=ec:raise ValueError('hardware/software configuration changed across sessions')
            else:write_new(out/'environment_contract.json',ec)
            write_new(session/'environment.json',initial_all)
            atomic(out/'suite_status.json',{'status':'running','session':str(session),'completed_conditions':len(rows)})
            try:
                tok=ServerTokenizer(client,base['model'])
                packets,info=await compile_packets(records,tok,base)
                write_new(session/'packet_preparation.json',info)
                if [q['id'] for q in packets]!=p['selected_source_ids']:raise ValueError('packet order changed')
                pool=ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'))
                await asyncio.get_running_loop().run_in_executor(pool,worker_ready)
                last_R=None;warm_index=0
                for cell in pending:
                    if own_identity()!=identity or inspect(root,True)!=expected:raise ValueError('source/profile changed mid-run')
                    current_all=await all_gate(original,client,runtime)
                    live.same_engines(current_all,initial_all)
                    if environment_contract(current_all)!=ec:raise ValueError('runtime config changed mid-run')
                    R=cell['replicas'];initial=current_all[:R];cfg=config_for(base,cell)
                    if last_R!=R:
                        warm_index+=1
                        warm_path=session/(f'WARM_R{R}_{warm_index}_'+p['run_uuid'][:8])
                        offsets=inactive_offsets(runtime,R)
                        await warmup(original,client,runtime,initial,cfg,warm_path,tok)
                        atomic(warm_path/'inactive_proof.json',inactive_proof(runtime,offsets))
                        last_R=R
                    lo,hi=cell['packet_slice'];ps=packets[lo:hi]
                    repetitions=int(cell.get('packet_repetitions',1))
                    if repetitions>1:
                        if cell.get('measurement_mode')!='fixed_token_replay':
                            raise ValueError('repeated packets only allowed in nonsemantic fixed-token stress suite')
                        ps=ps*repetitions
                    if len(ps)!=cell['workflow_count']:
                        raise ValueError('configured workflow_count does not match selected packets')
                    hetero_manifest=None
                    if cell.get('heterogeneous_workload'):
                        from .heterogeneous import select_packets
                        ps,hetero_manifest=select_packets(packets,cell['popularity_profile'],seed=510140)
                        if len(ps)!=cell['workflow_count']:
                            raise ValueError('heterogeneous workload count mismatch')
                    nodes,materializer,proof=await instantiate(ps,tok,cfg,cell['seed'],priors)
                    fixed_mode=cell.get('measurement_mode')=='fixed_token_replay'
                    if fixed_mode:
                        # One native token ID for deterministic *tail-only* padding.
                        pad_ids=await tok.encode(' .')
                        work_sha=convert_nodes(nodes,pad_ids[-1],heterogeneous=bool(hetero_manifest))
                        if hetero_manifest:
                            total_work=sum(n['output'] for n in nodes.values())
                            if total_work!=hetero_manifest['total_declared_output_tokens']:
                                raise ValueError('predeclared per-DAG output work mismatched')
                        if len({n['hash'] for n in nodes.values()})!=len(nodes):
                            raise ValueError('preflight rejected duplicate native request hashes')
                        materializer=None
                        proof.update(workload_mode='fixed_token_replay',
                            fixed_node_input_tokens=INPUT_TOKENS,
                            fixed_node_output_tokens=(None if hetero_manifest else OUTPUT_TOKENS),
                            fixed_output_budget_by_DAG=(hetero_manifest['per_shape_output_tokens'] if hetero_manifest else None),
                            popularity_manifest_sha256=(hetero_manifest['manifest_digest'] if hetero_manifest else None),
                            fixed_work_sha256=work_sha,
                            child_input_uses_parent_answer=False,
                            semantic_quality_applicable=False)
                    external=external_work(ps,proof)
                    for prev in rows:
                        if all(prev[k]==cell[k] for k in ('replicas','rate','repeat','block')) and prev['external_work_sha256']!=external:
                            raise ValueError('paired public workload or arrivals changed before execution')
                    attempt=out/'attempts'/cell['label']
                    attempt.mkdir(parents=True,exist_ok=False)
                    folder=attempt/('RAW_'+p['run_uuid'][:12])
                    atomic(attempt/'status.json',{'status':'running','spec':cell,'automatic_retry':False})
                    touched.append(cell['label']);offsets=inactive_offsets(runtime,R)
                    print('WF_CONDITION_START='+json.dumps(cell),flush=True)
                    if hetero_manifest is not None:
                        # IMPORTANT: native execute() must CREATE the RAW dir with
                        # exist_ok=False. Never create it via atomic() beforehand.
                        # Preserve the preregistered manifest at attempt level.
                        if folder.exists():
                            raise FileExistsError('RAW folder exists before native execution: '+str(folder))
                        atomic(attempt/'predeclared_popularity_manifest.json',hetero_manifest)
                    cfg['wf_worker_failure_dir']=str(folder/'worker_failures')
                    native_method='D' if cell['method'] in ('DAG','DAG-Lite','DAG-V2','DAG-A','DAG-W','DAG-H','Ours') else 'A'
                    with bindings(cell['method']):
                        summary,events=await live.execute(client,runtime,initial,cfg,nodes,folder,
                            native_method,model,executor=pool if native_method=='D' else None,
                            materializer=materializer,force_caps=None)
                    if hetero_manifest is not None:
                        # This second copy is part of accepted RAW evidence.
                        # Native executor now owns RAW directory creation.
                        if not folder.is_dir():
                            raise ValueError('native executor returned without RAW evidence directory')
                        atomic(folder/'predeclared_popularity_manifest.json',hetero_manifest)
                    # Labels are read by original quality code only after full native completion.
                    quality=({'semantic_quality':{'final_em':None,'final_f1':None,
                           'quality_scope':'nonsemantic fixed-token DAG replay',
                           'not_an_agent_quality_measure':True},'typed_response_rows':[]}
                         if fixed_mode else original.quality_evaluate(nodes,dataset,summary['cohort_duration_s']))
                    summary.update(quality)
                    summary.update(public_method=cell['method'],native_method=native_method,
                       dataset='MuSiQue-Ans-v1.0',known_DAG=True,source_questions=[q['id'] for q in ps],
                       replicas=R,arrival_rate=cell['rate'],workflow_sched_revision=REVISION,
                       budget_selection_enabled=False,public_dispatch_loop='original_periodic_200ms',
                       workload_mode=('fixed_token_replay' if fixed_mode else 'natural_json'),
                       popularity_profile=cell.get('popularity_profile'),
                       fixed_output_by_DAG=bool(hetero_manifest))
                    atomic(folder/'quality.json',quality)
                    atomic(folder/'typed_responses.json',quality['typed_response_rows'])
                    atomic(folder/'semantic_proof.json',proof)
                    atomic(folder/'summary.json',summary)
                    audit=validate_condition(folder,summary,nodes,cell)
                    if fixed_mode and audit['fixed_work_sha256']!=work_sha:
                        raise ValueError('fixed replay work checksum drift')
                    atomic(folder/'scheduling_audit.json',audit)
                    if args.suite=='dabs-r234-balanced-64':
                        from .evidence import capture_evidence
                        receipt=capture_evidence(folder,summary,nodes,cell)
                        summary['kv_evidence_summary']=receipt['cache']
                        summary['latency_decomposition_source']='native summary mean_critical_path_components_ms'
                        atomic(folder/'summary.json',summary)

                    atomic(folder/'inactive_proof.json',inactive_proof(runtime,offsets))
                    compact=original_compact(summary)
                    row={**cell,'folder':str(folder.relative_to(out)), 'summary':compact,
                         'summary_sha256':digest(summary),'external_work_sha256':external,
                         'fixed_work_sha256':audit['fixed_work_sha256'],
                         'files':condition_files(folder),'session':str(session.relative_to(out))}
                    rows.append(row);atomic(out/'index.json',rows)
                    atomic(attempt/'status.json',{'status':'accepted','summary_sha256':row['summary_sha256']})
                    print('WF_CONDITION_ACCEPTED='+json.dumps({**cell,**{k:compact.get(k) for k in
                        ('mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','cohort_workflows_s',
                         'requests','output_tokens','semantic_quality','decision_wall_ms')}},ensure_ascii=False),flush=True)
                success=True
            except BaseException as exc:
                atomic(session/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'automatic_retry':False,'touched_conditions':touched})
                atomic(out/'suite_status.json',{'status':'failed','error':repr(exc),'completed_conditions':len(rows)})
                raise
            finally:
                if pool:pool.shutdown(wait=True,cancel_futures=True)
                # Disable our control after native conditions. No model service stop/restart.
                atomic(runtime/'joint18v2/control.json',{'enabled':False})
                try:drained=await live.wait_drained(runtime,initial_all)
                except Exception:drained=False
                try:unchanged=inspect(root,True)==expected and own_identity()==identity
                except Exception:unchanged=False
                cleanup={'backend_drained':drained,'source_files_unchanged':unchanged,
                         'backend_code_changed':False,'old_results_overwritten':False,
                         'new_method_sources_in_separate_directory':True}
                atomic(session/'cleanup.json',cleanup)
                print('WF_CLEANUP='+json.dumps(cleanup),flush=True)
                status=('complete' if len(rows)==len(design) else 'paused') if success and drained and unchanged else 'failed'
                atomic(out/'suite_status.json',{'status':status,'completed_conditions':len(rows),'expected':len(design)})
                result=report.write(out,rows,design,status)
                if args.suite=='dabs-r234-balanced-64':
                    from .extended_report import write_extended
                    write_extended(out,rows,design,status)

                print('WF_RUN_DIR='+str(out),flush=True)
                print('WF_RUN_STATUS='+status,flush=True)
                if not drained or not unchanged:raise RuntimeError('cleanup/drift check failed; do not start next suite')
    return result

def export_run(out):
    out=Path(out).resolve();p,rows,status=read_run(out)
    target=out.parent/(out.name+'_results.zip')
    if target.exists():raise ValueError('export exists; do not overwrite a previous evidence archive')
    # Include all raw timing/plan/backend evidence, not only favorable summaries.
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_symlink():raise ValueError('symlink not allowed in exported result tree')
            if path.is_file():z.write(path,str(path.relative_to(out)))
    target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n')
    print('WF_EXPORT='+str(target),flush=True)
