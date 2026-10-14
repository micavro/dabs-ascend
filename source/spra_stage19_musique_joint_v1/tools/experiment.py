#!/usr/bin/env python3
"""Selector repair experiment entrypoint. No stage, restart or install action."""
from __future__ import annotations
import argparse,asyncio,copy,fcntl,hashlib,json,multiprocessing,os,sys,time,uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint18v2.common import load,atomic,digest,token_hash,VERSION
from joint18v2.client import gate,execute,token_pool,wait_drained
from joint18v2.workload import build
from joint18v2.report import comparison
from joint18v2.search import REVISION
from joint18v2.lifetime_probe import REVISION as LIFETIME_REVISION
OLD_PILOT='PILOT_20261009T075649Z_7646b651'
MODEL_REL='spra_stage18_joint_core_v1/runs/CALIBRATE_20261008T152619Z_edfd9e63/host_model.json'
OLD_NAME='spra_stage18_v2_joint_integration_v1'
HOOK_SHA='78ca1d716c5701a9f83ba7dc0b22669b6b50a38bd54626a56642130cd1684418'
ORDERS=[['A','D','B','Decoupled','Joint'],['D','B','Decoupled','Joint','A'],
        ['Joint','Decoupled','A','B','D'],['B','A','Joint','D','Decoupled']]
CHECK_FILES=('joint18v2/client.py','joint18v2/actions.py','joint18v2/search.py','joint18v2/model.py',
     'core18v2/engine.py','joint18v2/common.py','joint18v2/transport.py','joint18v2/workload.py',
     'joint18v2/latency_metrics.py','runtime/joint18v2_hook.py','configs/development.json')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compact(s):return {k:v for k,v in s.items() if k not in ('workflow_rows','unfinished_workflow_curve','request_latency_rows')}
def salt(pool):
    n=uuid.uuid4().int;out=[]
    for _ in range(12):n,i=divmod(n,len(pool));out.append(pool[i])
    return out

def source_check(source):
    expected=load(ROOT/'analysis/baseline_hashes.json');actual={};bad=[]
    for rel in CHECK_FILES:
        p=Path(source)/rel
        actual[rel]=sha(p) if p.is_file() else None
        if actual[rel]!=expected[rel]:bad.append(rel)
    if bad:raise RuntimeError('Original integration source differs or is absent: '+', '.join(bad)+'. No source/live files modified; do not redeploy or bypass hashes.')
    return {'source':str(Path(source).resolve()),'checked_hashes':actual,'pass':True,'source_modified':False}

def code_identity():
    return {str(p.relative_to(ROOT)):sha(p) for d in ('joint18v2','core18v2','tools','configs')
            for p in sorted((ROOT/d).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json')}

def prepare(a):
    from joint18v2.service_fit import read_rows,fit_phase_model
    source=a.source;sourceinfo=source_check(source)
    old=a.pilot_run or source/'runs'/OLD_PILOT
    legacy_path=a.legacy_model or ROOT.parent/MODEL_REL
    if not old.is_dir() or not legacy_path.is_file():
        raise RuntimeError(f'Archived input missing. Need {old} and {legacy_path}. Use explicit --pilot-run/--legacy-model; do NOT rerun GPU calibration.')
    status=load(old/'suite_status.json')
    if status.get('status')!='complete':raise RuntimeError('old pilot is not complete')
    folders=sorted(p for p in old.glob('B[1-4]_*') if p.is_dir())
    if len(folders)!=20:raise RuntimeError(f'need all 20 archived conditions, got {len(folders)}')
    signatures=set();rows=[];rej={};files={}
    for folder in folders:
        s=load(folder/'summary.json');mf=load(folder/'manifest.json')
        method=mf['method'];block=int(folder.name.split('_')[0][1:]);signatures.add((block,method))
        if not s.get('condition_valid') or s.get('requests')!=96 or s.get('output_tokens')!=9216:
            raise RuntimeError('unexpected original6 N16 old condition '+str(folder))
        rr,rejections=read_rows(folder);rows+=rr;rej[folder.name]=rejections
        for n in ('backend.jsonl','manifest.json','summary.json'):files[str(folder/n)]=sha(folder/n)
    if signatures!={(b,m) for b in range(1,5) for m in ('A','D','B','Decoupled','Joint')}:
        raise RuntimeError('old pilot method/block coverage mismatch')
    model,report=fit_phase_model(rows,load(legacy_path),4)
    stamp='PREPARED_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'prepared'/stamp;out.mkdir(parents=True,exist_ok=False)
    atomic(out/'phase_model.json',model);atomic(out/'phase_cv_report.json',report)
    with (out/'host_samples.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    unchanged=all(sha(Path(p))==h for p,h in files.items())
    if not unchanged:raise RuntimeError('source archive changed while preparing; artifact not ready')
    manifest={'revision':REVISION,'source_check':sourceinfo,'training_pilot':str(old.resolve()),
      'training_source_hashes':files,'legacy_model_path':str(legacy_path.resolve()),'legacy_sha256':sha(legacy_path),
      'model_sha256':sha(out/'phase_model.json'),'model_id':model['model_id'],'rows':len(rows),
      'rejection_counts':rej,'fit_uses_JCT_labels':False,'new_measured_run_used':False,
      'GPU_REQUESTS':0,'NETWORK_REQUESTS':0,'LIVE_RUNTIME_READS':0,'LIVE_CHANGES':0,
      'source_files_unchanged':unchanged,'counterfactual_rank_certified':False}
    atomic(out/'manifest.json',manifest)
    ROOT.joinpath('prepared').mkdir(exist_ok=True);atomic(ROOT/'prepared/latest.json',{'directory':str(out.resolve())})
    phase_summary={r:{ph:{'n':v['count'],'phase_fit_selected':v['phase_fit_selected'],
      'legacy_block_mae_ms':v['legacy_block_equal_mae_ms'],'new_block_mae_ms':v['candidate_block_equal_mae_ms']}
      for ph,v in phases.items()} for r,phases in report.items()}
    print('PREPARE_RESULT='+json.dumps({'directory':str(out),'phase_summary':phase_summary,
      'model_id':model['model_id'],'source_files_unchanged':unchanged,'gpu_requests':0,'live_changes':0}),flush=True)
    print('MODEL_FROZEN='+str(out/'phase_model.json'),flush=True)
    return out

def read_prepared(arg=None):
    folder=Path(arg) if arg else Path(load(ROOT/'prepared/latest.json')['directory'])
    mf=load(folder/'manifest.json');mp=folder/'phase_model.json'
    if sha(mp)!=mf['model_sha256']:raise RuntimeError('prepared model hash changed')
    m=load(mp)
    if m['model_id']!=mf['model_id']:raise RuntimeError('prepared model ID mismatch')
    return folder,m,mf

def canary_receipt(source,initial):
    for f in sorted((Path(source)/'runs').glob('CANARY_*/canary_result.json'),reverse=True):
        z=load(f)
        if z.get('pass') and z.get('lifetime_revision')==LIFETIME_REVISION and z.get('version')==VERSION and z.get('hook_hash')==HOOK_SHA:
            if load(f.parent/'cleanup.json').get('backend_drained') and load(f.parent/'suite_status.json').get('status')=='complete':
                return str(f.resolve())
    raise RuntimeError('No existing successful native cache/lifetime receipt for unchanged hook. Do not manufacture a receipt.')

def warm_nodes(cfg,pool,profile,nsalt):
    p,o=profile;out={}
    import random
    for r in range(4):
        rng=random.Random(189900+p+r);tokens=nsalt+[rng.choice(pool) for _ in range(p-len(nsalt))];k=f'warm/{r}'
        out[k]={'id':k,'wid':k,'name':'service_warmup','parents':[],'prompt':p,'output':o,'tokens':tokens,
            'hash':token_hash(tokens),'prefix_hash':None,'share':0,'arrival_offset':.02,'state':'blocked','replica':None,
            'computed':0,'output_done':0,'ready_at':None,'start':None,'finish':None,'warm_replica':r}
    return out

class FixedWarmup:
    triggered=False;record=None
    def apply(self,snap,base,ss,cfg):
        from joint18v2.actions import resources
        a=copy.deepcopy(base)
        for k in a['placement']:a['placement'][k]=snap['nodes'][k]['warm_replica']
        a.update(family='fixed_service_warmup',order=[],caps=[cfg['base_cap']]*len(ss))
        return resources(a,snap,cfg),None

def warm_worker():return os.getpid()

async def run(a):
    import httpx
    sourceinfo=source_check(a.source);prepared,model,model_info=read_prepared(a.prepared)
    cfg=load(ROOT/'configs/development.json')
    if a.blocks!=4:raise RuntimeError('this frozen development batch requires all four blocks')
    with (a.runtime/'experiment.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      async with httpx.AsyncClient(timeout=httpx.Timeout(180,connect=5,pool=10),trust_env=False,
          limits=httpx.Limits(max_connections=192,max_keepalive_connections=64,keepalive_expiry=2)) as client:
        initial=await gate(client,a.runtime,4)
        if any(x.get('hook_source_sha256')!=HOOK_SHA for x in initial):raise RuntimeError('unexpected backend hook; no restart performed')
        receipt=canary_receipt(a.source,initial)
        out=ROOT/'runs'/('SELECTOR_PILOT_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
        print('RESULT='+str(out),flush=True)
        code=code_identity();atomic(out/'environment.json',initial);atomic(out/'source_check.json',sourceinfo)
        atomic(out/'code_identity.json',code);atomic(out/'phase_model.json',model);atomic(out/'model_provenance.json',model_info)
        atomic(out/'frozen.json',{'revision':REVISION,'config':cfg,'orders':ORDERS,
           'block_seeds':[182501,182502,182503,182504], 'wave_offsets_change_across_blocks':False,
           'reference':'cap1024 regression reference; not retuned v2 optimum','native_canary_reused':receipt,
           'singletons_gate_joint':False,'warmup':{'profiles':[[512,32],[3328,64]],'requests':8,'included_in_measured_JCT':False},
           'new_native_hook':False,'restarts':0,'model_frozen_before_all_conditions':True})
        rows=[];executor=None
        try:
            pool=await token_pool(client,cfg['model'])
            # A fixed count, the same four-replica service path; never repeat until fast.
            for j,profile in enumerate(((512,32),(3328,64)),1):
                ns=warm_nodes(cfg,pool,profile,salt(pool))
                summary,_=await execute(client,a.runtime,initial,cfg,ns,out/'SERVICE_WARMUP'/f'P{j}','Forced',None,mechanism=FixedWarmup())
                counts={str(r):sum(n.get('replica')==r for n in ns.values()) for r in range(4)}
                if list(counts.values())!=[1]*4:raise RuntimeError('service warmup did not cover all replicas')
                atomic(out/'SERVICE_WARMUP'/f'P{j}'/'receipt.json',{'replica_counts':counts,'measured':False})
            print('SERVICE_WARMUP_COMPLETE=True REQUESTS=8 MEASURED=False',flush=True)
            executor=ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'))
            await asyncio.get_running_loop().run_in_executor(executor,warm_worker)
            for block,order in enumerate(ORDERS,1):
                atomic(out/f'block{block}_order.json',order)
                for position,method in enumerate(order,1):
                    if code_identity()!=code:raise RuntimeError('source changed during measured suite')
                    ns=build(cfg,pool,182500+block,salt(pool))
                    folder=out/f'B{block}_{position}_{method}'
                    summary,events=await execute(client,a.runtime,initial,cfg,ns,folder,method,model,executor=executor)
                    rows.append({'block':block,'position':position,'method':method,'summary':compact(summary)})
                    atomic(out/'progress.json',rows)
            report=comparison(rows)
            report['selector_revision']=REVISION
            report['coverage_by_method']={m:[r['summary'].get('selector_coverage') for r in rows if r['method']==m]
                for m in ('A','D','B','Decoupled','Joint')}
            report['reference_retuned']=False;report['old_results_pooled']=False
            atomic(out/'joint_result.json',report);atomic(out/'suite_status.json',{'status':'complete','measured_conditions':20})
            print('SELECTOR_JOINT_RESULT='+json.dumps(report,ensure_ascii=False),flush=True)
        except BaseException as ex:
            atomic(out/'suite_status.json',{'status':'failed','partial_conditions':len(rows),'error':repr(ex),'automatic_retry':False});raise
        finally:
            if executor:executor.shutdown(wait=True,cancel_futures=True)
            atomic(a.runtime/'joint18v2/control.json',{'enabled':False})
            try:drained=await wait_drained(a.runtime,initial)
            except Exception:drained=False
            cleanup={'backend_drained':drained,'control_disabled':True,'source_files_unchanged':code_identity()==code,'old_results_changed':False}
            atomic(out/'cleanup.json',cleanup);print('CLEANUP='+json.dumps(cleanup),flush=True)
            if not drained:raise RuntimeError('backend not drained; preserve results and stop')

async def check_gate(a):
    import httpx
    source_check(a.source)
    async with httpx.AsyncClient(timeout=10,trust_env=False) as client:
        ss=await gate(client,a.runtime,4)
        if any(s.get('hook_source_sha256')!=HOOK_SHA for s in ss):raise RuntimeError('unexpected backend hook')
        print('GATE_READY=True MODEL_GENERATION_REQUESTS=0 BACKEND_CHANGED=False')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','prepare','gate','run','report'])
    p.add_argument('--source',type=Path,default=ROOT.parent/OLD_NAME)
    p.add_argument('--runtime',type=Path,default=Path('/home/wmy/sci/spra_core_exp/joint_e1'))
    p.add_argument('--pilot-run',type=Path);p.add_argument('--legacy-model',type=Path);p.add_argument('--prepared',type=Path)
    p.add_argument('--blocks',type=int,default=4);p.add_argument('--run-dir',type=Path)
    a=p.parse_args()
    if a.command=='plan':
        print(json.dumps({'revision':REVISION,'CPU_REPLAY_upload_required':False,'backend_restart':False,
           'next':'prepare archived phase model -> same-version A/D/B/Decoupled/Joint x 4 paired blocks',
           'warm_mechanism_repeated':False,'singletons_gate_joint':False,'raw_JCT_cost_subtraction':False,
           'source':str(a.source),'config':load(ROOT/'configs/development.json')},indent=2,ensure_ascii=False));return
    if a.command=='prepare':prepare(a);return
    if a.command=='report':
        if not a.run_dir:p.error('--run-dir required')
        print((a.run_dir/'joint_result.json').read_text());return
    asyncio.run(check_gate(a) if a.command=='gate' else run(a))
if __name__=='__main__':main()
