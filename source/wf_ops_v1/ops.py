#!/usr/bin/env python3
"""Workflow experiment archive, verified comparison and quiet console sidecar.

Does NOT modify or import the frozen scheduling implementation. No GPU request
is sent except by an explicitly requested `logged -- run ... --confirm-gpu-use`.
Checkpoint reads files while holding the existing experiment locks when present.
No automatic retry, result deletion, baseline substitution or score selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path('/home/wmy/sci/spra_paper_experiments_github')
VERSION = 'workflow-archive-logging-sidecar-1'
METRICS = ('mean_jct_ms','p95_jct_ms','planned_arrival_mean_jct_ms',
           'mean_request_ttft_ms','p95_request_ttft_ms','mean_node_time_to_first_content_ms',
           'mean_client_tpot_proxy_ms','p95_client_tpot_proxy_ms',
           'cohort_workflows_s','cohort_tokens_s')
HIGHER = {'cohort_workflows_s','cohort_tokens_s'}
DROP = {'workflow_rows','unfinished_workflow_curve','request_latency_rows','quality_rows','typed_response_rows'}
HISTORICAL = ('PILOT_20261010T092749Z_421e356e','PILOT_20261011T185607Z_8e964348',
              'PILOT_20261011T185806Z_fc5b583b')
PROFILE = 'MUSIQUE_20261010T092711Z_be09912a'
CODE_EXT = {'.py','.sh','.json','.yaml','.yml','.toml','.ini','.cfg','.txt','.md','.csv','.diff','.zip'}
COMPACT_RAW = {'summary.json','quality.json','typed_responses.json','manifest.json',
 'semantic_proof.json','scheduling_audit.json','inactive_proof.json','request_latency.json',
 'request_latency_summary.json','plans.jsonl','controls.json','action_timing.json',
 'service_prediction_diagnostics.json','ready_events.json','queue_intervals.json',
 'status.json','failure.json','cleanup.json','environment.json','packet_preparation.json',
 'warmup_receipt.json','workflow_only_result.json','HISTORICAL_AD_AUDIT.json','summary.log','receipt.json','invocation.json','conditions.jsonl'}
MAX_REVIEW_FILE = 16*1024*1024
MAX_REVIEW_TOTAL = 256*1024*1024


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def csv_new(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def numeric(v):
    return type(v) in (int,float) and math.isfinite(v)


def safe_file(root,relative):
    rel=Path(relative)
    if rel.is_absolute() or not rel.parts or any(x in ('..','') for x in rel.parts):
        raise ValueError('unsafe relative file: '+str(relative))
    node=root
    for part in rel.parts:
        node=node/part
        if node.is_symlink(): raise ValueError('symlink refused: '+str(node))
    if not node.is_file():raise FileNotFoundError(node)
    return node


def inside(root,path):
    path=Path(path)
    if not path.is_absolute():path=root/path
    try:rel=path.relative_to(root)
    except ValueError:raise ValueError('path outside declared repository: '+str(path))
    node=root
    for part in rel.parts:
        node=node/part
        if node.is_symlink():raise ValueError('symlink refused: '+str(node))
    return path


def discover(root):
    apps=sorted(p for p in root.glob('spra_workflow_*') if p.is_dir() and not p.is_symlink())
    runs=[]
    for app in apps:
        if (app/'runs').is_dir():
            for run in sorted((app/'runs').iterdir()):
                if run.is_dir() and not run.is_symlink():runs.append(run)
    old=root/'spra_stage19_typed_handoff_v1/runs'
    if old.is_dir():
        for run in sorted(old.glob('PILOT_*')):
            if not run.is_symlink() and (run.name in HISTORICAL or (run/'workflow_only_result.json').is_file()):
                runs.append(run)
    return apps,list(dict.fromkeys(runs))


def inspect_run(run,verify_selected=True):
    """Verify saved schema, specs and summaries; do not claim raw-time re-audit."""
    p=load(run/'protocol.json');expected_hash=load(run/'protocol.sha256.json')['sha256']
    if file_sha(run/'protocol.json')!=expected_hash:raise ValueError('protocol hash mismatch')
    rows=load(run/'index.json');status=load(run/'suite_status.json')
    if status.get('status')=='running':raise ValueError('run is active; wait for a normal pause before checkpoint')
    if status.get('status') not in ('prepared','paused','complete','failed'):raise ValueError('unknown execution status')
    design=p['plan']['conditions'];by={d['label']:d for d in design}
    if len(by)!=len(design):raise ValueError('duplicate planned label')
    seen=set()
    for row in rows:
        label=row['label']
        if label in seen or label not in by:raise ValueError('duplicate or unplanned accepted label')
        seen.add(label)
        if any(row.get(k)!=v for k,v in by[label].items()):raise ValueError('accepted spec differs: '+label)
        s=row['summary']
        for key in ('mean_jct_ms','p95_jct_ms'):
            if not numeric(s.get(key)) or s[key]<=0:raise ValueError('bad '+key+': '+label)
        if s.get('condition_valid') is not True:raise ValueError('condition not valid: '+label)
        if s.get('budget_switches_total')!=0 or s.get('prefill_caps_applied')!=[4096]:
            raise ValueError('not fixed-4096 scheduling: '+label)
        expected_native='D' if row['method'] in ('DAG','DAG-Lite') else 'A'
        if s.get('native_method',s.get('method')) not in (None,expected_native):raise ValueError('native method mismatch')
        if verify_selected:
            f=safe_file(run,str(PurePosixPath(row['folder'])/'summary.json'));actual=load(f)
            if digest(actual)!=row['summary_sha256']:raise ValueError('summary content digest mismatch: '+label)
            if {k:v for k,v in actual.items() if k not in DROP}!=s:raise ValueError('index/raw summary mismatch: '+label)
            if row.get('files',{}).get('summary.json')!=file_sha(f):raise ValueError('summary file SHA mismatch')
    if status.get('completed_conditions',len(rows))!=len(rows):raise ValueError('status/index count mismatch')
    if status['status']=='complete' and len(rows)!=len(design):raise ValueError('false complete status')
    if (run/'extension_source.zip').is_file():
        with zipfile.ZipFile(run/'extension_source.zip') as z:
            identity=p.get('extension_identity',{})
            for name,h in identity.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('saved extension source mismatch: '+name)
    coverage=[]
    for R in sorted({d['replicas'] for d in design}):
        want=[d for d in design if d['replicas']==R];got=[r for r in rows if r['replicas']==R]
        coverage.append({'replicas':R,'accepted':len(got),'expected':len(want),
            'by_repeat':{str(t):{'accepted':sum(r['repeat']==t for r in got),'expected':sum(d['repeat']==t for d in want)}
             for t in sorted({d['repeat'] for d in want})}})
    return {'protocol':p,'rows':rows,'status':status,'coverage':coverage,
            'missing':[d['label'] for d in design if d['label'] not in seen]}


def comparisons(run_id,detail):
    p=detail['protocol'];design=p['plan']['conditions'];rows=detail['rows']
    groups=defaultdict(dict)
    for r in rows:groups[(r['replicas'],r['rate'],r['repeat'],r['block'])][r['method']]=r
    pairs=[]
    for (R,rate,t,b),ms in sorted(groups.items()):
        for method in ('DAG','DAG-Lite'):
            if method not in ms:continue
            for baseline in ('RR','P2C','Affinity')+(('DAG',) if method!='DAG' else ()):
                if baseline not in ms:continue
                c,a=ms[method],ms[baseline]
                if c['external_work_sha256']!=a['external_work_sha256']:
                    raise ValueError('paired external workload mismatch: '+c['label'])
                v={'run_id':run_id,'replicas':R,'rate':rate,'repeat':t,'block':b,
                   'candidate':method,'reference':baseline,'candidate_label':c['label'],'reference_label':a['label']}
                for k in METRICS:
                    cv,av=c['summary'].get(k),a['summary'].get(k)
                    v[k+'_gain_pct']=100*((cv/av-1) if k in HIGHER else (1-cv/av)) if numeric(cv) and numeric(av) and av>0 else None
                for k in ('final_em','final_f1'):
                    cv=c['summary'].get('semantic_quality',{}).get(k);av=a['summary'].get('semantic_quality',{}).get(k)
                    v[k+'_delta_pp']=100*(cv-av) if numeric(cv) and numeric(av) else None
                pairs.append(v)
    cs=defaultdict(list)
    for v in pairs:cs[(v['replicas'],v['rate'],v['candidate'],v['reference'])].append(v)
    cells=[]
    for (R,rate,m,b),vals in sorted(cs.items()):
        wanted=[d for d in design if d['replicas']==R and d['rate']==rate and d['method']==m]
        wantref={(d['repeat'],d['block']) for d in design if d['replicas']==R and d['rate']==rate and d['method']==b}
        exp={(d['repeat'],d['block']) for d in wanted}&wantref
        complete={(v['repeat'],v['block']) for v in vals}==exp and bool(exp)
        x={'run_id':run_id,'suite':p['plan']['suite'],'replicas':R,'rate':rate,'candidate':m,'reference':b,
           'pairs':len(vals),'expected_pairs':len(exp),'complete_comparison':complete,
           'session_status':detail['status']['status'],'performance_suite':p['plan']['suite'] not in ('smoke','agecheck')}
        for k in [f+'_gain_pct' for f in METRICS]+['final_em_delta_pp','final_f1_delta_pp']:
            ns=[v[k] for v in vals if numeric(v.get(k))]
            x[k]=statistics.mean(ns) if ns else None
            x[k+'_n']=len(ns)
        x['positive_jct_pairs']=sum(v['mean_jct_ms_gain_pct']>0 for v in vals)
        x['jct_at_least_10_pct']=complete and x['performance_suite'] and detail['status']['status'] in ('complete','paused') and x['mean_jct_ms_gain_pct']>=10
        x['selection_is_descriptive_not_holdout_validation']=True
        cells.append(x)
    return pairs,cells


def build_catalog(root,runs):
    registry=[];allconds=[];pairs=[];cells=[];details={};errors=[]
    for run in runs:
        run_id=str(run.relative_to(root))
        if not (run/'protocol.json').is_file():
            registry.append({'run_id':run_id,'kind':'legacy_or_unrecognized','status':'raw_preserved_not_recomputed',
                'path':str(run),'accepted':None,'expected':None});continue
        try:
            d=inspect_run(run);details[run_id]=d
            ps,cs=comparisons(run_id,d);pairs+=ps;cells+=cs
            registry.append({'run_id':run_id,'kind':'native_suite','status':d['status']['status'],
                'path':str(run),'accepted':len(d['rows']),'expected':len(d['protocol']['plan']['conditions']),
                'run_uuid':d['protocol'].get('run_uuid'),'protocol_sha256':file_sha(run/'protocol.json'),
                'suite':d['protocol']['plan']['suite'],'coverage':d['coverage']})
            for row in d['rows']:
                c={k:row[k] for k in ('label','replicas','rate','repeat','block','position','method','seed','router_seed','packet_slice','base_cap') if k in row}
                c.update(run_id=run_id,raw_folder=str(run/row['folder']),summary_sha256=row['summary_sha256'],
                    external_work_sha256=row['external_work_sha256'])
                c.update({k:row['summary'].get(k) for k in METRICS})
                c.update(output_tokens=row['summary'].get('output_tokens'),requests=row['summary'].get('requests'))
                allconds.append(c)
        except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
            errors.append({'run_id':run_id,'error':str(exc)})
            registry.append({'run_id':run_id,'kind':'native_suite','status':'verification_failed_not_ranked',
                'path':str(run),'accepted':None,'expected':None,'error':str(exc)})
    return registry,allconds,pairs,cells,details,errors


def note_catalog(out,registry,cells,errors):
    rows=['# Workflow实验档案与JCT比较','',
      '按原run分别统计，不跨代码版本或运行协议合并。完整条件比较才标记10%；单次/单Block峰值不替代工况平均。',
      '主指标：同副本、到达率、轮次、Block内JCT百分比下降，再取平均；负值为退化。',
      '请求TTFT、条件P95与客户端TPOT代理保持原口径。原模型可见UUID与输出差异仍存在。',
      '归档不包含模型权重/SSH密钥，不保证另一环境无需依赖即可运行。源码和原始文件归档与实验结论验证是两回事。','',
      '## 运行清单','', '|run|状态|已接受/计划|','|---|---|---|']
    for r in registry:rows.append(f"|{r['run_id']}|{r['status']}|{r.get('accepted')}/{r.get('expected')}|")
    rows+=['','## 完整负载工况的平均JCT收益','',
       '|run|R|λ|方法|基线|配对|JCT改善%|≥10%|','|---|---:|---:|---|---|---:|---:|---|']
    for c in cells:
        if c['candidate']!='DAG' or not c['performance_suite']:continue
        rows.append('|'+ '|'.join([c['run_id'],str(c['replicas']),str(c['rate']),c['candidate'],c['reference'],
            f"{c['pairs']}/{c['expected_pairs']}",f"{c['mean_jct_ms_gain_pct']:.2f}",
            '是' if c['jct_at_least_10_pct'] else ('未达' if c['complete_comparison'] else '不完整')])+'|')
    rows+=['','## 核对失败与范围','',json.dumps(errors,ensure_ascii=False,indent=2),
       '', '请保留full_raw.zip与ARCHIVE_MANIFEST.json。review.zip仅为分析回传材料，不替代本机全量归档。',
       '记录“当前已完成”需要index.json覆盖；本脚本不会启动缺失实验或重试失败attempt。']
    (out/'REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def walk_regular(path,excluded=()):
    if path.is_symlink():raise ValueError('symlink refused: '+str(path))
    if path.is_file():yield path;return
    for directory,dirs,files in os.walk(path,followlinks=False):
        for name in list(dirs):
            node=Path(directory)/name
            if node.is_symlink():raise ValueError('directory symlink refused: '+str(node))
            if name in set(excluded)|{'__pycache__','.git','.pytest_cache'}:dirs.remove(name)
        for name in sorted(files):
            node=Path(directory)/name
            if node.is_symlink():raise ValueError('file symlink refused: '+str(node))
            if node.is_file() and node.suffix not in {'.pyc','.pyo'}:yield node


def source_inventory(root,apps,runs,details):
    """All current workflow app files/results, and required historical native sources.

    Weights, external secrets and arbitrary unrelated experiment generations are
    intentionally not recursively copied. Every omission from this scope is recorded.
    """
    files={};missing=[]
    def add(path):
        path=inside(root,path)
        if not path.exists():missing.append(str(path));return
        for f in walk_regular(path):files[str(f.relative_to(root))]=f
    for a in apps:add(a)
    for r in runs:add(r)
    if (root/'wf_ops_output').is_dir():add(root/'wf_ops_output')
    if (root/'wf_ops_v1').is_dir():add(root/'wf_ops_v1')
    for name in ('src','tools','configs','metadata'):
        path=root/name
        if path.is_dir():
            for f in walk_regular(path):
                if f.suffix in CODE_EXT:files[str(f.relative_to(root))]=f
    for name in ('spra_stage19_musique_joint_v1','spra_stage19_typed_handoff_v1'):
        path=root/name
        if not path.is_dir():missing.append(str(path));continue
        for f in walk_regular(path,('runs','prepared','dataset','logs','offline_assets','data','third_party')):
            if f.suffix in CODE_EXT or f.name.startswith('SHA256'):files[str(f.relative_to(root))]=f
    references=set()
    for d in details.values():
        ck=d['protocol'].get('checkpoint',{})
        for key in ('prepared','dataset'):
            if ck.get(key):references.add(str(inside(root,ck[key])))
    references.add(str(root/'spra_stage19_typed_handoff_v1/prepared'/PROFILE))
    references.add(str(root/'spra_stage19_musique_joint_v1/dataset/compiled'))
    for r in sorted(references):add(Path(r))
    for app in apps:
        z=app.with_suffix('.zip')
        if z.is_file():add(z)
    # Do not copy model weights if unexpectedly placed inside an app/result tree.
    omitted=[]
    for n in list(files):
        if files[n].suffix.lower() in {'.safetensors','.bin','.pt','.pth','.ckpt','.gguf'}:
            omitted.append({'path':n,'reason':'model_weight_or_binary_excluded'});del files[n]
    return files,missing,omitted


def unchanged_signature(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)


def write_zip_stream(path,items,accepted_hashes=None):
    """Hash the exact bytes stored. Stop if a file changes; publish after checks."""
    manifest={};accepted_hashes=accepted_hashes or {}
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,source in sorted(items.items()):
            before=unchanged_signature(source);h=hashlib.sha256();count=0
            with source.open('rb') as f,z.open(name,'w',force_zip64=True) as dest:
                for b in iter(lambda:f.read(1024*1024),b''):
                    h.update(b);dest.write(b);count+=len(b)
            if before!=unchanged_signature(source) or count!=before[2]:raise ValueError('file changed while archiving: '+str(source))
            if name in accepted_hashes and h.hexdigest()!=accepted_hashes[name]:raise ValueError('accepted raw-file hash mismatch: '+name)
            manifest[name]={'sha256':h.hexdigest(),'bytes':count,'source':str(source)}
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps({'schema':VERSION,'files':manifest},ensure_ascii=False,indent=2)+'\n')
    return manifest


def checkpoint(root,out,full=False):
    root=root.resolve(strict=True)
    if out.exists():raise FileExistsError('checkpoint directory already exists; never overwrite: '+str(out))
    out=out.absolute();apps,runs=discover(root)
    if out.is_relative_to(root) and not out.is_relative_to(root/'wf_archives'):
        raise ValueError('inside repository, checkpoint output must be under wf_archives/')
    if not runs:raise ValueError('no workflow experiment run directories found')
    if any(out.is_relative_to(p) for p in apps+runs):raise ValueError('checkpoint must be outside captured app/result trees')
    # Shared native experiment locks, not a deployment or service change.
    with ExitStack() as stack:
        locks=set()
        for run in runs:
            if (run/'protocol.json').is_file():
                p=load(run/'protocol.json');rt=p.get('runtime')
                if rt:
                    lp=Path(rt)/'experiment.lock'
                    if lp.is_symlink():raise ValueError('runtime lock symlink')
                    if not lp.is_file():raise ValueError('native experiment lock missing: '+str(lp))
                    locks.add(lp)
                if (run/'suite_status.json').is_file() and load(run/'suite_status.json').get('status')=='running':
                    raise ValueError('active run; checkpoint after clean pause: '+str(run))
        for lp in sorted(locks):
            handle=stack.enter_context(lp.open('rb'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registry,conditions,pairs,cells,details,errors=build_catalog(root,runs)
        files,missing,omitted=source_inventory(root,apps,runs,details)
        if not files:raise ValueError('no files to preserve')
        checks={}
        metadata={}
        for run_id,d in details.items():
            run=root/run_id
            for key in ('protocol.json','index.json','suite_status.json'):
                metadata[run/key]=file_sha(run/key)
            for row in d['rows']:
                for name,h in row['files'].items():
                    rel=str(PurePosixPath(run_id)/row['folder']/name)
                    safe_file(root,rel);checks[rel]=h
        size=sum(p.stat().st_size for p in files.values())
        out.parent.mkdir(parents=True,exist_ok=True)
        need=(size if full else min(size,MAX_REVIEW_TOTAL))+64*1024*1024
        if shutil.disk_usage(out.parent).free<need:raise OSError('insufficient worst-case archive disk space; no source deleted')
        out.mkdir(exist_ok=False)
        try:
            put_json(out/'catalog.json',{'schema':VERSION,'repo':str(root),'runs':registry,'errors':errors,
                'scope':'workflow request-scheduling lineage plus 3 known typed pilots; not all unrelated joint experiments',
                'source_missing':missing,'model_weight_exclusions':omitted,'archive_mode':'full' if full else 'review_only'})
            csv_new(out/'run_registry.csv',registry);csv_new(out/'condition_registry.csv',conditions)
            csv_new(out/'paired_metrics.csv',pairs);csv_new(out/'cell_gains.csv',cells)
            csv_new(out/'ten_percent_candidates.csv',[c for c in cells if c['jct_at_least_10_pct']])
            put_json(out/'cell_gains.json',cells)
            note_catalog(out,registry,cells,errors)
            items={};used=0;review_omissions=[]
            runparts=[str(r.relative_to(root))+'/' for r in runs]
            for name,f in sorted(files.items()):
                isrun=any(name.startswith(r) for r in runparts)
                include=(not isrun and '/logs/' not in name and not name.startswith('wf_ops_output/')) or f.name in COMPACT_RAW or (
                    isrun and f.name in {'protocol.json','protocol.sha256.json','index.json','suite_status.json',
                      'environment_contract.json','extension_source.zip','REPORT.md','RESULT.json','cell_gains.csv','conditions.csv','paired_metrics.csv'})
                n=f.stat().st_size
                if not include:review_omissions.append({'path':name,'reason':'raw_or_log_kept_in_full_archive_only'});continue
                if n>MAX_REVIEW_FILE or used+n>MAX_REVIEW_TOTAL:
                    review_omissions.append({'path':name,'reason':'compact_size_limit'});continue
                items['repo/'+name]=f;used+=n
            if full:
                print(f'ARCHIVE_BEGIN files={len(files)} source_bytes={size} mode=full',flush=True)
                m=write_zip_stream(out/'full_raw.partial.zip',{'repo/'+n:f for n,f in files.items()},
                    {'repo/'+n:h for n,h in checks.items()})
                put_json(out/'full_manifest.json',{'files':m,'accepted_files_hash_checked':len(checks),
                    'raw_timing_recomputed':False,'all_selected_source_files_archived':True})
            else:
                put_json(out/'full_manifest.json',{'full_archive_created':False,'source_files_inventory':{
                    n:{'bytes':f.stat().st_size,'source':str(f)} for n,f in files.items()},'raw_files_hashed':False})
            put_json(out/'review_omissions.json',{'files':review_omissions,'missing_sources':missing,
                'full_archive_is_required_for_all_raw_data':True})
            for f in sorted(out.iterdir()):
                if f.is_file() and f.suffix in {'.json','.md','.csv'}:items['catalog/'+f.name]=f
            write_zip_stream(out/'review.partial.zip',items,{'repo/'+n:h for n,h in checks.items() if 'repo/'+n in items})
            for f,h in metadata.items():
                if file_sha(f)!=h:raise ValueError('run metadata changed during checkpoint: '+str(f))
            for base in (['full_raw'] if full else [])+['review']:
                temp=out/(base+'.partial.zip');dest=out/(base+'.zip');temp.rename(dest)
                with (out/(base+'.zip.sha256')).open('x') as f:f.write(file_sha(dest)+'  '+dest.name+'\n')
            receipt={'schema':VERSION,'status':'complete','full_archive_created':full,
                'review':str(out/'review.zip'),'full':str(out/'full_raw.zip') if full else None,
                'verified_suite_count':len(details),'unranked_runs':len(errors),
                'run_count':len(runs),'conditions_in_verified_indexes':len(conditions),'new_gpu_requests':0,
                'frozen_scheduler_edited':False,'model_weights_included':False,'source_missing':missing}
            put_json(out/'receipt.json',receipt)
            print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
            for r in registry:
                print('RUN',r['run_id'],r['status'],f"{r.get('accepted')}/{r.get('expected')}",flush=True)
                for cov in r.get('coverage',[]):print(' COVERAGE',json.dumps(cov,ensure_ascii=False),flush=True)
            return receipt
        except BaseException as exc:
            put_json(out/'CHECKPOINT_FAILED.json',{'error':repr(exc),'partial_archives_not_valid':True,
                'source_results_not_modified':True})
            raise


class Console:
    def __init__(self,screen,jsonl,csvfile):
        self.screen=screen;self.jsonl=jsonl
        self.writer=csv.DictWriter(csvfile,fieldnames=['label','replicas','rate','repeat','block','method',
            'mean_jct_ms','p95_jct_ms','mean_request_ttft_ms','cohort_workflows_s','requests','output_tokens',
            'decision_wall_mean_ms','final_em','final_f1'])
        self.writer.writeheader();self.csvfile=csvfile;self.accepted=0;self.typed=0;self.errors=0;self.pair={};self.emitted=set()
    def show(self,text):
        self.screen.write(text+'\n');self.screen.flush()
    def feed(self,raw):
        line=raw.decode('utf-8',errors='replace').rstrip('\r\n');kind,sep,payload=line.partition('=')
        if kind=='TYPED_RESPONSE':self.typed+=1;return
        if kind in ('WF_CONDITION_ACCEPTED','WF_CONDITION_START','CONDITION_DONE','WF_CLEANUP'):
            try:d=json.loads(payload)
            except (ValueError,TypeError):
                self.errors+=1;self.show('LOG_PARSE_WARNING '+kind+'; exact bytes retained in full.log');return
            if kind=='WF_CONDITION_START':self.show('START '+str(d.get('label')));return
            if kind=='CONDITION_DONE':
                if d.get('method')=='Forced':self.show(f"WARMUP requests={d.get('requests')} JCT={d.get('mean_jct_ms',0):.2f} ms (not measured)")
                return
            if kind=='WF_CLEANUP':self.show('CLEANUP '+json.dumps(d,ensure_ascii=False));return
            self.accepted+=1
            self.jsonl.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n');self.jsonl.flush()
            row={k:d.get(k) for k in self.writer.fieldnames};row['decision_wall_mean_ms']=d.get('decision_wall_ms',{}).get('mean')
            row['final_em']=d.get('semantic_quality',{}).get('final_em');row['final_f1']=d.get('semantic_quality',{}).get('final_f1')
            self.writer.writerow(row);self.csvfile.flush()
            self.show(f"DONE +{self.accepted:03d} {d.get('label')} | JCT={d.get('mean_jct_ms',0):.2f}ms "
                f"P95={d.get('p95_jct_ms',0):.2f}ms TTFT={d.get('mean_request_ttft_ms',0):.2f}ms "
                f"WF/s={d.get('cohort_workflows_s',0):.3f} out={d.get('output_tokens')}")
            key=tuple(d.get(k) for k in ('replicas','rate','repeat','block'))
            ms=self.pair.setdefault(key,{});ms[d['method']]=d
            if 'DAG' in ms:
                for b in ('RR','P2C','Affinity'):
                    if b not in ms or (key,b) in self.emitted:continue
                    a=ms[b]['mean_jct_ms'];v=100*(1-ms['DAG']['mean_jct_ms']/a)
                    self.show(f'PAIR R{key[0]} L{key[1]:g} T{key[2]} B{key[3]} DAG/{b}: JCT {v:+.2f}% (single pair)')
                    self.emitted.add((key,b))
            return
        if not line:return
        # Short ordinary messages/errors remain visible. Very long dumps are kept
        # byte-for-byte in full.log rather than flooded to the terminal.
        self.show(line if len(line)<=800 else line[:240]+' ... [full line in full.log]')


def logged(app,logroot,argv):
    app=app.resolve(strict=True)
    if not (app/'run.sh').is_file():raise FileNotFoundError(app/'run.sh')
    if not argv or argv[0] not in ('run','gate','report','export','check','freeze','plan','benchmark'):
        raise ValueError('after -- pass the original command, e.g. run --suite grid ...')
    if argv[0]=='run' and '--confirm-gpu-use' not in argv:raise ValueError('GPU run requires explicit --confirm-gpu-use')
    if logroot.resolve().is_relative_to(app):raise ValueError('sidecar log directory must stay outside frozen application')
    folder=logroot/('LOG_'+stamp());folder.mkdir(parents=True,exist_ok=False)
    source_before={str(p.relative_to(app)):file_sha(p) for p in walk_regular(app,('runs','logs','frozen'))
                   if p.suffix in {'.py','.sh'} and ('reference' not in p.relative_to(app).parts)}
    command=['bash','run.sh',*argv]
    manifest={'schema':VERSION,'cwd':str(app),'argv':command,'started_utc':datetime.now(timezone.utc).isoformat(),
        'sidecar_sha256':file_sha(Path(__file__)),'child_python':sys.executable,'child_uses_original_run_sh':True,'new_algorithm':False,
        'environment':{k:os.environ.get(k) for k in ('CONDA_DEFAULT_ENV','CONDA_PREFIX','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')},
        'source_before':source_before}
    put_json(folder/'invocation.json',manifest)
    print('SIDECAR_DIR='+str(folder),flush=True)
    interrupted=[]
    with (folder/'full.log').open('xb') as full,(folder/'summary.log').open('x',encoding='utf-8') as summary,\
         (folder/'conditions.jsonl').open('x',encoding='utf-8') as events,(folder/'conditions.csv').open('x',encoding='utf-8-sig',newline='') as cf:
        class Screen:
            def write(self,text):
                summary.write(text)
                try:sys.stdout.write(text)
                except BrokenPipeError:pass
            def flush(self):
                summary.flush()
                try:sys.stdout.flush()
                except BrokenPipeError:pass
        display=Console(Screen(),events,cf)
        child_env=os.environ.copy()
        child_env['PATH']=str(Path(sys.executable).parent)+os.pathsep+child_env.get('PATH','')
        child=subprocess.Popen(command,cwd=str(app),env=child_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        old={}
        def forward(signum,frame):
            interrupted.append(signum)
            try:os.killpg(child.pid,signum)
            except ProcessLookupError:pass
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            old[sig]=signal.signal(sig,forward)
        try:
            assert child.stdout is not None
            for raw in iter(child.stdout.readline,b''):
                full.write(raw);full.flush();display.feed(raw)
            rc=child.wait()
        except BaseException:
            forward(signal.SIGINT,None);child.wait();raise
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
        os.fsync(full.fileno())
    source_after={n:file_sha(app/n) for n in source_before}
    receipt={'schema':VERSION,'child_returncode':rc,'signals_forwarded':interrupted,
        'accepted_lines_this_invocation':display.accepted,'suppressed_response_lines':display.typed,
        'parse_warnings':display.errors,'source_unchanged':source_before==source_after,
        'raw_console_sha256':file_sha(folder/'full.log'),'summary_console_sha256':file_sha(folder/'summary.log'),
        'same_algorithm_not_identical_timing_guarantee':True,'completed_utc':datetime.now(timezone.utc).isoformat()}
    put_json(folder/'receipt.json',receipt)
    print('LOG_RECEIPT='+json.dumps(receipt,ensure_ascii=False),flush=True)
    print('FULL_LOG='+str(folder/'full.log'),flush=True)
    print('SUMMARY_LOG='+str(folder/'summary.log'),flush=True)
    return rc if rc>=0 else 128-rc


def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate archive names')
        manifest=json.loads(z.read('ARCHIVE_MANIFEST.json'))['files']
        if set(names)!=set(manifest)|{'ARCHIVE_MANIFEST.json'}:raise ValueError('manifest member set mismatch')
        for name,expected in manifest.items():
            if name.startswith('/') or '..' in PurePosixPath(name).parts:raise ValueError('unsafe archive member')
            h=hashlib.sha256();n=0
            with z.open(name) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
            if h.hexdigest()!=expected['sha256'] or n!=expected['bytes']:raise ValueError('archive member mismatch: '+name)
    print('ARCHIVE_VERIFY_OK files='+str(len(manifest)),flush=True)
    return len(manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('checkpoint');p.add_argument('--repo',type=Path,default=ROOT);p.add_argument('--out',type=Path)
    p.add_argument('--full',action='store_true',help='preserve all raw files within workflow experiment scope on GPU')
    p=sub.add_parser('logged');p.add_argument('--app',type=Path,default=ROOT/'spra_workflow_sched_v2_1')
    p.add_argument('--log-root',type=Path,default=ROOT/'wf_ops_output');p.add_argument('args',nargs=argparse.REMAINDER)
    p=sub.add_parser('verify');p.add_argument('archive',type=Path)
    args=parser.parse_args()
    try:
        if args.command=='checkpoint':checkpoint(args.repo,args.out or args.repo/'wf_archives'/('CHECKPOINT_'+stamp()),args.full)
        elif args.command=='verify':verify_archive(args.archive)
        else:
            rest=args.args[1:] if args.args and args.args[0]=='--' else args.args
            return logged(args.app,args.log_root,rest)
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        print('STOP: '+str(exc),file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
