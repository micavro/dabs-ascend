"""Source-only handoff loader. Explicitly replaces historical result dependencies.

The algorithm/native execution modules are retained. A new, named input profile
is verified rather than requiring an old fit report or historical score table.
No GPU, service, model, cache or clock validation is bypassed here.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
from .common import HERE, load, sha, write_new

KIT = HERE.parent.parent
_MODULE = None

def profile_dir():
    return Path(os.environ.get('WF_PORT_PROFILE', str(KIT/'profiles/reference'))).resolve()

def reproduction_module():
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location('_workflow_port_fixed_contract', HERE/'reference/workflow_only.py')
        m = importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        _MODULE=m
    return _MODULE

def source_map(root):
    root=Path(root).resolve();prof=profile_dir();mf=load(prof/'manifest.json');paths={}
    for name in mf['application_source_hashes']:
        ns,rel=name.split('/',1)
        if ns not in ('source','adapter') or '..' in Path(rel).parts or Path(rel).is_absolute():
            raise ValueError('invalid application source namespace/path')
        p=root/('spra_stage19_musique_joint_v1' if ns=='source' else 'spra_stage19_typed_handoff_v1')/rel
        if p.is_symlink():raise ValueError('source symlinks must be resolved explicitly: '+str(p))
        paths[name]=p
    for name in ('manifest.json',*mf['files']):
        if Path(name).name!=name:raise ValueError('invalid profile file name')
        paths['profile/'+name]=prof/name
    paths['dataset/manifest.json']=root/'spra_stage19_musique_joint_v1/dataset/compiled/manifest.json'
    paths['adapter/dependency_identity.json']=root/'spra_stage19_typed_handoff_v1/dependency_identity.json'
    paths['reproduction/workflow_only.py']=HERE/'reference/workflow_only.py'
    return paths

def inspect(root:Path, require_data=True):
    root=Path(root).resolve();prof=profile_dir();mf=load(prof/'manifest.json')
    if mf.get('schema')!='workflow-port-profile-1':raise ValueError('use a named port profile, not an old result directory')
    paths=source_map(root);bad=[]
    for n,h in mf['application_source_hashes'].items():
        if not paths[n].is_file() or sha(paths[n])!=h:bad.append(n)
    for n,h in mf['files'].items():
        if not (prof/n).is_file() or sha(prof/n)!=h:bad.append('profile/'+n)
    if bad:raise ValueError('profile/source identity mismatch; create an explicit new adaptation profile: '+', '.join(bad))
    ds=root/'spra_stage19_musique_joint_v1/dataset/compiled';dm=load(ds/'manifest.json')
    am=load(KIT/'provenance/input_assets.json')
    if sha(ds/'manifest.json')!=am['source_dataset_manifest_sha256']:raise ValueError('original dataset selection/manifest changed')
    states={}
    for n in ('pilot.public.jsonl','evaluation_gold.jsonl'):
        p=ds/n;ok=p.is_file() and not p.is_symlink() and sha(p)==dm['files'][n]
        states[n]='hash_match' if ok else 'missing_or_hash_mismatch'
        if require_data and not ok:raise ValueError('required input absent/changed: '+str(p)+'; run port.py export-on-gpu on the source server')
    if require_data:
        rr=[json.loads(x) for x in (ds/'pilot.public.jsonl').read_text().splitlines() if x.strip()]
        if len(rr)!=64 or len({x['id'] for x in rr})!=64:raise ValueError('expected original64 distinct inputs')
    return {'revision':'workflow-source-handoff-loader-1','source_files_verified':len(mf['application_source_hashes']),
       'typed_source':str(root/'spra_stage19_typed_handoff_v1'),
       'scheduler_source':str(root/'spra_stage19_musique_joint_v1'),
       'prepared':str(prof),'profile_id':mf['profile_id'],'profile_manifest_sha256':sha(prof/'manifest.json'),
       'model_sha256':mf['files']['model.json'],'priors_sha256':mf['files']['priors.json'],
       'dataset_manifest_sha256':sha(ds/'manifest.json'),'data_files':states,
       'typed_dependency_identity_sha256':sha(paths['adapter/dependency_identity.json']),
       'gpu_generation_requests':0,'backend_accessed':False,'reference_profile_for_target':mf.get('target_calibration_required',True)}

def load_original(root):
    root=Path(root).resolve();source=root/'spra_stage19_musique_joint_v1';typed=root/'spra_stage19_typed_handoff_v1'
    os.environ['STAGE19_SOURCE']=str(source);sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('workflow_d0_original_typed',typed/'tools/run_typed.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    # Revisions are not pretended to be the old hook after an explicit target profile is created.
    m.HOOK=load(profile_dir()/'manifest.json')['hook_sha256']
    return m

def load_profile(original,root):
    inspect(root,require_data=False);p=profile_dir();mf=load(p/'manifest.json')
    return p,load(p/'model.json'),load(p/'priors.json'),mf

def freeze(root,out):
    ident=inspect(root,require_data=True);paths=source_map(root);files={n:sha(p) for n,p in paths.items()};out=Path(out)
    if out.exists():
        r=load(out/'receipt.json')
        if r['files']!=files or r['checkpoint']!=ident or sha(out/'D0_reference.zip')!=r['archive_sha256']:
            raise ValueError('existing frozen inputs differ; use new run and profile directory')
        return r
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(out/'D0_reference.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,p in paths.items():z.write(p,n)
    r={'schema':'workflow-port-freeze-1','checkpoint':ident,'files':files,
       'archive_sha256':sha(out/'D0_reference.zip'),'GPU_generation_requests':0,
       'original_source_modified':False,'historical_output_trace_frozen':False}
    write_new(out/'receipt.json',r);return r
