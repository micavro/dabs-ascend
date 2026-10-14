"""Read-only checkpoint verification and loading of the actually deployed engine."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
from .common import HERE, PROFILE, load, sha, digest, write_new

_MODULE = None

def reproduction_module():
    global _MODULE
    if _MODULE is None:
        path = HERE/'reference/workflow_only.py'
        spec = importlib.util.spec_from_file_location('_frozen_workflow_only_v1', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MODULE = mod
    return _MODULE

def inspect(root: Path, require_data=True):
    root = root.resolve()
    result = reproduction_module().inspect_sources(root, require_data=require_data)
    result['typed_dependency_identity_sha256'] = sha(root/'spra_stage19_typed_handoff_v1/dependency_identity.json')
    return result

def load_original(root):
    return reproduction_module().load_original(Path(root))

def load_profile(original, root):
    return original.prepared(SimpleNamespace(prepared=Path(root)/'spra_stage19_typed_handoff_v1/prepared'/PROFILE))

def source_map(root):
    root = Path(root)
    typed = root/'spra_stage19_typed_handoff_v1'
    source = root/'spra_stage19_musique_joint_v1'
    prof = typed/'prepared'/PROFILE
    mf = load(prof/'manifest.json')
    result = {}
    for name in mf['application_source_hashes']:
        kind, rel = name.split('/', 1)
        result[name] = (source if kind == 'source' else typed)/rel
    for name in ('manifest.json', *mf['files']): result['profile/'+name] = prof/name
    result['adapter/dependency_identity.json'] = typed/'dependency_identity.json'
    result['dataset/manifest.json'] = source/'dataset/compiled/manifest.json'
    result['reproduction/workflow_only.py'] = HERE/'reference/workflow_only.py'
    result['reproduction/spra_workflow_d0_repro_v1.zip'] = HERE/'reference/spra_workflow_d0_repro_v1.zip'
    return result

def freeze(root, out):
    """Archive code and model files, never weights or keys. No inference calls."""
    identity = inspect(Path(root), require_data=True)
    files = source_map(root)
    contents = {n: sha(p) for n, p in files.items()}
    out = Path(out)
    if out.exists():
        receipt = load(out/'receipt.json')
        if receipt['files'] != contents or receipt['checkpoint'] != identity:
            raise ValueError('frozen checkpoint differs; do not overwrite/edit expected hashes')
        if sha(out/'D0_reference.zip') != receipt['archive_sha256']:
            raise ValueError('frozen archive changed')
        return receipt
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(out/'D0_reference.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name, path in files.items(): z.write(path, name)
    receipt = {'schema':'workflow-D0-readonly-freeze-1', 'checkpoint':identity,
               'files':contents, 'archive_sha256':sha(out/'D0_reference.zip'),
               'GPU_generation_requests':0, 'original_source_modified':False,
               'historical_output_trace_frozen':False}
    write_new(out/'receipt.json', receipt)
    return receipt
