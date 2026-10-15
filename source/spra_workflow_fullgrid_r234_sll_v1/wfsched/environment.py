"""Read-only client/host inventory; records failures and does not import CUDA."""
from __future__ import annotations
from datetime import datetime, timezone
import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from .common import write_new, sha

def capture(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,errors='replace',timeout=15,check=False)
        return {'argv':argv,'returncode':p.returncode,'stdout':p.stdout[:100000],'stderr':p.stderr[:10000]}
    except (OSError,subprocess.TimeoutExpired) as e:return {'argv':argv,'unavailable':True,'error':str(e)}

def collect(app: Path) -> Path:
    versions={}
    for name in ('torch','vllm','transformers','httpx','numpy','ray','lm-format-enforcer'):
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    root=app/'inventory';root.mkdir(exist_ok=True)
    out=root/('host_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]+'.json')
    value={'schema':'workflow-readonly-host-inventory-1','collected_at_utc':datetime.now(timezone.utc).isoformat(),
      'host':platform.node(),'os':platform.platform(),'python':sys.version,'python_executable':sys.executable,
      'client_packages':versions,'environment':{k:os.environ.get(k) for k in ('CONDA_PREFIX','CONDA_DEFAULT_ENV','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')},
      'cpu':capture(['lscpu']), 'memory':capture(['free','-b']),
      'gpu':capture(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used','--format=csv,noheader,nounits']),
      'gpu_processes':capture(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']),
      'nvcc':capture(['nvcc','--version']),
      'limits':['This is collection-time state, not reconstructed historical environment.',
                'Client environment packages do not establish model-server/container versions.',
                'Combine with per-run environment_contract.json and session environment.json for active service mapping.',
                'Replica index is not assumed to equal physical GPU index; no exclusivity established.'],
      'generation_requests':0,'sources_modified':False}
    write_new(out,value)
    out.with_suffix('.json.sha256').write_text(sha(out)+'  '+out.name+'\n')
    return out
