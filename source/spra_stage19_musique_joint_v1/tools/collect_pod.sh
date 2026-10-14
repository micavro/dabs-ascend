#!/usr/bin/env bash
# Read-only collection. Does not restart, patch, create controls or issue model requests.
set -uo pipefail
NS="${NAMESPACE:-spra}"
POD="${POD:-spra-vllm-0}"
BASE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$BASE/reports/BOOT_${POD}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "$OUT"
capture() {
  local file="$1"; shift
  "$@" >"$OUT/$file" 2>&1
  local rc=$?
  printf '%s exit=%s\n' "$file" "$rc" >>"$OUT/commands.log"
  return 0
}
capture pod.json kubectl --request-timeout=20s -n "$NS" get pod "$POD" -o json
capture describe.txt kubectl --request-timeout=20s -n "$NS" describe pod "$POD"
capture current.log kubectl --request-timeout=20s -n "$NS" logs "$POD" --all-containers=true --timestamps=true --tail=300
capture previous.log kubectl --request-timeout=20s -n "$NS" logs "$POD" --all-containers=true --previous=true --timestamps=true --tail=300
capture events.txt kubectl --request-timeout=20s -n "$NS" get events --field-selector "involvedObject.name=$POD" --sort-by=.metadata.creationTimestamp
# A process-less/crashing container can reject exec; this is recorded, not retried as inference.
kubectl --request-timeout=20s -n "$NS" exec -i "$POD" -- python3 - <<'PY' >"$OUT/runtime_files.txt" 2>&1
import hashlib,json,os
from pathlib import Path
root=Path('/spra-e1')
for rel in ('launcher.py','joint18v2_hook.py','joint18v2/bootstrap.py'):
    p=root/rel
    print(json.dumps({'file':str(p),'exists':p.is_file(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}))
pod=os.environ.get('POD_NAME','spra-vllm-0')
for p in sorted((root/'joint18v2').glob(pod+'.boot.*.json'))[-8:]:
    print('BOOT_RECORD',p);print(p.read_text())
p=root/'joint18v2'/(pod+'.status.json')
print('RUNTIME_STATUS_EXISTS',p.is_file())
if p.is_file():
    d=json.loads(p.read_text());print(json.dumps({k:d.get(k) for k in ('version','recovery_revision','launch_id','pid','mono_ns','fatal_engine_error','running','waiting')},default=str))
PY
printf 'runtime_files.txt exit=%s\n' "$?" >>"$OUT/commands.log"
python3 - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);summary={}
try:
    x=json.loads((out/'pod.json').read_text());summary['uid']=x.get('metadata',{}).get('uid');s=x.get('status',{});summary['phase']=s.get('phase');summary['conditions']=s.get('conditions',[])
    summary['containers']=[{k:c.get(k) for k in ('name','ready','restartCount','state','lastState')} for c in s.get('containerStatuses',[])]
except Exception as e:summary['pod_json_error']=repr(e)
logs='\n'.join((out/f).read_text(errors='replace') for f in ('current.log','previous.log'))
summary['signatures']={k:k in logs for k in ('bootstrapping phase','Address already in use','Engine process failed to start','ModuleNotFoundError','UnsupportedCache','CUDA out of memory','JOINT18V2_STARTUP')}
summary['interpretation']='signatures are clues, not an automatic root-cause or recovery decision'
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('BOOT_DIAGNOSTIC='+json.dumps(summary,ensure_ascii=False))
PY
printf 'BOOT_DIAGNOSTIC_DIR=%s\nREAD_ONLY=True POD_DELETIONS=0 MODEL_REQUESTS=0\n' "$OUT"
# Commands are best-effort, and their failures are preserved in commands.log.
