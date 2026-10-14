#!/usr/bin/env bash
# Explicit operation: run only after applying the recovery patch and stopping experiments.
set -euo pipefail
: "${TARGETS:?Set TARGETS explicitly; use TARGETS=0 first}"
NS="${NAMESPACE:-spra}"
DIR="$(cd "$(dirname "$0")" && pwd)"
for i in $TARGETS; do
  case "$i" in 0|1|2|3) ;; *) echo 'Invalid replica index' >&2;exit 2;; esac
  POD="spra-vllm-$i"
  OLD="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.metadata.uid}')"
  CHECK="${PRECHECK_POD:-$POD}"
  if [ "$CHECK" != "$POD" ];then
    # An explicit healthy sibling can inspect the shared source when pod0 cannot
    # accept exec. Require same node and running image digest, not just a name.
    TMP="$(mktemp -d)"
    kubectl -n "$NS" get pod "$POD" -o json >"$TMP/target.json"
    kubectl -n "$NS" get pod "$CHECK" -o json >"$TMP/check.json"
    python3 - "$TMP" <<'PY2'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);a=json.loads((p/'target.json').read_text());b=json.loads((p/'check.json').read_text())
assert a['spec']['nodeName']==b['spec']['nodeName'], 'precheck sibling must be on the same node'
def image(d):
    cs=d.get('status',{}).get('containerStatuses',[])
    assert len(cs)==1 and cs[0].get('imageID'), 'exact single-container image digest required'
    return cs[0]['imageID']
assert image(a)==image(b), 'precheck sibling has a different image digest'
print('SIBLING_PREFLIGHT='+json.dumps({'target':a['metadata']['name'],'checked_on':b['metadata']['name'],'same_node':True,'same_image_digest':True}))
PY2
    rm -rf "$TMP"
  fi
  # Do not delete again on a precheck failure; collect logs and stop.
  if ! kubectl -n "$NS" exec "$CHECK" -- env PYTHONDONTWRITEBYTECODE=1 python3 -B /spra-e1/joint18v2_hook.py --precheck; then
    NAMESPACE="$NS" POD="$POD" bash "$DIR/collect_pod.sh" || true
    echo 'STOP: precheck failed or container not executable; no deletion performed.' >&2
    exit 1
  fi
  kubectl -n "$NS" delete pod "$POD" --wait=true
  NEW=''
  for attempt in $(seq 1 120);do
    NEW="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.metadata.uid}' 2>/dev/null || true)"
    if [ -n "$NEW" ] && [ "$NEW" != "$OLD" ];then break;fi
    sleep 1
  done
  if [ -z "$NEW" ] || [ "$NEW" = "$OLD" ];then
    NAMESPACE="$NS" POD="$POD" bash "$DIR/collect_pod.sh" || true
    echo "STOP: no replacement UID for $POD; do not repeat delete" >&2;exit 1
  fi
  if ! kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=300s;then
    NAMESPACE="$NS" POD="$POD" bash "$DIR/collect_pod.sh" || true
    echo "STOP: $POD not Ready; diagnostics captured; no canary was started" >&2;exit 1
  fi
  # Verify that Ready is also accompanied by live, patched, Scheduler-owned status.
  if ! kubectl -n "$NS" exec -i "$POD" -- python3 - <<'PY'
import hashlib,json,os,time
from pathlib import Path
p=Path('/spra-e1/joint18v2')/(os.environ['POD_NAME']+'.status.json')
d=json.loads(p.read_text())
assert d.get('recovery_revision')=='stage18v2-boot-metrics-fix-1', d.get('recovery_revision')
assert d.get('hook_source_sha256')==hashlib.sha256(Path('/spra-e1/joint18v2_hook.py').read_bytes()).hexdigest(), 'old hook process'
assert 0<=time.monotonic_ns()-d['mono_ns']<3_000_000_000, 'status heartbeat stale'
assert (Path('/proc')/str(d['pid'])).exists(), 'status PID not live'
assert not d.get('fatal_engine_error'), d.get('fatal_engine_error')
assert d.get('launch_id'), 'missing guarded bootstrap identity'
print('PATCHED_RUNTIME_READY='+json.dumps({k:d[k] for k in ('pod','pid','launch_id','recovery_revision','hook_source_sha256')}))
PY
  then
    NAMESPACE="$NS" POD="$POD" bash "$DIR/collect_pod.sh" || true
    echo 'STOP: Kubernetes Ready without matching patched runtime; no canary started' >&2;exit 1
  fi
  echo "RESTART_OK pod=$POD old_uid=$OLD new_uid=$NEW"
done
