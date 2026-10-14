#!/usr/bin/env bash
set -euo pipefail
: "${TARGETS:?Set TARGETS explicitly; first use TARGETS=0}"
NAMESPACE="${NAMESPACE:-spra}"
for i in $TARGETS; do
  case "$i" in 0|1|2|3) ;; *) echo 'Invalid replica index' >&2; exit 2;; esac
  p="spra-vllm-$i"
  old="$(kubectl -n "$NAMESPACE" get pod "$p" -o jsonpath='{.metadata.uid}')"
  kubectl -n "$NAMESPACE" exec "$p" -- env PYTHONDONTWRITEBYTECODE=1 python3 -B /spra-e1/joint18v2_hook.py --precheck
  kubectl -n "$NAMESPACE" delete pod "$p" --wait=true
  new=""
  for attempt in $(seq 1 120); do
    new="$(kubectl -n "$NAMESPACE" get pod "$p" -o jsonpath='{.metadata.uid}' 2>/dev/null || true)"
    if [ -n "$new" ] && [ "$new" != "$old" ]; then break; fi
    sleep 1
  done
  if [ -z "$new" ] || [ "$new" = "$old" ]; then echo "STOP: replacement not seen for $p" >&2; exit 1; fi
  kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$p" --timeout=300s
  printf 'RESTART_OK pod=%s old_uid=%s new_uid=%s\n' "$p" "$old" "$new"
done
