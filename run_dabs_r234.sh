#!/usr/bin/env bash
# Host entry for the complete 20261015 experiment, using four existing services.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DEVICE_IDS="${DEVICE_IDS:-0,1,2,3}"
MODE=benchmark
if (( $# > 0 )) && [[ "$1" != --* ]]; then MODE="$1"; shift; fi
case "$MODE" in
  native-*) exec python3 "$ROOT/tools/native_dabs.py" "$MODE" "$@" ;;
  prepare)
    (( $# == 0 )) || { echo 'prepare takes environment settings only; use probe for extra options.' >&2; exit 1; }
    python3 "$ROOT/tools/inspect_dabs_repair.py" --wait-ready "${READY_TIMEOUT:-300}"
    exec bash "$ROOT/run_dabs_existing.sh" probe --suite r234
    ;;
  inspect) exec python3 "$ROOT/tools/inspect_dabs_repair.py" "$@" ;;
  report) exec python3 "$ROOT/tools/report_dabs_r234.py" "$@" ;;
  explain) exec python3 "$ROOT/tools/explain_dabs_r234.py" "$@" ;;
  diagnose)
    MODE=benchmark
    PAUSE=(--max-conditions 8)
    for arg in "$@"; do
      case "$arg" in --max-conditions|--max-conditions=*|--resume|--resume=*) PAUSE=();; esac
    done
    set -- --execution-scope http-limited "${PAUSE[@]}" "$@"
    ;;
  probe|benchmark|check|plan|test) ;;
  *) echo 'Use prepare/inspect/probe/explain/report/diagnose/benchmark/check/plan/test' >&2; exit 1;;
esac
for arg in "$@"; do
  case "$arg" in --suite|--suite=*) echo 'This entry uses the fixed R234 suite.' >&2; exit 1;; esac
done
exec bash "$ROOT/run_dabs_existing.sh" "$MODE" --suite r234 "$@"
