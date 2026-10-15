#!/usr/bin/env bash
# Host entry for the complete 20261015 experiment, using four existing services.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DEVICE_IDS="${DEVICE_IDS:-0,1,2,3}"
MODE=benchmark
if (( $# > 0 )) && [[ "$1" != --* ]]; then MODE="$1"; shift; fi
case "$MODE" in benchmark|check|plan|test) ;; *) echo 'Use benchmark/check/plan/test' >&2; exit 1;; esac
for arg in "$@"; do
  case "$arg" in --suite|--suite=*) echo 'This entry uses the fixed R234 suite.' >&2; exit 1;; esac
done
exec bash "$ROOT/run_dabs_existing.sh" "$MODE" --suite r234 "$@"
