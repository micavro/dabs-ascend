#!/usr/bin/env bash
# Two operator commands for the existing four Ascend services.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec python3 "$ROOT/tools/run_native_workflow.py" "$@"
