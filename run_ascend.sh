#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" "$ROOT/tools/run_ascend.py" "$@"
