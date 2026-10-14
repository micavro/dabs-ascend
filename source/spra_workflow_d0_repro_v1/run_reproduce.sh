#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
LOG=$(mktemp "$PWD/logs/reproduce_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX.log")
printf 'LOG=%s\n' "$LOG"
PYTHONDONTWRITEBYTECODE=1 python -B workflow_only.py reproduce "$@" 2>&1 | tee "$LOG"
