#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -B -u tools/run_typed.py profile "$@"
