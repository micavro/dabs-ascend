#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -B -u tools/run_semantic.py profile "$@"
