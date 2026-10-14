#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c SHA256SUMS
python -B -m unittest discover -s tests -v
python -B workflow_only.py inspect "$@"
