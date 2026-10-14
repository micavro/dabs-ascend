#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
python -B tools/run_typed.py plan
