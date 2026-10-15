#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export WF_SOURCE_ROOT="${WF_SOURCE_ROOT:-$(cd .. && pwd)}"
export PYTHONPATH="$WF_SOURCE_ROOT/spra_stage19_musique_joint_v1:$WF_SOURCE_ROOT/spra_stage19_typed_handoff_v1${PYTHONPATH:+:$PYTHONPATH}"
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python -B run.py check --repo "$WF_SOURCE_ROOT"
