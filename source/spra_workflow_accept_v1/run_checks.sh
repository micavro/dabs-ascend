#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Local unit tests use only the retained source, no serving/GPU requests.
export WF_SOURCE_ROOT="${WF_SOURCE_ROOT:-$(cd .. && pwd)}"
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python -B run.py check --repo "$WF_SOURCE_ROOT"
