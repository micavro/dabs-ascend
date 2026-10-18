#!/usr/bin/env python3
"""Run from the repository root, on host for plan or in the CPU controller."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))

if __name__ == '__main__':
    from dabs_ascend.fast_one import main
    raise SystemExit(main())
