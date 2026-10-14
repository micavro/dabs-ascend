#!/usr/bin/env python3
"""Linux Ascend entry; test/help also work without an accelerator."""
from pathlib import Path
import os
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ascend"))

if __name__ == "__main__":
    # Frozen source uses locale-default reads. Enable UTF-8 without editing it,
    # including spawn children and Windows CPU-only validation.
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if not sys.flags.utf8_mode:
        raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]]))
    from dabs_ascend.cli import main
    raise SystemExit(main())
