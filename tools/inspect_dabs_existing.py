#!/usr/bin/env python3
"""Host-only Docker snapshot. No model loading, container mutation or Env logging."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/ascend"))
from dabs_ascend.existing import container_name, deployment_records
from dabs_ascend.runtime import device_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--container-prefix", default="wushibo-qwen-small-npu")
    args = parser.parse_args()
    names = [container_name(args.container_prefix, d) for d in device_ids(args.devices)]
    rows = json.loads(subprocess.check_output(["sudo", "docker", "inspect", *names], text=True))
    redacted = deployment_records(rows)
    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"deployment_existing_{stamp}_{uuid.uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(redacted, handle, ensure_ascii=False, indent=2)
    print(path)


if __name__ == "__main__":
    main()
