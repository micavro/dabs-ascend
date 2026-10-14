#!/usr/bin/env python3
"""Package this project for Linux without outputs, credentials or local caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/ascend"))
from dabs_ascend.common import sha, verify_original_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/DABS_Ascend_Linux_v1.zip")
    args = parser.parse_args()
    target = args.out.expanduser().resolve()
    if target.exists() or target.with_suffix(target.suffix + ".sha256").exists():
        parser.error("Output or checksum already exists; choose a new --out")
    identity = verify_original_assets()
    paths = [ROOT / name for name in ("README.md", "README_先读我.md", "README_existing.md", "run_ascend.sh", "run_dabs_existing.sh",
             "requirements-ascend.txt", "requirements-controller.txt", "requirements-offline.txt",
             ".gitignore", ".gitattributes")]
    for directory in ("source", "profiles", "provenance", "evidence", "docs", "tools", "runtime_local",
                      "experiments/ascend", "tests/ascend"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    files = {}
    for path in paths:
        if "manonly" in path.name.lower() or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.name.startswith(".env") or path.suffix in (".pem", ".key") or path.name in ("key.txt", "id_rsa", "id_ed25519"):
            continue
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Refuse file outside project: " + str(path))
        files[path.relative_to(ROOT).as_posix()] = path
    for name in ("run_ascend.sh", "run_dabs_existing.sh"):
        if b"\r" in (ROOT / name).read_bytes():
            raise ValueError(name + " must use Linux LF line endings")
    prefix = "DABS_Ascend_Linux_v1/"
    manifest = {"schema": "dabs-ascend-delivery-1", "files": {name: sha(path) for name, path in sorted(files.items())},
                "original_source_files_verified": identity["source_check"]["source_files_verified"],
                "local_test_command": "python tools/run_ascend.py test",
                "npu_validation_performed": False, "model_weights_included": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(prefix + name)
            info.create_system = 3
            info.external_attr = ((0o100755 if name.endswith(".sh") else 0o100644) << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        archive.writestr(prefix + "ASCEND_DELIVERY_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(target) as archive:
        damaged = archive.testzip()
        if damaged:
            raise ValueError("Archive CRC failure: " + damaged)
    with target.with_suffix(target.suffix + ".sha256").open("x", encoding="utf-8") as handle:
        handle.write(sha(target) + "  " + target.name + "\n")
    print(json.dumps({"archive": str(target), "files": len(files) + 1, "bytes": target.stat().st_size,
                      "sha256": sha(target), "npu_validation_performed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
