from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "source/spra_stage19_musique_joint_v1/dataset/compiled"


def source_paths():
    paths = [ROOT / "source" / name for name in (
        "spra_stage19_musique_joint_v1", "spra_stage19_typed_handoff_v1",
        "spra_workflow_sched_v2_1")]
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return paths


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    paths = []
    for directory in source_paths():
        paths.extend(p for p in directory.rglob("*.py") if "manonly" not in p.name.lower())
    paths.extend((ROOT / "experiments/ascend").rglob("*.py"))
    paths.extend((ROOT / 'source/spra_workflow_fullgrid_r234_sll_v1').rglob('*.py'))
    paths.extend([ROOT / "tools/run_ascend.py", ROOT / "run_ascend.sh",
                  ROOT / "tools/run_dabs_existing.py", ROOT / "tools/inspect_dabs_existing.py", ROOT / "run_dabs_existing.sh",
                  ROOT / "profiles/reference/configuration.json", ROOT / 'run_dabs_r234.sh',
                  ROOT / 'provenance/r234_import.json'])
    paths.extend(ROOT / 'tools' / name for name in ('inspect_dabs_repair.py', 'probe_dabs_container.py', 'report_dabs_r234.py', 'explain_dabs_r234.py'))
    paths.extend([ROOT / 'tools/native_dabs.py', ROOT / 'experiments/ascend/dabs_native/upstream.json',
                  ROOT / 'tools/run_native_workflow.py', ROOT / 'tools/native_setup.py', ROOT / 'run_native.sh'])
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def verify_original_assets():
    source_paths()
    from wfsched.legacy import inspect
    try:
        original = inspect(ROOT / "source", require_data=True)
    except ValueError as exc:
        original = {"matches_archived_checkpoint": False, "provenance_note": str(exc)}
    # Input data checks below protect the fixed experimental workload, not the
    # controller/source deployment identity.
    assets = read_json(ROOT / "provenance/input_assets.json")
    for name, expected in assets["files"].items():
        if sha(DATASET / name) != expected:
            raise ValueError("Original dataset changed: " + name)
    if sha(DATASET / "manifest.json") != assets["source_dataset_manifest_sha256"]:
        raise ValueError("Original dataset manifest changed")
    return {"source_check": original, "dataset_sha256": dict(assets["files"]),
            "manifest_sha256": assets["source_dataset_manifest_sha256"]}


def configuration(replicas, native_resources=None, client_max_inflight=2):
    cfg = read_json(ROOT / "profiles/reference/configuration.json")
    from dabs_native.timing_policy import configuration as timing_configuration
    cfg.update(timing_configuration())
    cfg.update(replicas=replicas, base_cap=4096, wf_router_seed=20261014,
               wf_public_method="DAG", apply_delay_ms=0.,
               phase_model_required=False, cache_scenario="no_unverified_reuse")
    if native_resources is not None:
        from dabs_native.contracts import validate_resources
        resources = validate_resources(native_resources)
        cfg.update(execution_scope='native', block_size=resources['block_size'],
                   base_cap=resources['step_tokens'], hard_total=resources['step_tokens'],
                   caps=[resources['step_tokens']], max_model_len=resources['context_tokens'],
                   max_num_seqs=resources['server_max_seqs'], client_max_inflight=client_max_inflight,
                   long_prefill_token_threshold=resources.get('long_prefill_token_threshold', 0),
                   effective_native_resources=resources, phase_model_required=True)
        cfg.update(native_cache_query_limit=8, native_cache_query_token_limit=8192)
    return cfg


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)
