from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import traceback
import unittest
import uuid

from . import REVISION
from .common import DATASET, ROOT, configuration, read_json, sha, source_identity, source_paths, verify_original_assets, write_json
from .runtime import MODEL_NAME, Servers, device_ids, device_locks, preflight


def parser(existing=False):
    p = argparse.ArgumentParser(description="DABS Linux Ascend functional port. Reuse the existing MFE container; no automatic installs.")
    p.add_argument("mode", nargs="?", choices=("smoke", "benchmark", "check", "test", "plan", 'probe'), default="smoke")
    p.add_argument("--model", type=Path, default=Path(os.environ.get("MODEL_PATH", "/x/Qwen2.5-1.5B-Instruct")))
    p.add_argument("--devices", default=os.environ.get("DEVICE_IDS", "0,1,2,3"))
    p.add_argument("--out", type=Path)
    p.add_argument("--base-port", type=int, default=int(os.environ.get("BASE_PORT", "18000")))
    p.add_argument("--memory", type=float, default=.75, help="vLLM memory fraction (same conservative MFE default)")
    p.add_argument("--startup-timeout", type=float, default=900.)
    p.add_argument("--condition-timeout", type=float, default=600.)
    p.add_argument("--request-timeout", type=float, default=180.)
    p.add_argument("--max-inflight", type=int, default=8, help="logical client limit per replica, not measured KV capacity")
    p.add_argument("--suite", choices=("smoke", "blocks", "case", "grid", 'r234', 'dabs-r234-balanced-64'), default=None)
    p.add_argument('--max-conditions', type=int, help='R234 only: clean pause after N additional conditions')
    p.add_argument('--resume', type=Path, help='R234 only: resume a cleanly paused output directory')
    p.add_argument('--execution-scope', choices=('native', 'http-limited'), default='native',
                   help='R234: native requires verified hooks; http-limited is diagnostic, not full mechanism reproduction')
    p.add_argument('--native-evidence', type=Path, help='native acceptance.json; default uses outputs/native-latest.json raw paths')
    p.add_argument('--exploratory-calibration', action='store_true', help='Exploratory run with measured prediction error; not formal acceptance')
    p.add_argument('--http-keepalive', type=int, default=None,
                   help='R234 idle HTTP pool size; 0 preserves the reported site workaround (not a performance claim)')
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--rate", type=float, default=2.)
    p.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=20261014)
    p.set_defaults(service_mode="managed")
    if existing:
        p.description = "DABS controller for existing Ascend HTTP services; no model process ownership."
        p.set_defaults(service_mode="existing", base_port=8001, model=Path("/models/qwen"),
                       devices=os.environ.get("DEVICE_IDS", "0,1,2,3,4,5,6,7"),
                       max_inflight=2, condition_timeout=1800., request_timeout=300.)
        p.add_argument("--deployment-manifest", type=Path)
        p.add_argument("--container-prefix", default="wushibo-qwen-small-npu")
    return p


def prepare_args(argv=None, existing=False):
    p = parser(existing=existing)
    args = p.parse_args(argv)
    try:
        args.devices = device_ids(args.devices)
    except ValueError as exc:
        p.error(str(exc))
    args.model = args.model.expanduser().resolve()
    args.suite = args.suite or ("smoke" if args.mode == "smoke" else "blocks")
    if args.suite == 'r234':
        args.suite = 'dabs-r234-balanced-64'
    if args.exploratory_calibration and (not existing or args.mode != 'benchmark' or
            args.suite != 'dabs-r234-balanced-64' or args.execution_scope != 'native' or
            args.resume is not None or args.native_evidence is None):
        p.error('exploratory-calibration requires existing native R234 benchmark, raw evidence and a new output')
    if args.suite == 'dabs-r234-balanced-64':
        args.http_keepalive = 0 if args.http_keepalive is None else args.http_keepalive
        if not 0 <= args.http_keepalive <= 256:
            p.error('http-keepalive must be 0..256')
        if len(args.devices) != 4:
            p.error('R234 needs exactly four selected services/devices; each condition uses the first R')
        supplied = list(sys.argv[1:] if argv is None else argv)
        if any(x.split('=')[0] in ('--rate', '--blocks', '--seed', '--repeats') for x in supplied):
            p.error('R234 uses the frozen 96-condition plan; rate/blocks/seed/repeats cannot be overridden')
        if args.max_conditions is not None and not 1 <= args.max_conditions <= 96:
            p.error('max-conditions must be 1..96')
        if args.resume is not None and (args.out is not None or args.mode != 'benchmark'):
            p.error('Use benchmark --resume PATH without --out')
    elif args.resume is not None or args.max_conditions is not None or args.http_keepalive is not None or args.execution_scope != 'native':
        p.error('resume/max-conditions/http-keepalive/execution-scope are only available for suite r234')
    if args.mode == "smoke" and args.suite != "smoke":
        p.error("Use benchmark --suite ... for larger suites")
    if args.mode == 'probe' and (args.suite != 'dabs-r234-balanced-64' or not existing or args.resume or args.max_conditions):
        p.error('probe requires existing services, suite r234, without resume/max-conditions')
    args.repeats = args.repeats if args.repeats is not None else (8 if args.suite == "case" else 1 if args.suite == "smoke" else 2)
    import math
    if args.repeats < 1 or not math.isfinite(args.rate) or args.rate < .5:
        p.error("repeats >=1 and rate >=0.5 required")
    if any(x not in (1, 2, 3, 4) for x in args.blocks) or len(set(args.blocks)) != len(args.blocks):
        p.error("blocks must be distinct members of 1,2,3,4")
    if not .05 <= args.memory <= .95 or not 1 <= args.max_inflight <= 64:
        p.error("memory must be 0.05..0.95 and max-inflight 1..64")
    port_span = max(args.devices) + 1 if existing else len(args.devices)
    if not 1024 <= args.base_port <= 65536 - port_span:
        p.error("Selected port range is invalid")
    if existing:
        from .existing import container_name
        try:
            container_name(args.container_prefix, args.devices[0])
        except ValueError as exc:
            p.error(str(exc))
        if args.mode not in ("test", "plan") and args.deployment_manifest is None:
            p.error("--deployment-manifest is required for existing-service inference/check")
        if args.deployment_manifest is not None:
            args.deployment_manifest = args.deployment_manifest.expanduser().resolve()
    if any(not math.isfinite(x) or x <= 0 for x in (args.startup_timeout, args.condition_timeout, args.request_timeout)):
        p.error("Timeouts must be finite positive seconds")
    if args.suite == "case" and (len(args.devices) != 4 or args.rate != 2.):
        p.error("B3 case requires four replicas and rate=2")
    return args


def make_plan(args):
    if args.suite == 'dabs-r234-balanced-64':
        from .r234 import plan
        return plan()
    from .scheduler import METHODS
    conditions = []
    replica_counts = list(range(1, min(4, len(args.devices)) + 1)) if args.suite == "grid" else [len(args.devices)]
    rates = [1., 2., 4., 8.] if args.suite == "grid" else [args.rate]
    blocks = [3] if args.suite == "case" else [0] if args.suite == "smoke" else args.blocks
    for replicas in replica_counts:
        for rate in rates:
            for repetition in range(1, args.repeats + 1):
                for block in blocks:
                    seed = args.seed + repetition * 100 + block
                    methods = list(METHODS)
                    random.Random(seed).shuffle(methods)
                    for method in methods:
                        conditions.append({"label": f"R{replicas}_L{rate:g}_T{repetition}_B{block}_{method}",
                                           "method": method, "replicas": replicas, "rate": rate,
                                           "repeat": repetition, "block": block, "seed": seed,
                                           "split": "smoke" if block == 0 else "pilot",
                                           "slice": [0, 2] if block == 0 else [(block - 1) * 16, block * 16]})
    return conditions


async def execute_suite(args, out, plan, summaries, initial_identity):
    import httpx
    from .backend import Backend, Tokenizer
    from .calibration import calibrate
    from .runner import run_condition
    from .scheduler import planner_ready
    from semantic19.dataset import load_public
    from typed19.workflow import compile_packets

    if args.service_mode == "existing":
        from .existing import ExistingServers
        servers = ExistingServers(args, out)
    else:
        servers = Servers(args, out)
    pool = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=10),
                                 trust_env=False, limits=httpx.Limits(max_connections=256, max_keepalive_connections=128)) as client:
        try:
            await servers.start(client)
            backend = Backend(client, servers.endpoints)
            write_json(out / "api_contract.json", await backend.inspect())
            write_json(out / "canary_summary.json", await backend.canaries(out))
            tok = Tokenizer(client, servers.endpoints[0])
            model = await calibrate(backend, tok, out)
            cfg = configuration(len(args.devices))
            cfg.update(model=MODEL_NAME, max_num_seqs=args.max_inflight,
                       condition_wall_s=args.condition_timeout, wf_router_seed=args.seed)
            prepared = {}
            for split in sorted({s["split"] for s in plan}):
                rows, manifest = load_public(DATASET, split)
                packets, info = await compile_packets(rows, tok, cfg)
                prepared[split] = packets
                write_json(out / (split + "_preparation.json"), {"info": info, "dataset_manifest": manifest,
                                                               "ids": [row["id"] for row in rows]})
            pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
            await asyncio.get_running_loop().run_in_executor(pool, planner_ready)
            for index, spec in enumerate(plan, 1):
                if source_identity() != initial_identity:
                    write_json(out / "source_provenance_changes.json", source_identity())
                    initial_identity = source_identity()
                verify_original_assets()
                servers.check_alive()
                current_cfg = {**cfg, "replicas": spec["replicas"], "arrival_rate": spec["rate"]}
                lo, hi = spec["slice"]
                current_cfg["workflow_count"] = hi - lo
                packets = prepared[spec["split"]][lo:hi]
                if len(packets) != hi - lo:
                    raise ValueError("Declared dataset slice is incomplete")
                print(f"ASCEND_CONDITION {index}/{len(plan)} {spec['label']}", flush=True)
                summary = await run_condition(backend, tok, packets, model, current_cfg, spec,
                                              out / "conditions" / spec["label"], pool, servers.check_alive)
                summaries.append(summary)
                write_json(out / "index.json", summaries)
                print("ASCEND_CONDITION_DONE=" + json.dumps({"label": spec["label"], "jct_mean_ms": summary["jct_mean_ms"],
                                                            "functional_pass": summary["functional_pass"]}), flush=True)
                if not summary["functional_pass"]:
                    raise RuntimeError("Answer/dependency contract failed; full results retained; no automatic retry")
            servers.check_alive()
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            if args.service_mode == "existing":
                await servers.aclose()
            else:
                servers.close()


def main(argv=None, *, existing=False):
    args = prepare_args(argv, existing=existing)
    source_paths()
    if args.mode == "test":
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests/ascend"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.suite == 'dabs-r234-balanced-64':
        if args.mode == 'probe':
            from .r234_probe import main as probe_main
            return probe_main(args)
        from .r234_run import main as r234_main
        return r234_main(args)
    plan = make_plan(args)
    if args.mode == "plan":
        print(json.dumps({"suite": args.suite, "conditions": len(plan), "plan": plan,
                          "real_inference": False, "strict_gpu39_reproduction": False}, indent=2))
        return 0
    out = args.out or ROOT / "outputs" / ("ascend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    out = out.expanduser().resolve()
    if out.exists():
        print(f"ERROR: output already exists; preserve it and choose a new --out: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=False)
    print("ASCEND_OUTPUT=" + str(out), flush=True)
    summaries, status, error = [], "failed", None
    identity = source_identity()
    write_json(out / "protocol.json", {"revision": REVISION, "mode": args.mode, "suite": args.suite,
                                      "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                      "plan": plan, "source_sha256": identity,
                                      "dataset_manifest_sha256": sha(DATASET / "manifest.json"),
                                      "strict_gpu39_reproduction": False})
    write_json(out / "index.json", summaries)
    write_json(out / "status.json", {"status": "preflight"})
    prior_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        write_json(out / "original_integrity.json", verify_original_assets())
        if args.service_mode == "existing":
            from .existing import preflight as existing_preflight
            existing_preflight(args, out)
        else:
            preflight(args, out)
        if args.mode == "check":
            write_json(out / "status.json", {"status": "environment_checked", "real_inference": False})
            print("ASCEND_ENVIRONMENT_CHECK_PASS (no model generation)", flush=True)
            return 0
        lock_dir = ROOT / "outputs/.existing_controller_locks" if existing else None
        with device_locks(args.devices, directory=lock_dir):
            write_json(out / "status.json", {"status": "running"})
            asyncio.run(execute_suite(args, out, plan, summaries, identity))
        status = "complete"
    except BaseException as exc:
        error = repr(exc)
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("ASCEND_FAILED=" + error, file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, prior_sigterm)
        if args.mode != "check" or error:
            from .report import make_report
            report = make_report(out, plan, summaries, status, error)
            write_json(out / "status.json", {"status": status, "error": error,
                                             "functional_test_pass": report["functional_test_pass"]})
    if args.mode == "check":
        return 1
    print("ASCEND_REPORT=" + str(out / "TEST_REPORT.md"), flush=True)
    return 0 if report["functional_test_pass"] else 1
