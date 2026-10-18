"""One-scenario entry point. Reuses LIVE evidence; never installs or calibrates."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import multiprocessing
import math
import traceback
import os
from pathlib import Path
import sys
import uuid

from .fast_io import write_json
from .fast_policy import REVISION, warm_worker

METHODS = ('RR', 'SLL', 'Ayo', 'Ours')
DEFAULT_METHODS = ','.join(METHODS)
EXPERIMENT_REVISION = 'ascend-fast-one-r4l8-four-methods-v2'


def configure(cfg):
    return {**cfg, 'epoch_s': .2, 'continuation_period_ms': 200.,
            'fast_poll_s': .2, 'ready_coalesce_s': .003,
            'fast_plan_budget_s': .08, 'fast_snapshot_age_s': .5,
            'fast_commit_reserve_s': .1, 'fast_observe_timeout_s': .5,
            'fast_harvest_s': 5., 'fast_starvation_s': 5.,
            'selector_revision': REVISION, 'native_batch_rollout': False,
            'fast_single_scenario': True}


def plan(methods=METHODS):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('methods must be a unique ordered subset of ' + DEFAULT_METHODS)
    import importlib
    import importlib.util
    root = Path(__file__).resolve().parents[3]
    folder = root / 'source/spra_workflow_fullgrid_r234_sll_v1/wfsched'
    alias = '_dabs_fast_plan_frozen'
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    design = importlib.import_module(alias + '.design')
    original = design.plan('dabs-r234-balanced-64')['conditions']
    source = {s['method']: s for s in original if (s['replicas'], s['rate'], s['repeat']) == (4, 8, 1)}
    rows = []
    for position, method in enumerate(methods, 1):
        row = dict(source[method])
        row['archived_gpu_base_cap'] = row.pop('base_cap')
        row['effective_engine_step_budget'] = 'read_from_live_capabilities_not_archived_GPU_value'
        row.update(method=method, position=position, label=f'R4_L8_T1_B0_{method}_Fast',
                   fast_adaptation=True, strict_v5_reproduction=False)
        rows.append(row)
    return rows


def resolve_evidence(value):
    """Relocate paths ONLY within the explicitly supplied evidence directory.

    Support old /project/... paths after a read-only /evidence mount. Never
    silently pick the newest unrelated model or follow stale absolute paths.
    """
    root = Path(value).expanduser().resolve()
    if root.is_file():
        manifest = root
        root = root.parent
    else:
        manifest = root / 'evidence.json'
    if not root.is_dir():
        raise ValueError('evidence directory not found: ' + str(root))
    paths = {}
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf-8'))
        for key, value in record.get('paths', {}).items():
            if key not in ('native-probe-one', 'native-probe-four', 'native-calibrate'):
                continue
            raw = Path(value)
            candidates = [root / raw.name]
            if not raw.is_absolute():
                candidates.insert(0, root / raw)
            if raw.is_absolute() and raw.is_relative_to(root):
                candidates.insert(0, raw)
            matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir() and p.resolve().is_relative_to(root)))
            if len(matches) == 1:
                paths[key] = str(matches[0])
    # Old workflow evidence manifests can use relative numbered subdirectories.
    # Select by raw role record, not by PASS strings or modification time.
    for role, replicas in (('native-probe-one', 1), ('native-probe-four', 4)):
        if role not in paths:
            matches = []
            for p in root.rglob('probe.json'):
                try:
                    if json.loads(p.read_text())['replicas'] == replicas and (p.parent / 'native_events.jsonl').is_file():
                        matches.append(p.parent.resolve())
                except (KeyError, ValueError):
                    continue
            if len(set(matches)) != 1:
                raise ValueError(f'{role}: require exactly one raw probe directory inside {root}')
            paths[role] = str(matches[0])
    if 'native-calibrate' not in paths:
        matches = [p.parent.resolve() for p in root.rglob('native_profile.json')
                   if all((p.parent / name).is_file() for name in ('native_events.jsonl', 'calibration_samples.json', 'calibration_plan.json', 'capabilities.json'))]
        if len(set(matches)) != 1:
            raise ValueError('require exactly one complete raw calibration directory inside --input')
        paths['native-calibrate'] = str(matches[0])
    for role, directory in paths.items():
        if not Path(directory).is_relative_to(root):
            raise ValueError('evidence role escapes supplied directory: ' + role)
    return paths


def baseline(state, cfg, method):
    from .native_algorithms import baseline as original
    # Preserve original Ayo/RR/SLL routing rules; only the execution loop is new.
    if method not in METHODS:
        raise ValueError('unknown fast-one method: ' + str(method))
    return original(state, cfg, method)


def make_report(out, summaries):
    columns = ['method', 'jct_mean_ms', 'jct_p95_ms', 'workflows_s', 'output_tokens_s',
               'completed_requests', 'output_tokens', 'functional_pass',
               'selected', 'confirmed', 'fast_complete_comparisons']
    with (out / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for s in summaries:
            d = s['diagnostics']
            writer.writerow({'method': s['spec']['method'], **{k: s[k] for k in columns[1:8]},
                             'selected': d.get('nonreference_selected', 0),
                             'confirmed': d.get('nonreference_execution_confirmed', 0),
                             'fast_complete_comparisons': d.get('fast_complete_comparisons', 0)})
    rows = {s['spec']['method']: s for s in summaries}
    comparisons = []
    if 'Ours' in rows:
        for method, s in rows.items():
            if method == 'Ours':
                continue
            comparisons.append({'baseline': method, 'method': 'Ours',
                'jct_mean_improvement_pct': 100 * (s['jct_mean_ms'] - rows['Ours']['jct_mean_ms']) / s['jct_mean_ms'],
                'jct_p95_improvement_pct': 100 * (s['jct_p95_ms'] - rows['Ours']['jct_p95_ms']) / s['jct_p95_ms'],
                'throughput_improvement_pct': 100 * (rows['Ours']['workflows_s'] - s['workflows_s']) / s['workflows_s']})
    write_json(out / 'comparisons.json', comparisons)
    lines = ['Ascend fast-one: R=4, lambda=8, T1 only',
             'Experiment revision: ' + EXPERIMENT_REVISION,
             'Completed methods: ' + ','.join(rows),
             'Ours reference is internal; not an extra baseline experiment.',
             'Completed workload is NOT evidence of a performance win.',
             'No native V5 batch rollout; request-event approximation; admission/route only.',
             'Not directly comparable with historical 200/1500ms runs. One repeat is not a significance test.', '']
    for s in summaries:
        lines.append(f"{s['spec']['method']}: mean_JCT={s['jct_mean_ms']/1000:.3f}s, "
                     f"p95_JCT={s['jct_p95_ms']/1000:.3f}s, requests={s['completed_requests']}, "
                     f"selected={s['diagnostics'].get('nonreference_selected',0)}, "
                     f"confirmed={s['diagnostics'].get('nonreference_execution_confirmed',0)}")
        lines.append('timings_ms=' + json.dumps(s['timings_ms'], ensure_ascii=False))
        lines.append('diagnostics=' + json.dumps(s['diagnostics'], ensure_ascii=False, sort_keys=True))
    for row in comparisons:
        lines.append('paired=' + json.dumps(row, ensure_ascii=False))
    (out / 'test_result.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


async def execute(args, out, specs):
    import httpx
    from .common import configuration, source_paths, DATASET, verify_original_assets
    source_paths()
    from . import r234
    from .backend import Backend, Tokenizer
    from .native_client import NativeClient
    from .native_gate import verify_live
    from .native_algorithms import adapted
    from .fast_condition import run_condition
    from semantic19.dataset import load_public
    from typed19.workflow import compile_packets, instantiate

    paths = resolve_evidence(args.input)
    write_json(out / 'evidence_paths.json', paths)
    # Read-only data identity validation. No service calibration/probe traffic.
    verify_original_assets()
    endpoints = [f'http://127.0.0.1:{args.base_port + i}' for i in range(4)]
    summaries, pool = [], None
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=10), trust_env=False,
                                 limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)) as client:
        backend = Backend(client, endpoints)
        write_json(out / 'api_contract.json', await backend.inspect(fixed=True))
        backend.native = NativeClient(client, endpoints, out / 'preflight.jsonl')
        try:
            gate = await verify_live(backend.native, paths)
            write_json(out / 'native_gate.json', gate)
            if not gate['pass']:
                raise ValueError('Existing evidence no longer valid: ' + str(gate['missing']) +
                                 '. No automatic install, restart or recalibration was attempted.')
            model = json.loads((Path(paths['native-calibrate']) / 'native_profile.json').read_text())
            write_json(out / 'frozen_profile.json', model)
            caps = backend.native.capabilities
            cfg = configure(configuration(4, caps[0]['resources'], 2))
            cfg.update(condition_wall_s=args.condition_timeout, arrival_rate=8., workflow_count=64,
                       input_profile='fixed_3584_balanced_DAG', output_contract='fixed_by_graph_shape',
                       guided_decoding=None, parent_binding='dependency_order_only',
                       mode='nonsemantic_fixed_token_replay')
            write_json(out / 'effective_config.json', cfg)
            write_json(out / 'native_capabilities.json', caps)
            await backend.native.drain(4)
            while await backend.native.harvest(4, wait_complete=True):
                pass
            tok = Tokenizer(client, endpoints[0])
            rows, dataset_manifest = load_public(DATASET, 'pilot')
            packets, info = await compile_packets(rows, tok, cfg)
            packets, workload = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
            write_json(out / 'workload.json', workload)
            write_json(out / 'preparation.json', {'info': info, 'dataset': dataset_manifest})
            # Warm imports before arrivals, on parent and persistent worker.
            for name in ('routing', 'ayo_topology', 'semantic_selector'):
                adapted(name)
            pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn'))
            await asyncio.get_running_loop().run_in_executor(pool, warm_worker)
            for i, spec in enumerate(specs, 1):
                # Every call validates engine identity; do not refit/reprobe per method.
                await backend.native.inspect()
                await backend.native.drain(4)
                current = {**cfg, 'wf_router_seed': spec['router_seed'], 'wf_public_method': spec['method'],
                           'wf_routing_policy': spec['method']}
                nodes, _, proof = await instantiate(packets, tok, {**current, 'block_size': 16}, spec['seed'], output_prior=None)
                proof.update(await r234.convert(nodes, tok))
                print(f"FAST_CONDITION {i}/{len(specs)} {spec['label']}", flush=True)
                s = await run_condition(backend, nodes, proof, model, current, spec,
                                        out / 'conditions' / spec['label'], pool, baseline)
                if not r234.audit_work(nodes) or len(nodes) != 216 or len({n['wid'] for n in nodes.values()}) != 64:
                    raise ValueError('Original fixed-work audit failed')
                for old in summaries:
                    if old['external_work_fingerprint'] != s['external_work_fingerprint'] or old['arrival_offsets'] != s['arrival_offsets']:
                        raise ValueError('Method pairing differs: workload or scheduled arrivals')
                summaries.append(s)
                write_json(out / 'index.json', summaries)
                make_report(out, summaries)
                print(f"FAST_DONE {i}/{len(specs)} mean_JCT_ms={s['jct_mean_ms']:.3f}", flush=True)
            write_json(out / 'status.json', {'status': 'completed', 'conditions': len(summaries),
                       'functional_pass': True, 'performance_pass': None, 'formal_acceptance': False,
                       'scope': 'one-scenario engineering screen only'})
            return summaries
        finally:
            backend.native.flush_evidence(durable=True)
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)


def show_logs(directory=None):
    """Read one run summary, including failure state; never issue service calls."""
    root = Path(__file__).resolve().parents[3]
    if directory is None:
        choices = sorted(p for p in (root / 'outputs').glob('native_fast-one_*') if p.is_dir())
        if not choices:
            print('No fast-one results in this checkout; run plan first.', file=sys.stderr)
            return 2
        directory = choices[-1]
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        print('Result directory not found: ' + str(directory), file=sys.stderr)
        return 2
    print('FAST_RESULT_DIRECTORY=' + str(directory))
    shown = False
    for name in ('status.json', 'test_result.txt', 'error_traceback.txt'):
        file = directory / name
        if file.is_file():
            print('\n=== ' + name + ' ===')
            print(file.read_text(encoding='utf-8'), end='\n')
            shown = True
    if not shown:
        print('No summary yet. Inspect the preserved console log in outputs/.')
        return 2
    return 0


def parser():
    p = argparse.ArgumentParser(description='R4/L8/T1 single scenario; no automatic calibration or grid')
    p.add_argument('command', choices=('plan', 'run', 'logs'))
    p.add_argument('--input', type=Path, help='current service-instance native_test evidence directory')
    p.add_argument('--methods', default=DEFAULT_METHODS, help='ordered subset of RR,SLL,Ayo,Ours; default runs these four once')
    p.add_argument('--base-port', type=int, default=8001)
    p.add_argument('--condition-timeout', type=float, default=1800.)
    p.add_argument('--request-timeout', type=float, default=300.)
    p.add_argument('--out', type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'logs':
        return show_logs(args.out)
    methods = args.methods.split(',')
    if len(set(methods)) != len(methods) or not methods or any(m not in METHODS for m in methods):
        raise SystemExit('--methods must be a unique ordered subset of ' + DEFAULT_METHODS)
    if args.base_port < 1 or args.base_port + 3 > 65535 or any(not math.isfinite(v) or v <= 0 for v in (args.condition_timeout, args.request_timeout)):
        raise SystemExit('Invalid port or timeout')
    specs = plan(methods)
    if args.command == 'plan':
        print(json.dumps({'experiment_revision': EXPERIMENT_REVISION, 'scenario': 'R4_L8_T1',
                          'methods': methods, 'condition_count': len(specs), 'conditions': specs, 'automatic_grid': False,
                          'generation_requests': 216 * len(specs), 'workflows': 64 * len(specs),
                          'output_tokens': 13184 * len(specs), 'new_calibration': False}, ensure_ascii=False, indent=2))
        return 0
    if args.input is None:
        raise SystemExit('run requires --input <current native_test evidence directory>; no auto model selection')
    from .common import ROOT
    from .runtime import device_locks
    out = args.out or ROOT / 'outputs' / ('native_fast-one_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'plan.json', specs)
    write_json(out / 'status.json', {'status': 'preflight'})
    write_json(out / 'protocol.json', {'revision': REVISION, 'base_commit': '7b672c0496d587c4884ec7823f459898fe1f0ef0',
        'experiment_revision': EXPERIMENT_REVISION, 'methods': methods,
        'generation_conditions': len(specs), 'scenario': 'R4_L8_T1', 'execution': 'real atomic new admissions',
        'full_v5_rollout': False, 'submitted_migration': False, 'engine_queue_reorder': False,
        'new_calibration': False, 'automatic_grid': False, 'formal_acceptance': False,
        'command': sys.argv, 'host_command': os.environ.get('DABS_HOST_COMMAND'),
        'controller_image': os.environ.get('DABS_CONTROLLER_IMAGE')})
    source = Path(__file__).parent
    write_json(out / 'fast_source_identity.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('fast_*.py'))})
    try:
        # Same project device lock directory as the existing R234 launcher.
        with device_locks([0, 1, 2, 3], ROOT / 'outputs/.existing_controller_locks'):
            asyncio.run(execute(args, out, specs))
    except BaseException as exc:
        write_json(out / 'status.json', {'status': 'failed', 'error': repr(exc), 'performance_pass': None})
        (out / 'error_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
        print('FAST_FAILED: ' + repr(exc), file=sys.stderr)
        print('Evidence retained: ' + str(out), file=sys.stderr)
        return 1
    print('FAST_REPORT=' + str(out / 'test_result.txt'), flush=True)
    host = os.environ.get('DABS_HOST_PROJECT')
    if host and out.is_relative_to(ROOT):
        print('HOST_FAST_REPORT=' + str(Path(host) / out.relative_to(ROOT) / 'test_result.txt'), flush=True)
    return 0
