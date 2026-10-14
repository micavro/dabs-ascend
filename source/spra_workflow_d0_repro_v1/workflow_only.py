#!/usr/bin/env python3
"""Pure A/D remeasurement through the existing typed19 native executor.

No backend implementation, model, profile or deployment is replaced. No fixture
mode is provided. inspect/audit-history/selftests do not contact a model service.
The GPU path imports the ORIGINAL run_typed.py and calls its run() directly;
only the measured method list/order, post-run checks and report are changed.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/home/wmy/sci/spra_paper_experiments_github')
PILOT = 'PILOT_20261010T092749Z_421e356e'
PROFILE = 'MUSIQUE_20261010T092711Z_be09912a'
REVISION = 'workflow-D0-native-remeasurement-1'
FROZEN_MANIFEST_SHA = '01ecc079475af81a12f6af8692b744c599590921a0eda6bdfe9e8298e67337c6'
METHODS = ['A', 'D']
# Relative A/D order from the original four five-method blocks; no other method runs.
HISTORICAL_ORDERS = [['A', 'D'], ['D', 'A'], ['D', 'A'], ['D', 'A']]
FIELDS = ('mean_jct_ms', 'p95_jct_ms', 'mean_request_ttft_ms',
          'p95_request_ttft_ms', 'mean_node_time_to_first_content_ms',
          'mean_client_tpot_proxy_ms', 'p95_client_tpot_proxy_ms',
          'cohort_workflows_s', 'cohort_tokens_s')


def load(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(x) -> bool:
    return type(x) in (float, int) and math.isfinite(x)


def save_new(p: Path, value) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def source_paths(root: Path):
    return (root / 'spra_stage19_typed_handoff_v1',
            root / 'spra_stage19_musique_joint_v1')


def inspect_sources(root: Path, prepared: Path | None = None, require_data=False):
    """Static identity check. Does not import the old engine or read gold contents."""
    typed, source = source_paths(root)
    prepared = prepared or typed / 'prepared' / PROFILE
    if sha(prepared / 'manifest.json') != FROZEN_MANIFEST_SHA:
        raise ValueError('not the located historical profile manifest; do not edit expected hashes')
    mf = load(prepared / 'manifest.json')
    if mf.get('application_revision') != 'stage19-typed-handoff-1':
        raise ValueError('wrong application revision; do not reuse an XML profile')
    if mf.get('best_fixed_cap') != 4096:
        raise ValueError('historical checkpoint requires the frozen 4096 cap')
    if mf.get('profile_configuration', {}).get('arrival_rate') != 2.0:
        raise ValueError('historical checkpoint requires the frozen rate-2 profile')
    if mf.get('profile_configuration', {}).get('replicas') != 4:
        raise ValueError('historical checkpoint requires four replicas')
    expected = mf['application_source_hashes']
    if len(expected) != 48:
        raise ValueError('expected the actual 48-file D0 checkpoint, not a replacement manifest')
    bad = []
    for name, want in expected.items():
        if name.startswith('source/'):
            p = source / name[len('source/'):]
        elif name.startswith('adapter/'):
            p = typed / name[len('adapter/'):]
        else:
            raise ValueError('unknown source namespace: ' + name)
        if not p.is_file() or sha(p) != want:
            bad.append(name)
    for name, want in mf['files'].items():
        p = prepared / name
        if not p.is_file() or sha(p) != want:
            bad.append('profile/' + name)
    if bad:
        raise ValueError('source/profile drift; do not edit hashes: ' + ', '.join(bad))
    ds = source / 'dataset' / 'compiled'
    if sha(ds / 'manifest.json') != mf['dataset_manifest_sha256']:
        raise ValueError('compiled dataset manifest changed')
    dm = load(ds / 'manifest.json')
    if dm.get('counts', {}).get('pilot') != 64:
        raise ValueError('expected historical 64-question pilot, not B1 repeated four times')
    data_status = {}
    for name in ('pilot.public.jsonl', 'evaluation_gold.jsonl'):
        p = ds / name
        good = p.is_file() and sha(p) == dm['files'][name]
        data_status[name] = 'hash_match' if good else 'missing_or_hash_mismatch'
        if require_data and not good:
            raise ValueError('required existing dataset missing/changed: ' + str(p))
    if require_data:
        records = [json.loads(s) for s in (ds / 'pilot.public.jsonl').read_text().splitlines() if s.strip()]
        if len(records) != 64 or len({r['id'] for r in records}) != 64:
            raise ValueError('pilot must contain 64 unique questions')
    return {'revision': REVISION, 'source_files_verified': len(expected),
            'typed_source': str(typed), 'scheduler_source': str(source),
            'prepared': str(prepared), 'dataset': str(ds),
            'model_sha256': mf['files']['model.json'],
            'priors_sha256': mf['files']['priors.json'],
            'profile_manifest_sha256': sha(prepared / 'manifest.json'),
            'dataset_manifest_sha256': sha(ds / 'manifest.json'),
            'data_files': data_status, 'historical_run': str(typed / 'runs' / PILOT),
            'gpu_generation_requests': 0, 'backend_accessed': False}


def paired_report(rows, expected_blocks=4):
    groups = {}
    for x in rows:
        b = x['block']; m = x['method']
        if m not in METHODS:
            raise ValueError('only A and D are allowed in pure scheduling report')
        if m in groups.setdefault(b, {}):
            raise ValueError('duplicate block/method result')
        groups[b][m] = x['summary']
    pairs = []
    for b, g in sorted(groups.items()):
        if set(g) != set(METHODS):
            continue
        if not all(finite(g[m].get('mean_jct_ms')) and g[m]['mean_jct_ms'] > 0 for m in METHODS):
            raise ValueError('nonfinite/nonpositive mean JCT')
        gains = {}
        for field in FIELDS:
            a, d = g['A'].get(field), g['D'].get(field)
            higher = field in ('cohort_workflows_s', 'cohort_tokens_s')
            gains[field] = (100 * ((d / a - 1) if higher else (1 - d / a))
                            if finite(a) and a > 0 and finite(d) else None)
        q = {}
        for field in ('final_em', 'final_f1'):
            av = g['A'].get('semantic_quality', {}).get(field)
            dv = g['D'].get('semantic_quality', {}).get(field)
            q[field + '_delta_pp'] = 100 * (dv - av) if finite(av) and finite(dv) else None
        pairs.append({'block': b, 'gains_D_vs_A_percent': gains,
                      'quality_delta_pp': q,
                      'actual_output_tokens': {m: g[m].get('output_tokens') for m in METHODS},
                      'absolute_mean_jct_ms': {m: g[m]['mean_jct_ms'] for m in METHODS}})
    aggregates = {}
    for field in FIELDS:
        vals = [p['gains_D_vs_A_percent'][field] for p in pairs if p['gains_D_vs_A_percent'][field] is not None]
        aggregates[field] = {'pairs': len(vals), 'mean_gain_pct': statistics.mean(vals) if vals else None,
                             'positive_pairs': sum(v > 0 for v in vals)}
    return {'revision': REVISION, 'methods': METHODS, 'complete': len(pairs) == expected_blocks,
            'accepted_conditions': len(rows),
            'actual_measured_requests': sum(x['summary'].get('requests', 0) for x in rows),
            'paired_blocks': pairs, 'D_vs_A': aggregates,
            'scope': '64 historical dev questions in four disjoint blocks; fresh execution, not unseen questions',
            'primary': 'mean of within-block percentage reductions of mean workflow JCT',
            'batching_policy': 'same fixed 4096; native continuous batching retained',
            'p95': 'within-condition P95 comparisons; not pooled P95',
            'tpot': 'client proxy, not CUDA token ITL',
            'capacity_claimed': False, 'all_negative_results_retained': True,
            'guaranteed_10_percent': False}


def validate_fixed_summary(summary, count=None):
    if summary.get('condition_valid') is not True:
        raise ValueError('condition not valid')
    if summary.get('budget_invariants') is not True:
        raise ValueError('missing/failed native budget invariants')
    if summary.get('prefill_caps_applied') != [4096]:
        raise ValueError('native cap differs from fixed 4096 or has missing evidence')
    if summary.get('budget_switches_total') != 0:
        raise ValueError('unexpected native budget switch')
    matches, checked = summary.get('native_next_batch_matches'), summary.get('native_next_batch_checked')
    if type(checked) is not int or checked <= 0 or matches != checked:
        raise ValueError('native batch-check mismatch/missing evidence; preserve raw records')
    if count is not None and (summary.get('requests') != count or summary.get('request_correlation') != f'{count}/{count}'):
        raise ValueError('request count/correlation mismatch')


def load_original(root: Path):
    typed, source = source_paths(root)
    os.environ['STAGE19_SOURCE'] = str(source.resolve())
    # The old dependency module pins source hashes; no identity checks are disabled.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('workflow_d0_original_typed', typed / 'tools/run_typed.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker_ready():
    """Importable under multiprocessing spawn; no predictor or model modification."""
    return os.getpid()


def configure_remeasurement(module, direction, receipt, checker='legacy'):
    """Thin in-process orchestration changes; ORIGINAL run/execute/choose reused."""
    orders = copy.deepcopy(HISTORICAL_ORDERS)
    if direction == 'reverse':
        orders = [list(reversed(order)) for order in orders]
    module.METHODS = list(METHODS)
    module.ORDERS = orders
    module.progress_summary = paired_report
    module.worker_ready = worker_ready
    original_execute = module.execute
    original_atomic = module.atomic
    out_holder = {}
    initial_checker = module.native_client.compare_iteration
    if checker == 'tail-corrected':
        from tail_check import compare_iteration
        module.native_client.compare_iteration = compare_iteration

    def traced_atomic(path, value):
        path = Path(path)
        if path.name == 'frozen_protocol.json':
            value = copy.deepcopy(value)
            value['workflow_only_remeasurement'] = {
                **receipt, 'methods': METHODS, 'orders': orders,
                'source_scheduler_modified': False, 'executor_modified_on_disk': False,
                'original_20_condition_suite_replayed_exactly': False,
                'difference_from_history': 'A/D-only remeasurement; other three methods omitted, original block/arrival seeds retained',
                'checker': checker, 'no_adaptive_actions': True,
                'working_directory_or_source_ROOT_not_reassigned': True}
            out_holder['directory'] = path.parent
            print('WORKFLOW_REPRO_RUN=' + str(path.parent), flush=True)
        return original_atomic(path, value)

    async def checked_execute(client, runtime, initial, cfg, nodes, folder, method, model=None, **kwargs):
        if method not in ('A', 'D', 'Forced'):
            raise ValueError('unexpected method; adaptive conditions are forbidden')
        if cfg.get('base_cap') != 4096 or len(initial) != 4 or cfg.get('replicas') != 4:
            raise ValueError('first reproduction is locked to R4/fixed4096')
        if method in METHODS and (kwargs.get('force_caps') is not None or kwargs.get('mechanism') is not None):
            raise ValueError('force_caps bypasses D planning in original executor: do not use it')
        summary, events = await original_execute(client, runtime, initial, cfg, nodes, folder, method, model, **kwargs)
        validate_fixed_summary(summary, len(nodes))
        if method in METHODS:
            controls = load(Path(folder) / 'controls.json')
            if not controls or any(c.get('caps') != [4096] * 4 or c.get('totals') != [4096] * 4 for c in controls):
                raise ValueError('published controls do not match fixed-cap contract')
            for name, n in nodes.items():
                if n.get('state') != 'done' or not finite(n.get('finish')):
                    raise ValueError('unfinished request: ' + name)
            save_new(Path(folder) / 'workflow_only_audit.json', {
                'fixed_caps_verified': True, 'requests': len(nodes), 'checker': checker,
                'schedule_source_unchanged': True, 'native_budget_switches': 0})
        return summary, events

    module.atomic = traced_atomic
    module.execute = checked_execute
    return out_holder, original_execute, original_atomic, initial_checker


def audit_history(root: Path, run: Path | None, out: Path):
    """Read only the located run. Never regenerates missing results."""
    typed, _ = source_paths(root)
    run = (run or typed / 'runs' / PILOT).resolve()
    if not run.is_dir():
        raise FileNotFoundError('historical run is not present in this copy: ' + str(run))
    protocol = load(run / 'frozen_protocol.json')
    cfg = protocol['config']
    if any(cfg.get(k) != v for k, v in [('replicas', 4), ('arrival_rate', 2.0), ('base_cap', 4096)]):
        raise ValueError('located run does not match historical R4/rate2/fixed4096 configuration')
    rows = []
    retained = [run / n for n in ('frozen_protocol.json', 'code_identity.json', 'dataset_manifest.json', 'index.json', 'joint_result.json', 'suite_status.json', 'cleanup.json', 'environment.json') if (run / n).is_file()]
    for path in sorted(run.glob('B*/summary.json')):
        manifest = load(path.parent / 'manifest.json')
        m = manifest.get('method')
        if m not in METHODS:
            continue
        block = int(path.parent.name.split('_')[0][1:])
        summary = load(path)
        input_nodes = manifest.get('input_nodes', {})
        validate_fixed_summary(summary, len(input_nodes))
        rows.append({'block': block, 'method': m, 'summary': summary})
        for name in ('summary.json', 'manifest.json', 'nodes.json', 'controls.json', 'semantic_proof.json', 'quality.json'):
            p = path.parent / name
            if p.is_file():
                retained.append(p)
    report = paired_report(rows)
    if len(rows) != 8 or not report['complete']:
        raise ValueError('historical A/D summaries incomplete; no missing row imputed')
    report['origin'] = str(run)
    report['raw_backend_recomputed'] = False
    report['evidence_level'] = 'saved condition summaries plus fixed-cap counters; full backend.jsonl re-audit not performed'
    out.mkdir(parents=True, exist_ok=False)
    save_new(out / 'HISTORICAL_AD_AUDIT.json', report)
    save_new(out / 'FILES.json', {str(p.relative_to(run)): sha(p) for p in retained})
    with zipfile.ZipFile(out / 'historical_AD_evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in retained:
            # Original source files are never edited, and no arbitrary external paths are followed.
            if p.is_symlink() or not p.resolve().is_relative_to(run):
                raise ValueError('unsafe historical file path: ' + str(p))
            if p.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('unexpected large selected evidence file: ' + str(p))
            z.write(p, 'historical_run/' + str(p.relative_to(run)))
        z.write(out / 'HISTORICAL_AD_AUDIT.json', 'HISTORICAL_AD_AUDIT.json')
        z.write(out / 'FILES.json', 'FILES.json')
    print(json.dumps({'report': str(out / 'HISTORICAL_AD_AUDIT.json'),
                      'evidence': str(out / 'historical_AD_evidence.zip'),
                      'D_vs_A': report['D_vs_A'], 'gpu_generation_requests': 0}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('inspect', 'audit-history', 'gate', 'reproduce'))
    p.add_argument('--repo', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--runtime', type=Path, default=Path('/home/wmy/sci/spra_core_exp/joint_e1'))
    p.add_argument('--prepared', type=Path)
    p.add_argument('--history-run', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--order', choices=('historical-relative', 'reverse'), default='historical-relative')
    p.add_argument('--checker', choices=('legacy', 'tail-corrected'), default='tail-corrected',
                   help='Only changes the independent post-execution checker, not schedule/native execution')
    p.add_argument('--confirm-gpu-use', action='store_true')
    a = p.parse_args()
    if a.command == 'audit-history':
        out = a.out or HERE / 'audits' / ('history_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
        audit_history(a.repo, a.history_run, out)
        return
    receipt = inspect_sources(a.repo, a.prepared, require_data=a.command == 'reproduce')
    if a.command == 'inspect':
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return
    if a.command == 'reproduce' and not a.confirm_gpu_use:
        p.error('reproduce sends 12 warmup + 426 measured requests; require --confirm-gpu-use')
    original = load_original(a.repo)
    # Check the original loader against its exact frozen application identity before GPU access.
    prepared_path = Path(receipt['prepared'])
    original.prepared(SimpleNamespace(prepared=prepared_path))
    ns = SimpleNamespace(command='gate' if a.command == 'gate' else 'pilot',
                         rate=2.0, runtime=a.runtime, dataset=Path(receipt['dataset']),
                         prepared=prepared_path, legacy_model=None)
    if a.command == 'gate':
        asyncio.run(original.run(ns))
        return
    receipt['inspection_gpu_generation_requests'] = receipt.pop('gpu_generation_requests')
    receipt['inspection_backend_accessed'] = receipt.pop('backend_accessed')
    receipt.update(wrapper_sha256=sha(Path(__file__)), independent_checker_sha256=sha(HERE / 'tail_check.py'),
                   GPU_measurement_claimed_in_delivery=False)
    state, old_execute, old_atomic, old_checker = configure_remeasurement(original, a.order, receipt, a.checker)
    try:
        asyncio.run(original.run(ns))
        out = state['directory']
        report = load(out / 'joint_result.json')
        save_new(out / 'workflow_only_result.json', report)
        save_new(out / 'workflow_reproduction_receipt.json', receipt)
        if not report.get('complete') or report.get('accepted_conditions') != 8:
            raise RuntimeError('incomplete pure scheduling remeasurement')
        print('WORKFLOW_ONLY_RESULT=' + json.dumps(report, ensure_ascii=False), flush=True)
        print('WORKFLOW_ONLY_OUTPUT=' + str(out), flush=True)
    finally:
        original.execute = old_execute
        original.atomic = old_atomic
        original.native_client.compare_iteration = old_checker


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as ex:
        print('STOP: ' + str(ex), file=sys.stderr)
        raise SystemExit(2)
