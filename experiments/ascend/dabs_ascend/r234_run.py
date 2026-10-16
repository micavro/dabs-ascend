"""Auditable full R234 execution; only clean condition-boundary pauses resume."""
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid

from . import r234
from .common import (DATASET, ROOT, append_json, configuration, read_json, sha,
                     source_identity, verify_original_assets, write_json)
from .runtime import MODEL_NAME, device_locks


def utc():
    return datetime.now(timezone.utc).isoformat()


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def arguments(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def stable_arguments(args):
    return {k: v for k, v in arguments(args).items()
            if k not in ('out', 'resume', 'max_conditions', 'deployment_manifest', 'mode', 'native_evidence', 'model')}


def native_paths(args):
    if getattr(args, 'native_evidence', None):
        return read_json(args.native_evidence)['paths']
    path = ROOT / 'outputs/native-latest.json'
    latest = read_json(path) if path.exists() else {}
    return {k: latest[k] for k in ('native-probe-one', 'native-probe-four', 'native-calibrate',
                                  'native-invalidate', 'native-restart-check') if k in latest}


def preserve_native_freeze(out, gate, model, resume):
    from dabs_native.contracts import engine_epoch, resource_values
    def scientific_binding(value):
        return {'identities': [engine_epoch(i) for i in value['identities']],
                'resources': [resource_values(r) for r in value['resources']],
                **{k: value.get(k) for k in ('models', 'validation', 'timing_target', 'host_step_equivalence')}}
    binding = scientific_binding(model)
    path = out / 'native_freeze.json'
    if resume:
        if scientific_binding(read_json(out / 'ascend_profile.json')) != binding:
            raise ValueError('Native model/resources/engine changed since pause; start a new experiment directory')
    else:
        write_json(path, {**binding, 'provenance': {'artifacts': gate['artifacts'], 'paths': gate['paths']}})


def audit_completed(out, plan):
    summaries = read_json(out / 'index.json')
    if len(summaries) >= len(plan):
        raise ValueError('Run is already complete')
    for index, summary in enumerate(summaries):
        if summary['spec'] != plan[index] or not summary['functional_pass']:
            raise ValueError('Completed index is not a valid prefix of the frozen plan')
        directory = out / 'conditions' / summary['spec']['label']
        hashes = read_json(directory / 'artifacts.json')
        expected_names = {'config.json', 'spec.json', 'initial_nodes.json', 'input_proof.json', 'nodes.json',
                          'requests.jsonl', 'decisions.jsonl', 'summary.json', 'quality.json', 'status.json'}
        if summary.get('execution_scope') == 'native':
            expected_names.add('native_events.jsonl')
        if set(hashes) != expected_names:
            raise ValueError('Incomplete raw artifacts for ' + summary['spec']['label'])
        for name, expected in hashes.items():
            if sha(directory / name) != expected:
                raise ValueError('Completed raw artifact changed: ' + str(directory / name))
        if read_json(directory / 'summary.json') != summary:
            raise ValueError('Index differs from raw summary')
    directories = {p.name for p in (out / 'conditions').iterdir()} if (out / 'conditions').exists() else set()
    if directories != {s['spec']['label'] for s in summaries}:
        raise ValueError('Unindexed or incomplete condition directories; keep this run and start a new one')
    return summaries


def verify_pairing(summaries, summary):
    for prior in summaries:
        if prior['external_work_fingerprint'] != summary['external_work_fingerprint']:
            raise ValueError('Fixed work differs across methods/conditions')
        if prior['spec']['rate'] == summary['spec']['rate'] and prior['arrival_offsets'] != summary['arrival_offsets']:
            raise ValueError('Paired scheduled arrivals differ')


def deployment_signature(args, out):
    env = read_json(out / 'environment.json')
    # The host inspector changes its timestamp on every invocation. Compare
    # actual services/options/model identity, not the inspection timestamp.
    deployment = [{k: v for k, v in row.items() if k != 'served_model'} for row in env.get('deployment', [])]
    return {'devices': args.devices, 'model_config': read_json(args.model / 'config.json'),
            'controller_image': os.environ.get('DABS_CONTROLLER_IMAGE'),
            'deployment': deployment, 'service_mode': args.service_mode}


async def execute(args, out, plan, summaries, identity):
    import httpx
    from .backend import Backend, Tokenizer
    from .calibration import calibrate
    from .runner import run_condition
    from semantic19.dataset import load_public
    from typed19.workflow import compile_packets
    if args.service_mode == 'existing':
        from .existing import ExistingServers
        servers = ExistingServers(args, out)
    else:
        from .runtime import Servers
        servers = Servers(args, out)
    pool = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=10), trust_env=False,
                                 limits=httpx.Limits(max_connections=256, max_keepalive_connections=args.http_keepalive)) as client:
        try:
            await servers.start(client)
            backend = Backend(client, servers.endpoints)
            write_json(out / 'api_contract.json', await backend.inspect(fixed=True))
            native = args.execution_scope == 'native'
            if native:
                from .native_client import NativeClient
                from .native_gate import verify_live
                from .native_algorithms import transformation_manifest
                write_json(out / 'native_transformations.json', transformation_manifest())
                backend.native = NativeClient(client, servers.endpoints, out / 'native_preflight.jsonl')
                caps = await backend.native.inspect()
                if getattr(args, 'exploratory_calibration', False):
                    from .native_explore import verify
                    gate, model = await verify(backend.native, native_paths(args))
                else:
                    gate = await verify_live(backend.native, native_paths(args))
                    if not gate['pass']:
                        raise ValueError('native evidence gate failed: ' + str(gate['missing']))
                    model = read_json(Path(native_paths(args)['native-calibrate']) / 'native_profile.json')
                preserve_native_freeze(out, gate, model, args.resume)
                write_json(out / 'native_gate.json', gate)
                await backend.native.drain(4)
                write_json(out / 'native_capabilities.json', caps)
            append_json(out / 'process.jsonl', {'event': 'canaries_start', 'utc': utc(), 'resumed': bool(args.resume)})
            write_json(out / 'canary_summary.json', await backend.fixed_canaries(out))
            if native:
                await backend.native.drain(4)
                while await backend.native.harvest(4, wait_complete=True):
                    pass
            tok = Tokenizer(client, servers.endpoints[0])
            if native:
                write_json(out / 'ascend_profile.json', model)
                write_json(out / 'calibration_identity.json', {'ascend_profile.json': sha(out / 'ascend_profile.json')})
            elif args.resume:
                model = read_json(out / 'ascend_profile.json')
            else:
                model = await calibrate(backend, tok, out, fixed=True)
                write_json(out / 'calibration_identity.json', {name: sha(out / name)
                           for name in ('ascend_profile.json', 'calibration_rows.json')})
            cfg = configuration(4, caps[0]['resources'], args.max_inflight) if native else configuration(4)
            if native:
                protocol = read_json(out / 'protocol.json')
                protocol.update(limits=[('EXPLORATORY: prediction accuracy not accepted; see native_gate.json for measured errors and model transfer.'
                    if gate.get('exploratory') else 'Native capability/evidence gate passed; performance equivalence is not implied.'),
                    '200 ms finite priority window; original FIFO continuation and 5 s age guard.',
                    'Ayo uses conservative reserved KV blocks divided by physical capacity.',
                    'Service model target: ' + model['timing_target'] + '; device events are auxiliary. See native-async-calibration.md for approximation limits.'],
                    timing_target=model['timing_target'], original_gpu_timing_equivalent=model.get('original_gpu_timing_equivalent', False),
                    effective_resources=[c['resources'] for c in caps], client_max_inflight=args.max_inflight,
                    exploratory=bool(gate.get('exploratory')), formal_acceptance=gate['pass'])
                write_json(out / 'protocol.json', protocol)
                repair = read_json(out / 'repair_protocol.json')
                repair['native_mechanism_test_pass'] = gate['pass']
                repair['exploratory'] = bool(gate.get('exploratory'))
                repair['calibration_scope'] = model['timing_target']
                write_json(out / 'repair_protocol.json', repair)
            if not native:
                cfg['max_num_seqs'] = args.max_inflight
            cfg.update(model=MODEL_NAME, condition_wall_s=args.condition_timeout,
                       wf_router_seed=190500, input_profile='fixed_3584_balanced_DAG',
                       output_contract='fixed_by_graph_shape', guided_decoding=None,
                       parent_binding='dependency_order_only', mode='nonsemantic_fixed_token_replay')
            rows, manifest = load_public(DATASET, 'pilot')
            packets, info = await compile_packets(rows, tok, cfg)
            packets, workload = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
            if args.resume and read_json(out / 'workload.json') != workload:
                raise ValueError('Compiled workload differs from paused run')
            write_json(out / 'workload.json', workload)
            write_json(out / 'preparation.json', {'info': info, 'dataset_manifest': manifest,
                                                'source_ids': [r['id'] for r in rows]})
            pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn'))
            await asyncio.get_running_loop().run_in_executor(pool, r234.planner_ready)
            first = len(summaries)
            stop = min(len(plan), first + (args.max_conditions or len(plan)))
            for index in range(first, stop):
                spec = plan[index]
                if source_identity() != identity:
                    append_json(out / 'process.jsonl', {'event': 'source_provenance_changed', 'source_sha256': source_identity()})
                    identity = source_identity()
                r234.verify_import()
                servers.check_alive()
                current = {**cfg, 'replicas': spec['replicas'], 'arrival_rate': spec['rate'],
                           'workflow_count': spec['workflow_count'], 'wf_router_seed': spec['router_seed'],
                           'wf_public_method': spec['method'], 'wf_routing_policy': spec['method']}
                directory = out / 'conditions' / spec['label']
                append_json(out / 'process.jsonl', {'event': 'condition_start', 'utc': utc(), 'index': index + 1,
                            'spec': spec, 'files': ['dabs_ascend.runner.run_condition', 'typed19.workflow.instantiate',
                                'dabs_r234_frozen.fixed_work.convert_nodes', 'dabs_ascend.r234.baseline',
                                'dabs_ascend.r234.plan_ours' if spec['method'] == 'Ours' else None],
                            'output': str(directory), 'logical_to_physical_devices': args.devices[:spec['replicas']]})
                print(f"R234_CONDITION {index + 1}/96 {spec['label']}", flush=True)
                summary = await run_condition(backend, tok, packets, model, current, spec, directory, pool, servers.check_alive)
                if native and gate.get('exploratory'):
                    summary.update(exploratory=True, formal_acceptance=False,
                                   model_transfer_from_replica_0=gate['model_transfer_from_replica_0'])
                    write_json(directory / 'summary.json', summary)
                write_json(directory / 'config.json', current)
                write_json(directory / 'spec.json', spec)
                verify_pairing(summaries, summary)
                if not summary['functional_pass']:
                    raise ValueError('Fixed work or dependency check failed; no automatic retry')
                names = ('config.json', 'spec.json', 'initial_nodes.json', 'input_proof.json', 'nodes.json',
                         'requests.jsonl', 'decisions.jsonl', 'summary.json', 'quality.json', 'status.json')
                if native:
                    names += ('native_events.jsonl',)
                write_json(directory / 'artifacts.json', {name: sha(directory / name) for name in names})
                summaries.append(summary)
                write_json(out / 'index.json', summaries)
                if len(summaries) % 4 == 0:
                    group = summaries[-4:]
                    audit = {'last_condition': spec['label'], 'methods': [s['spec']['method'] for s in group],
                             'workload_pass': all(s['functional_pass'] for s in group),
                             'workload_identity_equal': len({s['external_work_fingerprint'] for s in group}) == 1,
                             'engine_mechanisms_pass': native and gate['pass'], 'execution_scope': args.execution_scope}
                    append_json(out / 'group_audits.jsonl', audit)
                    if set(audit['methods']) != set(r234.METHODS) or not audit['workload_pass'] or not audit['workload_identity_equal']:
                        raise ValueError('Four-method group audit failed')
                append_json(out / 'process.jsonl', {'event': 'condition_complete', 'utc': utc(), 'index': index + 1,
                            'label': spec['label'], 'mean_jct_ms': summary['jct_mean_ms'], 'output': str(directory)})
                from .r234_report import make_report
                make_report(out, plan, summaries, 'running')
                print(f"R234_DONE {index + 1}/96 mean_JCT_ms={summary['jct_mean_ms']:.2f} output_tokens={summary['output_tokens']}", flush=True)
            servers.check_alive()
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            if args.service_mode == 'existing':
                await servers.aclose()
            else:
                servers.close()


def main(args):
    plan = r234.plan()
    if args.mode == 'plan':
        print(json.dumps({'suite': r234.SUITE, 'conditions': 96, 'plan': plan, 'limits': r234.LIMITS,
                          'requests_per_condition': 216, 'output_tokens_per_condition': 13184,
                          'inference_executed': False}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == 'benchmark' and args.execution_scope != 'http-limited':
        from .repair import mechanism_gate
        if getattr(args, 'exploratory_calibration', False):
            from .native_explore import evaluate
            try:
                gate, _ = evaluate(native_paths(args))
            except (ValueError, KeyError, OSError) as exc:
                print('EXPLORATORY_REFUSED=' + str(exc), file=sys.stderr)
                return 2
        else:
            gate = mechanism_gate(native_paths(args))
        if args.service_mode != 'existing' or not (gate['pass'] or gate.get('exploratory_allowed')):
            print('R234_NATIVE_GATE=' + json.dumps(gate, ensure_ascii=False), file=sys.stderr)
            return 2
    prefix = 'r234_native_' if args.execution_scope == 'native' else 'r234_ascend_'
    out = args.resume or args.out or ROOT / 'outputs' / (prefix + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    out = out.expanduser().resolve()
    identity = source_identity()
    try:
        if args.resume:
            if read_json(out / 'status.json')['status'] != 'paused':
                raise ValueError('Only a clean --max-conditions pause can resume; failed/interrupted runs must be preserved')
            protocol = read_json(out / 'protocol.json')
            if protocol['plan'] != plan or protocol['stable_arguments'] != stable_arguments(args):
                raise ValueError('Frozen plan or deployment parameters differ from paused run')
            for name, expected in read_json(out / 'calibration_identity.json').items():
                if sha(out / name) != expected:
                    append_json(out / 'process.jsonl', {'event': 'calibration_file_fingerprint_changed', 'file': name, 'actual': sha(out / name)})
            summaries = audit_completed(out, plan)
        else:
            out.mkdir(parents=True, exist_ok=False)
            summaries = []
            write_json(out / 'protocol.json', {'suite': r234.SUITE, 'plan': plan, 'source_sha256': identity,
                       'stable_arguments': stable_arguments(args), 'limits': r234.LIMITS,
                       'workflows_per_condition': 64, 'requests_per_condition': 216,
                       'input_tokens_per_request': 3584, 'output_tokens_per_condition': 13184,
                       'strict_native_reproduction': False, 'automatic_retry': False})
            write_json(out / 'repair_protocol.json', {
                'revision': '20261016-native-v023-v1', 'execution_scope': args.execution_scope,
                'plan_revision': 'R4-L8_then_R2-L16_rotating_methods_mirrored_round2',
                'http_keepalive': args.http_keepalive, 'native_mechanism_test_pass': False,
                'calibration_scope': 'pending native preflight' if args.execution_scope == 'native' else 'HTTP TTFT/TPOT proxy; not native batch-step validation',
                'group_audit_conditions': 4, 'first_checkpoint_conditions': 8,
                'previous_results_combinable': False})
            write_json(out / 'index.json', [])
    except Exception as exc:
        print('R234_REFUSED=' + str(exc), file=sys.stderr)
        return 2
    lock_dir = ROOT / 'outputs/.existing_controller_locks' if args.service_mode == 'existing' else None
    try:
        with device_locks(args.devices, directory=lock_dir):
            if args.resume:
                if read_json(out / 'status.json')['status'] != 'paused':
                    raise ValueError('Run changed while acquiring device locks')
                summaries = audit_completed(out, plan)
            with (out / 'console.log').open('a', encoding='utf-8', buffering=1) as logfile:
                with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
                    print('R234_OUTPUT=' + str(out), flush=True)
                    return run_recorded(args, out, plan, summaries, identity)
    except Exception as exc:
        print('R234_REFUSED=' + str(exc), file=sys.stderr)
        return 2


def run_recorded(args, out, plan, summaries, identity):
    from .r234_report import make_report, procedure
    try:
        git = subprocess.run(['git', 'rev-parse', '--show-toplevel', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, encoding='utf-8', timeout=5)
        lines = git.stdout.splitlines()
        revision = lines[1] if git.returncode == 0 and len(lines) == 2 and Path(lines[0]).resolve() == ROOT else None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    record = {'utc': utc(), 'event': 'invocation', 'arguments': arguments(args), 'argv': sys.argv,
              'cwd': os.getcwd(), 'python': sys.version, 'executable': sys.executable, 'git_head': revision,
              'host_project': os.environ.get('DABS_HOST_PROJECT'), 'host_command': os.environ.get('DABS_HOST_COMMAND'),
              'controller_image': os.environ.get('DABS_CONTROLLER_IMAGE'), 'output_directory': str(out)}
    append_json(out / 'process.jsonl', record)
    procedure(out, record)
    status, error = 'failed', None
    prior = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupt)
    try:
        write_json(out / 'status.json', {'status': 'preflight', 'started_utc': record['utc']})
        write_json(out / 'import_integrity.json', r234.verify_import())
        write_json(out / 'original_integrity.json', verify_original_assets())
        if args.service_mode == 'existing':
            from .existing import preflight
        else:
            from .runtime import preflight
        preflight(args, out)
        signature = deployment_signature(args, out)
        if args.resume and read_json(out / 'deployment_identity.json') != signature:
            raise ValueError('Service deployment changed since pause')
        write_json(out / 'deployment_identity.json', signature)
        if args.mode == 'check':
            status = 'environment_checked'
        else:
            write_json(out / 'status.json', {'status': 'running'})
            asyncio.run(execute(args, out, plan, summaries, identity))
            status = 'complete' if len(summaries) == 96 else 'paused'
    except BaseException as exc:
        error = repr(exc)
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('R234_FAILED=' + error, file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, prior)
        report = make_report(out, plan, summaries, status, error)
        write_json(out / 'status.json', {'status': status, 'finished_utc': utc(), 'completed_conditions': len(summaries),
                                       'functional_test_pass': report['functional_test_pass'], 'error': error})
        append_json(out / 'process.jsonl', {'event': 'invocation_end', 'utc': utc(), 'status': status,
                                          'completed_conditions': len(summaries), 'error': error})
    print('R234_REPORT=' + str(out / 'TEST_REPORT.md'), flush=True)
    if status == 'paused':
        relative = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out)
        print('CLEAN_PAUSE: remaining=' + str(96 - len(summaries)), flush=True)
        print('RESUME: bash run_dabs_r234.sh --execution-scope ' + args.execution_scope + ' --resume ' + relative, flush=True)
    return 0 if status in ('paused', 'environment_checked') or report['functional_test_pass'] else 1
