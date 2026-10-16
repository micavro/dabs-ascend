#!/usr/bin/env python3
"""Linux native adapter deployment/control. Never starts/stops a model service."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_ascend.common import read_json, write_json, source_paths

MODES = ('native-install-check', 'native-install', 'native-verify', 'native-rollback',
         'native-check', 'native-check-four', 'native-probe-one', 'native-probe-four', 'native-calibrate',
         'native-calibrate-one', 'native-calibration-smoke', 'native-timing-audit', 'native-refit',
         'native-invalidate', 'native-restart-arm', 'native-restart-check', 'native-accept')


def docker(*args, **kwargs):
    return subprocess.run(['sudo', 'docker', *args], text=True, check=True, **kwargs)


def output_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('--out/--evidence must be inside this project outputs directory')
    return path


def mounted_path(value):
    return '/project/' + output_path(value).relative_to(ROOT).as_posix()


def host(args):
    if sys.platform != 'linux':
        raise ValueError('Deployment requires the Linux host; CPU tests run separately')
    if args.mode in MODES[:4]:
        selected = [args.replica] if args.replica is not None else list(range(4))
        if args.mode in ('native-install', 'native-rollback') and args.replica is None:
            raise ValueError('select exactly ONE replica with --replica 0..3; no automatic bulk mutation')
        out = output_path(args.out) if args.out else ROOT / 'outputs' / (args.mode + '-' + uuid.uuid4().hex[:10])
        out.mkdir(parents=True)
        python = os.environ.get('CONTROLLER_PYTHON', '/usr/local/python3.12.13/bin/python')
        ok = True
        records = []
        for replica in selected:
            container = 'wushibo-qwen-small-npu' + str(replica)
            stage = '/tmp/dabs-native-' + uuid.uuid4().hex
            docker('exec', container, 'mkdir', '-p', stage)
            docker('cp', str(ROOT / 'experiments/ascend/dabs_native'), container + ':' + stage + '/dabs_native')
            mode = {'native-install-check': 'check', 'native-install': 'install', 'native-verify': 'verify', 'native-rollback': 'rollback'}[args.mode]
            command = ['sudo', 'docker', 'exec', '-i', '-e', 'PYTHONPATH=' + stage, container, python, '-m', 'dabs_native.install', mode]
            if mode == 'verify':
                # Verify the installed adapter, not the staging copy's receipt.
                command = ['sudo', 'docker', 'exec', container, python, '-m', 'dabs_native.install', 'verify']
            if mode == 'rollback':
                if not args.backup:
                    raise ValueError('--backup must be the exact path printed by install')
                command += ['--backup', args.backup]
            result = subprocess.run(command, capture_output=True, text=True)
            (out / ('replica-' + str(replica) + '.txt')).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            print(result.stdout, result.stderr)
            try:
                value = json.loads(result.stdout)
            except ValueError:
                value = {'error': 'installer did not return JSON; see replica text log'}
            records.append({'replica': replica, 'exit_code': result.returncode, 'result': value})
            ok &= result.returncode == 0
        write_json(out / 'result.json', {'pass': ok, 'replicas': records})
        print('NATIVE_OUTPUT=' + str(out))
        print('No services restarted. No model configuration or inference stack upgrade performed.')
        return 0 if ok else 2
    image = docker('inspect', '--format', '{{.Image}}', 'wushibo-qwen-small-npu0', capture_output=True).stdout.strip()
    command = ['sudo', 'docker', 'run', '--rm', '--pull', 'never', '--network', 'host',
        '--user', str(os.getuid()) + ':' + str(os.getgid()), '--mount', 'type=bind,src=' + str(ROOT) + ',dst=/project',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUTF8=1', '-e', 'PYTHONUNBUFFERED=1', '--workdir', '/project', '--entrypoint',
        os.environ.get('CONTROLLER_PYTHON', '/usr/local/python3.12.13/bin/python'), image,
        'tools/native_dabs.py', args.mode, '--worker']
    if args.replica is not None:
        command += ['--replica', str(args.replica)]
    if args.out:
        command += ['--out', mounted_path(args.out)]
    if args.evidence:
        command += ['--evidence', mounted_path(args.evidence)]
    if args.input:
        # Offline audit may read an older checkout. Mount it read-only; never
        # change/ack/repackage the original files or connect to model services.
        source = args.input.resolve()
        if not source.is_dir():
            raise ValueError('--input must be an existing calibration directory')
        at = command.index('--workdir')
        command[at:at] = ['--mount', 'type=bind,src=' + str(source) + ',dst=/audit-input,readonly']
        command += ['--input', '/audit-input']
    return subprocess.call(command)


async def worker(args, out):
    import httpx
    from dabs_ascend.backend import Backend, Tokenizer
    from dabs_ascend.native_client import NativeClient
    from dabs_ascend.native_probe import probe, invalidation_probe
    from dabs_ascend.native_gate import verify_live
    from dabs_ascend.native_calibration import run_calibration
    if args.mode == 'native-refit':
        from dabs_ascend.native_calibration import refit_existing
        result = refit_existing(args.input, out)
        result.update(pass_=result['retrospective_error_check_pass'], diagnostic_only=True)
        result['pass'] = result.pop('pass_')
        write_json(out / 'result.json', result)
        summary = json.dumps(result, ensure_ascii=False, indent=2)
        (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
        print(summary)
        return 0 if result['pass'] else 2
    if args.mode == 'native-timing-audit':
        from dabs_ascend.native_timing import audit, read_events, reconstruct, collection_floors
        rows = list(read_events(args.input / 'native_events.jsonl'))
        report = audit(rows)
        write_json(out / 'timing_audit.json', report)
        plan_file = args.input / 'calibration_plan.json'
        if plan_file.exists():
            record = read_json(plan_file)
            floors = collection_floors(rows, record) if record.get('collection_start') else None
            samples, excluded = reconstruct(rows, record['plan'], require_new_fields=False, after_steps=floors)
            report.update(candidate_completion_intervals=len(samples), excluded_intervals=excluded,
                          diagnostic_only=True, acceptance_granted=False)
        write_json(out / 'timing_audit.json', report)
        write_json(out / 'result.json', {'pass': True, 'diagnostic_only': True, 'acceptance_granted': False})
        summary = json.dumps(report, ensure_ascii=False, indent=2)
        (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
        print(summary)
        print('NATIVE_OUTPUT=' + str(out))
        return 0
    async with httpx.AsyncClient(timeout=300, trust_env=False, limits=httpx.Limits(max_connections=128, max_keepalive_connections=0)) as client:
        endpoints = ['http://127.0.0.1:' + str(8001 + r) for r in range(4)]
        # A single-replica check/probe is possible before patching the other three.
        if args.mode in ('native-check', 'native-probe-one', 'native-restart-arm', 'native-restart-check',
                         'native-calibrate-one', 'native-calibration-smoke'):
            endpoints = [endpoints[args.replica or 0]]
        native = NativeClient(client, endpoints, out / 'native_events.jsonl')
        caps = await native.inspect()
        write_json(out / 'capabilities.json', caps)
        from dabs_ascend.native_algorithms import transformation_manifest
        write_json(out / 'native_transformations.json', transformation_manifest())
        backend = Backend(client, endpoints)
        backend.native = native
        tokenizer = Tokenizer(client, endpoints[0])
        latest_file = ROOT / 'outputs/native-latest.json'
        latest = read_json(latest_file) if latest_file.exists() else {}
        if args.mode in ('native-check', 'native-check-four'):
            result = {'pass': True, 'native_execution': True, 'capabilities': caps, 'inference_executed': False}
        elif args.mode in ('native-probe-one', 'native-probe-four'):
            count = 1 if args.mode.endswith('one') else 4
            result = await probe(backend, tokenizer, out, count)
            from dabs_ascend.native_gate import evaluate_probe
            result = {'checks': evaluate_probe(out, count)}
            result['pass'] = all(result['checks'].values())
        elif args.mode in ('native-calibrate', 'native-calibrate-one', 'native-calibration-smoke'):
            result = await run_calibration(backend, tokenizer, out, diagnostic=args.mode == 'native-calibration-smoke')
        elif args.mode == 'native-invalidate':
            result = await invalidation_probe(native, tokenizer, backend, out)
        elif args.mode == 'native-restart-arm':
            result = await native.call(0, 'snapshot')
            write_json(ROOT / 'outputs/native-restart-arm.json', result)
            print('Snapshot saved. Manually restart only replica 0 when permitted, then native-restart-check. This command sends no restart signal.')
        elif args.mode == 'native-restart-check':
            old = read_json(ROOT / 'outputs/native-restart-arm.json')
            receipt = await native.call(0, 'apply', action=native.action(old))
            result = {'old_identity': old['identity'], 'new': caps[0], 'receipt': receipt}
            result['pass'] = (caps[0]['identity'].get('previous_engine_instance') == old['identity']['engine_instance']
                and receipt['status'] == 'rejected' and receipt.get('reason') == 'engine_restarted')
            write_json(out / 'restart.json', result)
        else:
            paths = read_json(output_path(args.evidence))['paths'] if args.evidence else {
                k: latest[k] for k in ('native-probe-one', 'native-probe-four', 'native-calibrate', 'native-invalidate', 'native-restart-check') if k in latest}
            result = await verify_live(native, paths)
            write_json(out / 'acceptance.json', result)
        latest[args.mode] = str(out)
        write_json(latest_file, latest)
        write_json(out / 'result.json', result)
        summary = {'mode': args.mode, 'output': str(out), 'pass': result.get('pass'),
                   'physical_replica': args.replica if args.replica is not None else (0 if len(endpoints) == 1 else None),
                   'live_verified': result.get('live_verified', False),
                   'checks': result.get('checks'), 'not_run': result.get('not_run', []), 'missing': result.get('missing') or result.get('failures'),
                   'diagnostic_only': result.get('diagnostic_only', False),
                   'association_pass': result.get('association_pass'),
                   'calibration_pass': result.get('calibration_pass'),
                   'inference_executed': args.mode in ('native-probe-one', 'native-probe-four', 'native-calibrate',
                       'native-calibrate-one', 'native-calibration-smoke', 'native-invalidate')}
        (out / 'SUMMARY.txt').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if result.get('pass') is False else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--replica', type=int, choices=range(4))
    parser.add_argument('--backup')
    parser.add_argument('--out', type=Path, help='new output directory inside project outputs; never overwrite')
    parser.add_argument('--evidence', type=Path, help='native-accept only: explicit JSON with raw evidence paths')
    parser.add_argument('--input', type=Path, help='native-timing-audit/native-refit: old calibration directory, read-only')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.evidence and args.mode != 'native-accept':
        parser.error('--evidence is only for native-accept')
    if (args.mode in ('native-timing-audit', 'native-refit')) != bool(args.input):
        parser.error('--input is required only for native-timing-audit/native-refit')
    if args.mode not in (*MODES[:4], 'native-check', 'native-calibrate-one', 'native-calibration-smoke') and args.replica not in (None, 0):
        parser.error('--replica 1..3 is for install/check/verify/rollback; active one-replica probes use replica 0, four-replica probes use all four')
    if not args.worker:
        return host(args)
    source_paths()
    out = output_path(args.out) if args.out else ROOT / 'outputs' / (args.mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-') + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    try:
        return asyncio.run(worker(args, out))
    except Exception as exc:
        import traceback
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(out / 'result.json', {'pass': False, 'failures': [str(exc)]})
        (out / 'SUMMARY.txt').write_text('mode: ' + args.mode + '\npass: false\nerror: ' + str(exc) + '\n', encoding='utf-8')
        print('NATIVE_FAILED=' + repr(exc), 'NATIVE_OUTPUT=' + str(out))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
