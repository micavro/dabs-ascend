#!/usr/bin/env python3
"""Linux native adapter deployment/control. Never starts/stops a model service."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_ascend.common import read_json, write_json, source_paths

MODES = ('native-install-check', 'native-install', 'native-verify', 'native-rollback',
         'native-check', 'native-probe-one', 'native-probe-four', 'native-calibrate',
         'native-invalidate', 'native-restart-arm', 'native-restart-check', 'native-accept')


def docker(*args, **kwargs):
    return subprocess.run(['sudo', 'docker', *args], text=True, check=True, **kwargs)


def host(args):
    if sys.platform != 'linux':
        raise ValueError('Deployment requires the Linux host; CPU tests run separately')
    if args.mode in MODES[:4]:
        selected = [args.replica] if args.replica is not None else list(range(4))
        if args.mode in ('native-install', 'native-rollback') and args.replica is None:
            raise ValueError('select exactly ONE replica with --replica 0..3; no automatic bulk mutation')
        out = ROOT / 'outputs' / (args.mode + '-' + uuid.uuid4().hex[:10])
        out.mkdir(parents=True)
        python = os.environ.get('CONTROLLER_PYTHON', '/usr/local/python3.12.13/bin/python')
        credentials = ROOT / 'outputs/native-control.json'
        keys = read_json(credentials) if credentials.exists() else {'keys': {}}
        ok = True
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
            secret = None
            if mode == 'install':
                keys['keys'].setdefault(str(replica), secrets.token_hex(32))
                write_json(credentials, keys)
                os.chmod(credentials, 0o600)
                secret = keys['keys'][str(replica)]
            result = subprocess.run(command, input=secret, capture_output=True, text=True)
            (out / ('replica-' + str(replica) + '.txt')).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            print(result.stdout, result.stderr)
            ok &= result.returncode == 0
        print('NATIVE_OUTPUT=' + str(out))
        print('No services restarted. No model configuration or inference stack upgrade performed.')
        return 0 if ok else 2
    image = docker('inspect', '--format', '{{.Image}}', 'wushibo-qwen-small-npu0', capture_output=True).stdout.strip()
    command = ['sudo', 'docker', 'run', '--rm', '--pull', 'never', '--network', 'host',
        '--user', str(os.getuid()) + ':' + str(os.getgid()), '--mount', 'type=bind,src=' + str(ROOT) + ',dst=/project',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUTF8=1', '--workdir', '/project', '--entrypoint',
        os.environ.get('CONTROLLER_PYTHON', '/usr/local/python3.12.13/bin/python'), image,
        'tools/native_dabs.py', args.mode, '--worker']
    if args.replica is not None:
        command += ['--replica', str(args.replica)]
    return subprocess.call(command)


async def worker(args, out):
    import httpx
    from dabs_ascend.backend import Backend, Tokenizer
    from dabs_ascend.native_client import NativeClient
    from dabs_ascend.native_probe import probe, invalidation_probe
    from dabs_ascend.native_gate import verify_live
    from dabs_ascend.native_calibration import run_calibration
    async with httpx.AsyncClient(timeout=300, trust_env=False, limits=httpx.Limits(max_connections=128, max_keepalive_connections=0)) as client:
        endpoints = ['http://127.0.0.1:' + str(8001 + r) for r in range(4)]
        # A single-replica check/probe is possible before patching the other three.
        if args.mode in ('native-check', 'native-probe-one', 'native-restart-arm', 'native-restart-check') and args.replica in (None, 0):
            endpoints = endpoints[:1]
        native = NativeClient(client, endpoints, ROOT / 'outputs/native-control.json', out / 'native_events.jsonl')
        caps = await native.inspect()
        write_json(out / 'capabilities.json', caps)
        from dabs_ascend.native_algorithms import transformation_manifest
        write_json(out / 'native_transformations.json', transformation_manifest())
        backend = Backend(client, endpoints)
        backend.native = native
        tokenizer = Tokenizer(client, endpoints[0])
        latest_file = ROOT / 'outputs/native-latest.json'
        latest = read_json(latest_file) if latest_file.exists() else {}
        if args.mode == 'native-check':
            result = {'pass': True, 'native_execution': True, 'capabilities': caps, 'inference_executed': False}
        elif args.mode in ('native-probe-one', 'native-probe-four'):
            count = 1 if args.mode.endswith('one') else 4
            result = await probe(backend, tokenizer, out, count)
            from dabs_ascend.native_gate import evaluate_probe
            result = {'checks': evaluate_probe(out, count)}
            result['pass'] = all(result['checks'].values())
        elif args.mode == 'native-calibrate':
            result = await run_calibration(backend, tokenizer, out)
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
            paths = {k: latest[k] for k in ('native-probe-one', 'native-probe-four', 'native-calibrate', 'native-invalidate', 'native-restart-check') if k in latest}
            result = await verify_live(native, paths)
            write_json(out / 'acceptance.json', result)
        latest[args.mode] = str(out)
        write_json(latest_file, latest)
        write_json(out / 'result.json', result)
        summary = {'mode': args.mode, 'output': str(out), 'pass': result.get('pass'),
                   'live_verified': result.get('live_verified', False),
                   'checks': result.get('checks'), 'missing': result.get('missing') or result.get('failures'),
                   'inference_executed': args.mode in ('native-probe-one', 'native-probe-four', 'native-calibrate', 'native-invalidate')}
        (out / 'SUMMARY.txt').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if result.get('pass') is False else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--replica', type=int, choices=range(4))
    parser.add_argument('--backup')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode not in MODES[:4] and args.replica not in (None, 0):
        parser.error('--replica 1..3 is for install/check/verify/rollback; active one-replica probes use replica 0, four-replica probes use all four')
    if not args.worker:
        return host(args)
    source_paths()
    out = ROOT / 'outputs' / (args.mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-') + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    try:
        return asyncio.run(worker(args, out))
    except Exception as exc:
        import traceback
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('NATIVE_FAILED=' + repr(exc), 'NATIVE_OUTPUT=' + str(out))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
