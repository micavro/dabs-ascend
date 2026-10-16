"""Offline, exact-source installer. Run explicitly inside ONE selected container."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import sysconfig
from .contracts import digest

PACKAGE = Path(__file__).resolve().parent
PATCHED = ('vllm/v1/engine/core.py', 'vllm/entrypoints/openai/api_server.py',
           'vllm/v1/core/sched/output.py', 'vllm_ascend/worker/worker.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_file(path, data):
    """Prepare beside the target; never leave a truncated vendor module."""
    temporary = path.with_name(path.name + '.dabs-tmp-' + secrets.token_hex(8))
    try:
        with temporary.open('xb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def patch_source(name, data):
    """Four additive hooks; no replacement of the vendor scheduler algorithms."""
    text = data.decode('utf-8')
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    if name.endswith('/engine/core.py'):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCore')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        at, new = fn.end_lineno, '\n        from dabs_native.engine import install_engine\n        install_engine(self)\n'
    elif name.endswith('/api_server.py'):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_app')
        returns = [n for n in fn.body if isinstance(n, ast.Return)]
        if len(returns) != 1:
            raise ValueError('ambiguous build_app return')
        at, new = returns[0].lineno - 1, '    from dabs_native.api import install_api\n    install_api(app)\n'
    elif name.endswith('/sched/output.py'):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SchedulerOutput')
        at, new = cls.end_lineno, '\n    dabs_batch_id: str | None = None\n    dabs_batch_meta: dict | None = None\n'
    elif name.endswith('/worker/worker.py'):
        at, new = len(lines), '\nfrom dabs_native.worker import install_worker_class\ninstall_worker_class(NPUWorker)\n'
    else:
        return data
    lines.insert(at, new)
    result = ''.join(lines).encode('utf-8')
    compile(result, name, 'exec')
    return result


def installed_path(name):
    package = name.split('/')[0]
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError('package source not available: ' + package)
    return Path(next(iter(spec.submodule_search_locations))).joinpath(*name.split('/')[1:])


def check(allow_patched=False):
    manifest = json.loads((PACKAGE / 'upstream.json').read_text())
    versions = {p: metadata.version(p) for p in manifest['versions']}
    errors, files = [], {}
    for package, values in manifest['versions'].items():
        if versions[package] not in values:
            errors.append({'package': package, 'actual_version': versions[package], 'supported': values})
    receipt_path = PACKAGE / 'installation.json'
    receipt = json.loads(receipt_path.read_text()) if allow_patched and receipt_path.exists() else {}
    for name, expected in manifest['files'].items():
        try:
            path = installed_path(name)
            actual = sha(path.read_bytes())
            accepted = receipt.get('patched_sha256', {}).get(name, expected) if allow_patched else expected
            files[name] = {'path': str(path), 'expected': accepted, 'actual': actual}
            if actual != accepted:
                errors.append({'file': name, 'expected': accepted, 'actual': actual})
        except (OSError, ValueError) as exc:
            errors.append({'file': name, 'error': type(exc).__name__})
    if allow_patched:
        for name, expected in receipt.get('adapter_sha256', {}).items():
            if sha((PACKAGE / name).read_bytes()) != expected:
                errors.append({'adapter_file': name, 'error': 'adapter_modified'})
        if not receipt.get('adapter_sha256'):
            errors.append({'error': 'installation_receipt_missing'})
    return {'compatible': not errors, 'versions': versions, 'files': files,
            'errors': errors, 'upstream': manifest['upstream'], 'installation': receipt}


def install(auth_key):
    result = check()
    if not result['compatible']:
        raise ValueError(json.dumps(result['errors']))
    target = Path(sysconfig.get_paths()['purelib']) / 'dabs_native'
    if target.exists():
        raise ValueError('adapter directory exists; verify or rollback the previous installation first')
    backup = target.parent / ('dabs_native_backup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    if backup.exists():
        raise ValueError('backup collision')
    changed = {name: (Path(result['files'][name]['path']), patch_source(name, installed_path(name).read_bytes()))
               for name in PATCHED}
    backup.mkdir()
    for name, (path, data) in changed.items():
        copy = backup / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    shutil.copytree(PACKAGE, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'installation.json', 'auth.key'))
    receipt = {'schema': 1, 'backup': str(backup), 'upstream': result['upstream'],
               'adapter_path': str(target),
               'original_sha256': {n: result['files'][n]['actual'] for n in PATCHED},
               'patched_sha256': {n: sha(d) for n, (_, d) in changed.items()},
               'paths': {n: str(p) for n, (p, _) in changed.items()},
               'adapter_sha256': {p.name: sha(p.read_bytes()) for p in target.iterdir() if p.suffix in ('.py', '.json')}}
    (target / 'auth.key').write_text(auth_key)
    os.chmod(target / 'auth.key', 0o600)
    (target / 'installation.json').write_text(json.dumps(receipt, indent=2))
    (backup / 'installation.json').write_text(json.dumps(receipt, indent=2))
    # Prepare all content before applying; on failure restore only our own writes.
    applied = []
    try:
        for name, (path, data) in changed.items():
            if sha(path.read_bytes()) != receipt['original_sha256'][name]:
                raise ValueError('source changed during installation: ' + name)
            replace_file(path, data)
            applied.append(name)
    except BaseException:
        for name in applied:
            path = changed[name][0]
            if sha(path.read_bytes()) == receipt['patched_sha256'][name]:
                replace_file(path, (backup / name).read_bytes())
        raise
    return {'installed': True, 'backup': str(backup), 'requires_manual_restart': True,
            'services_restarted': False, 'installation_digest': digest(receipt)}


def rollback(backup):
    backup = Path(backup).resolve()
    receipt = json.loads((backup / 'installation.json').read_text())
    target = Path(receipt['adapter_path']).resolve()
    if (target.name != 'dabs_native' or target.parent != backup.parent
            or not target.is_dir() or (backup / 'retired_adapter').exists()
            or Path(receipt['backup']).resolve() != backup):
        raise ValueError('unsafe adapter archive path; nothing restored')
    if any(set(receipt[k]) != set(PATCHED) for k in ('paths', 'patched_sha256', 'original_sha256')):
        raise ValueError('incomplete rollback manifest; nothing restored')
    for name, expected in receipt['patched_sha256'].items():
        path = Path(receipt['paths'][name]).resolve()
        if path != installed_path(name).resolve():
            raise ValueError('rollback target differs from installed package: ' + name)
        # A failed install may already have restored some or all vendor files.
        if sha(path.read_bytes()) not in (expected, receipt['original_sha256'][name]):
            raise ValueError('modified patched source; refusing overwrite: ' + name)
        if sha((backup / name).read_bytes()) != receipt['original_sha256'][name]:
            raise ValueError('backup integrity failure: ' + name)
    for name in receipt['paths']:
        replace_file(Path(receipt['paths'][name]), (backup / name).read_bytes())
    shutil.move(str(target), str(backup / 'retired_adapter'))
    return {'rolled_back': True, 'requires_manual_restart': True,
            'note': 'Adapter preserved under backup/retired_adapter; vendor hooks restored. No service was restarted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'verify', 'install', 'rollback'))
    parser.add_argument('--backup')
    args = parser.parse_args()
    if args.mode == 'install':
        import sys
        key = sys.stdin.read().strip()
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Supply the host-generated 32-byte hex authentication key on stdin')
        result = install(key)
    elif args.mode == 'rollback':
        result = rollback(args.backup)
    else:
        result = check(allow_patched=args.mode == 'verify')
    print(json.dumps(result, indent=2))
    return 0 if result.get('compatible', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
