"""Capability-based installer with reversible backups. Run explicitly inside ONE selected container."""
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
    """Idempotent additive hooks; validate the actual insertion points."""
    text = data.decode('utf-8')
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    def named(nodes, kind, value):
        found = [n for n in nodes if isinstance(n, kind) and n.name == value]
        if len(found) != 1:
            raise ValueError('missing/ambiguous native hook: ' + name + ':' + value)
        return found[0]
    def hooked(node, function, argument):
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == function
                   and len(n.args) == 1 and isinstance(n.args[0], ast.Name) and n.args[0].id == argument
                   for n in ast.walk(node))
    if name.endswith('/engine/core.py'):
        cls = named(tree.body, ast.ClassDef, 'EngineCore')
        fn = named(cls.body, ast.FunctionDef, '__init__')
        if hooked(fn, 'install_engine', 'self'):
            return data
        at, new = fn.end_lineno, '\n        from dabs_native.engine import install_engine\n        install_engine(self)\n'
    elif name.endswith('/api_server.py'):
        fn = named(tree.body, ast.FunctionDef, 'build_app')
        returns = [n for n in fn.body if isinstance(n, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Name) or returns[0].value.id != 'app':
            raise ValueError('ambiguous build_app return: require one direct return app')
        if hooked(fn, 'install_api', 'app'):
            return data
        at, new = returns[0].lineno - 1, '    from dabs_native.api import install_api\n    install_api(app)\n'
    elif name.endswith('/sched/output.py'):
        cls = named(tree.body, ast.ClassDef, 'SchedulerOutput')
        fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        new = ''.join('\n    ' + key + ': ' + annotation + ' = None\n' for key, annotation in
                      [('dabs_batch_id', 'str | None'), ('dabs_batch_meta', 'dict | None')] if key not in fields)
        if not new:
            return data
        at = cls.end_lineno
    elif name.endswith('/worker/worker.py'):
        cls = named(tree.body, ast.ClassDef, 'NPUWorker')
        named(cls.body, ast.FunctionDef, 'execute_model')
        if hooked(tree, 'install_worker_class', 'NPUWorker'):
            return data
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
    versions, notes, files, errors = {}, [], {}, []
    for package, expected in manifest['versions'].items():
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
        if versions[package] not in expected:
            notes.append({'package': package, 'actual_version': versions[package], 'reference_versions': expected})
    # Only the four hook targets are needed for installation. Do not recapture
    # the whole environment or treat diagnostic fingerprints as compatibility.
    for name in PATCHED:
        try:
            path = installed_path(name)
            data = path.read_bytes()
            patched = patch_source(name, data)
            files[name] = {'path': str(path), 'actual': sha(data), 'reference': manifest['files'].get(name),
                           'hooks_present': patched == data}
            if sha(data) != manifest['files'].get(name):
                notes.append({'file': name, 'note': 'source differs from archived reference; hook structure checked'})
        except (OSError, ValueError, SyntaxError, UnicodeError) as exc:
            errors.append({'file': name, 'missing_capability': str(exc)})
    receipt_path = PACKAGE / 'installation.json'
    receipt = json.loads(receipt_path.read_text()) if allow_patched and receipt_path.exists() else {}
    return {'compatible': not errors, 'versions': versions, 'files': files, 'notes': notes,
            'errors': errors, 'upstream': manifest['upstream'], 'installation': receipt}


def install():
    result = check()
    if not result['compatible']:
        raise ValueError(json.dumps(result['errors']))
    target = Path(sysconfig.get_paths()['purelib']) / 'dabs_native'
    backup = target.parent / ('dabs_native_backup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    changed = {name: (Path(result['files'][name]['path']), installed_path(name).read_bytes()) for name in PATCHED}
    patched = {name: patch_source(name, original) for name, (_, original) in changed.items()}
    backup.mkdir()
    for name, (path, original) in changed.items():
        copy = backup / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(original)
    if target.exists():
        shutil.copytree(target, backup / 'previous_adapter')
    staged = backup / 'new_adapter'
    # Copy only shipped module/manifest files, never deployment-local state.
    staged.mkdir()
    shipped = [p for p in PACKAGE.iterdir() if p.suffix == '.py' or p.name == 'upstream.json']
    for source in shipped:
        shutil.copy2(source, staged / source.name)
    # Preserve the engine epoch across upgrades for restart-invalidity probes.
    if (target / 'engine_epoch.json').exists():
        shutil.copy2(target / 'engine_epoch.json', staged / 'engine_epoch.json')
    receipt = {'schema': 2, 'backup': str(backup), 'upstream': result['upstream'], 'versions': result['versions'],
               'adapter_path': str(target), 'original_sha256': {n: sha(d) for n, (_, d) in changed.items()},
               'patched_sha256': {n: sha(d) for n, d in patched.items()},
               'paths': {n: str(p) for n, (p, _) in changed.items()},
               'adapter_sha256': {p.name: sha(p.read_bytes()) for p in shipped}}
    for location in (staged, backup):
        (location / 'installation.json').write_text(json.dumps(receipt, indent=2))
    applied = []
    displaced = backup / 'displaced_adapter'
    try:
        if target.exists():
            target.rename(displaced)
        staged.rename(target)
        for name, (path, original) in changed.items():
            # Race protection only, not a version/hash allowlist.
            if path.read_bytes() != original:
                raise ValueError('source changed DURING installation: ' + name + '; retry after concurrent editing stops')
            replace_file(path, patched[name])
            applied.append(name)
    except BaseException:
        for name in applied:
            path = changed[name][0]
            if path.read_bytes() == patched[name]:
                replace_file(path, changed[name][1])
        if target.exists():
            target.rename(backup / 'failed_adapter')
        if displaced.exists():
            displaced.rename(target)
        raise
    return {'installed': True, 'backup': str(backup), 'requires_manual_restart': True,
            'services_restarted': False, 'provenance': result['notes']}


def rollback(backup):
    backup = Path(backup).resolve()
    receipt = json.loads((backup / 'installation.json').read_text())
    target = Path(receipt['adapter_path']).resolve()
    if (target.name != 'dabs_native' or target.parent != backup.parent or not target.is_dir()
            or (backup / 'retired_adapter').exists() or Path(receipt['backup']).resolve() != backup):
        raise ValueError('unsafe adapter archive path; nothing restored')
    if set(receipt['paths']) != set(PATCHED):
        raise ValueError('incomplete rollback paths; nothing restored')
    for name, value in receipt['paths'].items():
        if Path(value).resolve() != installed_path(name).resolve():
            raise ValueError('rollback target differs from installed package: ' + name)
        if not (backup / name).is_file():
            raise ValueError('missing rollback source: ' + name)
    # Preserve current edits before restoring, irrespective of their hashes.
    for name, value in receipt['paths'].items():
        saved = backup / 'before_rollback' / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(value, saved)
        replace_file(Path(value), (backup / name).read_bytes())
    target.rename(backup / 'retired_adapter')
    if (backup / 'previous_adapter').is_dir():
        shutil.copytree(backup / 'previous_adapter', target)
    return {'rolled_back': True, 'requires_manual_restart': True,
            'note': 'Current sources and adapter preserved in backup; previous deployment restored. No service restarted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'verify', 'install', 'rollback'))
    parser.add_argument('--backup')
    args = parser.parse_args()
    if args.mode == 'install':
        result = install()
    elif args.mode == 'rollback':
        result = rollback(args.backup)
    else:
        result = check(allow_patched=args.mode == 'verify')
    print(json.dumps(result, indent=2))
    return 0 if result.get('compatible', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
