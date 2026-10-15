from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
RUNTIME = HERE.parent.parent/'runtime_local'
PROFILE = 'MUSIQUE_20261010T092711Z_be09912a'
METHODS = ('RR', 'P2C', 'Affinity', 'DAG', 'DAG-Lite', 'DAG-A', 'DAG-W', 'DAG-H', 'Ayo', 'SLL', 'Ours')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def finite(x):
    return type(x) in (int, float) and math.isfinite(x)

def own_identity():
    names = ['run.py', 'run.sh', 'run_checks.sh', 'reference/workflow_only.py']
    names += [str(p.relative_to(HERE)) for p in (HERE/'wfsched').rglob('*.py')]
    names += [str(p.relative_to(HERE)) for p in (HERE/'plans').glob('*.json')]
    return {name: sha(HERE/name) for name in sorted(set(names))}

def condition_files(folder):
    folder = Path(folder)
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and not p.is_symlink()}

def verify_files(folder, expected):
    if condition_files(folder) != expected:
        raise ValueError('condition file set/hash changed; preserve evidence: ' + str(folder))
