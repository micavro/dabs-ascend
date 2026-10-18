"""Small stdlib-only output helpers for the isolated fast experiment."""
import json
import os
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def percentile(values, q):
    if not values:
        return None
    a = sorted(values)
    j = (len(a) - 1) * q
    lo = int(j)
    return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (j - lo)


def distribution(values):
    return {'n': len(values), 'p50': percentile(values, .5), 'p95': percentile(values, .95),
            'max': max(values) if values else None}
