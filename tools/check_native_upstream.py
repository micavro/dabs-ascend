#!/usr/bin/env python3
"""Check pinned vendor sources and additive patch anchors, without vLLM/NPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_native.install import patch_source, PATCHED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline', type=Path, help='directory containing exact upstream files')
    args = p.parse_args()
    manifest = json.loads((ROOT / 'experiments/ascend/dabs_native/upstream.json').read_text())
    def verify(item):
        name, expected = item
        if args.offline:
            data = (args.offline / name).read_bytes()
        else:
            package = 'vllm-ascend' if name.startswith('vllm_ascend/') else 'vllm'
            url = 'https://raw.githubusercontent.com/vllm-project/' + package + '/' + manifest['upstream'][package] + '/' + name
            with urlopen(url, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('pinned source differs: ' + name)
        if name in PATCHED:
            patched = patch_source(name, data)
            if patched == data:
                raise ValueError('patch did not apply: ' + name)
            compile(patched, name, 'exec')
        return name
    with ThreadPoolExecutor(6) as executor:
        checked = list(executor.map(verify, manifest['files'].items()))
    print(json.dumps({'official_sources_checked': len(checked), 'patches_compiled': len(PATCHED),
                      'site_verified': False, 'NPU_used': False}, indent=2))


if __name__ == '__main__':
    main()
