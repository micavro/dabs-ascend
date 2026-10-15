"""Read-only payload executed via docker exec Python stdin. Stdlib only.

The host prepends deployment.py (server_options). No torch/vllm import, no
model loading, no environment dump, and no changes to a running server.
"""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
from pathlib import Path
import sys


def collect():
    result = {'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'packages': {}, 'processes': [], 'sources': {}}
    for name in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers', 'httpx', 'numpy'):
        try:
            result['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result['packages'][name] = None
    for path in sorted(Path('/proc').glob('[0-9]*/cmdline')):
        try:
            argv = path.read_bytes().decode('utf-8', 'replace').strip('\0').split('\0')
            if not any('vllm' in a.lower() for a in argv) or not any('--' in a for a in argv):
                continue
            result['processes'].append({'pid': int(path.parent.name),
                                        'server_options': server_options(argv)})
        except (OSError, ValueError):
            continue
    wanted = {
        'vllm': ('vllm/v1/core/sched/scheduler.py', 'vllm/v1/core/kv_cache_manager.py',
                 'vllm/v1/core/kv_cache_utils.py', 'vllm/v1/core/block_pool.py',
                 'vllm/v1/request.py', 'vllm/v1/engine/core.py', 'vllm/v1/engine/llm_engine.py',
                 'vllm/core/scheduler.py', 'vllm/entrypoints/openai/api_server.py'),
        'vllm-ascend': ('vllm_ascend/core/scheduler.py', 'vllm_ascend/core/scheduler_dynamic_batch.py',
                        'vllm_ascend/patch/platform/patch_scheduler.py', 'vllm_ascend/platform.py'),
    }
    for package, names in wanted.items():
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            continue
        for name in names:
            path = Path(dist.locate_file(name))
            if not path.is_file():
                # Editable installations may point at a checkout outside site-packages.
                # Finding a top-level package spec does not import the model runtime.
                spec = importlib.util.find_spec(name.split('/')[0])
                if spec is not None and spec.submodule_search_locations:
                    path = Path(next(iter(spec.submodule_search_locations))).joinpath(*name.split('/')[1:])
            if path.is_file() and path.stat().st_size <= 2_000_000:
                data = path.read_bytes()
                result['sources'][name] = {'sha256': hashlib.sha256(data).hexdigest(),
                                           'text': data.decode('utf-8', 'replace')}
    return result


if __name__ == '__main__':
    print(json.dumps(collect(), ensure_ascii=False))
