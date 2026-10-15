#!/usr/bin/env python3
"""Collect selected existing services and local code without any generation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.request import build_opener, ProxyHandler
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_ascend.common import sha, write_json
from dabs_ascend.existing import container_name, deployment_records
from dabs_ascend.runtime import device_ids
from dabs_ascend.telemetry import parse_metrics, metric_pressure


def command(argv, *, stdin=None, timeout=30):
    process = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                             encoding='utf-8', timeout=timeout)
    if process.returncode:
        # Do not persist raw Docker errors/arguments which may contain secrets.
        raise RuntimeError('Read-only command failed (exit %d)' % process.returncode)
    return process.stdout


def snapshot_project(project, out):
    """Copy only project code, never outputs/.git/.env or arbitrary documents."""
    project = project.expanduser().resolve()
    if not (project / 'run_dabs_r234.sh').is_file():
        raise ValueError('Expected DABS project: ' + str(project))
    candidates = [project / n for n in ('run_ascend.sh', 'run_dabs_existing.sh', 'run_dabs_r234.sh')]
    for folder in ('experiments/ascend', 'tools', 'profiles', 'provenance'):
        # os.walk does not follow directory symlinks.
        for directory, dirs, names in os.walk(project / folder, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith('.') and 'manonly' not in d.lower() and d != '__pycache__']
            candidates.extend(Path(directory) / n for n in names if Path(n).suffix in ('.py', '.sh', '.json'))
    hashes = {}
    for path in candidates:
        if path.is_symlink() or not path.is_file() or 'manonly' in str(path).lower():
            continue
        try:
            path.resolve().relative_to(project)
        except ValueError:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        name = path.relative_to(project).as_posix()
        dest = out / 'project_source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[name] = sha(dest)
    return {'path': str(project), 'files_sha256': hashes}


def collect(args):
    devices = device_ids(args.devices)
    if len(devices) != 4:
        raise ValueError('Select exactly four services')
    names = [container_name(args.container_prefix, d) for d in devices]
    out = args.out or ROOT / 'outputs' / ('repair_probe_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    out = out.expanduser().resolve()
    if out.exists():
        raise ValueError('Output already exists; preserve it and use a new directory')
    out.mkdir(parents=True, exist_ok=False)
    facts = {'utc': datetime.now(timezone.utc).isoformat(), 'host_platform': platform.platform(),
             'devices': devices, 'generation_requests': 0, 'services_modified': False,
             'errors': [], 'services': [], 'native_mechanisms_verified': False,
             'limitations': ['Configuration is evidence, not native cache/queue/action validation.',
                             'Shared NPU use is not excluded; arrange an experiment window with other users.']}
    try:
        facts['project'] = snapshot_project(args.project, out)
    except (OSError, ValueError) as exc:
        facts['project'] = {'path': str(args.project), 'files_sha256': {}}
        facts['errors'].append({'stage': 'project_snapshot', 'error_type': type(exc).__name__})
    payload = (ROOT / 'experiments/ascend/dabs_ascend/deployment.py').read_text(encoding='utf-8') + '\n' + (ROOT / 'tools/probe_dabs_container.py').read_text(encoding='utf-8')
    opener = build_opener(ProxyHandler({}))
    for device, name in zip(devices, names):
        row = {'name': name, 'physical_device': device, 'port': args.base_port + device}
        print('INSPECT=' + name, flush=True)
        for field, operation in (
            ('deployment', lambda: deployment_records(json.loads(command(['sudo', 'docker', 'inspect', name])))[0]),
            ('runtime', lambda: json.loads(command(['sudo', 'docker', 'exec', '-i', name, args.python, '-'], stdin=payload, timeout=45))),
        ):
            try:
                row[field] = operation()
            except Exception as exc:
                row[field] = None
                facts['errors'].append({'service': name, 'stage': field, 'error_type': type(exc).__name__})
        if row.get('runtime'):
            for filename, source in row['runtime'].get('sources', {}).items():
                relative = Path(filename)
                if relative.is_absolute() or '..' in relative.parts or 'manonly' in filename.lower():
                    raise ValueError('Unsafe installed source path')
                text = source.pop('text', None)
                if text is not None:
                    target = out / 'installed_sources' / name / (filename + '.txt')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding='utf-8')
                    source['saved_file'] = target.relative_to(out).as_posix()
        for route in ('/health', '/v1/models', '/openapi.json', '/metrics'):
            try:
                with opener.open('http://127.0.0.1:%d%s' % (row['port'], route), timeout=10) as response:
                    data = response.read(4_000_001)
                if len(data) > 4_000_000:
                    raise ValueError('Response too large')
                if route == '/openapi.json':
                    spec = json.loads(data)
                    row['api_info'] = spec.get('info')
                    row['api_routes'] = sorted(spec.get('paths', {}))
                    row['openapi_sha256'] = hashlib.sha256(data).hexdigest()
                elif route == '/v1/models':
                    row['models'] = [{k: m.get(k) for k in ('id', 'max_model_len')} for m in json.loads(data)['data']]
                elif route == '/metrics':
                    row['metrics'] = parse_metrics(data.decode('utf-8', 'replace'))
                    row['reported_pressure'] = metric_pressure(row['metrics'])
                else:
                    row['health_ok'] = True
            except Exception as exc:
                detail = {'service': name, 'stage': route, 'error_type': type(exc).__name__}
                if route == '/metrics':
                    row['metrics_error'] = detail
                else:
                    facts['errors'].append(detail)
        facts['services'].append(row)
    try:
        (out / 'npu-smi.txt').write_text(command(['npu-smi', 'info']), encoding='utf-8')
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        facts['npu_smi_error_type'] = type(exc).__name__
    model = args.model.expanduser().resolve()
    facts['model'] = {'path': str(model), 'metadata_sha256': {}, 'weights': {}}
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'model.safetensors.index.json'):
        if (model / name).is_file():
            try:
                facts['model']['metadata_sha256'][name] = sha(model / name)
            except OSError as exc:
                facts['errors'].append({'stage': 'model_metadata', 'file': name, 'error_type': type(exc).__name__})
    for path in model.glob('*.safetensors'):
        facts['model']['weights'][path.name] = {'bytes': path.stat().st_size, 'sha256': None}
    facts['complete'] = not facts['errors'] and all(
        (s.get('runtime') or {}).get('packages', {}).get('vllm') and
        (s.get('runtime') or {}).get('packages', {}).get('vllm-ascend') for s in facts['services'])
    write_json(out / 'inspection.json', facts)
    summary = summary_text(facts, out)
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    (out / 'README.txt').write_text('Read-only inspection; no generation, server restart or mechanism PASS.\n'
                                  'Copy/paste SUMMARY.txt first. No archive is created and no upload is needed.\n'
                                  'Details: inspection.json; installed_sources/**/*.txt; project_source/.\n'
                                  'Docker Env and unfiltered launch commands are excluded.\n', encoding='utf-8')
    print('REPAIR_INSPECTION=' + str(out / 'inspection.json'))
    print('REPAIR_SUMMARY=' + str(out / 'SUMMARY.txt'))
    print(summary)
    print('INSPECTION_COMPLETE=' + str(bool(facts['complete'])))
    return 0 if facts['complete'] else 1


def summary_text(facts, out):
    lines = ['DABS 现场采集摘要（复制本文件文字即可，无需上传文件）',
             'UTC: ' + facts['utc'], '目录: ' + str(out),
             '被记录的项目: ' + facts['project']['path'],
             '采集完整: ' + str(bool(facts['complete'])),
             '推理请求: 0；修改/重启服务: 否；原生机制验证: 未完成',
             '主机: ' + facts['host_platform']]
    runner = out / 'project_source/experiments/ascend/dabs_ascend/r234_run.py'
    if runner.exists():
        values = re.findall(r'max_keepalive_connections\s*=\s*([^\s,)]+)', runner.read_text(encoding='utf-8'))
        lines.append('现场代码 keepalive 设置: ' + ', '.join(values))
    for service in facts['services']:
        runtime = service.get('runtime') or {}
        deployment = service.get('deployment') or {}
        lines += ['', '%s device=%s port=%s' % (service['name'], service['physical_device'], service['port']),
                  'image: ' + str(deployment.get('image_id')), 'container: ' + str(deployment.get('container_id')),
                  'packages: ' + json.dumps(runtime.get('packages', {}), ensure_ascii=False),
                  'python: ' + str(runtime.get('python')) + ' executable=' + str(runtime.get('executable')),
                  'docker_options: ' + json.dumps(deployment.get('server_options', {}), ensure_ascii=False),
                  'process_options: ' + json.dumps(runtime.get('processes', []), ensure_ascii=False),
                  'models: ' + json.dumps(service.get('models'), ensure_ascii=False),
                  'health: ' + str(service.get('health_ok', False)),
                  'reported_pressure: ' + json.dumps(service.get('reported_pressure'), ensure_ascii=False),
                  'metric_names: ' + json.dumps(sorted({m['name'] for m in service.get('metrics', [])})),
                  'installed_sources: ' + json.dumps({n: s['sha256'] for n, s in runtime.get('sources', {}).items()})]
    lines += ['', 'model: ' + json.dumps(facts['model'], ensure_ascii=False),
              'errors: ' + json.dumps(facts['errors'], ensure_ascii=False),
              'NPU 资源情况另见 npu-smi.txt；不能据此认定独占。',
              '需要引擎源码时，按指定文件复制 installed_sources 中的 .txt；不要上传或打包。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default=os.environ.get('DEVICE_IDS', '0,1,2,3'))
    parser.add_argument('--container-prefix', default=os.environ.get('CONTAINER_PREFIX', 'wushibo-qwen-small-npu'))
    parser.add_argument('--base-port', type=int, default=int(os.environ.get('BASE_PORT', '8001')))
    parser.add_argument('--python', default=os.environ.get('CONTROLLER_PYTHON', '/usr/local/python3.12.13/bin/python'))
    parser.add_argument('--model', type=Path, default=Path(os.environ.get('MODEL_PATH', '/data/model/Qwen2.5-1.5B-Instruct')))
    parser.add_argument('--project', type=Path, default=Path(os.environ.get('DABS_PREVIOUS_PROJECT', str(ROOT))))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('Run inspect on the Linux host')
    return collect(args)


if __name__ == '__main__':
    raise SystemExit(main())
