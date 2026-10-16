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
import time
from urllib.error import HTTPError, URLError
from urllib.request import build_opener, ProxyHandler
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_ascend.common import append_json, configuration, sha, write_json
from dabs_ascend.deployment import resource_comparison
from dabs_ascend.existing import container_name, deployment_records, validate_manifest
from dabs_ascend.runtime import device_ids
from dabs_ascend.telemetry import parse_metrics, metric_pressure


class CommandFailure(RuntimeError):
    def __init__(self, code, category):
        super().__init__('Read-only command failed')
        self.code, self.category = code, category


def command(argv, *, stdin=None, timeout=30, include_stderr=False):
    process = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', timeout=timeout)
    if process.returncode:
        # Do not persist raw Docker errors/arguments which may contain secrets.
        error = process.stderr.lower()
        category = next((label for text, label in (
            ('is not running', 'container_not_running'),
            ('no such object', 'container_not_found'), ('no such container', 'container_not_found'),
            ('executable file not found', 'python_not_found'),
            ('no such file or directory', 'executable_or_file_not_found'),
            ('permission denied', 'permission_denied'), ('password is required', 'sudo_authentication_required'),
            ('cannot connect to the docker daemon', 'docker_daemon_unavailable'),
        ) if text in error), 'command_failed')
        raise CommandFailure(process.returncode, category)
    # docker logs can write server output to stderr even when exit status is 0.
    return process.stdout + ('\n' + process.stderr if include_stderr else '')


def error_detail(exc):
    result = {'error_type': type(exc).__name__}
    if isinstance(exc, CommandFailure):
        result.update(exit_code=exc.code, reason=exc.category)
    elif isinstance(exc, HTTPError):
        result.update(http_status=exc.code, reason='http_error')
    else:
        cause = exc.reason if isinstance(exc, URLError) else exc
        result['reason'] = type(cause).__name__
        if isinstance(cause, OSError) and cause.errno is not None:
            result['errno'] = cause.errno
    return result


def wait_services(args, names, out, opener):
    """Poll only selected containers; never start or restart an external service."""
    started = time.monotonic()
    deadline = started + args.wait_ready
    while True:
        services = []
        for device, name in zip(device_ids(args.devices), names):
            row = {'name': name, 'physical_device': device, 'port': args.base_port + device,
                   'health_ok': False}
            try:
                row['deployment'] = deployment_records(json.loads(command(['sudo', 'docker', 'inspect', name])))[0]
                dep = row['deployment']
                state = dep['state']
                if dep['running'] is not True or state.get('Paused') or state.get('Restarting') or state.get('Dead'):
                    row['readiness'] = 'container_not_ready'
                else:
                    from types import SimpleNamespace
                    validate_manifest([dep], SimpleNamespace(devices=[device], base_port=args.base_port,
                                                            container_prefix=args.container_prefix))
                    row['readiness'] = 'waiting_for_health'
                    try:
                        timeout = min(2., max(.1, deadline - time.monotonic())) if args.wait_ready else 2.
                        with opener.open('http://127.0.0.1:%d/health' % row['port'], timeout=timeout) as response:
                            response.read(1024)
                        row.update(health_ok=True, readiness='ready')
                    except Exception as exc:
                        row['health_error'] = error_detail(exc)
            except Exception as exc:
                row['readiness'] = 'deployment_unavailable_or_invalid'
                row['deployment_error'] = error_detail(exc)
            services.append(row)
        elapsed = round(time.monotonic() - started, 3)
        event = {'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': elapsed, 'services': services}
        append_json(out / 'readiness.jsonl', event)
        print('READINESS %.1fs %s' % (elapsed, ' '.join('%s=%s' % (r['name'], r['readiness']) for r in services)), flush=True)
        if all(r['health_ok'] for r in services):
            break
        if any(r['readiness'] not in ('ready', 'waiting_for_health') for r in services):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(min(args.poll_interval, max(0., deadline - time.monotonic())))
    return services


def filtered_logs(text):
    """Bounded diagnostic excerpts; omit request payloads and credential lines."""
    lines = []
    sensitive = re.compile(r'api[_ -]?key|authorization|bearer\s|password|secret|access[_ -]?token|'
                           r'prompt[\"\x27]?\s*[:=]|completion[\"\x27]?\s*[:=]|request[_ ]body', re.I)
    wanted = re.compile(r'error|exception|traceback|failed|killed|fatal|out of memory|loading|loaded|'
                        r'initializ|starting|started|startup|listening|uvicorn|engine|cache|compil|'
                        r'graph captur|HCCL|CANN|version', re.I)
    for line in text.splitlines():
        line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line)
        if sensitive.search(line):
            lines.append('[credential or request-content line omitted]')
        elif wanted.search(line):
            lines.append(line[:1200])
    return lines


def collect_failure_logs(row, out, tail):
    dep = row.get('deployment') or {}
    if not dep or tail == 0:
        return
    started = (dep.get('state') or {}).get('StartedAt') or ''
    # A validated Docker timestamp limits logs to this launch, not yesterday's run.
    since = started if re.fullmatch(r'20\d\d-\d\d-\d\dT[0-9:.]+Z', started) else '30m'
    try:
        raw = command(['sudo', 'docker', 'logs', '--timestamps', '--since', since,
                       '--tail', str(tail), row['name']], include_stderr=True)
        lines = filtered_logs(raw)
        relative = 'service_logs/' + row['name'] + '.txt'
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Filtered recent diagnostics, since=' + since + '\n'
                        'Not a complete log; omitted lines may contain request data or credentials.\n'
                        + '\n'.join(lines) + '\n', encoding='utf-8')
        row['recent_logs'] = {'file': relative, 'since': since, 'filtered': True,
                              'requested_tail': tail, 'excerpt': lines[-12:]}
    except Exception as exc:
        row['logs_error'] = error_detail(exc)


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
             'readiness_policy': {'wait_seconds': args.wait_ready, 'poll_interval': args.poll_interval},
             'limitations': ['Configuration is evidence, not native cache/queue/action validation.',
                             'Shared NPU use is not excluded; arrange an experiment window with other users.']}
    try:
        facts['project'] = snapshot_project(args.project, out)
    except (OSError, ValueError) as exc:
        facts['project'] = {'path': str(args.project), 'files_sha256': {}}
        facts['errors'].append({'stage': 'project_snapshot', 'error_type': type(exc).__name__})
    payload = (ROOT / 'experiments/ascend/dabs_ascend/deployment.py').read_text(encoding='utf-8') + '\n' + (ROOT / 'tools/probe_dabs_container.py').read_text(encoding='utf-8')
    opener = build_opener(ProxyHandler({}))
    services = wait_services(args, names, out, opener)
    for row in services:
        name = row['name']
        print('INSPECT=' + name, flush=True)
        if not row['health_ok']:
            facts['errors'].append({'service': name, 'stage': 'readiness', 'readiness': row['readiness'],
                                    **row.get('health_error', row.get('deployment_error', {}))})
            collect_failure_logs(row, out, args.log_tail)
        dep = row.get('deployment') or {}
        row['resource_comparison'] = resource_comparison(dep.get('server_options', {}),
                                                         configuration(4)['base_cap'], args.max_inflight)
        row['runtime'] = None
        # A live but unhealthy server can still provide package/source evidence.
        if dep.get('running') and row['readiness'] in ('ready', 'waiting_for_health'):
            try:
                row['runtime'] = json.loads(command(['sudo', 'docker', 'exec', '-i', name, args.python, '-'], stdin=payload, timeout=45))
            except Exception as exc:
                facts['errors'].append({'service': name, 'stage': 'runtime', **error_detail(exc)})
        else:
            row['runtime_skipped'] = 'Selected container is not available for exec; no interpreter failure inferred.'
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
        for route in ('/v1/models', '/openapi.json', '/metrics') if row['health_ok'] else ():
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
                detail = {'service': name, 'stage': route, **error_detail(exc)}
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
                                  'Startup diagnostics: readiness.jsonl; service_logs/*.txt (filtered).\n'
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
                  'state: ' + json.dumps(deployment.get('state'), ensure_ascii=False),
                  'restart_count: ' + str(deployment.get('restart_count')),
                  'readiness: ' + str(service.get('readiness')),
                  'packages: ' + json.dumps(runtime.get('packages', {}), ensure_ascii=False),
                  'python: ' + str(runtime.get('python')) + ' executable=' + str(runtime.get('executable')),
                  'docker_options: ' + json.dumps(deployment.get('server_options', {}), ensure_ascii=False),
                  'process_options: ' + json.dumps(runtime.get('processes', []), ensure_ascii=False),
                  'resource_comparison: ' + json.dumps(service.get('resource_comparison'), ensure_ascii=False),
                  'models: ' + json.dumps(service.get('models'), ensure_ascii=False),
                  'health: ' + str(service.get('health_ok', False)),
                  'reported_pressure: ' + json.dumps(service.get('reported_pressure'), ensure_ascii=False),
                  'metric_names: ' + json.dumps(sorted({m['name'] for m in service.get('metrics', [])})),
                  'installed_sources: ' + json.dumps({n: s['sha256'] for n, s in runtime.get('sources', {}).items()})]
        if service.get('runtime_skipped'):
            lines.append('runtime_skipped: ' + service['runtime_skipped'])
        if service['resource_comparison']['step_budget_matches'] is False:
            lines.append('配置差异：服务端批次 token 上限与控制器假设不同；未自动改动服务或算法参数。')
        logs = service.get('recent_logs')
        if logs:
            lines.append('recent_logs: %s since=%s（已过滤，只列末尾 12 行）' % (logs['file'], logs['since']))
            lines.extend(logs['excerpt'] or ['（最近日志中未提取到诊断行；不能据此判断启动成功）'])
        if service.get('logs_error'):
            lines.append('logs_error: ' + json.dumps(service['logs_error']))
        if deployment.get('state', {}).get('ExitCode') == 137:
            lines.append('退出码 137 的原因未确定；OOM=false 不能单独排除其他进程或系统层内存问题。')
    lines += ['', 'model: ' + json.dumps(facts['model'], ensure_ascii=False),
              'errors: ' + json.dumps(facts['errors'], ensure_ascii=False),
              'NPU 资源情况另见 npu-smi.txt；不能据此认定独占。',
              '需要引擎源码时，按指定文件复制 installed_sources 中的 .txt；不要上传或打包。']
    if not all(s['health_ok'] for s in facts['services']):
        lines += ['下一步：先复制本 SUMMARY.txt；尚未开始 72 请求探针。',
                  '已退出/暂停/重启中的容器需先处理服务状态；运行但未就绪的服务需结合本次启动日志判断。',
                  '本工具不会自动启动容器，也不会把旧缓存命中日志作为当前服务就绪证明。']
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
    parser.add_argument('--wait-ready', type=int, default=0, help='Health wait in seconds; inspect defaults to 0, prepare to 300')
    parser.add_argument('--poll-interval', type=float, default=5.)
    parser.add_argument('--log-tail', type=int, default=80, help='Recent failed-service log lines to filter; 0 disables')
    parser.add_argument('--max-inflight', type=int, default=int(os.environ.get('MAX_INFLIGHT', '2')))
    args = parser.parse_args()
    if not 0 <= args.wait_ready <= 3600 or not 1 <= args.poll_interval <= 30 or not 0 <= args.log_tail <= 500 or args.max_inflight < 1:
        parser.error('Invalid wait/poll/log/client limit')
    if platform.system() != 'Linux':
        parser.error('Run inspect on the Linux host')
    return collect(args)


if __name__ == '__main__':
    raise SystemExit(main())
