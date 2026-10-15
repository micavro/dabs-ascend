"""Small real-HTTP probes for tomorrow's diagnosis; no algorithm/core changes."""
import asyncio
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
import uuid

from .common import ROOT, append_json, percentile, write_json
from .repair import mechanism_gate
from .runtime import device_locks
from .telemetry import sample


def overlap(streams):
    events = sorted([(s['start'], 1) for s in streams] + [(s['finish'], -1) for s in streams])
    count = peak = 0
    for _, delta in events:
        count += delta
        peak = max(peak, count)
    return peak


async def run_case(backend, replica, prompts, budgets, label, out, seed=190500):
    """Independent requests start together, each uses a unique client ID."""
    records = []
    stopped = asyncio.Event()

    async def observe(stage):
        row = await sample(backend.client, backend.endpoints[replica], replica)
        row.update(case=label, stage=stage)
        records.append(row)
        append_json(out / 'service_metrics.jsonl', row)

    async def monitor():
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=1.)
            except asyncio.TimeoutError:
                await observe('during')

    await observe('before')
    observer = asyncio.create_task(monitor())
    nodes = [{'id': label + '-' + str(i), 'tokens': tokens, 'max_output_tokens': budget, 'fixed_workload_replay': True}
             for i, (tokens, budget) in enumerate(zip(prompts, budgets))]
    tasks = [asyncio.create_task(backend.generate(replica, node, seed, out / 'requests.jsonl', schema=None)) for node in nodes]
    try:
        streams = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        stopped.set()
        await observer
    await observe('after')
    duration = max(s['finish'] for s in streams) - min(s['start'] for s in streams)
    row = {'case': label, 'replica': replica, 'requests': len(nodes), 'output_budgets': budgets,
           'http_overlap_peak': overlap(streams), 'wall_s': duration,
           'ttft_mean_ms': statistics.mean(s['ttft_text_proxy_ms'] for s in streams),
           'ttft_p95_ms': percentile([s['ttft_text_proxy_ms'] for s in streams], .95),
           'completion_tokens': sum(s['usage']['completion_tokens'] for s in streams),
           'client_request_ids': [s['client_request_id'] for s in streams],
           'server_response_ids': [s.get('server_response_id') for s in streams],
           'prompt_sha256': [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts],
           'metrics_samples': len(records), 'metrics_available': any(r['available'] for r in records),
           'per_prefix_hit_verified': False, 'engine_applied_order': None,
           'timing_scope': 'client HTTP; not engine batch-step time'}
    for key in ('running', 'waiting', 'running_plus_2_waiting'):
        values = [r['pressure'][key] for r in records if r['pressure'][key] is not None]
        row['reported_' + key + '_peak'] = max(values) if values else None
    if len(set(row['client_request_ids'])) != len(nodes):
        raise ValueError('Duplicate client request ID')
    append_json(out / 'cases.jsonl', row)
    print('PROBE_STATE=' + json.dumps({'case': label, 'before': records[0]['pressure'], 'after': records[-1]['pressure'],
          'metrics_before': records[0]['metrics'], 'metrics_after': records[-1]['metrics']}), flush=True)
    print('PROBE_CASE=' + json.dumps({k: row[k] for k in ('case', 'replica', 'requests', 'http_overlap_peak', 'ttft_mean_ms', 'metrics_available')}), flush=True)
    return row


async def execute(args, out, rows):
    import httpx
    from .backend import Backend, Tokenizer
    from .existing import ExistingServers
    servers = ExistingServers(args, out)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=10), trust_env=False,
                                limits=httpx.Limits(max_connections=32, max_keepalive_connections=args.http_keepalive)) as client:
        try:
            await servers.start(client)
            backend = Backend(client, servers.endpoints)
            write_json(out / 'api_contract.json', await backend.inspect(fixed=True))
            reference = None
            tokenizers = [Tokenizer(client, endpoint) for endpoint in servers.endpoints]
            for tokenizer in tokenizers:
                tokens = await tokenizer.chat_tokens('DABS probe tokenizer check.', 'Continue the text.')
                if reference is not None and tokens != reference:
                    raise ValueError('Replica canonical tokenizer mismatch')
                reference = tokens
            tokenizer = tokenizers[0]
            pad = (await tokenizer.encode(' .'))[-1]
            namespace = uuid.uuid4().hex

            async def prompt(label):
                tokens = await tokenizer.chat_tokens('DABS diagnostic namespace ' + namespace + ' ' + label, 'Continue the text.')
                if len(tokens) >= 3584:
                    raise ValueError('Unexpected probe prompt length')
                return tokens + [pad] * (3584 - len(tokens))

            for replica in range(4):
                shared = await prompt('cache-r' + str(replica))
                new = await prompt('new-namespace-r' + str(replica))
                for label, target, tokens in (
                    ('first', replica, shared), ('repeat_same_prefix', replica, shared),
                    ('same_prefix_other_replica', (replica + 1) % 4, shared),
                    ('new_namespace', replica, new),
                ):
                    servers.check_alive()
                    rows.append(await run_case(backend, target, [tokens], [16], 'cache-r%d-%s' % (replica, label), out))
            for replica in range(4):
                for label, budgets in (('serial16', [16]), ('serial128', [128]),
                                       ('parallel16', [16] * 4), ('parallel128', [128] * 4),
                                       ('mixed', [16, 128, 16, 128])):
                    # Unique namespaces avoid assuming warm cache across these
                    # load probes. Mixed means mixed requested lengths, NOT proof
                    # of a native mixed prefill/decode batch.
                    prompts = [await prompt('load-r%d-%s-%d' % (replica, label, i)) for i in range(len(budgets))]
                    servers.check_alive()
                    rows.append(await run_case(backend, replica, prompts, budgets, 'load-r%d-%s' % (replica, label), out))
            servers.check_alive()
        finally:
            await servers.aclose()


def save_summary(out, rows, status, error):
    result = {'status': status, 'completed_cases': len(rows), 'planned_cases': 36,
              'completed_requests': sum(r['requests'] for r in rows), 'planned_requests': 72,
              'native_mechanism_pass': False, 'error': error, 'cases': rows,
              'limits': ['Global cache metrics do not prove per-prefix residency.',
                         'HTTP overlap does not prove native batch concurrency.',
                         'Shared hardware/background load can affect timings.',
                         'This probe is not a calibrated native phase model or a formal experiment.']}
    write_json(out / 'probe_summary.json', result)
    lines = ['DABS 小探针摘要（复制文字即可；不打包、不上传）',
             '目录: ' + str(out), '状态: ' + status,
             '完成 %d/36 组、%d/72 请求；原生机制验证: 未完成' % (len(rows), result['completed_requests']),
             '全局 cache 指标/HTTP 时延不能证明逐前缀命中或引擎队列应用。',
             'case | replica | requests | HTTP overlap | TTFT mean ms | reported running/waiting peak']
    for r in rows:
        lines.append('%s | %s | %s | %s | %.2f | %s/%s' % (r['case'], r['replica'], r['requests'],
                     r['http_overlap_peak'], r['ttft_mean_ms'], r['reported_running_peak'], r['reported_waiting_peak']))
    if error:
        lines.append('错误: ' + error)
    lines += ['详细接口观测: service_metrics.jsonl；请求: requests.jsonl；环境: environment.json',
              '不要依据单次冷热时延差异宣称缓存已接通；逐前缀接口仍需匹配现场版本。']
    (out / 'PROBE_SUMMARY.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print('PROBE_SUMMARY=' + str(out / 'PROBE_SUMMARY.txt'), flush=True)


def main(args):
    if args.service_mode != 'existing' or args.resume or args.max_conditions:
        raise ValueError('probe uses existing services, no resume/max-conditions')
    out = (args.out or ROOT / 'outputs' / ('r234_probe_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])).resolve()
    out.mkdir(parents=True, exist_ok=False)
    from .r234_run import Tee
    with (out / 'console.log').open('a', encoding='utf-8', buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            return run_recorded(args, out)


def run_recorded(args, out):
    rows, status, error = [], 'failed', None
    from .common import source_identity
    from .existing import preflight
    write_json(out / 'probe_protocol.json', {'source_sha256': source_identity(), 'devices': args.devices,
               'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
               'host_project': os.environ.get('DABS_HOST_PROJECT'), 'host_command': os.environ.get('DABS_HOST_COMMAND'),
               'request_timeout_s': args.request_timeout,
               'plan': '16 prefix-repeat checks + 56 serial/parallel length requests; 36 cases total',
               'http_keepalive': args.http_keepalive, 'max_simultaneous_requests': 4,
               'server_configuration_changed': False, 'formal_matrix_started': False,
               'native_gate': mechanism_gate()})
    (out / 'REPRODUCE.md').write_text('# 小探针执行记录\n\n'
        '命令：`' + (os.environ.get('DABS_HOST_COMMAND') or ' '.join(sys.argv)) + '`\n\n'
        '入口：run_dabs_r234.sh → run_dabs_existing.sh → tools/run_dabs_existing.py → dabs_ascend.r234_probe。\n'
        '预检：existing.py；tokenizer/生成：backend.py；只读服务指标：telemetry.py。\n'
        '四服务共 36 组、72 个请求；原始请求、状态采样、摘要和控制台均保存在本目录。\n'
        '容器 /project 对应宿主机项目目录；宿主机路径和源文件哈希见 probe_protocol.json。\n'
        '只复制 PROBE_SUMMARY.txt 及需要的日志文字，不打包或上传。\n', encoding='utf-8')
    try:
        with device_locks(args.devices, directory=ROOT / 'outputs/.existing_controller_locks'):
            preflight(args, out)
            asyncio.run(execute(args, out, rows))
            status = 'complete'
    except BaseException as exc:
        error = repr(exc)
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        save_summary(out, rows, status, error)
        write_json(out / 'status.json', {'status': status, 'error': error})
    return 0 if status == 'complete' else 1
