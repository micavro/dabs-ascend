"""Measured R234 tables and an offline browser view; no invented native data."""
from __future__ import annotations

import csv
import html
import io
import statistics

from .common import read_json, write_json
from .r234 import LIMITS, METHODS


def write_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding='utf-8-sig')


def paired_rows(summaries):
    by_key = {(s['spec']['replicas'], s['spec']['rate'], s['spec']['repeat'], s['spec']['method']): s for s in summaries}
    rows = []
    for (replicas, rate, repeat, method), ours in by_key.items():
        if method != 'Ours':
            continue
        for base in METHODS[:-1]:
            reference = by_key.get((replicas, rate, repeat, base))
            if reference is None:
                continue
            same = (reference['external_work_fingerprint'] == ours['external_work_fingerprint']
                    and reference['arrival_offsets'] == ours['arrival_offsets'])
            gain = ((reference['jct_mean_ms'] - ours['jct_mean_ms']) / reference['jct_mean_ms'] * 100) if same else None
            rows.append({'replicas': replicas, 'rate': rate, 'repeat': repeat, 'reference': base,
                         'paired_work_pass': same, 'ours_jct_ms': ours['jct_mean_ms'],
                         'reference_jct_ms': reference['jct_mean_ms'], 'ours_jct_reduction_pct': gain})
    return rows


def procedure(out, record):
    if record['arguments'].get('ours'):
        (out/'REPRODUCE.md').write_text(
            '# Ours 正常限时运行\n\n1500ms 周期/快照，1200ms 规划；有效动作经引擎检查后实际执行。\n'
            '仅运行原计划 Ours 子序列；不是四方法性能比较。详细汇总见 test_result.txt。\n'
            '命令：' + (record['host_command'] or ' '.join(record['argv'])) + '\n', encoding='utf-8')
        return
    if record['arguments'].get('ours_timing'):
        (out/'REPRODUCE.md').write_text(
            '# Ours default 实时负载 + 无预算后台规划计时\n\n'
            '复用指定预测模型；不运行探针、canary、重标定或规则自检。实际请求仅执行 Ours default。\n'
            '后台规划与评分无 CPU 截止；过期或提前返回的规划动作均不应用；只保留一个后台任务，不排队。\n'
            '完整记录 conditions/*/decisions.jsonl；汇总 shadow_timing_summary.json 与 COPY.txt。\n'
            '命令：' + (record['host_command'] or ' '.join(record['argv'])) + '\n', encoding='utf-8')
        return
    lines = ['# 本次复现过程', '', '本文件和 process.jsonl 自动生成；运行结束后保留整个输出目录。', '',
             f"- 本次调用开始（UTC）：{record['utc']}",
             f"- 主机项目目录：`{record['host_project'] or record['cwd']}`",
             f"- 容器输出目录：`{out}`",
             '- 容器 `/project/outputs/` 对应主机项目的 `outputs/`。',
             f"- Git 提交：`{record['git_head'] or 'ZIP 下载无 Git 元数据；以 protocol.json 中逐文件 SHA256 为准'}`",
             f"- 运行命令：`{record['host_command'] or ' '.join(record['argv'])}`",
             '- 完整参数、Python 版本、镜像 ID 与所有调用历史：`process.jsonl`。', '',
             '## 实际执行链', '',
             '1. `run_dabs_r234.sh` → `run_dabs_existing.sh`：读取已选服务的 Docker 映射，启动临时 CPU 控制容器。',
             '2. `tools/inspect_dabs_existing.py`：将服务、卡号、端口和启动选项保存到主机 `outputs/deployment_existing_*.json`。',
             '3. `tools/run_dabs_existing.py` → `dabs_ascend.cli` → `dabs_ascend.r234_run`：校验冻结源码、模型、环境。',
             '4. `dabs_ascend.backend`：逐服务验证 3584 输入以及 16/32/48/64/128 输出，随后 `calibration.py` 采集本机 HTTP 时延。',
             '5. 新包 `wfsched/design.py` 提供 96 条件；`heterogeneous.py` 选取 balanced 工作流；`fixed_work.py` 填充固定输入。',
             '6. `dabs_ascend.runner` 在 DAG 依赖满足后提交 HTTP 请求；`r234.py` 调用新包 RR/SLL、Ayo、V5 逻辑。',
             '7. `dabs_ascend.r234_report` 每完成一个条件更新汇总、配对比较、网页与 CSV。', '',
             '## 保存位置及查看顺序', '',
             '- `TEST_REPORT.md`：完成数量、固定工作量检查、方法差异和比较结论。',
             '- `RESULTS.html`：离线打开的指标表与曲线；只有实测完成条件才显示数据。',
             '- `metrics.csv`、`paired_comparisons.csv`：逐条件指标与同 R/到达率/轮次的 Ours 对照差值。',
             '- `workflow_metrics.csv`、`latency_decomposition.csv`、`replica_load.csv`：每个工作流、客户端关键路径与副本 HTTP 负载。',
             '- `protocol.json`：固定实验计划、运行参数、源码 SHA256；`workload.json`：24 个源样本的重加权序列。',
             '- `environment.json`、`servers.json`、`api_contract.json`：现场环境、物理映射和接口。',
             '- `console.log`、`process.jsonl`：控制台、调用时间、条件开始结束、执行模块与输出路径。',
             '- `conditions/<条件>/`：`spec.json`、`config.json`、`initial_nodes.json`、`input_proof.json`、`requests.jsonl`、`decisions.jsonl`、`nodes.json`、`quality.json`、`summary.json`、`status.json`、`artifacts.json`。',
             '- 失败或中断：查看 `error.txt` 和最后一个条件的 `status.json`；原始过程保留，不自动重试或跳过。', '',
             '## 结果口径', '',
             '96/96 且检查通过才是完整跑完。`--max-conditions 4` 只完成前四个集成条件，状态为 paused。',
             'Ours 收益 = (基线平均 JCT − Ours 平均 JCT) / 基线平均 JCT；正值为改善，负值为退化。',
             '这是固定 token DAG 负载，无问答准确率。原生 KV 和引擎阶段指标不可得，保持 null。', '']
    if record['arguments'].get('execution_scope') == 'native':
        lines[lines.index('4. `dabs_ascend.backend`：逐服务验证 3584 输入以及 16/32/48/64/128 输出，随后 `calibration.py` 采集本机 HTTP 时延。')] = '4. 原生门禁重新计算原始事件、核对引擎实例、配置和独立标定；加载冻结模型，计时口径见 protocol.json，正式数据不参与拟合。'
        lines[-2] = '原生状态、KV 查询、引擎回执和带 batch ID 的引擎 host-step 与辅助 NPU forward 时间保存在 native_events.jsonl；HTTP TTFT/TPOT 仍只标作客户端代理。'
    if record['arguments'].get('screening'):
        from .r234 import screening_plan
        count = len(screening_plan(record['arguments']['screening']))
        lines = [f'本次为诊断筛查，共 {count} 条；工作量完成与规划完整评分独立判断，不拼入或续接正式矩阵。'
                 if text.startswith('96/96 且检查通过') else text for text in lines]
        lines[2:2] = ['诊断筛查：' + record['arguments']['screening'] + '。仅运行预定 T1 四方法组，不拼入或续接 96 条矩阵；使用 run_native.sh screen 时，完整评分结论另见上级工作流目录 planning_audit.json。', '']
    (out / 'REPRODUCE.md').write_text('\n'.join(lines), encoding='utf-8')


def chart(summaries, metric, title):
    # Standalone SVG, no plotting dependencies on the offline controller.
    pieces = []
    colors = ('#3366cc', '#d97706', '#059669', '#c026d3')
    for replicas in (2, 3, 4):
        points = {}
        for method in METHODS:
            for rate in (4, 8, 12, 16):
                values = [s[metric] for s in summaries if s['spec']['replicas'] == replicas and s['spec']['method'] == method and s['spec']['rate'] == rate]
                if values:
                    points[(method, rate)] = statistics.mean(values)
        if not points:
            continue
        top = max(points.values()) * 1.1 or 1
        svg = [f'<svg viewBox="0 0 600 300" role="img" aria-label="R{replicas} {html.escape(title)}">',
               f'<text x="60" y="22">R={replicas} {html.escape(title)}（已完成轮次均值）</text>',
               '<path d="M60 40 V245 H560" fill="none" stroke="#777"/>']
        for j in range(5):
            value = top * j / 4
            y = 245 - 190 * j / 4
            svg += [f'<text x="4" y="{y:.1f}" font-size="11">{value:.1f}</text>',
                    f'<path d="M60 {y:.1f} H560" stroke="#ddd"/>']
        for method, color in zip(METHODS, colors):
            line = []
            for rate in (4, 8, 12, 16):
                if (method, rate) not in points:
                    continue
                x, y = 80 + (rate - 4) * 38, 245 - points[(method, rate)] / top * 190
                line.append(f'{x:.1f},{y:.1f}')
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{method} λ={rate}: {points[(method, rate)]:.2f}</title></circle>')
            svg.append(f'<polyline points="{" ".join(line)}" fill="none" stroke="{color}"/>')
        svg.extend(f'<text x="{80 + (rate - 4) * 38}" y="265">{rate}</text>' for rate in (4, 8, 12, 16))
        svg += ['<text x="210" y="290">到达率（工作流/秒）</text>', '</svg>']
        pieces.append(''.join(svg))
    return ''.join(pieces)


def make_report(out, plan, summaries, status, error=None, *, source_out=None):
    source_out = source_out or out
    protocol = read_json(source_out / 'protocol.json') if (source_out / 'protocol.json').exists() else {}
    if protocol.get('ours_timing'):
        from .native_shadow_timing import make_report as timing_report
        return timing_report(out, plan, summaries, status, error)
    if protocol.get('ours_only'):
        from .native_ours_report import make_report as ours_report
        return ours_report(out, plan, summaries, status, error)
    screening = protocol.get('screening')
    from .r234 import screening_plan
    expected = len(screening_plan(screening)) if screening else 96
    plan_valid = plan == screening_plan(screening) if screening else len(plan) == 96
    env = read_json(source_out / 'environment.json') if (source_out / 'environment.json').exists() else {}
    canaries = read_json(source_out / 'canary_summary.json') if (source_out / 'canary_summary.json').exists() else []
    native_gate = read_json(source_out / 'native_gate.json') if (source_out / 'native_gate.json').exists() else {}
    exploratory = native_gate.get('exploratory') is True
    native = bool(native_gate.get('pass') and native_gate.get('live_verified'))
    observed_native = native or bool(exploratory and native_gate.get('mechanisms_live_verified'))
    profile = read_json(source_out / 'ascend_profile.json') if (source_out / 'ascend_profile.json').exists() else {}
    canary_ok = (len(canaries) == 20 and all(r['passed'] for r in canaries)
                 and {(r['replica'], r['fixed_output_budget']) for r in canaries} == {(r, n) for r in range(4) for n in (16, 32, 48, 64, 128)})
    rows, workflows, stages, loads, reasons = [], [], [], [], []
    keys = ('completed_workflows', 'completed_requests', 'input_tokens', 'output_tokens', 'jct_mean_ms', 'jct_p95_ms',
            'workflows_s', 'output_tokens_s', 'ttft_text_proxy_mean_ms', 'ttft_text_proxy_p95_ms', 'tpot_text_proxy_mean_ms',
            'condition_wall_s', 'fixed_work_pass', 'dependency_handoff_pass', 'v5_decisions',
            'v5_candidate_generated', 'v5_candidate_scored', 'selector_scored_decisions',
            'nonreference_actions', 'expired_plan_count', 'busy_planner_count', 'planner_mean_wall_ms')
    for s in summaries:
        spec = {k: s['spec'][k] for k in ('label', 'replicas', 'rate', 'repeat', 'method')}
        rows.append({**spec, **{k: s.get(k) for k in keys}, 'native_kv_hit_rate': None,
                     'native_prefill_ms': None, 'native_decode_ms': None})
        diagnostics = s.get('score_diagnostics', {})
        rows[-1].update({k: diagnostics.get(k) for k in ('reference_score_started', 'reference_score_rankable',
                                                        'candidate_score_started', 'candidate_score_rankable', 'guard_rejected',
                                                        'native_atomic_fallback_decisions', 'native_deferred_admissions')})
        rows[-1].update(client_applied_nonreference_actions=s.get('nonreference_actions') if s.get('action_accounting_version') == 2 else None,
                        engine_applied_nonreference_actions=s.get('engine_applied_nonreference_actions'),
                        action_accounting_version=s.get('action_accounting_version', 1))
        for kind in ('score_failure_reasons', 'commit_reject_reasons', 'native_fallback_reasons',
                     'planner_stop_reasons', 'candidate_generation_reasons'):
            reasons.extend({**spec, 'stage': kind, 'reason': reason, 'count': count}
                           for reason, count in diagnostics.get(kind, {}).items())
        for w in s['workflow_rows']:
            workflows.append({**spec, **{k: v for k, v in w.items() if k != 'critical_path_ms'}})
            stages.append({**spec, 'wid': w['wid'], 'jct_ms': w['jct_ms'], **w['critical_path_ms']})
        loads.extend({**spec, **r} for r in s.get('replica_rows', []))
    pairs = paired_rows(summaries)
    count_ok = plan_valid and len(summaries) == expected and [s['spec'] for s in summaries] == plan
    work_ok = bool(summaries) and all(s['functional_pass'] and s['fixed_work_pass'] for s in summaries)
    passed = status == 'complete' and count_ok and work_ok and env.get('passed', False) and canary_ok and all(p['paired_work_pass'] for p in pairs) and len(pairs) == expected // 4 * 3
    ours = [s for s in summaries if s['spec']['method'] == 'Ours']
    report = {'suite': 'dabs-r234-balanced-64', 'status': status, 'completed_conditions': len(summaries),
              'exploratory': exploratory, 'formal_acceptance': bool(native and not exploratory and not screening),
              'screening': screening, 'diagnostic_only': bool(screening),
              'screening_completed': bool(screening and passed), 'full_matrix_complete': bool(not screening and passed),
              'calibration_error_deferred': native_gate.get('calibration_error_deferred', []),
              'model_transfer_from_replica_0': native_gate.get('model_transfer_from_replica_0', False),
              'declared_conditions': expected, 'functional_test_pass': bool(passed),
              'fixed_work_pass': work_ok, 'http_inference_verified': canary_ok,
              'real_npu_verified': native, 'native_mechanism_test_pass': bool(native and passed),
              'strict_native_reproduction_pass': bool(native and passed and not screening and not native_gate.get('not_run', [])
                                                      and profile.get('original_gpu_timing_equivalent') is True),
              'timing_target': profile.get('timing_target') if observed_native else None,
              'original_gpu_timing_equivalent': profile.get('original_gpu_timing_equivalent', False) if native else False,
              'native_optional_tests_not_run': native_gate.get('not_run', []),
              'native_acceptance_scope': native_gate.get('scope'),
              'native_cache_observed': observed_native, 'native_kv_hit_rate': None,
              'ours_conditions_without_scoring': [s['spec']['label'] for s in ours if not s['selector_scored_decisions']],
              'ours_conditions_without_rankable_candidate': [s['spec']['label'] for s in ours if not s['v5_candidate_scored']],
              'ours_selected_nonreference_actions': sum(s.get('selected_nonreference_actions', s['nonreference_actions']) for s in ours),
              'ours_client_applied_nonreference_actions': (sum(s['nonreference_actions'] for s in ours)
                                                          if ours and all(s.get('action_accounting_version') == 2 for s in ours) else None),
              'ours_engine_applied_nonreference_actions': sum(s.get('engine_applied_nonreference_actions', 0) for s in ours) if observed_native else None,
              'limits': ([
                  'Native EngineCore snapshots/cache queries/receipts are in conditions/*/native_events.jsonl; HTTP timing remains explicitly separate.',
                  'Timing target: ' + str(profile.get('timing_target')) + '. Async completion accounting is a new throughput proxy, not the original GPU host-step target. NPU forward events remain auxiliary.',
                  'Only unsubmitted routing, admission permits and safe waiting/partial-prefill queue slots are supported. Decode/inflight requests cannot be moved.',
                  'Input/DAG/seeds/96-condition order remain frozen. Negative gains, zero deviations and failures are retained.',
                  'Mechanism validation is distinct from reproducing NVIDIA performance numbers.',
                  'Optional destructive probes NOT_RUN: ' + ', '.join(native_gate.get('not_run', []))] if observed_native else LIMITS) + env.get('limits', []), 'error': error}
    if exploratory:
        report['limits'].insert(0, 'EXPLORATORY ONLY: prediction accuracy not accepted; model transfer=' + str(native_gate.get('model_transfer_from_replica_0')))
    if screening:
        report['limits'] = [text for text in report['limits'] if '96-condition' not in text and 'Same 96 condition' not in text]
        report['limits'].insert(0, 'SCREENING ONLY: predetermined T1 subset; original specs and within-group method order retained. Not full-matrix or performance acceptance.')
    write_json(out / 'acceptance.json', report)
    for filename, values in [('metrics.csv', rows), ('paired_comparisons.csv', pairs),
                             ('workflow_metrics.csv', workflows), ('latency_decomposition.csv', stages),
                             ('replica_load.csv', loads), ('diagnostic_reasons.csv', reasons)]:
        write_csv(out / filename, values)
    if not reasons:
        (out / 'diagnostic_reasons.csv').write_text('label,replicas,rate,repeat,method,stage,reason,count\n', encoding='utf-8-sig')
    lines = ['# R234 昇腾运行报告', '', f'状态：{status}；完成 **{len(summaries)}/{expected}** 个条件。',
             f"完整负载运行检查：{'PASS' if passed else '未通过或尚未完成'}。", '',
             '每条件 64 工作流 / 216 请求 / 774144 输入 token / 13184 输出 token。',
             f'本次计划：{expected*64} 工作流 / {expected*216} 请求 / {expected*13184} 输出 token（不含预检、预热和校准）。', '',
             '过程与保存路径见 [REPRODUCE.md](REPRODUCE.md)，可视化见 [RESULTS.html](RESULTS.html)。', '',
             '## Ours 与配对基线比较', '',
             '正值表示 Ours 的平均 JCT 更低，负值表示更慢；仅纳入已完成配对，未完成不能作完整实验结论。', '',
             f'| 基线 | 已完成配对 / {expected//4} | 平均 JCT 改善 |', '|---|---:|---:|']
    for method in METHODS[:-1]:
        values = [p['ours_jct_reduction_pct'] for p in pairs if p['reference'] == method and p['paired_work_pass']]
        value = f'{statistics.mean(values):+.2f}%' if values else '尚无'
        lines.append(f'| {method} | {len(values)} / {expected//4} | {value} |')
    lines += ['', f"Ours 已执行条件：{len(ours)}；无完整参考评分的条件：{len(report['ours_conditions_without_scoring'])}；无完整候选评分的条件：{len(report['ours_conditions_without_rankable_candidate'])}。",
              f"选择非基线动作：{report['ours_selected_nonreference_actions']}；客户端实际产生不同提交序列/路由：{report['ours_client_applied_nonreference_actions']}；引擎应用次数：{report['ours_engine_applied_nonreference_actions']}。",
              '旧版 nonreference_actions 只计选择且未过期，不能当作引擎已应用；None 表示缺少证据。',
              'v5_candidate_scored 只计完整、可排名的候选；开始次数另见 metrics.csv，失败原因见 diagnostic_reasons.csv。',
              '无可执行候选、评分超时或收益不足时会保留基线；完整负载通过不等于 Ours 性能提升。', '',
              '## 原生指标与适配边界', ''] + ['- ' + text for text in report['limits']]
    if error:
        lines += ['', '错误：`' + error.replace('`', "'") + '`；详见 error.txt。']
    if screening:
        lines[2:2] = ['**诊断筛查，不是 96 条正式实验。工作量完成与 Ours 完整评分分开判定；单轮结果不证明性能优势。**', '']
    if exploratory:
        lines[2:2] = ['**探索性试跑：暂不要求预测误差合格，不计正式验收。**',
                      '共用副本 0 预测模型：' + str(native_gate.get('model_transfer_from_replica_0')),
                      '原始误差保留于 native_gate.json 与 ascend_profile.json。', '']
    (out / 'TEST_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    columns = ('label', 'jct_mean_ms', 'jct_p95_ms', 'workflows_s', 'output_tokens', 'nonreference_actions')
    table = '<table><thead><tr>' + ''.join('<th>' + c + '</th>' for c in columns) + '</tr></thead><tbody>'
    for row in rows:
        table += '<tr>' + ''.join('<td>' + html.escape(f'{row[c]:.2f}' if isinstance(row[c], float) else str(row[c])) + '</td>' for c in columns) + '</tr>'
    table += '</tbody></table>'
    document = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>R234 results</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px;color:#172033;background:#f5f7fb}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #ddd}svg{max-width:530px;background:white;margin:8px}h1{color:#123452}</style>'
    boundary = '原生证据见 native_events.jsonl；引擎 host-step、辅助 NPU forward 和 HTTP 时延分开记录' if observed_native else '原生 KV / 引擎阶段时间未观测'
    document += f'<h1>R234 昇腾实验：{len(summaries)}/{expected}</h1><p>状态：{html.escape(status)}。固定 token DAG 回放；{boundary}。</p><p>RR 蓝 · SLL 橙 · Ayo 绿 · Ours 紫。图表只展示已完成条件；完成前不能据此筛选条件。</p>'
    document += chart(summaries, 'jct_mean_ms', '平均 JCT (ms)') + chart(summaries, 'workflows_s', '完成吞吐 (工作流/s)')
    document += '<h2>逐条件结果</h2>' + table + '<p>过程：REPRODUCE.md · 结论：TEST_REPORT.md · 配对比较：paired_comparisons.csv</p></html>'
    if exploratory:
        document = document.replace('<h1>', '<p><strong>探索性试跑：预测误差未获验收；可能共用副本 0 的估计。不是正式复现结果。</strong></p><h1>', 1)
    if screening:
        document = document.replace('<h1>', '<p><strong>诊断筛查，不是 96 条正式实验；工作量完成不等于 Ours 完整评分或性能优势。</strong></p><h1>', 1)
    (out / 'RESULTS.html').write_text(document, encoding='utf-8')
    return report
