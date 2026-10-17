"""Bounded Ours-only report: workload, scoring and actual effects stay separate."""
from collections import Counter
import json

from .common import read_json, write_json
from .r234 import ours_timing_plan
from dabs_native.timing_policy import configuration


def make_report(out, plan, summaries, status, error=None):
    def read(name, default):
        path = out / name
        return read_json(path) if path.is_file() else default
    protocol = read('protocol.json', {})
    expected = ours_timing_plan(limit=protocol.get('timing_count'))
    gate = read('native_gate.json', {})
    canaries = read('canary_summary.json', [])
    canary_ok = (len(canaries) == 20 and all(c.get('passed') for c in canaries)
                 and {(c['replica'], c['fixed_output_budget']) for c in canaries}
                 == {(r, n) for r in range(4) for n in (16, 32, 48, 64, 128)})
    workload_ok = (plan == expected and len(summaries) == len(plan)
                   and [s['spec'] for s in summaries] == plan
                   and all(all(s.get(k) is True for k in
                               ('functional_pass', 'fixed_work_pass', 'dependency_handoff_pass')) for s in summaries))
    passed = bool(status == 'complete' and not error and workload_ok
                  and gate.get('pass') and gate.get('live_verified')
                  and read('environment.json', {}).get('passed') and canary_ok)
    counts, observation, retries, commits = Counter(), Counter(), Counter(), Counter()
    rows = []
    for summary in summaries:
        diagnostics = summary.get('runtime_diagnostics', {})
        counts.update(diagnostics.get('stages', {}))
        observation.update(diagnostics.get('observation_errors', {}))
        retries.update(diagnostics.get('observation_retries', {}))
        commits.update(diagnostics.get('commit_reasons', {}))
        rows.append({'condition': summary['spec']['label'],
                     'jct_mean_ms': summary['jct_mean_ms'],
                     'engine_applied_nonreference': summary.get('engine_applied_nonreference_actions', 0),
                     'runtime': diagnostics, 'completion': summary.get('completion_diagnostics', {})})
    applied = sum(s.get('engine_applied_nonreference_actions', 0) for s in summaries)
    report = {'status': status, 'ours_only': True, 'default_only_execution': False,
              'formal_acceptance': False, 'full_matrix_complete': False,
              'functional_test_pass': passed, 'workload_complete': workload_ok,
              'declared_conditions': len(plan), 'completed_conditions': len(summaries),
              'timing_configuration': configuration(), 'counts': dict(counts),
              'observation_errors': dict(observation), 'observation_retries': dict(retries),
              'commit_reasons': dict(commits), 'engine_applied_nonreference': applied,
              'conditions': rows, 'error': error,
              'interpretation': 'Raw counts and timings only; scheduling effectiveness and performance are reviewed offline.'}
    write_json(out/'acceptance.json', report)
    write_json(out/'ours_run_summary.json', report)
    lines = ['DABS Ours 正常限时实验汇总（不是 default-only 计时）',
             '周期=1500ms；快照=1500ms；规划软上限=1200ms；提交预留=25ms。',
             '状态=' + status + '；完成=' + str(len(summaries)) + '/' + str(len(plan)),
             '工作量/环境/原生门禁通过=' + str(passed),
             'Ours 子序列保留原种子和到达；无候选和零收益照实记录，不做四方法性能结论。',
             '不自动判断调度有效性，不设候选/应用次数门槛；带回日志后人工分析。',
             '评分完成、选中、按时返回、实际应用是不同计数；晚返回诊断不重复算作新决策。',
             '总阶段=' + json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
             '实际非参考动作=' + str(applied),
             '观测最终失败=' + json.dumps(dict(observation), ensure_ascii=False, sort_keys=True),
             '观测重取=' + json.dumps(dict(retries), sort_keys=True),
             '提交拒绝=' + json.dumps(dict(commits), sort_keys=True),
             '各条件耗时为独立分布，不能平均这些分位数当总体分位数。']
    for row in rows:
        lines += ['', row['condition'], json.dumps(row, ensure_ascii=False, sort_keys=True)]
    if error:
        lines += ['错误=' + str(error)]
    lines += ['', 'COPY.txt 为64KiB分片目录；原始事件在 conditions/*/*.jsonl。']
    (out/'test_result.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'TEST_REPORT.md').write_text('# Ours 正常限时实验\n\n详见同目录 test_result.txt 与 ours_run_summary.json。\n',
                                    encoding='utf-8')
    return report
