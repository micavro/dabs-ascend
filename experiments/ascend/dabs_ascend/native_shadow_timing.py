"""Live default-only Ours workload with one unbounded, non-executable planner.

No CPU deadline, score slice, kill timer, retries or backlog. Model support,
resource constraints, surrogate event/horizon guards and candidates stay intact.
"""
import asyncio
from collections import Counter
import math
from pathlib import Path
import time

from .common import percentile, read_json, write_json
from .copy_log import compact, emit, top_counts

CONTRACT = 'ours-default-live-unbounded-shadow-v1'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def distribution(values):
    values = [v for v in values if finite(v)]
    if not values:
        return {'n': 0}
    return {'n': len(values), 'p50': round(percentile(values, .5), 3),
            'p95': round(percentile(values, .95), 3), 'p99': round(percentile(values, .99), 3),
            'max': round(max(values), 3)}


def profile_path(source):
    """Find a model, not a PASS flag. No raw probe/hash audit or recalibration."""
    source = Path(source).resolve()
    if source.is_file() and source.name in ('ascend_profile.json', 'native_profile.json'):
        return source
    # A timing run stages the exact predictor here; prefer it to its child copy.
    staged = source/'timing_input/ascend_profile.json'
    if staged.is_file():
        return staged
    direct = [source/name for name in ('ascend_profile.json', 'native_profile.json') if (source/name).is_file()]
    nested = [source/child/'ascend_profile.json' for child in ('r234_screening', 'r234_native', 'r234_exploratory', 'r234_ours_timing', 'r234_ours')
              if (source/child/'ascend_profile.json').is_file()]
    candidates = direct or nested
    if len(candidates) != 1:
        raise ValueError('--input needs one prior ours/ours-timing/screen/verify/experiment result or a specific ascend_profile.json/native_profile.json')
    return candidates[0]


def run_unbounded(state, model, cfg, context):
    """Child process. Return timings/diagnostics ONLY; strip executable actions."""
    from .native_algorithms import choose
    started, elapsed_start = time.monotonic(), time.perf_counter()
    try:
        result = choose(state, model, cfg, None, unbounded_timing=True)
    except Exception as exc:
        result = {'evaluations': [], 'v5_stop_reason': 'shadow_worker_error',
                  'worker_error': type(exc).__name__ + ': ' + str(exc)}
    elapsed_ms = (time.perf_counter()-elapsed_start)*1000
    finished = time.monotonic()
    result.pop('action', None)
    for evaluation in result.get('evaluations', []):
        evaluation.pop('action', None)
    window = context['window']
    dispatch_ms = (started-window['prepared_at'])*1000
    total_ms = (finished-window['prepared_at'])*1000
    captured = context.get('earliest_snapshot_at')
    reserve = cfg['commit_reserve_s']*1000
    reference = [e for e in result.get('evaluations', []) if e.get('cell') == '00']
    candidates = [e for e in result.get('evaluations', []) if e.get('cell') != '00']
    score_ms = lambda e: e.get('diagnostic', {}).get('evaluator_elapsed_ms')
    ref_ms = score_ms(reference[0]) if reference else None
    candidate_ms = sum(score_ms(e) for e in candidates if finite(score_ms(e))) if candidates else None
    ref_complete = bool(reference and reference[0].get('complete'))
    generated = result.get('v5_candidate_generated', 0)
    result['shadow_timing'] = {
        'contract': CONTRACT, 'diagnostic_only': True, 'action_usable': False,
        'worker_wall_ms': elapsed_ms, 'dispatch_ms': dispatch_ms,
        'front_ms': context['front_ms'], 'from_window_ready_ms': total_ms,
        'original_budget_ms': window['budget_ms'],
        'original_worker_budget_ms': (window['deadline']-started)*1000,
        'exceeded_original_deadline': finished > window['deadline'],
        'overrun_ms': max(0., (finished-window['deadline'])*1000),
        # Same observation, hypothetical serial observe -> plan -> reserve.
        # This is not a changed-period run or a promise of engine acceptance.
        'cycle_need_estimate_ms': context['front_ms']+total_ms+reserve,
        'snapshot_need_estimate_ms': None if captured is None else (finished-captured)*1000+reserve,
        'commit_reserve_ms': reserve, 'reference_ms': ref_ms, 'candidate_total_ms': candidate_ms,
        'proposal_ms': result.get('proposal_elapsed_ms'),
        'guard_ms': result.get('unbounded_timing', {}).get('guard_ms'),
        'reference_build_ms': result.get('unbounded_timing', {}).get('reference_build_ms'),
        'nodes': len(state['nodes']), 'ready': len(state['ready']),
        'reference_complete': ref_complete,
        'all_scores_complete': bool(generated and ref_complete and len(candidates) == generated
                                    and all(e.get('complete') for e in candidates)),
        'generated': generated, 'candidate_started': len(candidates),
        'candidate_complete': sum(bool(e.get('complete')) for e in candidates),
        'worker_finished_at': finished, 'window_prepared_at': window['prepared_at'],
        'original_deadline': window['deadline'], 'includes_return_delivery': False}
    return result


def record_return(future):
    """Stamp delivery on the parent loop, including result IPC/loop delay."""
    if future.cancelled() or future.exception() is not None:
        return
    timing = future.result()['shadow_timing']
    if timing.get('includes_return_delivery'):
        return
    received = time.monotonic()
    delay = max(0., (received-timing['worker_finished_at'])*1000)
    timing.update(parent_received_at=received, return_delay_ms=delay, includes_return_delivery=True,
                  from_window_ready_ms=(received-timing['window_prepared_at'])*1000,
                  exceeded_original_deadline=received > timing['original_deadline'],
                  overrun_ms=max(0., (received-timing['original_deadline'])*1000))
    timing['cycle_need_estimate_ms'] += delay
    if timing['snapshot_need_estimate_ms'] is not None:
        timing['snapshot_need_estimate_ms'] += delay


async def await_natural_completion(future, progress, interval=30):
    """asyncio.wait does not cancel the job on timeout, unlike wait_for."""
    while not future.done():
        done, _ = await asyncio.wait({future}, timeout=interval)
        if not done:
            progress()
    return await future


class ShadowLog:
    def __init__(self, run, directory, label):
        self.run, self.directory, self.label = run, Path(directory), label
        self.counts, self.reasons, self.examples = Counter(), Counter(), Counter()
        self.groups = {name: {} for name in ('all', 'with_candidates', 'all_scores_complete')}
        self.slow = []
        self.detail_bytes = 0
        self.saved_at = 0.

    def started(self):
        self.counts['started'] += 1

    def busy(self):
        self.counts['skipped_busy_no_queue'] += 1

    def add(self, row):
        result = row['result']
        timing = result['shadow_timing']
        self.counts['returned'] += 1
        self.counts['exceeded_original_deadline'] += timing['exceeded_original_deadline']
        self.counts['with_candidates'] += bool(timing['generated'])
        self.counts['all_scores_complete'] += timing['all_scores_complete']
        self.counts['reference_complete'] += timing['reference_complete']
        self.counts['candidate_started'] += timing['candidate_started']
        self.counts['candidate_complete'] += timing['candidate_complete']
        self.counts['worker_errors'] += bool(result.get('worker_error'))
        self.reasons[result.get('v5_stop_reason', 'unknown')] += 1
        for evaluation in result.get('evaluations', []):
            if not evaluation.get('complete'):
                self.reasons['score:' + str(evaluation.get('reason'))] += 1
        for name, group in self.groups.items():
            if name == 'with_candidates' and not timing['generated']:
                continue
            if name == 'all_scores_complete' and not timing['all_scores_complete']:
                continue
            fields = ('worker_wall_ms', 'dispatch_ms', 'return_delay_ms', 'from_window_ready_ms') if name == 'all' else (
                'worker_wall_ms', 'dispatch_ms', 'return_delay_ms', 'front_ms', 'from_window_ready_ms', 'reference_build_ms',
                'proposal_ms', 'reference_ms', 'candidate_total_ms', 'guard_ms',
                'cycle_need_estimate_ms', 'snapshot_need_estimate_ms')
            for key in fields:
                if finite(timing.get(key)):
                    group.setdefault(key, []).append(timing[key])
        sample = {'i': row['index'], 'ms': {k: timing.get(k) for k in (
            'worker_wall_ms', 'original_budget_ms', 'dispatch_ms', 'return_delay_ms', 'front_ms', 'proposal_ms',
            'reference_ms', 'candidate_total_ms', 'guard_ms', 'cycle_need_estimate_ms', 'snapshot_need_estimate_ms')},
            'expired': timing['exceeded_original_deadline'], 'all_complete': timing['all_scores_complete'],
            'nodes': timing['nodes'], 'ready': timing['ready'], 'generated': timing['generated'],
            'stop': result.get('v5_stop_reason'), 'error': result.get('worker_error'),
            'scores': [{'cell': e.get('cell'), 'complete': e.get('complete'), 'reason': e.get('reason'),
                        'ms': e.get('diagnostic', {}).get('evaluator_elapsed_ms'),
                        'stages_ms': e.get('diagnostic', {}).get('stages_ms'),
                        'progress': e.get('diagnostic', {}).get('progress')} for e in result.get('evaluations', [])]}
        if timing['generated']:
            self.slow = sorted(self.slow+[sample], key=lambda s: s['ms']['worker_wall_ms'], reverse=True)[:3]
        kind = 'complete_comparison' if timing['all_scores_complete'] else 'incomplete_comparison' if timing['generated'] else 'no_candidate'
        if result.get('worker_error'):
            kind = 'worker_error'
        if self.examples[kind] < (2 if timing['generated'] else 1):
            import json
            bounded = compact(sample)
            size = len(json.dumps(bounded, ensure_ascii=False).encode('utf-8'))
            if self.detail_bytes+size <= 10*1024:
                emit(self.run, 'SHADOW', condition=self.label, kind=kind, **bounded)
                self.examples[kind] += 1
                self.detail_bytes += size
        if time.monotonic() >= self.saved_at:
            self.save(False)
            self.saved_at = time.monotonic()+30

    def save(self, final):
        summary = {'contract': CONTRACT, 'default_only_execution': True, 'planner_actions_applied': 0,
                   'cpu_budget_limit_ms': None, 'score_slice_limit_ms': None,
                   'final': final, 'counts': dict(self.counts), 'stop_reasons': dict(self.reasons),
                   'ms': {name: {k: distribution(v) for k, v in group.items()} for name, group in self.groups.items()},
                   'slowest_candidate_calls': self.slow,
                   'limits': 'Natural algorithm return, not necessarily complete scores; model/event/horizon failures stay failures. '
                             'Busy windows are skipped, not queued. Timing is on a default-only live trajectory. '
                             'cycle/snapshot need estimates include the existing commit reserve, not a guarantee of new-period JCT or valid actions.'}
        write_json(self.directory/'shadow_timing_summary.json', summary)
        return summary

    def finish(self):
        self.counts['unfinished'] = self.counts['started']-self.counts['returned']
        summary = self.save(True)
        emit(self.run, 'SHADOW_SUMMARY', condition=self.label, counts=self.counts, reasons=top_counts(self.reasons),
             default_only=True, planner_actions_applied=0, budget_limit='none', skipped_windows='busy; no queue')
        for name, timings in summary['ms'].items():
            emit(self.run, 'SHADOW_TIMING', condition=self.label, group=name, ms=timings)
        for sample in self.slow:
            emit(self.run, 'SHADOW_SLOW', condition=self.label, **compact(sample))
        from .copy_log import folder
        if (folder(self.run)/'parts.json').is_file():
            packed = lambda d: [d.get(k) for k in ('n', 'p50', 'p95', 'p99', 'max')]
            emit(self.run, 'SHADOW_BRIEF', label=self.label, counts=dict(self.counts), reasons=dict(self.reasons),
                 complete_ms={k: packed(v) for k, v in summary['ms']['all_scores_complete'].items()
                              if k in ('worker_wall_ms', 'cycle_need_estimate_ms', 'snapshot_need_estimate_ms')})
        return summary


def make_report(out, plan, summaries, status, error=None):
    """Diagnostic result, never four-method/96-condition or planning acceptance."""
    complete = status == 'complete' and [s['spec'] for s in summaries] == plan
    report = {'status': status, 'declared_conditions': len(plan), 'completed_conditions': len(summaries),
              'functional_test_pass': complete and all(s['functional_pass'] for s in summaries),
              'diagnostic_only': True, 'formal_acceptance': False, 'ours_shadow_timing': True,
              'default_only_execution': True, 'planner_actions_applied': 0, 'error': error,
              'conditions': [{'label': s['spec']['label'], 'timing': s.get('shadow_timing'),
                              'requests': s['completed_requests'], 'output_tokens': s['output_tokens']} for s in summaries]}
    write_json(out/'shadow_timing_report.json', report)
    write_json(out/'acceptance.json', report)
    lines = ['Ours 实时无预算后台规划计时；实际请求始终 default；不是性能比较或机制验收。',
             '状态=%s；已完成=%s/%s' % (status, len(summaries), len(plan)),
             '不运行额外规则自检、探针、canary、重标定或完整评分验收。实际动作的新鲜度/资源/回执安全检查不变。',
             '规划与单次评分均无 CPU 截止；过期结果只记日志，永远不应用；同一时刻最多一个规划，无积压。',
             '条件结束后等待最后一个后台任务自然完成；忙时未采样的窗口有计数。',
             'copy_logs/COPY.txt 为分文件目录；COPY_00_overview_* 为总览，COPY_序号_条件_* 为该条件完整摘要，每份至多 64KiB。',
             '条件摘要 SHADOW_TIMING 分 all / with_candidates / all_scores_complete；用于预算判断优先看后两组。总览数组为 [n,p50,p95,p99,max]，单位 ms。',
             'cycle_need_estimate_ms 与 snapshot_need_estimate_ms 含返回通信/事件循环延迟和原 25ms 提交预留；只估计当前轨迹，不保证新周期的效果。']
    import json
    for condition in report['conditions']:
        lines.append(condition['label'] + ' ' + json.dumps(compact(condition['timing']), ensure_ascii=False, separators=(',', ':')))
    if error:
        lines.append('ERROR ' + str(error))
    (out/'test_result.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'TEST_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report
