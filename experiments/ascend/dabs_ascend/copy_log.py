"""Small, append-only text for onsite copy/paste; raw evidence stays separate."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def compact(value):
    """Bound individual fields, never replace full raw JSONL evidence."""
    if isinstance(value, dict):
        items = list(value.items())
        result = {str(k)[:100]: compact(v) for k, v in items[:24]}
        if len(items) > 24:
            result['_omitted_fields'] = len(items) - 24
        return result
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value[:4]] + ([{'omitted': len(value)-4}] if len(value) > 4 else [])
    if isinstance(value, str):
        return value if len(value) <= 140 else value[:137] + '...'
    if isinstance(value, float):
        return round(value, 3)
    return value


def top_counts(counts, limit=16):
    items = Counter(counts).most_common()
    result = dict(items[:limit])
    if len(items) > limit:
        result['_other_count'] = sum(v for _, v in items[limit:])
        result['_other_keys'] = len(items)-limit
    return result


def folder(run):
    run = Path(run)
    # A workflow's numbered probes and R234 child share ONE copy folder.
    owner = run.parent if (run.parent / 'run.json').is_file() else run
    record = owner / 'run.json'
    if record.is_file():
        # Resuming the same scientific run keeps its original copy folder.
        value = json.loads(record.read_text(encoding='utf-8')).get('copy_log_owner')
        if value:
            owner = Path(value)
    path = owner / 'copy_logs'
    path.mkdir(parents=True, exist_ok=True)
    return path


def emit(run, event, **fields):
    path = folder(run) / 'COPY.txt'
    new = not path.exists()
    def value(v):
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, ensure_ascii=False, separators=(',', ':'))
        return ' '.join(str(v).split())
    with path.open('a', encoding='utf-8') as stream:
        if new:
            stream.write('DABS复制摘要 v3；P=30秒进度；D=原因；S=评分/候选详情；C=结果；DIAG=漏斗统计；TIMING=毫秒p50/p95。\n'
                         '每条件D至多6条(同类3)、S至多3条、SELECT至多2条、回执2条；中途摘要总限12KiB，结尾汇总另留。省略有计数，完整证据在conditions/*/*.jsonl；wall_ms不是单次评分耗时。\n')
        clock = datetime.now(timezone.utc).strftime('%H:%M:%SZ')
        stream.write(clock + ' ' + event + ' ' + ' '.join(k + '=' + value(v) for k, v in fields.items() if v is not None) + '\n')
    # One read-only command can print the current run without finding directories.
    outputs = next((p for p in path.parents if p.name == 'outputs'), None)
    if outputs is not None:
        from .common import write_json
        write_json(outputs / 'native-entry-last-copy.json', {'file': path.relative_to(outputs.parent).as_posix()})


class ConditionLog:
    def __init__(self, run, label):
        self.run, self.label = run, label
        self.counts, self.examples = Counter(), Counter()
        self.progress_at = 0.
        self.timings = {}
        self.funnel, self.limits = Counter(), Counter()
        self.query_totals, self.opportunities = Counter(), Counter()
        self.detail_bytes = 0
        self.emitted = Counter()

    def timing(self, key, value):
        if isinstance(value, (int, float)):
            self.timings.setdefault(key, []).append(value)

    def detail(self, event, **fields):
        fields = compact(fields)
        size = len(json.dumps(fields, ensure_ascii=False).encode('utf-8')) + len(self.label) + 80
        allowance = 12 * 1024 if event in ('S', 'SELECT') else 6 * 1024
        if self.detail_bytes + size > allowance:
            self.counts['copy_byte_budget_suppressed'] += 1
            return
        self.detail_bytes += size
        self.emitted[event] += 1
        emit(self.run, event, condition=self.label, **fields)

    def decision(self, row):
        result = row.get('result', {})
        state = row.get('snapshot', {})
        measured = {k: state[k] for k in ('native_prepare_ms', 'native_rpc_ms', 'native_projection_ms') if k in state}
        measured.update(state.get('native_prepare_parts_ms', {}))
        measured.update({k: result[k] for k in ('planning_dispatch_delay_ms', 'planning_worker_budget_ms', 'wall_ms',
                                               'proposal_elapsed_ms', 'post_selection_diagnostics_ms') if k in result})
        for key, value in measured.items():
            self.timing(key, value)
        self.funnel.update(result.get('candidate_diagnostics', {}))
        query = state.get('native_query_budget', {})
        for key in ('ready_requests', 'covered_requests', 'ready_at_snapshot', 'covered_at_snapshot', 'new_ready_after_query'):
            self.query_totals[key] += query.get(key, 0)
        for key in ('ready', 'reference_admitted', 'reference_held', 'mutable', 'source_known_ready', 'source_known_admitted', 'source_known_held'):
            self.opportunities[key] += result.get('opportunity_diagnostics', {}).get(key, 0)
        window = row.get('planning_window', {})
        self.limits.update(window.get('binding_limits', []))
        self.timing('planning_budget_ms', window.get('budget_ms'))
        reason = result.get('v5_stop_reason') or ('planner_busy' if row.get('planner_busy') else
                 'planning_window_expired' if row.get('discarded_expired') else 'baseline')
        fallback = row.get('native_fallback_reason')
        fallback = ','.join(map(str, fallback)) if isinstance(fallback, list) else fallback
        if reason == 'baseline' and fallback:
            reason = 'native_fallback:' + str(fallback)
        self.counts[reason] += 1
        evals = result.get('evaluations', [])
        for evaluation in evals:
            prefix = 'ref_' if evaluation.get('cell') == '00' else 'candidate_'
            diagnostic = evaluation.get('diagnostic', {})
            self.timing(prefix + 'cpu_wall_ms', evaluation.get('cpu_wall_ms'))
            for field in ('slice_budget_ms', 'evaluator_elapsed_ms', 'slice_overrun_ms'):
                self.timing(prefix + field, diagnostic.get(field))
            for field, value in diagnostic.get('stages_ms', {}).items():
                self.timing(prefix + field + '_ms', value)
        self.counts['ref_ok'] += sum(e.get('cell') == '00' and bool(e.get('complete')) for e in evals)
        self.counts['candidate_ok'] += sum(e.get('cell') != '00' and bool(e.get('complete')) for e in evals)
        self.counts['generated'] += result.get('v5_candidate_generated', 0)
        self.counts['late'] += bool(row.get('discarded_expired'))
        if not row.get('discarded_expired') and not row.get('event') and not fallback:
            self.counts['on_time_ref_ok'] += sum(e.get('cell') == '00' and bool(e.get('complete')) for e in evals)
            self.counts['on_time_candidate_ok'] += sum(e.get('cell') != '00' and bool(e.get('complete')) for e in evals)
        selected = result.get('selected_cell', '00') != '00'
        self.counts['selected'] += selected
        # Reserve separate rare-event samples: initial no-candidate rows must
        # not consume the entire opportunity/scorer evidence allowance.
        if evals or result.get('v5_candidate_generated'):
            sample_kind = ('complete_pair' if any(e.get('cell') == '00' and e.get('complete') for e in evals)
                           and any(e.get('cell') != '00' and e.get('complete') for e in evals)
                           else 'partial_score' if evals else 'generated_without_score')
            sample_key = ('score_sample', sample_kind)
            if not self.examples[sample_key]:
                self.examples[sample_key] += 1
                self.detail('S', i=row.get('index'), reason=reason, expired=bool(row.get('discarded_expired')),
                    kind=sample_kind,
                    limit=window.get('binding_limits'), remaining_ms=window.get('limit_remaining_ms'),
                    budget_ms=window.get('budget_ms'), worker_ms=result.get('planning_worker_budget_ms'),
                    wall_ms=result.get('wall_ms'), proposal_ms=result.get('proposal_elapsed_ms'),
                    scores=[{'cell': e.get('cell'), 'ok': e.get('complete'), 'cost_ms': e.get('cost_ms'),
                             'reason': e.get('reason'), 'cpu_ms': e.get('cpu_wall_ms'),
                             **e.get('diagnostic', {})} for e in evals],
                    opportunity=result.get('opportunity_diagnostics'), funnel=result.get('candidate_diagnostics'),
                    guard_rejections=result.get('v5_guard_details'),
                    samples=result.get('candidate_diagnostic_samples'), rejections=result.get('admission_rejection_samples'))
            else:
                self.counts['S_suppressed'] += 1
        if selected and self.emitted['SELECT'] < 2:
            self.detail('SELECT', i=row.get('index'), cell=result.get('selected_cell'),
                        expired=bool(row.get('discarded_expired')), fallback=fallback,
                        gain_ms=result.get('v5_gain_model_ms'), threshold_ms=result.get('v5_required_gain_ms'),
                        proposal=result.get('action', {}).get('proposal'),
                        receipts=[r['receipt'].get('status') for r in row.get('native_receipts', [])])
        key = (reason, bool(row.get('event')), selected, fallback, row.get('native_deferred_reason'))
        self.examples[key] += 1
        if self.examples[key] > 3 or self.emitted['D'] >= 6 or (reason == 'baseline' and not selected):
            self.counts['D_suppressed'] += 1
            return
        receipts = [{k: r['receipt'].get(k) for k in ('status', 'reason', 'applied_step')
                     if r['receipt'].get(k) is not None} for r in row.get('native_receipts', [])]
        state = row.get('snapshot', {})
        cache = Counter(e.get('state', 'unknown') for e in state.get('cache_evidence', []))
        self.detail('D', i=row.get('index'), record_type=row.get('event'), reason=reason,
             budget_ms=round(row.get('planning_window', {}).get('budget_ms', 0), 2),
             generated=result.get('v5_candidate_generated'), selected=result.get('selected_cell'),
             limit=window.get('binding_limits'),
             cache=dict(cache), pressure=state.get('pressure'), observe_errors=state.get('native_observation_errors'),
             prediction_errors=state.get('native_prediction_errors'),
             ms={k: round(v, 2) for k, v in measured.items() if isinstance(v, (int, float)) and v},
             query={k: v for k, v in state.get('native_query_budget', {}).items() if k in (
                 'ready_requests', 'unique_queries', 'query_tokens_per_replica', 'covered_requests', 'unqueried_requests',
                 'ready_at_snapshot', 'covered_at_snapshot', 'new_ready_after_query')},
             causes=result.get('candidate_diagnostics'), fallback=row.get('native_fallback_reason'),
             opportunity=result.get('opportunity_diagnostics'), samples=result.get('candidate_diagnostic_samples'),
             rejections=result.get('admission_rejection_samples'),
             deferred=row.get('native_deferred_reason'), submitted=len(row.get('applied_submission_order', [])), receipts=receipts)

    def receipt(self, index, replica, receipt):
        key = ('final_receipt', replica, receipt.get('status'))
        self.examples[key] += 1
        self.counts['final_receipt:' + str(receipt.get('status'))] += 1
        if self.emitted['RECEIPT'] < 2:
            self.detail('RECEIPT', i=index, replica=replica,
                 **{k: receipt[k] for k in ('status', 'applied_step', 'scheduled_step', 'reason', 'rejected') if k in receipt})
        else:
            self.counts['receipt_suppressed'] += 1

    def progress(self, row):
        now = time.monotonic()
        if now < self.progress_at:
            return
        self.progress_at = now + 30
        self.detail('P', completed=row['completed_requests'], submitted=row['submitted_requests'],
             inflight=row['client_inflight'], ready=row['ready_requests'], observe_ms=round(row.get('observe_ms') or 0, 2),
             expired=row['expired_decisions'], fallback=row['atomic_fallbacks'], deferred=row['deferred_admissions'],
             planner={k: self.counts[k] for k in ('generated', 'on_time_ref_ok', 'on_time_candidate_ok', 'selected', 'late')})

    def complete(self, summary):
        from .common import percentile
        timing = {k: {'n': len(v), 'p50': round(percentile(v, .5), 2), 'p95': round(percentile(v, .95), 2)}
                  for k, v in self.timings.items()}
        emit(self.run, 'TIMING', condition=self.label, ms=timing)
        emit(self.run, 'DIAG', condition=self.label, limits=top_counts(self.limits),
             funnel=compact(top_counts(self.funnel, 22)), query_observation_totals=dict(self.query_totals),
             opportunity_decision_totals=dict(self.opportunities),
             detail_records=dict(self.emitted), detail_bytes=self.detail_bytes)
        emit(self.run, 'C', condition=self.label, work=summary.get('fixed_work_pass'), dag=summary['dependency_handoff_pass'],
             requests=summary['completed_requests'], output=summary['output_tokens'],
             jct_ms=round(summary['jct_mean_ms'], 2), p95_ms=round(summary['jct_p95_ms'], 2),
             generated=summary.get('v5_candidate_generated'), scored=summary.get('v5_candidate_scored'),
             selected=summary.get('selected_nonreference_actions'), applied=summary.get('engine_applied_nonreference_actions'),
             route=summary.get('engine_nonreference_routing_actions'), queue=summary.get('engine_nonreference_queue_actions'),
             planner=compact(top_counts(self.counts, 22)),
             diagnostics=compact({k: (top_counts(v, 6) if isinstance(v, dict) else v)
                                  for k, v in (summary.get('score_diagnostics') or {}).items()}))
