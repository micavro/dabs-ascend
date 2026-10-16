"""Small, append-only text for onsite copy/paste; raw evidence stays separate."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time


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
            stream.write('DABS复制摘要 v1；普通文本，不打包。P=30秒进度；D=原因/回执样本；C=条件最终统计。\n'
                         '仅摘要：相同原因累计计数，D每类最多3例；完整证据仍在conditions/*/*.jsonl。\n')
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

    def decision(self, row):
        result = row.get('result', {})
        reason = result.get('v5_stop_reason') or ('planner_busy' if row.get('planner_busy') else
                 'planning_window_expired' if row.get('discarded_expired') else 'baseline')
        fallback = row.get('native_fallback_reason')
        fallback = ','.join(map(str, fallback)) if isinstance(fallback, list) else fallback
        if reason == 'baseline' and fallback:
            reason = 'native_fallback:' + str(fallback)
        self.counts[reason] += 1
        evals = result.get('evaluations', [])
        self.counts['ref_ok'] += sum(e.get('cell') == '00' and bool(e.get('complete')) for e in evals)
        self.counts['candidate_ok'] += sum(e.get('cell') != '00' and bool(e.get('complete')) for e in evals)
        self.counts['generated'] += result.get('v5_candidate_generated', 0)
        self.counts['late'] += bool(row.get('discarded_expired'))
        if not row.get('discarded_expired') and not row.get('event') and not fallback:
            self.counts['on_time_ref_ok'] += sum(e.get('cell') == '00' and bool(e.get('complete')) for e in evals)
            self.counts['on_time_candidate_ok'] += sum(e.get('cell') != '00' and bool(e.get('complete')) for e in evals)
        selected = result.get('selected_cell', '00') != '00'
        self.counts['selected'] += selected
        key = (reason, bool(row.get('event')), selected, fallback, row.get('native_deferred_reason'))
        self.examples[key] += 1
        if self.examples[key] > 3 or (reason == 'baseline' and not selected):
            return
        receipts = [{k: r['receipt'].get(k) for k in ('action_id', 'status', 'reason', 'applied_step')
                     if r['receipt'].get(k) is not None} for r in row.get('native_receipts', [])]
        state = row.get('snapshot', {})
        cache = Counter(e.get('state', 'unknown') for e in state.get('cache_evidence', []))
        emit(self.run, 'D', condition=self.label, i=row.get('index'), record_type=row.get('event'), reason=reason,
             budget_ms=round(row.get('planning_window', {}).get('budget_ms', 0), 2),
             generated=result.get('v5_candidate_generated'), selected=result.get('selected_cell'),
             scores=[(e.get('cell'), e.get('complete'), e.get('cost_ms'), e.get('reason')) for e in evals],
             cache=dict(cache), pressure=state.get('pressure'), observe_errors=state.get('native_observation_errors'),
             causes=result.get('candidate_diagnostics'), fallback=row.get('native_fallback_reason'),
             deferred=row.get('native_deferred_reason'), submitted=len(row.get('applied_submission_order', [])), receipts=receipts)

    def receipt(self, index, replica, receipt):
        key = ('final_receipt', replica, receipt.get('status'))
        self.examples[key] += 1
        self.counts['final_receipt:' + str(receipt.get('status'))] += 1
        if self.examples[key] <= 3:
            emit(self.run, 'RECEIPT', condition=self.label, i=index, replica=replica,
                 **{k: receipt[k] for k in ('action_id', 'status', 'applied_step', 'scheduled_step',
                     'priority_scheduled_order', 'reason', 'rejected') if k in receipt})

    def progress(self, row):
        now = time.monotonic()
        if now < self.progress_at:
            return
        self.progress_at = now + 30
        emit(self.run, 'P', condition=self.label, completed=row['completed_requests'], submitted=row['submitted_requests'],
             inflight=row['client_inflight'], ready=row['ready_requests'], observe_ms=round(row.get('observe_ms') or 0, 2),
             expired=row['expired_decisions'], fallback=row['atomic_fallbacks'], deferred=row['deferred_admissions'],
             planner=dict(self.counts))

    def complete(self, summary):
        emit(self.run, 'C', condition=self.label, work=summary.get('fixed_work_pass'), dag=summary['dependency_handoff_pass'],
             requests=summary['completed_requests'], output=summary['output_tokens'],
             jct_ms=round(summary['jct_mean_ms'], 2), p95_ms=round(summary['jct_p95_ms'], 2),
             generated=summary.get('v5_candidate_generated'), scored=summary.get('v5_candidate_scored'),
             selected=summary.get('selected_nonreference_actions'), applied=summary.get('engine_applied_nonreference_actions'),
             route=summary.get('engine_nonreference_routing_actions'), queue=summary.get('engine_nonreference_queue_actions'),
             planner=dict(self.counts), diagnostics=summary.get('score_diagnostics'))
