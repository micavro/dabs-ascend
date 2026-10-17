"""Recompute on-time scoring coverage from decisions, separate from NPU gates."""
from collections import Counter
import json
import math
from pathlib import Path

from .common import read_json
from .native_algorithms import executable_key
from .native_planning import CONTRACT


def audit(directory, expected_conditions, *, screening=None):
    directory = Path(directory)
    from .r234 import plan, screening_plan
    summaries = read_json(directory / 'index.json')
    status = read_json(directory / 'status.json')
    missing, rows = [], []
    if status.get('status') != ('complete' if screening else 'paused') or status.get('completed_conditions') != expected_conditions:
        missing.append('not_a_complete_screening_run' if screening else 'not_a_clean_condition_boundary_pause')
    expected = screening_plan(screening) if screening else plan()[:expected_conditions]
    if len(summaries) != expected_conditions or [s['spec'] for s in summaries] != expected:
        missing.append('completed_conditions_not_expected_screening_plan' if screening else 'completed_conditions_not_expected_plan_prefix')
    for summary in summaries:
        label = summary['spec']['label']
        if not all(summary.get(k) is True for k in ('functional_pass', 'fixed_work_pass', 'dependency_handoff_pass')):
            missing.append(label + ':work_or_DAG_failed')
        if summary['spec']['method'] != 'Ours':
            continue
        counts, reasons = Counter(), Counter()
        with (directory / 'conditions' / label / 'decisions.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                result = row.get('result', {})
                if row.get('event') == 'final_engine_receipt':
                    continue
                reasons[result.get('v5_stop_reason', 'planner_timeout_or_no_result')] += 1
                if row.get('discarded_expired') or row.get('event') or row.get('native_fallback_reason'):
                    counts['late_or_fallback'] += 1
                    continue
                if row.get('planning_window', {}).get('contract') != CONTRACT:
                    counts['missing_planning_contract'] += 1
                    continue
                finished = result.get('planning_finished_at')
                if not isinstance(finished, (int, float)) or not math.isfinite(finished) or finished > row['planning_window']['deadline']:
                    counts['missing_or_late_worker_completion'] += 1
                    continue
                def complete(e):
                    cost = e.get('cost_ms')
                    return (e.get('complete') is True and e.get('reason') == 'complete_surrogate' and
                            isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0)
                ref = next((e for e in result.get('evaluations', []) if e.get('cell') == '00' and complete(e)), None)
                if not ref:
                    continue
                counts['complete_reference'] += 1
                candidates = [e for e in result.get('evaluations', []) if e.get('cell') != '00' and complete(e)
                              and executable_key(e['action'], row['snapshot']) != executable_key(ref['action'], row['snapshot'])]
                counts['complete_candidates'] += len(candidates)
                counts['comparable_decisions'] += bool(candidates)
        if not counts['comparable_decisions']:
            missing.append(label + ':no_on_time_complete_reference_and_distinct_candidate_pair; inspect stop reasons')
        rows.append({'condition': label, **dict(counts), 'stop_reasons': dict(reasons)})
    if not rows:
        missing.append('no_Ours_condition')
    return {'pass': not missing, 'scope': 'raw_on_time_planning_coverage; not replacement for native/workload gates',
            'screening': screening, 'full_matrix_acceptance': False,
            'nonreference_selection_required': False, 'conditions': rows, 'missing': missing}
