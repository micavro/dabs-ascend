#!/usr/bin/env python3
"""Read old results into copyable text summaries, without rerunning or archiving."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import statistics
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_ascend.common import read_json, sha, write_json
from dabs_ascend.repair import aggregate_diagnostics


def explain(run, out, max_decisions=500):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists() or out == run or run in out.parents or out in run.parents:
        raise ValueError('Choose a new output directory outside the original run')
    if not 1 <= max_decisions <= 10000:
        raise ValueError('max-decisions must be 1..10000')
    if (run / 'index.json').is_file():
        summaries = read_json(run / 'index.json')
        table = [{**s, **s['spec']} for s in summaries]
        evidence = 'index.json'
    else:
        with (run / 'metrics.csv').open(encoding='utf-8-sig', newline='') as handle:
            table = list(csv.DictReader(handle))
        evidence = 'metrics.csv'
    out.mkdir(parents=True, exist_ok=False)
    keys = [(int(r['replicas']), float(r['rate']), int(r['repeat']), r['method']) for r in table]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate condition keys')
    lookup = dict(zip(keys, table))
    paired = {}
    for method in ('RR', 'SLL', 'Ayo'):
        gains = []
        for (replicas, rate, repeat, name), ours in lookup.items():
            baseline = lookup.get((replicas, rate, repeat, method))
            if name == 'Ours' and baseline is not None:
                value = float(baseline['jct_mean_ms'])
                if value <= 0:
                    raise ValueError('Invalid baseline JCT')
                gains.append(100 * (value - float(ours['jct_mean_ms'])) / value)
        paired[method] = {'pairs': len(gains), 'mean_jct_gain_pct': statistics.mean(gains) if gains else None}
    ours = [r for r in table if r['method'] == 'Ours']
    totals = {key: sum(int(r.get(key) or 0) for r in ours) for key in
              ('v5_decisions', 'v5_candidate_generated', 'v5_candidate_scored', 'nonreference_actions', 'expired_plan_count')}
    selected = [lookup[k] for k in ((4, 8., 1, 'Ours'), (2, 16., 1, 'Ours'), (3, 8., 1, 'Ours')) if k in lookup]
    details = []
    for row in selected:
        label = row['label']
        path = run / 'conditions' / label / 'decisions.jsonl'
        if run not in path.resolve().parents:
            raise ValueError('Unsafe condition path')
        detail = {'label': label, 'available': path.is_file(), 'records_read': 0,
                  'coverage': 'not_available', 'examples': [], 'cache_evidence_known_records': 0}
        if path.is_file():
            observations = []
            with path.open(encoding='utf-8') as handle:
                for index, line in enumerate(handle):
                    if index >= max_decisions:
                        detail['coverage'] = 'first_%d_records_only' % max_decisions
                        break
                    value = json.loads(line)
                    result = value.get('result', {})
                    state = value.get('snapshot') or {}
                    detail['cache_evidence_known_records'] += int(any(e.get('known') for e in state.get('cache_evidence', [])))
                    observations.append({'result': {k: result.get(k) for k in ('evaluations', 'decode_guard_rejections', 'selected_cell')
                                                     if result.get(k) is not None},
                                         'commit_reject_reason': value.get('commit_reject_reason')})
                    if len(detail['examples']) < 6 and (result.get('evaluations') or not detail['examples']):
                        detail['examples'].append({'index': value.get('index'), 'ready': len(state.get('ready', [])),
                            'pressure': state.get('pressure'), 'pressure_source': state.get('pressure_source', 'legacy client proxy'),
                            'capacity_blocks': state.get('capacity_blocks'), 'reserved_blocks': state.get('reserved_blocks'),
                            'state_counts': dict(Counter(n.get('state') for n in state.get('nodes', {}).values())),
                            'evaluations': [{k: e.get(k) for k in ('cell', 'complete', 'reason', 'cpu_wall_ms')} for e in result.get('evaluations', [])],
                            'selected_cell': result.get('selected_cell'), 'expired': value.get('discarded_expired', False),
                            'client_submission_count': len(value.get('applied_submission_order', [])),
                            'engine_applied': value.get('engine_applied')})
                else:
                    detail['coverage'] = 'all_recorded_lines; legacy_expired_worker_results_may_be_missing'
            detail['records_read'] = len(observations)
            detail['diagnostics'] = aggregate_diagnostics(observations)
        details.append(detail)
    result = {'source': str(run), 'evidence': evidence, 'evidence_sha256': sha(run / evidence),
              'rows': len(table), 'methods': dict(Counter(r['method'] for r in table)),
              'paired_summary': paired, 'ours_totals': totals, 'decision_samples': details,
              'native_fields_empty_rows': sum(all(r.get(k) in (None, '') for k in ('native_kv_hit_rate', 'native_prefill_ms', 'native_decode_ms')) for r in table),
              'raw_integrity_audited': False, 'source_files_modified': False,
              'limits': ['Counts in old CSV do not prove engine-applied actions.',
                         'CSV marked pass is not an independent audit of raw requests/dependencies.',
                         'Decision sampling is bounded; absence in this excerpt does not prove absence in the whole run.']}
    write_json(out / 'old_run_analysis.json', result)
    lines = ['DABS 旧结果诊断摘要（只复制文字，不打包）', '原目录: ' + str(run),
             '证据: %s；%d 条；方法数: %s' % (evidence, len(table), json.dumps(result['methods'])),
             'Ours 计数: ' + json.dumps(totals), '配对重算: ' + json.dumps(paired),
             '旧 nonreference_actions 不等于引擎应用；v5_candidate_scored 只计完整可排名候选。']
    for detail in details:
        lines += ['', detail['label'] + ': ' + detail['coverage'],
                  '读取条数=%d；有 known 缓存证据的记录=%d' % (detail['records_read'], detail['cache_evidence_known_records']),
                  '评分诊断=' + json.dumps(detail.get('diagnostics'), ensure_ascii=False)]
    lines += ['', '原始数据没有改写；未运行推理；未独立审计全部原始文件。',
              '代表性决策片段见 DECISIONS.txt，完整诊断见 old_run_analysis.json。']
    summary = '\n'.join(lines) + '\n'
    (out / 'OLD_RUN_SUMMARY.txt').write_text(summary, encoding='utf-8')
    (out / 'DECISIONS.txt').write_text(json.dumps(details, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(summary)
    print('OLD_RUN_SUMMARY=' + str(out / 'OLD_RUN_SUMMARY.txt'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--max-decisions', type=int, default=500, help='maximum recorded lines per selected condition')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / 'outputs' / ('r234_explain_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    explain(args.run, out, args.max_decisions)


if __name__ == '__main__':
    main()
