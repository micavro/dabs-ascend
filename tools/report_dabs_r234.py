#!/usr/bin/env python3
"""Regenerate derived reports in a NEW directory; never change raw experiments."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/ascend'))
from dabs_ascend.common import read_json, sha, write_json
from dabs_ascend.repair import aggregate_diagnostics
from dabs_ascend.r234_report import make_report


def regenerate(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source or source in out.parents or out in source.parents or out.exists():
        raise ValueError('Use a new output directory outside the original experiment')
    protocol, status = read_json(source / 'protocol.json'), read_json(source / 'status.json')
    summaries = copy.deepcopy(read_json(source / 'index.json'))
    if [s['spec'] for s in summaries] != protocol['plan'][:len(summaries)]:
        raise ValueError('Index differs from the recorded plan')
    verified = {}
    for summary in summaries:
        directory = source / 'conditions' / summary['spec']['label']
        if source not in directory.resolve().parents:
            raise ValueError('Unsafe condition path')
        manifest = read_json(directory / 'artifacts.json')
        required = {'summary.json', 'decisions.jsonl', 'requests.jsonl', 'nodes.json', 'spec.json', 'config.json',
                    'initial_nodes.json', 'input_proof.json', 'quality.json', 'status.json'}
        if set(manifest) != required:
            raise ValueError('Incomplete raw artifact manifest')
        for name, expected in manifest.items():
            path = directory / name
            if path.name != name or path.is_symlink() or sha(path) != expected:
                raise ValueError('Raw artifact changed: ' + name)
            verified[path.relative_to(source).as_posix()] = expected
        if summary != read_json(directory / 'summary.json'):
            raise ValueError('Index differs from raw summary')
        # Stream large logs; retain only diagnostics, never all snapshots/tokens.
        def records():
            with (directory / 'decisions.jsonl').open(encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        summary['score_diagnostics'] = aggregate_diagnostics(records())
        # Historical code discarded expired results. A zero from surviving logs
        # is only a lower bound, not proof no scoring occurred in the worker.
        summary['score_diagnostic_coverage'] = ('all_returned_results' if summary.get('action_accounting_version') == 2
                                                 else 'recorded_results_only; expired_worker_results_unavailable')
    out.mkdir(parents=True, exist_ok=False)
    report = make_report(out, protocol['plan'], summaries, status['status'], status.get('error'), source_out=source)
    write_json(out / 'REGENERATION.json', {'source_directory': str(source), 'raw_files_modified': False,
               'source_protocol_sha256': sha(source / 'protocol.json'), 'verified_artifacts': verified,
               'diagnostic_coverage': {s['spec']['label']: s['score_diagnostic_coverage'] for s in summaries},
               'inference_executed': False})
    (out / 'REPRODUCE.md').write_text('# 报告重生成\n\n原始实验：`' + str(source) + '`。\n\n'
        '未执行推理，未修改原始文件。验证详情见 REGENERATION.json。\n'
        '旧版丢弃的超时评分结果无法恢复，旧版动作计数不能证明引擎应用。\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or ROOT / 'outputs' / ('r234_report_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    regenerate(args.run, out)
    print('R234_REGENERATED_REPORT=' + str(out.resolve() / 'TEST_REPORT.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
