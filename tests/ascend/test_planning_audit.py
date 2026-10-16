"""CPU scoring coverage records, not native service acceptance evidence."""
import copy
from pathlib import Path
import tempfile
import time
import unittest

from dabs_ascend import r234
from dabs_ascend.common import append_json, read_json, write_json
from dabs_ascend.native_algorithms import choose
from dabs_ascend.native_planning import CONTRACT, run_planner
from dabs_ascend.planning_audit import audit
from test_native import model
from test_native_planning import opportunity


def fixture(directory, count=8):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    cfg, s = opportunity(); r234.planner_ready(True)
    window = {'contract': CONTRACT, 'prepared_at': time.monotonic(), 'deadline': time.monotonic() + 1}
    result = run_planner(choose, s, model(), cfg, window)
    summaries = []
    for spec in r234.plan()[:count]:
        summaries.append({'spec': spec, 'functional_pass': True, 'fixed_work_pass': True,
                          'dependency_handoff_pass': True, 'cpu_fixture_only': True})
        if spec['method'] == 'Ours':
            path = directory / 'conditions' / spec['label']; path.mkdir(parents=True, exist_ok=True)
            append_json(path / 'decisions.jsonl', {'snapshot': s, 'planning_window': window, 'result': result})
    write_json(directory / 'index.json', summaries)
    write_json(directory / 'status.json', {'status': 'paused', 'completed_conditions': count})


class PlanningAudit(unittest.TestCase):
    def test_real_complete_pair_required_selection_or_speedup_not_required(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d)
            result = audit(d, 8)
            self.assertTrue(result['pass'], result)
            self.assertFalse(result['nonreference_selection_required'])

    def test_expired_partial_nan_or_identical_action_cannot_pass(self):
        for defect in ('expired', 'partial', 'nan', 'same_action', 'late_worker', 'missing_worker'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as d:
                fixture(d)
                path = next(Path(d).glob('conditions/*/decisions.jsonl'))
                row = read_json(path)
                if defect == 'expired': row['discarded_expired'] = True
                elif defect == 'late_worker': row['result']['planning_finished_at'] = row['planning_window']['deadline'] + 1
                elif defect == 'missing_worker': row['result'].pop('planning_finished_at')
                else:
                    for e in row['result']['evaluations']:
                        if e['cell'] != '00':
                            if defect == 'partial': e['complete'] = False
                            if defect == 'nan': e['cost_ms'] = float('nan')
                            if defect == 'same_action': e['action'] = copy.deepcopy(row['result']['evaluations'][0]['action'])
                # JSONL remains one line, and a hand-written PASS cannot grant coverage.
                import json
                path.write_text(json.dumps(row) + '\n', encoding='utf-8')
                write_json(Path(d) / 'planning_audit.json', {'pass': True})
                self.assertFalse(audit(d, 8)['pass'])
