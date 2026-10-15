"""Regression: adaptive planner must never replace the native D0 reference."""
import unittest
from unittest.mock import patch
import joint18v2.client as live
from joint18v2.search import reference as native_reference
from wfsched.routing import reference as rr_p2c_reference
from wfsched.binding import bindings


class BindingFixRegression(unittest.TestCase):
    def test_dag_a_keeps_native_reference_and_adds_snapshot_counts(self):
        before = live.reference
        self.assertIs(before, native_reference)
        previous_snapshot = live.make_snapshot
        nodes = {
            'a': {'start': 1.0, 'state': 'submitted', 'replica': 1},
            'b': {'start': None, 'state': 'ready', 'replica': None},
            'c': {'start': 0.5, 'state': 'done', 'replica': 0},
        }
        with patch.object(live, 'make_snapshot', return_value={'replicas': 2}) as source:
            with bindings('DAG-A'):
                self.assertIs(live.reference, native_reference)
                self.assertIsNot(live.make_snapshot, source)
                snap = live.make_snapshot(nodes)
                self.assertEqual(snap['router_submitted_count'], 2)
                self.assertEqual(snap['router_ongoing'], [0, 1])
            self.assertIs(live.make_snapshot, source)
            self.assertIs(live.reference, native_reference)
        self.assertIs(live.make_snapshot, previous_snapshot)

    def test_dag_a_reference_actually_accepts_dag_config(self):
        from test_ours_v2 import V2Decision
        s, model, cfg = V2Decision().fixture(2)
        cfg['wf_routing_policy'] = 'DAG-A'
        s.update(router_submitted_count=0, router_ongoing=[0, 0])
        with bindings('DAG-A'):
            action = live.reference(s, cfg)
            self.assertEqual(action['family'], 'reference')
            self.assertEqual(len(action['caps']), 2)

    def test_rr_and_p2c_still_use_baseline_reference(self):
        prev_reference = live.reference
        for method in ('RR','P2C'):
            with bindings(method):
                self.assertIs(live.reference, rr_p2c_reference)
            self.assertIs(live.reference, prev_reference)

    def test_native_dag_keeps_native_reference(self):
        before = live.reference
        with bindings('DAG'):
            self.assertIs(live.reference, before)

if __name__=='__main__':
    unittest.main()
