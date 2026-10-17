"""CPU instrumentation regressions, never onsite performance evidence."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dabs_ascend import r234
from dabs_ascend.copy_log import ConditionLog, top_counts
from dabs_ascend.native_algorithms import adapted, choose
from dabs_ascend.native_client import NativeClient
from dabs_ascend.native_planning import planning_window
from dabs_ascend.scheduler import snapshot
from test_native import state, model
from test_native_planning import opportunity
from test_native_simplification import observation, capability


class DiagnosticsTests(unittest.TestCase):
    def test_engine_instrumentation_preserves_events_scores_and_inputs(self):
        cfg, s = opportunity()
        original = copy.deepcopy(s)
        action = r234.baseline(s, cfg, 'Ours')
        before = adapted('engine', diagnostics=False).evaluate(s, action, model(), cfg, record_events=True)
        after = adapted('engine').evaluate(s, action, model(), cfg, record_events=True)
        self.assertEqual(set(after['diagnostic_stage_ms']), {'validate', 'clone', 'setup', 'rollout'})
        self.assertTrue(all(v >= 0 for v in after['diagnostic_stage_ms'].values()))
        self.assertEqual(after['diagnostic_progress']['nodes_done'], len(s['nodes']))
        self.assertEqual(s, original)
        after.pop('diagnostic_stage_ms'); after.pop('diagnostic_progress')
        before.pop('cpu_wall_ms'); after.pop('cpu_wall_ms')
        self.assertEqual(before, after)

    def test_selector_diagnostics_preserve_candidates_and_chosen_action(self):
        cfg, s = opportunity()
        original = copy.deepcopy(s)
        r234.planner_ready(True)
        before = None
        original_adapter = adapted
        with patch('dabs_ascend.native_algorithms.adapted',
                   side_effect=lambda name: original_adapter(name, diagnostics=False)):
            before = choose(s, model(), cfg, time.monotonic() + 10)
        after = choose(s, model(), cfg, time.monotonic() + 10)
        for field in ('action', 'selected_cell', 'v5_candidate_generated', 'v5_candidate_scored',
                      'model_selection', 'v5_gain_model_ms', 'decode_guard_rejections'):
            self.assertEqual(before[field], after[field], field)
        clean = lambda result: [{k: v for k, v in e.items() if k not in ('cpu_wall_ms', 'diagnostic')}
                                for e in result['evaluations']]
        self.assertEqual(clean(before), clean(after))
        self.assertEqual(s, original)
        for evaluation in after['evaluations']:
            diagnostic = evaluation['diagnostic']
            self.assertGreater(diagnostic['slice_budget_ms'], 0)
            self.assertLessEqual(diagnostic['slice_budget_ms'], diagnostic['planner_remaining_at_score_ms'])
            self.assertGreaterEqual(diagnostic['evaluator_elapsed_ms'], sum(diagnostic['stages_ms'].values()))
        self.assertIn('source_known_admitted', after['opportunity_diagnostics'])

    def test_rejected_first_target_is_observed_but_not_fixed_or_retried(self):
        cfg, original = state()
        cfg.update(replicas=3, client_max_inflight=1)
        nodes = original['nodes']
        for n in nodes.values():
            n['compiled_groups'] = []
        nodes['b'].update(state='submitted', replica=1, start=1.)
        s = snapshot(nodes, {}, 3, cfg, 1, 1.)
        s.update(pressure=[8, 0, 1], capacity_blocks=[1000]*3, reserved_blocks=[0]*3,
                 native_mutable=[], native_queue_groups={str(r): {'running_prefill': [], 'waiting': ['b'] if r == 1 else []}
                                                        for r in range(3)})
        s['cache_evidence'] = [{'prefix_hash': 'a', 'replica': r, 'known': True,
                               'state': 'active_computed' if r == 0 else 'absent', 'tokens': 256 if r == 0 else 0}
                              for r in range(3)]
        result = choose(s, model(), cfg, time.monotonic()+10)
        self.assertEqual(result['v5_candidate_generated'], 0)
        self.assertEqual(result['evaluations'], [])
        self.assertEqual(result['candidate_diagnostics']['resource_projection_attempts'], 1)
        self.assertEqual(result['candidate_diagnostics']['alternatives_skipped_after_rejection'], 1)
        sample = result['candidate_diagnostic_samples'][0]
        self.assertEqual((sample['rejected_destination'], sample['skipped_destination']), (1, 2))
        self.assertEqual(sample['alternative_feasibility'], 'not_tested')
        self.assertEqual(result['admission_rejection_samples'][0]['lost_admissions'], ['a'])

    def test_binding_limits_report_negative_budget_without_extending_deadline(self):
        cfg, s = state(); s['now'] = 10.
        s['native_snapshots'] = [{'expires_ns': 10_050_000_000}]
        window = planning_window(s, cfg, 10.2, now=10.06)
        self.assertEqual(window['binding_limits'], ['snapshot_expiry'])
        self.assertLess(window['limit_remaining_ms']['snapshot_expiry'], 0)
        self.assertEqual(window['budget_ms'], 0)
        self.assertAlmostEqual(window['deadline'], 10.025)

    def test_three_held_only_shortlist_entries_are_counted_without_enumerating_more(self):
        cfg, original = state(); cfg['client_max_inflight'] = 1
        nodes = {f'k{i}': {**copy.deepcopy(original['nodes']['a']), 'id': f'k{i}', 'wid': f'k{i}',
                           'prefix_hash': f'k{i}', 'compiled_groups': []} for i in range(6)}
        for r in range(2):
            nodes[f'p{r}'] = {**copy.deepcopy(original['nodes']['b']), 'id': f'p{r}', 'wid': f'p{r}',
                             'compiled_groups': [], 'state': 'submitted', 'replica': r, 'start': 1.}
        s = snapshot(nodes, {}, 2, cfg, 1, 1.)
        s.update(pressure=[8, 0], capacity_blocks=[1000]*2, reserved_blocks=[0]*2,
                 native_mutable=[], native_queue_groups={str(r): {'running_prefill': [], 'waiting': [f'p{r}']}
                                                        for r in range(2)})
        s['cache_evidence'] = [{'prefix_hash': key, 'replica': r, 'known': True,
                               'state': 'active_computed' if r == 0 else 'absent', 'tokens': 256 if r == 0 else 0}
                              for key in s['ready'] for r in range(2)]
        result = choose(s, model(), cfg, time.monotonic()+10)
        funnel = result['candidate_diagnostics']
        self.assertEqual(funnel['pressure_cache_eligible_routes'], 6)
        self.assertEqual(funnel['shortlist_before_native_dedup'], 3)
        self.assertEqual(funnel['shortlist_at_cap'], 1)
        self.assertEqual(funnel['resource_projection_attempts'], 3)
        self.assertEqual(funnel['held_only_route_no_effect'], 3)
        self.assertEqual(result['v5_candidate_generated'], 0)
        self.assertEqual(result['evaluations'], [])

    def test_query_and_final_ready_counts_distinguish_nodes_becoming_ready_in_rpc(self):
        async def run():
            cfg, s = state(); nodes = s['nodes']
            for n in nodes.values():
                n['tokens'] = [7]*3584
            nodes['b']['state'] = 'blocked'
            native = NativeClient(None, ['r0', 'r1'])
            native.capabilities = [capability(), capability()]
            async def call(replica, op, **payload):
                self.assertEqual([q['query_id'] for q in payload['queries']], ['a'])
                nodes['b']['state'] = 'ready'
                row = observation(); now = time.monotonic_ns()
                row.update(captured_ns=now, expires_ns=now+120_000_000)
                return row
            native.call = call
            observed = await native._observe_once(nodes, cfg, 1, model())
            query = observed['native_query_budget']
            self.assertEqual((query['ready_requests'], query['covered_requests']), (1, 1))
            self.assertEqual((query['ready_at_snapshot'], query['covered_at_snapshot'], query['new_ready_after_query']), (2, 1, 1))
            self.assertTrue(all(k in observed['native_prepare_parts_ms'] for k in (
                'identity_ms', 'query_snapshot_ms', 'query_reference_ms', 'query_tail_ms', 'query_pack_ms')))
        asyncio.run(run())

    def test_copy_limits_distinct_reasons_and_preserves_rare_scores_and_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = ConditionLog(Path(tmp), 'R3_L8_T1_B0_Ours')
            for i in range(1000):
                log.decision({'index': i, 'result': {'v5_stop_reason': 'distinct_reason_' + str(i),
                    'candidate_diagnostics': {'proposal_calls': 1, 'no_legal_distinct_candidate': 1}}})
            for i, complete in enumerate((False, True)):
                row = {'index': 1000+i, 'result': {'v5_candidate_generated': 1, 'v5_stop_reason': 'scored',
                    'evaluations': [{'cell': cell, 'complete': complete, 'cpu_wall_ms': 4,
                        'diagnostic': {'slice_budget_ms': 10, 'evaluator_elapsed_ms': 5,
                                       'stages_ms': {'validate': 1, 'setup': 2, 'rollout': 1}}} for cell in ('00', '20')],
                    'wall_ms': 18}}
                log.decision(row)
            summary = {'fixed_work_pass': True, 'dependency_handoff_pass': True, 'completed_requests': 216,
                       'output_tokens': 13184, 'jct_mean_ms': 10, 'jct_p95_ms': 12}
            log.complete(summary)
            text = (Path(tmp)/'copy_logs/COPY.txt').read_text(encoding='utf-8')
            self.assertEqual(text.count(' D '), 6)
            self.assertEqual(text.count(' S '), 2)
            self.assertIn('complete_pair', text)
            self.assertIn('TIMING', text)
            self.assertIn('ref_cpu_wall_ms', text)
            self.assertEqual(log.funnel['proposal_calls'], 1000)
            self.assertEqual(log.counts['D_suppressed'], 996)
            self.assertLess(len(text.encode('utf-8')), 18*1024)
            self.assertIn('_other_count', text)
            self.assertEqual(log.timings['wall_ms'], [18, 18])
            self.assertEqual(log.timings['ref_cpu_wall_ms'], [4, 4])

    def test_copy_byte_cap_cannot_be_bypassed_by_many_large_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = ConditionLog(Path(tmp), 'Ours')
            for i in range(200):
                log.receipt(i, i, {'status': 'applied', 'reason': 'x'*100000,
                                  'priority_scheduled_order': ['a']*10000})
            text = (Path(tmp)/'copy_logs/COPY.txt').read_text(encoding='utf-8')
            self.assertEqual(text.count(' RECEIPT '), 2)
            self.assertEqual(log.counts['final_receipt:applied'], 200)
            self.assertEqual(log.counts['receipt_suppressed'], 198)
            self.assertLess(len(text.encode('utf-8')), 2000)

    def test_top_counts_retains_sum_when_many_reasons_are_suppressed(self):
        result = top_counts({str(i): i+1 for i in range(100)}, 16)
        self.assertEqual(sum(v for k, v in result.items() if k != '_other_keys'), 5050)
