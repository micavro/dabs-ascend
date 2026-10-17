"""Actual planner/scorer on CPU fixtures; never a substitute for NPU evidence."""
import copy
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dabs_ascend import r234
from dabs_ascend.native_algorithms import adapted, choose, executable_key
from dabs_ascend.native_planning import planning_window, run_planner, scorer_view
from dabs_ascend.native_scoring import next_batch
from test_native import state, model


def opportunity():
    cfg, s = state()
    s['pressure'] = [8, 0]
    s['cache_evidence'] = [{'prefix_hash': k, 'replica': r, 'tokens': 256 if r == 0 else 0,
        'known': True, 'state': 'active_computed' if r == 0 else 'absent', 'valid_until_ms': 120}
        for k in s['nodes'] for r in range(2)]
    return cfg, s


class NativePlanning(unittest.TestCase):
    def test_continuation_memoization_preserves_full_216_node_rollout_and_cache_expiry(self):
        from dabs_ascend.common import DATASET, configuration
        from dabs_ascend.scheduler import snapshot
        from test_native import resources
        from test_port import FakeTokenizer
        from semantic19.dataset import load_public
        from typed19.workflow import compile_packets, instantiate
        async def prepare():
            cfg = configuration(4, resources()); cfg.update(document_tokens_budget=2000, workflow_count=64)
            tok = FakeTokenizer(); rows, _ = load_public(DATASET, 'pilot')
            packets, _ = await compile_packets(rows, tok, cfg)
            packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
            nodes, _, _ = await instantiate(packets, tok, {**cfg, 'block_size': 16}, 190500)
            await r234.convert(nodes, tok)
            for n in nodes.values():
                n.update(arrival=1., ready_at=1., state='ready' if not n['parents'] else 'blocked')
            s = snapshot(nodes, {}, 4, cfg, 1, 1.)
            s.update(native_mutable=[], native_queue_groups={str(r): {'running_prefill': [], 'waiting': []} for r in range(4)},
                     capacity_blocks=[543]*4, native_observation_errors=[])
            return cfg, s
        cfg, s = asyncio.run(prepare())
        self.assertEqual(len(s['nodes']), 216)
        h = s['nodes'][s['ready'][0]]['prefix_hash']
        s['cache_evidence'] = [{'prefix_hash': h, 'replica': 0, 'tokens': 256, 'state': 'active_computed', 'valid_until_ms': 120}]
        s['future_cache_events'] = [{'at_ms': 450, 'state': 'evicted', 'replica': 0, 'prefix_hash': h}]
        for block, step in ((128, 1024), (16, 4096)):
            with self.subTest(block=block, step=step):
                current = {**cfg, 'block_size': block, 'hard_total': step, 'base_cap': step, 'caps': [step]}
                s['current_caps'] = [step]*4
                a = r234.baseline(s, current, 'Ours')
                old = adapted('engine', optimize=False).evaluate(s, a, model(), current, record_events=True)
                new = adapted('engine').evaluate(s, a, model(), current, record_events=True)
                self.assertTrue(old['complete'])
                old.pop('cpu_wall_ms'); new.pop('cpu_wall_ms')
                old.pop('diagnostic_stage_ms'); new.pop('diagnostic_stage_ms')
                self.assertEqual(old, new)  # all events, admissions, costs, cache counters, DAG completions

    def test_115ms_observation_does_not_consume_post_observation_budget_or_extend_tick(self):
        cfg, s = opportunity()
        s['now'] = 100.115  # old loop's `now` was 100.0
        s['native_snapshots'] = [{'expires_ns': 100_230_000_000}]
        w = planning_window(s, cfg, 100.2, now=100.115)
        self.assertAlmostEqual(w['deadline'], 100.175)
        self.assertAlmostEqual(w['budget_ms'], 60)
        self.assertEqual(w['fixed_action_deadline'], 100.2)
        self.assertLess(100.0 + cfg['planning_soft_limit_s'], w['prepared_at'])  # old bug

    def test_old_engine_snapshot_and_exhausted_fixed_tick_cannot_gain_new_lease(self):
        cfg, s = opportunity(); s['now'] = 100.
        s['native_snapshots'] = [{'expires_ns': 100_050_000_000}]
        self.assertEqual(planning_window(s, cfg, 100.2, now=100.06)['budget_ms'], 0)
        s['now'] = 100.21; s['native_snapshots'] = []
        self.assertEqual(planning_window(s, cfg, 100.2, now=100.21)['budget_ms'], 0)

    def test_ipc_delay_consumes_same_absolute_deadline(self):
        cfg, s = opportunity()
        with patch('dabs_ascend.native_planning.time.monotonic', return_value=10.2):
            def planner(s, m, c, deadline):
                self.assertEqual(deadline, 10.1)  # do not restart duration on receipt
                return {}
            result = run_planner(planner, s, model(), cfg, {'deadline': 10.1, 'prepared_at': 10.})
        self.assertAlmostEqual(result['planning_dispatch_delay_ms'], 200)

    def test_full_budget_can_generate_and_completely_score_a_legal_route(self):
        cfg, s = opportunity(); r234.planner_ready(True)
        result = choose(scorer_view(s), model(), cfg, time.monotonic() + .08)
        self.assertGreater(result['v5_candidate_generated'], 0)
        self.assertTrue(any(e['cell'] == '00' and e['complete'] for e in result['evaluations']))
        self.assertGreater(result['v5_candidate_scored'], 0)

    def test_insufficient_budget_is_not_reported_as_no_opportunity(self):
        cfg, s = opportunity()
        result = choose(s, model(), cfg, time.monotonic() - 1)
        self.assertEqual(result['v5_stop_reason'], 'insufficient_budget_before_proposals')
        self.assertEqual(result['evaluations'], [])

    def test_unknown_cache_does_not_invent_a_route_opportunity(self):
        cfg, s = state()
        for n in s['nodes'].values(): n['compiled_groups'] = []
        s['pressure'] = [8, 0]
        result = choose(s, model(), cfg, time.monotonic() + 1)
        self.assertEqual(result['v5_stop_reason'], 'no_distinct_executable_candidate')
        self.assertEqual(result['candidate_diagnostics']['source_cache_unknown'], 2)
        self.assertEqual(result['v5_candidate_generated'], 0)

    def test_capacity_is_not_relaxed_to_manufacture_a_candidate(self):
        cfg, s = opportunity(); cfg['client_max_inflight'] = 1
        result = choose(s, model(), cfg, time.monotonic() + 1)
        self.assertGreater(result['candidate_diagnostics'].get('admission_set_would_change', 0), 0)
        base = r234.baseline(s, cfg, 'Ours')
        for evaluation in result['evaluations']:
            self.assertEqual(set(evaluation['action']['admit']), set(base['admit']))

    def test_guard_out_of_range_keeps_scores_but_rejects_candidate(self):
        cfg, s = opportunity(); r234.planner_ready(True)
        with patch('dabs_ascend.native_scoring.next_batch', side_effect=ValueError('native_calibration_out_of_range:p:1')):
            result = choose(s, model(), cfg, time.monotonic() + 1)
        self.assertGreater(result['v5_candidate_scored'], 0)
        self.assertTrue(result['evaluations'])
        self.assertGreater(result['decode_guard_rejections'], 0)
        self.assertEqual(result['selected_cell'], '00')
        self.assertIn('native_calibration_out_of_range:p:1', result['candidate_diagnostics'])

    def test_compact_transport_keeps_all_semantics_and_next_batch_does_not_mutate(self):
        cfg, s = opportunity()
        s['native_snapshots'] = [{'queries': [{'tokens': [42] * 3584}]}]
        for e in s['cache_evidence']: e['native'] = {'tokens': [42] * 3584}
        view = scorer_view(s)
        self.assertEqual(set(view['nodes']), set(s['nodes']))
        self.assertEqual(view['nodes'], s['nodes'])
        self.assertNotIn('native_snapshots', view)
        a = r234.baseline(s, cfg, 'Ours')
        saved = copy.deepcopy(s)
        for r in range(2): next_batch(s, a, model(), r, cfg)
        self.assertEqual(s, saved)
        original = adapted('engine').evaluate(s, a, model(), cfg)
        compact = adapted('engine').evaluate(view, a, model(), cfg)
        for key in ('rankable', 'node_completion_ms', 'workflow_prediction_ms', 'sum_remaining_jct_ms'):
            self.assertEqual(original[key], compact[key])

    def test_held_route_and_inter_replica_list_order_are_not_executable_changes(self):
        cfg, s = state(); a = r234.baseline(s, cfg, 'Ours')
        b = copy.deepcopy(a); b['order'].reverse()
        self.assertEqual(executable_key(a, s), executable_key(b, s))

    def test_copy_log_coalesces_repeated_reasons_but_retains_actual_totals(self):
        from dabs_ascend.copy_log import ConditionLog
        with tempfile.TemporaryDirectory() as d:
            log = ConditionLog(Path(d), 'Ours')
            for i in range(100):
                log.decision({'index': i, 'result': {'v5_stop_reason': 'insufficient_budget_before_proposals'}})
            text = (Path(d) / 'copy_logs/COPY.txt').read_text(encoding='utf-8')
            self.assertEqual(text.count(' D '), 3)
            self.assertEqual(log.counts['insufficient_budget_before_proposals'], 100)
            self.assertLess(len(text), 1500)
