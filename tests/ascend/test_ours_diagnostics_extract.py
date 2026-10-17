"""Offline extraction fixtures; no model/service execution or user-data writes."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dabs_ascend.common import ROOT, write_json
from test_native_workflow import load_tool

extract = load_tool('extract_ours_diagnostics')


def jsonl(path, records):
    path.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')


def fixture(directory, nc_count=2, obs_count=1):
    directory.mkdir(parents=True)
    identity = {'engine_instance': 'fixture'}
    ref = {'replica': 0, 'snapshot_id': 'original', 'identity': identity, 'captured_ns': 10_000_000_000, 'expires_ns': 10_120_000_000}
    reasons = ['request_in_API_EngineCore_transit', 'request_completed_during_observation', 'unmapped_inflight_batch']
    state = {'now': 10.1, 'native_snapshot_refs': [ref], 'native_observation_errors': reasons,
             'nodes': {k: {'state': 'submitted', 'replica': 0} for k in ('mapping', 'incoming', 'engine_done', 'unknown', 'inside_lifetime')},
             'ready': []}
    records = []
    for i in range(nc_count):
        records += [{'index': i, 'snapshot': {'ready': ['ready'], 'nodes': {'ready': {'compiled_groups': []}}}},
                    {'event': 'unbounded_planner_timing', 'index': i, 'result': {
                        'v5_stop_reason': extract.NO_CANDIDATE, 'shadow_timing': {'generated': 0},
                        'candidate_diagnostics': {'proposal_calls': 1, 'shortlist_before_native_dedup': 1,
                            'no_distinct_native_effect': 1, 'no_effect_order_only': 1, 'distinct_executable_candidates': 0,
                            'pressure_cache_eligible_routes': 0},
                        'opportunity_diagnostics': {'ready': 1, 'source_known_ready': 1, 'mutable': 0, 'client_slots_after_reference': [0]},
                        'candidate_diagnostic_samples': [{'kind': 'native_duplicate', 'origin': 'order_only', 'held_only': False}]}}]
    for i in range(nc_count, nc_count+obs_count):
        records += [{'index': i, 'snapshot': state, 'planning_window': {'prepared_at': 10.11},
                     'fallback_snapshot': {'now': 99, 'native_snapshot_refs': []}},
                    {'event': 'unbounded_planner_timing', 'index': i, 'result': {
                        'v5_stop_reason': extract.UNUSABLE, 'native_fallback': reasons,
                        'shadow_timing': {'generated': 0}}}]
    records += [{'index': nc_count+obs_count, 'event': 'unbounded_planner_timing', 'result': {
        'shadow_timing': {'generated': 2, 'all_scores_complete': True}}}]
    jsonl(directory/'decisions.jsonl', records)
    mappings = [{'node_id': k, 'engine_request_id': 'rid-'+k, 'engine_admitted_ns': int(a*1e9), 'engine_finished_ns': int(b*1e9)}
                for k, a, b in [('mapping', 9., 9.99), ('incoming', 10.05, 10.5), ('engine_done', 9., 9.99),
                                ('inside_lifetime', 9., 10.5), ('done', 9., 10.02)]]
    events = [{'op': 'snapshot', 'replica': 0, 'client_start': 9.99, 'client_finish': 10.09, 'response': {
        **ref, 'running': [{'engine_request_id': 'rid-done', 'mapping': mappings[-1]}],
        'waiting': [], 'skipped_waiting': [], 'inflight_batches': [{'requests': [
            {'engine_request_id': 'rid-done'}, {'engine_request_id': 'rid-orphan'}]}]}}]
    for m in mappings:
        events.append({'op': 'mapping', 'replica': 0, 'client_start': 10.02 if m['node_id'] == 'mapping' else 10.4,
                       'client_finish': 10.14 if m['node_id'] == 'mapping' else 10.6, 'response': {'mappings': [m]}})
    jsonl(directory/'native_events.jsonl', events)
    write_json(directory/'nodes.json', {**{m['node_id']: {'finish': 10.15 if m['node_id'] == 'mapping' else 10.8,
        'stream': {'finish': 10.01 if m['node_id'] == 'mapping' else 10.3}} for m in mappings},
        'done': {'finish': 10.05, 'stream': {'finish': 10.03}}, 'unknown': {'finish': 10.8},
        'sensitive': {'tokens': ['SECRET_PROMPT_NOT_FOR_COPY']}})
    jsonl(directory/'requests.jsonl', [{'event': 'response', 'node': 'mapping', 'result': {
        'finish': 10.01, 'text': 'SECRET_PROMPT_NOT_FOR_COPY', 'native_mapping': {'mappings': [mappings[0]]}}}])


class ExtractionTests(unittest.TestCase):
    def test_lifecycle_join_uses_original_snapshot_and_distinguishes_causes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'condition'
            fixture(root)
            result = extract.collect(root)
            self.assertEqual(result['counts']['no_candidate'], 2)
            self.assertEqual(result['counts']['observation_unusable'], 1)
            self.assertEqual(result['stages']['all_removed_as_native_duplicates'], 2)
            expected = ['http_done_waiting_mapping_RPC', 'not_engine_admitted_at_capture',
                        'engine_finished_before_capture_client_still_submitted', 'transit_lifecycle_evidence_insufficient',
                        'missing_queue_inside_recorded_engine_lifetime', 'client_completed_between_capture_and_observe',
                        'inflight_unmapped_after_completed_filter', 'inflight_request_absent_from_snapshot_queues']
            self.assertEqual(result['lifecycle'], dict.fromkeys(expected, 1))
            self.assertEqual(result['lifecycle_calls'], dict.fromkeys(expected, 1))
            self.assertEqual(result['transit_capture'], {'pre_engine_admission': 1, 'post_engine_finish': 2,
                                                        'inside_recorded_lifetime': 1, 'insufficient_lifecycle_evidence': 1})
            self.assertEqual(result['transit_client']['http_done_mapping_RPC_pending'], 1)
            self.assertEqual(result['gaps'], {})
            sample = next(s for s in result['samples'] if s['kind'] == 'OBS:http_done_waiting_mapping_RPC')
            self.assertEqual(sample['ms_from_engine_capture']['http_done'], 10)
            self.assertEqual(sample['ms_from_engine_capture']['client_done'], 150)
            self.assertEqual(sample['ms_from_engine_capture']['client_observed'], 100)

    def test_617_and_69_are_not_sample_capped_and_input_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            condition = root/'r234_ours_timing/conditions/R3_L8_T1_B0_Ours'
            fixture(condition, 617, 69)
            saved = {p: p.read_bytes() for p in condition.iterdir()}
            output = extract.report(root)
            self.assertIn('"no_candidate":617', output)
            self.assertIn('"observation_unusable":69', output)
            self.assertIn('"http_done_waiting_mapping_RPC":69', output)
            self.assertIn('"shadow_returns":687', output)
            self.assertIn('omitted_summary_lines=0', output)
            self.assertIn('"kind":"NC:', output)
            self.assertNotIn('SECRET_PROMPT_NOT_FOR_COPY', output)
            self.assertLessEqual(len(output.encode('utf-8')), 12*1024)
            self.assertEqual(saved, {p: p.read_bytes() for p in condition.iterdir()})

    def test_missing_and_damaged_logs_remain_explicit_not_inferred(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'condition'
            fixture(root)
            (root/'native_events.jsonl').unlink()
            with (root/'decisions.jsonl').open('a', encoding='utf-8') as stream:
                stream.write('{incomplete\n')
            result = extract.collect(root)
            self.assertEqual(result['gaps']['missing:native_events.jsonl'], 1)
            self.assertEqual(result['gaps']['invalid_row:decisions.jsonl'], 1)
            self.assertEqual(result['gaps']['missing_referenced_engine_snapshot'], 1)
            self.assertFalse(result['lifecycle'])

    def test_expiry_in_rpc_and_during_projection_are_separate(self):
        for expiry, expected in [(10.08, 'snapshot_expired_at_client_observe'), (10.12, 'snapshot_expired_during_projection')]:
            with self.subTest(expiry=expiry), tempfile.TemporaryDirectory() as d:
                root = Path(d)/'condition'
                fixture(root, 0, 1)
                decisions = list(extract.rows(root/'decisions.jsonl'))
                decisions[0]['snapshot']['native_projection_ms'] = 30
                jsonl(root/'decisions.jsonl', decisions)
                events = list(extract.rows(root/'native_events.jsonl'))
                events[0]['response']['expires_ns'] = int(expiry*1e9)
                jsonl(root/'native_events.jsonl', events)
                result = extract.collect(root)
                self.assertEqual(result['lifecycle'][expected], 1)

    def test_empty_shortlist_and_unknown_cache_have_separate_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'condition'
            fixture(root, 0, 0)
            base = {'event': 'unbounded_planner_timing', 'index': 1, 'result': {
                'v5_stop_reason': extract.NO_CANDIDATE,
                'candidate_diagnostics': {'proposal_calls': 1, 'shortlist_before_native_dedup': 0, 'source_cache_unknown': 3},
                'opportunity_diagnostics': {'ready': 3, 'source_known_ready': 0}}}
            other = copy.deepcopy(base); other['index'] = 2
            other['result']['candidate_diagnostics'] = {}
            jsonl(root/'decisions.jsonl', [base, other])
            result = extract.collect(root)
            self.assertEqual(result['stages'], {'empty_before_native_dedup': 1, 'missing_candidate_diagnostics': 1})
            self.assertEqual(result['funnel']['source_cache_unknown'], 3)
            self.assertEqual(result['funnel_calls']['source_cache_unknown'], 1)
            self.assertEqual(result['nc_flags']['ready_but_no_known_source_cache'], 2)

    def test_standalone_command_from_outside_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'condition'
            fixture(root)
            run = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT/'tools/extract_ours_diagnostics.py'),
                                  '--input', str(root)], cwd=d, capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('OBS_LIFECYCLE_CALLS', run.stdout)
            self.assertEqual(run.stderr, '')
