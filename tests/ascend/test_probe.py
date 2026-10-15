"""Endpoint observations, parallel probes and bounded old-log text extraction."""
import asyncio
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from dabs_ascend.common import ROOT, sha, write_json
from dabs_ascend.r234_probe import execute, save_summary
from dabs_ascend.telemetry import metric_pressure, parse_metrics
from test_port import FakeTokenizer, api_document


class TelemetryTests(unittest.TestCase):
    def test_aggregate_gauges_are_explicit_and_unknown_is_not_zero(self):
        rows = parse_metrics('vllm:num_requests_running{model_name="qwen",tenant="SECRET"} 3\n'
                             'vllm:num_requests_waiting 2\nvllm:prefix_cache_hits_total 4\n'
                             'unrelated_secret{key="SECRET"} 1\nvllm:kv_cache_usage_perc NaN\n')
        self.assertNotIn('SECRET', json.dumps(rows))
        self.assertEqual(metric_pressure(rows)['running_plus_2_waiting'], 7)
        self.assertFalse(metric_pressure(rows)['per_prefix_cache_evidence'])
        self.assertIsNone(metric_pressure([])['running_plus_2_waiting'])
        self.assertIsNone(metric_pressure(rows + rows)['running_plus_2_waiting'])


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_72_requests_have_fixed_work_unique_ids_and_real_http_overlap(self):
        requests = []
        groups = {}
        async def handler(request):
            path = request.url.path
            if path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'Qwen2.5-1.5B-Instruct'}]})
            if path == '/openapi.json':
                doc = api_document('min_tokens')
                doc['components']['schemas']['CompletionRequest']['properties']['ignore_eos'] = {}
                return httpx.Response(200, json=doc)
            if path == '/metrics':
                return httpx.Response(200, text='vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n')
            if path == '/v1/completions':
                body = json.loads(request.content)
                request_id = request.headers['x-request-id']
                requests.append((body, request_id))
                index = len(requests) - 1
                position = (index - 16) % 14
                if index >= 16 and position >= 2:
                    group = ((index - 16) // 14, (position - 2) // 4)
                    record = groups.setdefault(group, {'count': 0, 'ready': asyncio.Event()})
                    record['count'] += 1
                    if record['count'] == 4:
                        record['ready'].set()
                    await asyncio.wait_for(record['ready'].wait(), timeout=5)
                    # Windows monotonic can have a 15.6ms tick. Keep the mock
                    # requests open beyond one tick after the barrier.
                    await asyncio.sleep(.035)
                else:
                    await asyncio.sleep(.005)
                events = [{'id': request_id, 'choices': [{'index': 0, 'text': 'test', 'finish_reason': 'length'}]},
                          {'id': request_id, 'usage': {'prompt_tokens': len(body['prompt']), 'completion_tokens': body['max_tokens']}}]
                return httpx.Response(200, text=''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n')
            return httpx.Response(200)
        class Servers:
            def __init__(self, *args): self.endpoints = ['http://test%d' % r for r in range(4)]
            async def start(self, client): pass
            def check_alive(self): pass
            async def aclose(self): pass
        original = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows = []
            with patch('httpx.AsyncClient', side_effect=lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)), \
                 patch('dabs_ascend.existing.ExistingServers', Servers), \
                 patch('dabs_ascend.backend.Tokenizer', side_effect=lambda *a: FakeTokenizer()), redirect_stdout(io.StringIO()):
                await execute(SimpleNamespace(request_timeout=10, http_keepalive=0), out, rows)
                save_summary(out, rows, 'complete', None)
            self.assertEqual(len(requests), 72)
            self.assertEqual(len(rows), 36)
            self.assertEqual(len({r[1] for r in requests}), 72)
            self.assertTrue(all(len(p['prompt']) == 3584 and p['min_tokens'] == p['max_tokens'] and p['ignore_eos'] for p, _ in requests))
            self.assertEqual(rows[0]['prompt_sha256'], rows[1]['prompt_sha256'])
            self.assertEqual(rows[0]['prompt_sha256'], rows[2]['prompt_sha256'])
            self.assertNotEqual(rows[0]['prompt_sha256'], rows[3]['prompt_sha256'])
            self.assertTrue(all(row['http_overlap_peak'] == 4 for row in rows if row['requests'] == 4))
            self.assertTrue((out / 'service_metrics.jsonl').exists())
            self.assertTrue((out / 'PROBE_SUMMARY.txt').exists())
            self.assertEqual(list(out.rglob('*.zip')), [])


class OldLogTests(unittest.TestCase):
    def test_bounded_sampling_has_coverage_labels_and_leaves_raw_untouched(self):
        spec = importlib.util.spec_from_file_location('explain_tool', ROOT / 'tools/explain_dabs_r234.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, out = root / 'old', root / 'explain'
            summary = {'spec': {'label': 'R4_L8_T1_B0_Ours', 'replicas': 4, 'rate': 8, 'repeat': 1, 'method': 'Ours'},
                       'jct_mean_ms': 10, 'v5_candidate_scored': 1}
            write_json(old / 'index.json', [summary])
            log = old / 'conditions/R4_L8_T1_B0_Ours/decisions.jsonl'
            log.parent.mkdir(parents=True)
            log.write_text('\n'.join(json.dumps({'index': i, 'result': {'evaluations': [
                {'cell': '00', 'complete': False, 'reason': 'cpu_deadline'}]}}) for i in range(3)), encoding='utf-8')
            before = sha(log)
            with redirect_stdout(io.StringIO()):
                result = module.explain(old, out, 2)
            self.assertEqual(result['decision_samples'][0]['records_read'], 2)
            self.assertEqual(result['decision_samples'][0]['coverage'], 'first_2_records_only')
            self.assertEqual(result['decision_samples'][0]['diagnostics']['score_failure_reasons'], {'cpu_deadline': 2})
            self.assertEqual(before, sha(log))
            self.assertTrue((out / 'OLD_RUN_SUMMARY.txt').exists())
            self.assertEqual(list(out.rglob('*.zip')), [])


if __name__ == '__main__':
    unittest.main()
