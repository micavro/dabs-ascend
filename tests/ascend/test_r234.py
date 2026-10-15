"""New suite contracts and CPU integration; these are not NPU measurements."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout, nullcontext
import copy
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from dabs_ascend import r234
from dabs_ascend.backend import Backend
from dabs_ascend.cli import make_plan, prepare_args
from dabs_ascend.common import DATASET, ROOT, append_json, configuration, read_json, sha, write_json
from dabs_ascend.r234_report import make_report, paired_rows, procedure
from dabs_ascend.r234_run import audit_completed, verify_pairing
from dabs_ascend.r234_run import main as run_suite
from dabs_ascend.runner import run_condition
from dabs_ascend.scheduler import snapshot
from test_port import FakeTokenizer, model_for, api_document
from semantic19.dataset import load_public
from typed19.workflow import compile_packets, instantiate


class PlanTests(unittest.TestCase):
    def test_imported_373_files_unchanged(self):
        self.assertEqual(r234.verify_import()['verified_files'], 373)

    def test_exact_96_conditions_and_mirror_order(self):
        args = prepare_args(['plan', '--suite', 'r234', '--devices', '0,1,2,3'], existing=True)
        plan = make_plan(args)
        frozen = read_json(ROOT / 'source/spra_workflow_fullgrid_r234_sll_v1/plans/dabs-r234-balanced-64.json') if (ROOT / 'source/spra_workflow_fullgrid_r234_sll_v1/plans/dabs-r234-balanced-64.json').exists() else None
        self.assertEqual(len(plan), 96)
        self.assertEqual([s['method'] for s in plan[:4]], list(r234.METHODS))
        self.assertTrue(all(s['replicas'] == 4 and s['rate'] == 8 for s in plan[:4]))
        keys = lambda seq: [(s['replicas'], s['rate'], s['method']) for s in seq]
        self.assertEqual(keys(plan[:48]), keys(list(reversed(plan[48:]))))
        self.assertEqual({s['seed'] for s in plan}, {190500})
        self.assertEqual({s['router_seed'] for s in plan}, {950100, 950200})
        self.assertEqual({(s['replicas'], s['rate']) for s in plan}, {(r, a) for r in (2, 3, 4) for a in (4, 8, 12, 16)})
        if frozen:
            self.assertEqual(plan, frozen['conditions'])

    def test_protocol_overrides_rejected(self):
        for extra in (['--rate', '2'], ['--seed', '4'], ['--repeats', '1'], ['--devices', '0,1'], ['--max-conditions', '0']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                prepare_args(['plan', '--suite', 'r234', '--devices', '0,1,2,3'] + extra)

    def test_partial_report_never_claims_complete_or_native(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            report = make_report(out, r234.plan(), [], 'paused')
            self.assertFalse(report['functional_test_pass'])
            self.assertFalse(report['strict_native_reproduction_pass'])
            self.assertIsNone(report['native_kv_hit_rate'])
            self.assertIn('0/96', (out / 'TEST_REPORT.md').read_text(encoding='utf-8'))
            self.assertTrue((out / 'RESULTS.html').exists())

    def test_ready_order_is_applied_and_submitted_order_removed(self):
        nodes = {'a': {'ready_at': 1}, 'b': {'ready_at': 2}, 'c': {'ready_at': 3}}
        action = {'admit': ['a', 'b', 'c'], 'order': ['submitted', 'b', 'a']}
        projected = r234.ready_projection(action, {'ready': list(nodes)})
        self.assertEqual(projected['order'], ['b', 'a'])
        self.assertEqual(r234.submission_order(projected, nodes), ['b', 'a', 'c'])
        self.assertEqual(action['order'][0], 'submitted')

    def test_surrogate_cannot_promote_ready_requests_ahead_of_submitted(self):
        nodes = {'a': {'ready_at': 1, 'state': 'ready'}, 'b': {'ready_at': 2, 'state': 'ready'},
                 'old': {'ready_at': 0, 'state': 'submitted', 'start': 0, 'phase': 'prefill'}}
        state = {'nodes': nodes, 'ready': ['a', 'b']}
        action = {'admit': ['a', 'b'], 'order': ['b', 'a']}
        self.assertEqual(r234.scoring_action(action, state)['order'], ['old', 'b', 'a'])
        self.assertEqual(r234.scoring_action({**action, 'order': []}, state)['order'], ['old', 'a', 'b'])

    def test_resume_checks_raw_artifacts_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            plan = r234.plan()
            summary = {'spec': plan[0], 'functional_pass': True}
            case = out / 'conditions' / plan[0]['label']
            names = ('config.json', 'spec.json', 'initial_nodes.json', 'input_proof.json', 'nodes.json',
                     'requests.jsonl', 'decisions.jsonl', 'summary.json', 'quality.json', 'status.json')
            for name in names:
                write_json(case / name, summary if name == 'summary.json' else {})
            write_json(case / 'artifacts.json', {n: sha(case / n) for n in names})
            write_json(out / 'index.json', [summary])
            self.assertEqual(audit_completed(out, plan), [summary])
            (case / 'requests.jsonl').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                audit_completed(out, plan)

    def test_pairing_and_signed_comparison(self):
        base = {'spec': r234.plan()[0], 'external_work_fingerprint': 'a', 'arrival_offsets': [0, 1], 'jct_mean_ms': 100}
        ours = {**base, 'spec': r234.plan()[3], 'jct_mean_ms': 125}
        self.assertEqual(paired_rows([base, ours])[0]['ours_jct_reduction_pct'], -25)
        with self.assertRaisesRegex(ValueError, 'work differs'):
            verify_pairing([base], {**ours, 'external_work_fingerprint': 'changed'})


class FixedBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_sampling_payload_and_short_generation_failure(self):
        payloads = []
        completion = 16

        def handler(request):
            body = json.loads(request.content)
            payloads.append(body)
            events = [{'choices': [{'index': 0, 'text': 'text', 'finish_reason': 'length'}]},
                      {'choices': [], 'usage': {'prompt_tokens': len(body['prompt']), 'completion_tokens': completion}}]
            return httpx.Response(200, text=''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = Backend(client, ['http://fake'])
            backend.fields = ['guided_json']
            node = {'id': 'fixed', 'tokens': [1] * 3584, 'max_output_tokens': 16, 'fixed_workload_replay': True}
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / 'requests.jsonl'
                await backend.generate(0, node, 190500, log, schema=None)
                self.assertEqual(payloads[-1]['min_tokens'], 16)
                self.assertTrue(payloads[-1]['ignore_eos'])
                self.assertNotIn('guided_json', payloads[-1])
                self.assertIsNone(node['answer_contract'])
                completion = 8
                with self.assertRaisesRegex(ValueError, 'Fixed replay output length'):
                    await backend.generate(0, copy.deepcopy(node), 190500, log, schema=None)
                natural = {'id': 'natural', 'tokens': [1] * 32, 'max_output_tokens': 16}
                await backend.generate(0, natural, 190500, log, schema=None)
                self.assertNotIn('min_tokens', payloads[-1])
                self.assertNotIn('ignore_eos', payloads[-1])

    async def test_fixed_inspection_needs_no_json_constraint(self):
        def handler(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'Qwen2.5-1.5B-Instruct'}]})
            document = api_document('min_tokens')
            document['components']['schemas']['CompletionRequest']['properties']['ignore_eos'] = {}
            return httpx.Response(200, json=document)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await Backend(client, ['http://fake']).inspect(fixed=True)
            self.assertIsNone(result[0]['constraint_field'])


class SimulatedFixedBackend:
    async def generate(self, replica, node, seed, log, schema=None):
        if schema is not None or not node.get('fixed_workload_replay'):
            raise AssertionError('Expected fixed workload, no schema')
        start = time.monotonic()
        await asyncio.sleep(.001)
        finish = time.monotonic()
        node['answer_contract'] = None
        node['stream'] = {'text': 'CPU TEST ONLY', 'usage': {'prompt_tokens': len(node['tokens']),
                          'completion_tokens': node['max_output_tokens']}, 'finish_reason': 'length',
                          'start': start, 'finish': finish, 'ttft_text_proxy_ms': .1, 'tpot_text_proxy_ms': .1}
        append_json(log, {'cpu_test_only': True, 'node': node['id'], 'usage': node['stream']['usage']})
        return node['stream']


class RecordedExecutionTests(unittest.TestCase):
    def test_clean_pause_resume_executes_next_condition_and_preserves_raw_files(self):
        class FakeServers:
            def __init__(self, args, out):
                self.endpoints = ['http://cpu-test-only/' + str(i) for i in range(4)]
                self.out = out
            async def start(self, client):
                write_json(self.out / 'servers.json', {'cpu_test_only': True})
            def check_alive(self):
                pass
            async def aclose(self):
                write_json(self.out / 'cleanup.json', {'external_services_stopped': False})

        class FakeBackend(SimulatedFixedBackend):
            def __init__(self, client, endpoints):
                self.endpoints = endpoints
            async def inspect(self, fixed=False):
                return {'cpu_test_only': True}
            async def fixed_canaries(self, out):
                return [{'replica': r, 'fixed_output_budget': n, 'passed': True}
                        for r in range(4) for n in (16, 32, 48, 64, 128)]

        async def calibrate(backend, tokenizer, out, *, fixed=False):
            self.assertTrue(fixed)
            model = model_for(4)
            write_json(out / 'ascend_profile.json', model)
            write_json(out / 'calibration_rows.json', {'cpu_test_only': True})
            return model

        def preflight(args, out):
            write_json(out / 'environment.json', {'passed': True, 'deployment': [], 'cpu_test_only': True})

        cfg = configuration(4)
        cfg.update(document_tokens_budget=2000, epoch_s=.05, commit_reserve_s=.005)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'results'
            model = root / 'model'
            write_json(model / 'config.json', {'cpu_test_only': True})
            common = ['benchmark', '--suite', 'r234', '--devices', '0,1,2,3', '--model', str(model),
                      '--deployment-manifest', str(root / 'fake.json'), '--max-conditions', '1']
            with patch('dabs_ascend.r234_run.device_locks', return_value=nullcontext()), \
                 patch('dabs_ascend.existing.preflight', preflight), \
                 patch('dabs_ascend.existing.ExistingServers', FakeServers), \
                 patch('dabs_ascend.backend.Backend', FakeBackend), \
                 patch('dabs_ascend.backend.Tokenizer', lambda *a: FakeTokenizer()), \
                 patch('dabs_ascend.calibration.calibrate', calibrate), \
                 patch('dabs_ascend.r234_run.configuration', lambda *a: copy.deepcopy(cfg)), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                args = prepare_args(common + ['--out', str(out)], existing=True)
                self.assertEqual(run_suite(args), 0, (out / 'error.txt').read_text() if (out / 'error.txt').exists() else '')
                first = read_json(out / 'index.json')[0]
                self.assertEqual(read_json(out / 'status.json')['status'], 'paused')
                raw = sha(out / 'conditions' / first['spec']['label'] / 'requests.jsonl')
                args = prepare_args(common + ['--resume', str(out)], existing=True)
                self.assertEqual(run_suite(args), 0, (out / 'error.txt').read_text() if (out / 'error.txt').exists() else '')
                self.assertEqual(len(read_json(out / 'index.json')), 2)
                self.assertEqual(read_json(out / 'index.json')[1]['spec']['method'], 'SLL')
                self.assertEqual(sha(out / 'conditions' / first['spec']['label'] / 'requests.jsonl'), raw)
                self.assertEqual(len(audit_completed(out, r234.plan())), 2)
                self.assertIn('R234_CONDITION 2/96', (out / 'console.log').read_text())
                self.assertTrue((out / 'REPRODUCE.md').exists())
                self.assertFalse(read_json(out / 'acceptance.json')['functional_test_pass'])
                write_json(out / 'status.json', {'status': 'failed'})
                self.assertEqual(run_suite(args), 2)


class FullWorkloadTests(unittest.IsolatedAsyncioTestCase):
    async def setup_data(self):
        cfg = configuration(4)
        cfg.update(document_tokens_budget=2000, max_num_seqs=2, workflow_count=64, arrival_rate=1000,
                   epoch_s=.05, planning_soft_limit_s=.02, commit_reserve_s=.005, condition_wall_s=60.)
        tok = FakeTokenizer()
        rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        return cfg, tok, r234.frozen('heterogeneous').select_packets(packets, 'balanced')

    async def test_balanced_work_fingerprints_and_unknown_cache(self):
        cfg, tok, (packets, descriptor) = await self.setup_data()
        self.assertEqual(len(packets), 64)
        self.assertEqual(len(descriptor['source_packet_counts']), 24)
        self.assertEqual(descriptor['total_declared_output_tokens'], 13184)
        nodes, _, _ = await instantiate(packets, tok, cfg, 190500)
        proof = await r234.convert(nodes, tok)
        self.assertEqual(len(nodes), 216)
        self.assertEqual({len(n['tokens']) for n in nodes.values()}, {3584})
        self.assertEqual(len({n['hash'] for n in nodes.values()}), 216)
        other, _, _ = await instantiate(packets, tok, cfg, 190500)
        other_proof = await r234.convert(other, tok)
        self.assertEqual(proof['external_work_fingerprint'], other_proof['external_work_fingerprint'])
        self.assertNotEqual(next(iter(nodes.values()))['prefix_hash'], next(iter(other.values()))['prefix_hash'])
        now = time.monotonic()
        for n in nodes.values():
            n['arrival'] = now
            if not n['parents']:
                n.update(state='ready', ready_at=now)
        state = snapshot(nodes, {}, 4, cfg, 1, now)
        base = r234.baseline(state, cfg, 'Ours')
        selector = r234.frozen('v5_selector')
        opts = selector.proposals(state, model_for(4), cfg, base)
        self.assertFalse(any(o[2] == 'route_only' for o in opts))
        result = r234.plan_ours(state, model_for(4), cfg, time.monotonic() + 1.)
        self.assertIn('v5_revision', result)
        self.assertFalse(result['native_cache_observed'])
        for evaluation in result['evaluations']:
            self.assertTrue(set(evaluation['action']['order']) <= set(state['ready']))

    async def test_all_four_methods_complete_real_216_node_dag_on_mock_backend(self):
        cfg, tok, (packets, _) = await self.setup_data()
        summaries = []
        with tempfile.TemporaryDirectory() as directory, ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn')) as pool:
            await asyncio.get_running_loop().run_in_executor(pool, r234.planner_ready)
            for spec in r234.plan()[:4]:
                current = {**cfg, 'wf_router_seed': spec['router_seed']}
                summary = await run_condition(SimulatedFixedBackend(), tok, packets, model_for(4), current,
                                              spec, Path(directory) / spec['label'], pool)
                verify_pairing(summaries, summary)
                summaries.append(summary)
                self.assertTrue(summary['functional_pass'])
                self.assertEqual(summary['completed_workflows'], 64)
                self.assertEqual(summary['completed_requests'], 216)
                self.assertEqual(summary['input_tokens'], 774144)
                self.assertEqual(summary['output_tokens'], 13184)
                self.assertIsNone(summary['quality']['final_em'])
                self.assertTrue(all(r['peak_http_inflight'] <= 2 for r in summary['replica_rows']))
            self.assertGreater(summaries[-1]['v5_decisions'], 0)
            report = make_report(Path(directory), r234.plan(), summaries, 'paused')
            self.assertFalse(report['functional_test_pass'])
            self.assertEqual(report['completed_conditions'], 4)
            for filename in ('metrics.csv', 'paired_comparisons.csv', 'workflow_metrics.csv',
                             'latency_decomposition.csv', 'replica_load.csv', 'RESULTS.html'):
                self.assertTrue((Path(directory) / filename).is_file())


if __name__ == '__main__':
    unittest.main()
