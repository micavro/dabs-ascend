"""Repair safety and accounting contracts. No claims of real NPU mechanisms."""
from contextlib import redirect_stderr
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from dabs_ascend import r234
from dabs_ascend.cli import prepare_args
from dabs_ascend.common import ROOT, configuration, read_json, write_json
from dabs_ascend.deployment import server_options
from dabs_ascend.existing import deployment_records
from dabs_ascend.repair import admission_check, aggregate_diagnostics, effect_signature, mechanism_gate, with_budget
from dabs_ascend.r234_run import main
from dabs_ascend.scheduler import snapshot


def ready_nodes():
    return {key: {'wid': key, 'arrival': 1., 'ready_at': 1., 'parents': [], 'state': 'ready',
                  'prompt': 3584, 'max_output_tokens': 128, 'output': 128, 'replica': None,
                  'prefix_hash': 'shared', 'start': None}
            for key in ('a', 'b')}


def action():
    return {'admit': ['b', 'a'], 'order': ['b', 'a'], 'placement': {'a': 0, 'b': 0}, 'caps': [4096, 4096]}


class RepairContracts(unittest.TestCase):
    def test_native_fallback_counts_retain_expiry_and_observation_reasons(self):
        totals = aggregate_diagnostics([
            {'native_fallback_reason': 'expired_decision_window'},
            {'native_fallback_reason': ['snapshot_expired_in_transit', 'unmapped_inflight_batch']},
            {'native_fallback_reason': [], 'native_deferred_reason': 'stale_snapshot',
             'commit_reject_reason': 'fresh_native_fallback:stale_snapshot'}])
        self.assertEqual(totals['native_atomic_fallback_decisions'], 2)
        self.assertEqual(totals['native_fallback_reasons'], {'expired_decision_window': 1,
            'snapshot_expired_in_transit': 1, 'unmapped_inflight_batch': 1})
        self.assertEqual(totals['native_deferred_admissions'], 1)
        self.assertEqual(totals['commit_reject_reasons'], {'fresh_native_fallback:stale_snapshot': 1})

    def setUp(self):
        self.cfg = {**configuration(2), 'max_num_seqs': 2}
        self.nodes = ready_nodes()

    def test_membership_and_order_are_separate(self):
        self.assertIsNone(admission_check(action(), self.nodes, self.cfg, 1., 1.01))
        baseline = {**action(), 'admit': ['a', 'b'], 'order': ['a', 'b']}
        self.assertNotEqual(effect_signature(action(), self.nodes), effect_signature(baseline, self.nodes))
        other = {**action(), 'placement': {'a': 0, 'b': 1}}
        self.assertEqual(effect_signature(other, self.nodes), effect_signature({**other, 'order': ['a', 'b']}, self.nodes))

    def test_whole_batch_rejects_over_capacity_duplicates_stale_and_dependencies(self):
        self.assertEqual(admission_check(action(), self.nodes, {**self.cfg, 'max_num_seqs': 1}, 1., 1.01), 'client_capacity_changed')
        self.assertEqual(admission_check({**action(), 'admit': ['a', 'a']}, self.nodes, self.cfg, 1., 1.01), 'duplicate_admission')
        self.assertEqual(admission_check(action(), self.nodes, self.cfg, 1., 1. + self.cfg['snapshot_max_age_s'] + .01), 'stale_snapshot')
        self.nodes['b']['parents'] = ['a']
        self.assertEqual(admission_check(action(), self.nodes, self.cfg, 1., 1.01), 'dependency_not_complete')

    def test_block_size_context_and_step_budget_have_distinct_units(self):
        self.nodes['a'].update(state='submitted', replica=0, start=1.)
        cfg = {**self.cfg, 'block_size': 128, 'max_model_len': 8192}
        state = snapshot(self.nodes, {}, 2, cfg, 1, 1.01)
        self.assertEqual(state['reserved_blocks'], [29, 0])
        self.assertEqual(state['current_caps'], [4096, 4096])
        self.assertEqual(state['resource_units']['context_limit_tokens'], 8192)
        self.assertEqual(state['resource_units']['client_max_inflight'], 2)
        self.assertIsNone(state['native_pressure'])
        self.assertIsNone(state['resource_units']['physical_kv_capacity_blocks'])

    def test_duplicate_submission_is_not_silently_deduplicated(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            r234.submission_order({**action(), 'admit': ['a', 'a']}, self.nodes)

    def test_pressure_and_cache_are_not_invented_from_history(self):
        self.nodes['a'].update(state='submitted', replica=0, start=1.)
        state = snapshot(self.nodes, {'shared': [0]}, 2, self.cfg, 1, 1.01)
        self.assertEqual(state['pressure'], [1, 0])
        self.assertIn('proxy', state['pressure_source'])
        self.assertEqual(state['cache_evidence'], [])
        self.assertIsNone(r234.frozen('v5_selector').cache_tokens(state, 'b', 0))

    def test_scoring_attempts_completions_reasons_and_late_results(self):
        rows = [{'result': {'evaluations': [
            {'cell': '00', 'complete': True},
            {'cell': '10', 'complete': False, 'reason': 'cpu_deadline'},
            {'cell': '20', 'complete': True}], 'decode_guard_rejections': 1, 'selected_cell': '00'}},
            {'event': 'expired_plan_diagnostic', 'result': {'evaluations': [
                {'cell': '00', 'complete': False, 'reason': 'no_feasible_progress'}]},
             'commit_reject_reason': 'stale_snapshot'}]
        totals = aggregate_diagnostics(rows)
        self.assertEqual(totals['reference_score_started'], 2)
        self.assertEqual(totals['reference_score_rankable'], 1)
        self.assertEqual(totals['candidate_score_started'], 2)
        self.assertEqual(totals['candidate_score_rankable'], 1)
        self.assertEqual(totals['score_failure_reasons'], {'cpu_deadline': 1, 'no_feasible_progress': 1})
        self.assertEqual(totals['guard_rejected'], 1)

    def test_worker_budget_is_relative_not_parent_absolute_deadline(self):
        def planner(state, model, cfg, deadline):
            self.assertEqual(state['now'], 987654321.)
            self.assertAlmostEqual(deadline, 42.08)
            return {}
        with patch('dabs_ascend.repair.time.monotonic', return_value=42.):
            with_budget(planner, {'now': 987654321.}, {}, {}, .08)

    def test_native_gate_stops_before_preflight_or_inference(self):
        args = prepare_args(['benchmark', '--suite', 'r234'])
        with patch('dabs_ascend.r234_run.source_identity', side_effect=AssertionError('run created')), \
             redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 2)
        self.assertFalse(mechanism_gate()['pass'])
        self.assertFalse(mechanism_gate()['inference_executed'])


class InspectionTests(unittest.TestCase):
    def load_tool(self):
        spec = importlib.util.spec_from_file_location('repair_probe_test', ROOT / 'tools/inspect_dabs_repair.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_shell_entrypoint_and_cmd_options_without_secrets(self):
        args = ['bash', '-c', 'vllm serve /model --max-num-batched-tokens 4096 --max-num-seqs=64 --api-key secret --enable-prefix-caching']
        options = server_options(args)
        self.assertEqual(options['--max-num-batched-tokens'], '4096')
        self.assertEqual(options['--max-num-seqs'], '64')
        self.assertNotIn('secret', json.dumps(options))
        row = {'Name': '/selected', 'Id': 'x', 'Image': 'y', 'State': {'Running': True},
               'HostConfig': {}, 'NetworkSettings': {}, 'Config': {'Entrypoint': args, 'Cmd': []}}
        self.assertEqual(deployment_records([row])[0]['server_options'], options)

    def test_source_snapshot_preserves_site_changes_and_excludes_outputs_and_env(self):
        tool = self.load_tool()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'old'
            project.mkdir()
            (project / 'run_dabs_r234.sh').write_text('site launcher')
            write_json(project / 'outputs/private.json', {'private': True})
            (project / '.env').write_text('SECRET=not-for-bundle')
            module = project / 'experiments/ascend/dabs_ascend/r234_run.py'
            module.parent.mkdir(parents=True)
            module.write_text('max_keepalive_connections=0')
            result = tool.snapshot_project(project, root / 'probe')
            self.assertEqual(len(result['files_sha256']), 2)
            copied = root / 'probe/project_source/experiments/ascend/dabs_ascend/r234_run.py'
            self.assertEqual(copied.read_text(), module.read_text())
            self.assertFalse((root / 'probe/project_source/.env').exists())

    def test_probe_only_inspects_or_executes_readonly_python_and_collects_no_env(self):
        tool = self.load_tool()
        calls = []
        def command(argv, **kwargs):
            calls.append(argv)
            if argv[:3] == ['sudo', 'docker', 'inspect']:
                device = int(argv[-1][-1])
                return json.dumps([{'Name': '/' + argv[-1], 'Id': 'id', 'Image': 'image',
                    'State': {'Running': True}, 'HostConfig': {'Devices': [{'PathOnHost': '/dev/davinci%d' % device}]},
                    'NetworkSettings': {'Ports': {'8000/tcp': [{'HostPort': str(8001 + device)}]}},
                    'Config': {'Env': ['SECRET=hidden'], 'Cmd': ['--api-key', 'hidden']}}])
            if argv[:4] == ['sudo', 'docker', 'exec', '-i']:
                self.assertNotIn('import torch', kwargs['stdin'])
                return json.dumps({'packages': {'vllm': 'test', 'vllm-ascend': 'test'}, 'sources': {}})
            if argv == ['npu-smi', 'info']:
                return 'fixture only'
            self.fail('Unexpected command: ' + str(argv))
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, *args): return b'{"data":[{"id":"test","max_model_len":4096}],"paths":{}}'
        class Opener:
            def open(self, *args, **kwargs): return Response()
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'project'
            project.mkdir()
            (project / 'run_dabs_r234.sh').write_text('test')
            args = SimpleNamespace(devices='0,1,2,3', container_prefix='test-npu', out=root / 'probe',
                                   project=project, model=root / 'model', python='python3', base_port=8001,
                                   wait_ready=0, poll_interval=5., log_tail=80, max_inflight=2)
            with patch.object(tool, 'command', command), patch.object(tool, 'build_opener', return_value=Opener()):
                self.assertEqual(tool.collect(args), 0)
            saved = (root / 'probe/inspection.json').read_text()
            self.assertNotIn('hidden', saved)
            self.assertFalse(read_json(root / 'probe/inspection.json')['native_mechanisms_verified'])
            self.assertEqual(len(calls), 9)
            self.assertFalse((root / 'probe.zip').exists())
            self.assertTrue((root / 'probe/SUMMARY.txt').exists())
            self.assertIn('process_options:', (root / 'probe/SUMMARY.txt').read_text(encoding='utf-8'))


@unittest.skipUnless(platform.system() == 'Linux', 'Linux host entry tests')
class HostTextTools(unittest.TestCase):
    def test_prepare_waits_and_stops_before_probe_if_inspection_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / 'python3'
            log = root / 'arguments.json'
            python.write_text('#!' + sys.executable + '\nimport json, os, sys\n'
                              'open(os.environ["TEST_ARGUMENTS"], "w").write(json.dumps(sys.argv[1:]))\n'
                              'sys.exit(1)\n')
            python.chmod(0o755)
            sudo = root / 'sudo'
            sudo.write_text('#!/bin/sh\necho UNEXPECTED_DOCKER >&2\nexit 99\n')
            sudo.chmod(0o755)
            env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'],
                   'TEST_ARGUMENTS': str(log), 'READY_TIMEOUT': '120'}
            result = subprocess.run(['bash', str(ROOT / 'run_dabs_r234.sh'), 'prepare'],
                                    capture_output=True, text=True, timeout=10, env=env)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertNotIn('UNEXPECTED_DOCKER', result.stderr)
            self.assertEqual(json.loads(log.read_text()),
                             [str(ROOT / 'tools/inspect_dabs_repair.py'), '--wait-ready', '120'])

    def test_inspect_and_report_bypass_controller_and_preserve_space_in_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / 'python3'
            log = root / 'arguments.json'
            python.write_text('#!' + sys.executable + '\nimport json, os, sys\n'
                              'open(os.environ["TEST_ARGUMENTS"], "w").write(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            sudo = root / 'sudo'
            sudo.write_text('#!/bin/sh\nexit 99\n')
            sudo.chmod(0o755)
            env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'], 'TEST_ARGUMENTS': str(log)}
            for mode, tool in (('inspect', 'inspect_dabs_repair.py'), ('report', 'report_dabs_r234.py'),
                               ('explain', 'explain_dabs_r234.py')):
                result = subprocess.run(['bash', str(ROOT / 'run_dabs_r234.sh'), mode, '--out', '/path with spaces'],
                                        capture_output=True, text=True, timeout=10, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(log.read_text()), [str(ROOT / 'tools' / tool), '--out', '/path with spaces'])
            self.assertEqual(list(root.glob('*.zip')), [])


if __name__ == '__main__':
    unittest.main()
