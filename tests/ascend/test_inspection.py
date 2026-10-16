"""Field regressions: stopped containers, startup readiness and resource units."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

from dabs_ascend.common import ROOT, read_json
from dabs_ascend.deployment import resource_comparison


def load_tool():
    spec = importlib.util.spec_from_file_location('field_inspection', ROOT / 'tools/inspect_dabs_repair.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def raw_container(name, running=True):
    device = int(name[-1])
    return {'Name': '/' + name, 'Id': name, 'Image': 'image', 'RestartCount': 0,
            'State': {'Running': running, 'Status': 'running' if running else 'exited',
                      'ExitCode': 0 if running else 137, 'OOMKilled': False,
                      'StartedAt': '2026-10-16T04:29:53.381131Z', 'FinishedAt': None},
            'HostConfig': {'Devices': [{'PathOnHost': '/dev/davinci%d' % device}]},
            'NetworkSettings': {'Ports': {'8000/tcp': [{'HostPort': str(8001 + device)}]}},
            'Config': {'Cmd': ['--max-num-batched-tokens', '1024', '--max-num-seqs', '10',
                               '--max-model-len', '4096'], 'Env': ['TOKEN=DO_NOT_COPY']}}


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, *args): return b''


class FieldInspectionTests(unittest.TestCase):
    def test_stopped_services_preserve_state_and_logs_without_exec_or_http(self):
        tool = load_tool()
        calls = []
        def command(argv, **kwargs):
            calls.append(argv)
            if argv[:3] == ['sudo', 'docker', 'inspect']:
                return json.dumps([raw_container(argv[-1], running=False)])
            if argv[:3] == ['sudo', 'docker', 'logs']:
                self.assertTrue(kwargs['include_stderr'])
                self.assertIn('2026-10-16T04:29:53.381131Z', argv)
                return '2026-10-16 ERROR engine failed\nconfig api_key=DO_NOT_COPY\nprompt: DO_NOT_COPY\n'
            if argv == ['npu-smi', 'info']:
                return 'test'
            self.fail(str(argv))
        class Opener:
            def open(self, *args, **kwargs):
                raise AssertionError('No HTTP to stopped service')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'project'
            project.mkdir()
            (project / 'run_dabs_r234.sh').write_text('test')
            args = SimpleNamespace(devices='0,1,2,3', container_prefix='test-npu', base_port=8001,
                out=root / 'probe', project=project, model=root / 'model', python='python3',
                wait_ready=300, poll_interval=5., log_tail=80, max_inflight=2)
            with patch.object(tool, 'command', command), patch.object(tool, 'build_opener', return_value=Opener()), \
                 patch.object(tool.time, 'sleep', side_effect=AssertionError('No wait for exited services')), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(tool.collect(args), 1)
            facts = read_json(args.out / 'inspection.json')
            self.assertEqual(facts['generation_requests'], 0)
            self.assertFalse(facts['complete'])
            self.assertEqual(len(facts['errors']), 4)
            for row in facts['services']:
                self.assertEqual(row['deployment']['state']['ExitCode'], 137)
                self.assertFalse(row['deployment']['state']['OOMKilled'])
                self.assertEqual(row['readiness'], 'container_not_ready')
                self.assertIsNone(row['runtime'])
                self.assertFalse(row['resource_comparison']['step_budget_matches'])
            summary = (args.out / 'SUMMARY.txt').read_text(encoding='utf-8')
            self.assertIn('原因未确定', summary)
            self.assertIn('ERROR engine failed', summary)
            for path in args.out.rglob('*'):
                if path.is_file():
                    self.assertNotIn('DO_NOT_COPY', path.read_text(encoding='utf-8'))
            self.assertEqual(len(calls), 9)
            self.assertFalse(list(args.out.rglob('*.zip')))

    def test_wait_retries_reset_and_finishes_only_when_all_four_healthy(self):
        tool = load_tool()
        urls = []
        class Opener:
            def open(self, url, **kwargs):
                urls.append(url)
                if len(urls) <= 4:
                    raise URLError(ConnectionResetError(104, 'reset'))
                return Response()
        args = SimpleNamespace(devices='0,1,2,3', container_prefix='test-npu', base_port=8001,
                               wait_ready=300, poll_interval=5.)
        names = ['test-npu%d' % d for d in range(4)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(tool, 'command', side_effect=lambda argv: json.dumps([raw_container(argv[-1])])), \
             patch.object(tool.time, 'sleep') as sleep, redirect_stdout(io.StringIO()):
            rows = tool.wait_services(args, names, Path(directory), Opener())
            events = [json.loads(line) for line in (Path(directory) / 'readiness.jsonl').read_text().splitlines()]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['services'][0]['health_error']['errno'], 104)
        self.assertTrue(all(r['health_ok'] for r in rows))
        self.assertEqual(len(urls), 8)
        sleep.assert_called_once()

    def test_zero_wait_failure_and_wrong_port_never_count_as_ready(self):
        tool = load_tool()
        args = SimpleNamespace(devices='0,1,2,3', container_prefix='test-npu', base_port=8001,
                               wait_ready=0, poll_interval=5.)
        class Opener:
            def open(self, *args, **kwargs): raise URLError(ConnectionRefusedError(111, 'refused'))
        for bad_port in (False, True):
            def command(argv):
                row = raw_container(argv[-1])
                if bad_port:
                    row['NetworkSettings']['Ports'] = {}
                return json.dumps([row])
            with self.subTest(bad_port=bad_port), tempfile.TemporaryDirectory() as directory, \
                 patch.object(tool, 'command', command), patch.object(tool.time, 'sleep') as sleep, \
                 redirect_stdout(io.StringIO()):
                rows = tool.wait_services(args, ['test-npu%d' % d for d in range(4)], Path(directory), Opener())
                self.assertFalse(any(r['health_ok'] for r in rows))
                self.assertTrue(all(r['readiness'] == ('deployment_unavailable_or_invalid' if bad_port else 'waiting_for_health') for r in rows))
                sleep.assert_not_called()

    def test_command_errors_are_actionable_without_persisting_raw_secrets(self):
        tool = load_tool()
        result = subprocess.CompletedProcess([], 1, '', 'Container SECRET is not running')
        with patch.object(tool.subprocess, 'run', return_value=result):
            with self.assertRaises(tool.CommandFailure) as raised:
                tool.command(['sudo', 'docker', 'exec'])
        detail = tool.error_detail(raised.exception)
        self.assertEqual(detail['reason'], 'container_not_running')
        self.assertEqual(detail['exit_code'], 1)
        self.assertNotIn('SECRET', json.dumps(detail))
        result = subprocess.CompletedProcess([], 0, 'stdout', 'ERROR stderr')
        with patch.object(tool.subprocess, 'run', return_value=result):
            self.assertIn('ERROR stderr', tool.command(['sudo', 'docker', 'logs'], include_stderr=True))

    def test_unhealthy_services_stop_waiting_at_deadline(self):
        tool = load_tool()
        args = SimpleNamespace(devices='0,1,2,3', container_prefix='test-npu', base_port=8001,
                               wait_ready=6, poll_interval=5.)
        clock = [0.]
        class Opener:
            def open(self, *args, **kwargs): raise URLError(ConnectionResetError(104, 'reset'))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(tool, 'command', side_effect=lambda argv: json.dumps([raw_container(argv[-1])])), \
             patch.object(tool.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(tool.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
             redirect_stdout(io.StringIO()):
            rows = tool.wait_services(args, ['test-npu%d' % d for d in range(4)], Path(directory), Opener())
            self.assertEqual(clock[0], 6.)
            self.assertFalse(any(r['health_ok'] for r in rows))
            self.assertEqual(len((Path(directory) / 'readiness.jsonl').read_text().splitlines()), 3)

    def test_resource_limits_are_not_conflated_or_silently_changed(self):
        options = {'--max-num-batched-tokens': '1024', '--max-num-seqs': '10', '--max-model-len': '4096'}
        comparison = resource_comparison(options, 4096, 2)
        self.assertFalse(comparison['step_budget_matches'])
        self.assertEqual(comparison['server_max_num_seqs'], 10)
        self.assertEqual(comparison['client_max_inflight'], 2)
        self.assertIsNone(comparison['physical_kv_capacity_blocks'])
        self.assertIsNone(resource_comparison({}, 4096, 2)['step_budget_matches'])
        self.assertEqual(options['--max-num-batched-tokens'], '1024')


if __name__ == '__main__':
    unittest.main()
