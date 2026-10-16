"""CPU orchestration doubles only: these tests are not native/NPU acceptance."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dabs_ascend.common import ROOT, read_json, write_json


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load_tool('run_native_workflow')
native_cli = load_tool('native_dabs')


class NativeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commands = []
        self.failure = None
        self.completed = 96

    def destructive_evidence(self):
        paths = {}
        for mode in ('native-invalidate', 'native-restart-check'):
            directory = self.root / 'outputs' / mode
            directory.mkdir(parents=True, exist_ok=True)
            paths[mode] = '/project/outputs/' + mode
        write_json(self.root / 'outputs/native-latest.json', paths)

    def fake_launch(self, command, logfile):
        self.commands.append(command)
        mode = command[2]
        target = entry.Workflow.host_path(self.current, command[command.index('--out') + 1])
        target.mkdir(parents=True)
        logfile.write_text('CPU fake ' + mode, encoding='utf-8')
        if self.failure == mode:
            write_json(target / 'result.json', {'pass': False, 'failures': ['fixture failure']})
            return 7
        if mode == 'benchmark':
            write_json(target / 'status.json', {'status': 'complete', 'completed_conditions': self.completed,
                                               'functional_test_pass': True})
            return 0
        result = {'pass': True}
        if mode == 'native-accept':
            evidence = self.current.host_path(command[command.index('--evidence') + 1])
            paths = read_json(evidence)['paths']
            required = ('native-probe-one', 'native-probe-four', 'native-calibrate')
            missing = ['missing evidence: ' + name for name in required if name not in paths]
            result.update(missing=missing, live_verified=not missing, paths=paths)
            result['pass'] = not missing
            write_json(target / 'acceptance.json', result)
        write_json(target / 'result.json', result)
        return 0 if result['pass'] else 2

    def run_workflow(self, mode='test', invalidate_cache=False):
        self.current = entry.Workflow(self.root, mode, invalidate_cache)
        with patch.object(self.current, 'launch', side_effect=self.fake_launch), redirect_stdout(io.StringIO()):
            code = self.current.run()
        return self.current, code

    def test_test_then_experiment_binds_that_test_and_runs_all_conditions(self):
        self.destructive_evidence()
        tested, code = self.run_workflow()
        self.assertEqual(code, 0)
        self.assertEqual([c[2] for c in self.commands], ['native-check-four', 'native-probe-one',
                         'native-probe-four', 'native-calibrate', 'native-accept'])
        # Changing the shared latest pointer cannot substitute different evidence.
        write_json(self.root / 'outputs/native-latest.json', {'native-calibrate': '/project/outputs/unrelated'})
        run, code = self.run_workflow('experiment')
        self.assertEqual(code, 0)
        command = self.commands[-1]
        self.assertEqual(command[2], 'benchmark')
        self.assertEqual(command[command.index('--execution-scope') + 1], 'native')
        self.assertEqual(command[command.index('--native-evidence') + 1], '/project/' + tested.record['acceptance'])
        self.assertNotIn('--max-conditions', command)
        self.assertNotIn('--resume', command)
        self.assertEqual(run.record['status'], 'passed')
        self.assertTrue((run.out / 'r234_native/status.json').is_file())
        self.assertIn('96', (run.out / 'SUMMARY.txt').read_text(encoding='utf-8'))

    def test_failed_calibration_stops_and_invalidates_previous_test_pointer(self):
        self.destructive_evidence()
        previous, code = self.run_workflow()
        self.assertEqual(code, 0)
        self.failure = 'native-calibrate'
        count = len(self.commands)
        failed, code = self.run_workflow()
        self.assertEqual(code, 2)
        self.assertNotEqual(previous.out, failed.out)
        self.assertEqual(self.commands[-1][2], 'native-calibrate')
        self.assertEqual(len(self.commands) - count, 4)
        count = len(self.commands)
        blocked, code = self.run_workflow('experiment')
        self.assertEqual(code, 2)
        self.assertEqual(len(self.commands), count)
        self.assertEqual(blocked.record['next_commands'], ['bash run_native.sh test'])
        self.assertTrue((previous.out / 'run.json').exists())

    def test_normal_test_does_not_require_or_reuse_destructive_evidence(self):
        self.destructive_evidence()
        run, code = self.run_workflow()
        self.assertEqual(code, 0)
        self.assertNotIn('native-invalidate', [c[2] for c in self.commands])
        self.assertNotIn('native-restart-arm', [c[2] for c in self.commands])
        paths = read_json(run.out / 'evidence.json')['paths']
        self.assertEqual(set(paths), {'native-probe-one', 'native-probe-four', 'native-calibrate'})

    def test_cache_reset_only_with_explicit_option(self):
        self.destructive_evidence()
        run, code = self.run_workflow(invalidate_cache=True)
        self.assertEqual(code, 0)
        self.assertEqual([c[2] for c in self.commands][-2:], ['native-invalidate', 'native-accept'])
        self.assertTrue(run.record['explicit_cache_invalidation'])

    def test_new_directory_connects_without_any_credentials(self):
        run, code = self.run_workflow()
        self.assertEqual(code, 0)
        self.assertEqual(self.commands[0][2], 'native-check-four')
        self.assertEqual(run.record['next_commands'], ['bash run_native.sh experiment'])
        self.assertTrue((run.out / 'SUMMARY.txt').is_file())

    def test_missing_interface_reports_install_command_without_restart(self):
        self.failure = 'native-check-four'
        run, code = self.run_workflow()
        self.assertEqual(code, 2)
        self.assertEqual([c[2] for c in self.commands], ['native-check-four'])
        self.assertIn('bash run_native.sh install', run.record['next_commands'])

    def test_zero_exit_without_live_acceptance_cannot_pass(self):
        self.destructive_evidence()
        original = self.fake_launch
        def false_live(command, logfile):
            code = original(command, logfile)
            if command[2] == 'native-accept':
                target = self.current.host_path(command[command.index('--out') + 1])
                write_json(target / 'acceptance.json', {'pass': True, 'live_verified': False, 'missing': []})
            return code
        self.current = entry.Workflow(self.root, 'test')
        with patch.object(self.current, 'launch', side_effect=false_live), redirect_stdout(io.StringIO()):
            self.assertEqual(self.current.run(), 2)
        self.assertNotEqual(self.current.record['status'], 'passed')

    def test_partial_experiment_cannot_be_reported_as_complete(self):
        self.destructive_evidence()
        self.assertEqual(self.run_workflow()[1], 0)
        self.completed = 8
        run, code = self.run_workflow('experiment')
        self.assertEqual(code, 2)
        self.assertNotEqual(run.record['status'], 'passed')

    def test_interrupt_preserves_summary_and_failed_step(self):
        run = entry.Workflow(self.root, 'test')
        with patch.object(run, 'launch', side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            code = run.run()
        self.assertEqual(code, 130)
        self.assertEqual(read_json(run.out / 'run.json')['status'], 'interrupted')
        self.assertEqual(run.record['steps'][0]['status'], 'interrupted')
        self.assertTrue((run.out / 'error.txt').exists())

    def test_output_mapping_rejects_escape_and_translates_container_paths(self):
        run = entry.Workflow(self.root, 'test')
        self.assertEqual(run.host_path('/project/outputs/probe'), self.root / 'outputs/probe')
        for path in ('/project/../secret', 'outputs/../../secret', '/etc/passwd'):
            with self.assertRaises(ValueError):
                run.host_path(path)

    def test_native_docker_forwards_explicit_artifacts_without_npu_mapping(self):
        args = SimpleNamespace(mode='native-accept', replica=None, backup=None,
                               out=Path('outputs/one/accept'), evidence=Path('outputs/one/evidence.json'))
        with patch.object(native_cli, 'ROOT', self.root), patch.object(native_cli.sys, 'platform', 'linux'), \
             patch.object(native_cli.os, 'getuid', return_value=1000, create=True), \
             patch.object(native_cli.os, 'getgid', return_value=1000, create=True), \
             patch.object(native_cli, 'docker', return_value=SimpleNamespace(stdout='fixture-image\n')), \
             patch.object(native_cli.subprocess, 'call', return_value=0) as call:
            self.assertEqual(native_cli.host(args), 0)
        command = call.call_args.args[0]
        self.assertIn('/project/outputs/one/accept', command)
        self.assertIn('/project/outputs/one/evidence.json', command)
        self.assertNotIn('--device', command)
        self.assertNotIn('restart', command)

    def test_child_failure_streams_stderr_and_retains_exit_code(self):
        run = entry.Workflow(self.root, 'test')
        logfile = run.out / 'child.txt'
        command = [sys.executable, '-c', 'import sys; print("before failure"); print("failure detail", file=sys.stderr); sys.exit(3)']
        with redirect_stdout(io.StringIO()) as console:
            self.assertEqual(run.launch(command, logfile), 3)
        self.assertIn('failure detail', console.getvalue())
        self.assertIn('before failure', logfile.read_text(encoding='utf-8'))
        self.assertIn('failure detail', logfile.read_text(encoding='utf-8'))

    @unittest.skipUnless(platform.system() == 'Linux', 'Bash entry runs on the Linux deployment host')
    def test_bash_help_needs_no_docker(self):
        result = subprocess.run(['bash', str(ROOT / 'run_native.sh'), '--help'],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('experiment', result.stdout)
