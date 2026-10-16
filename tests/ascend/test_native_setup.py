"""Installation orchestration on CPU doubles; never connects to NPU services."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import sysconfig
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dabs_ascend.common import read_json, write_json
from test_native_workflow import entry, load_tool

setup = load_tool('native_setup')


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        package = self.root / 'experiments/ascend/dabs_native'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('# fixture\n')
        self.w = entry.Workflow(self.root, 'install')
        self.installer = setup.Installer(self.w)
        self.events = []
        self.states = [{'state': 'fresh'} for _ in range(4)]
        self.loaded = set()

    def step(self, mode, label, extra=()):
        replica = int(extra[-1]) if mode == 'native-install' else None
        self.events.append((mode, replica))
        out = self.w.out / (mode + '-' + str(replica))
        out.mkdir()
        if mode == 'native-install':
            write_json(out / 'result.json', {'pass': True, 'replicas': [{'replica': replica,
                'result': {'backup': '/vendor/dabs_native_backup_' + str(replica)}}]})
        return out

    def restart(self, replica):
        self.assertIn(str(replica), read_json(self.w.out / 'backups.json'))
        self.events.append(('restart', replica))
        self.loaded.add(replica)

    def run_setup(self, idle=None, restart=None):
        def inspect(replica):
            self.events.append(('inspect', replica))
            return copy.deepcopy(self.states[replica])
        with patch.object(self.installer, 'inspect', side_effect=inspect), \
             patch.object(self.installer, 'idle', side_effect=idle), \
             patch.object(self.installer, 'ready', side_effect=lambda r: {'loaded': True} if r in self.loaded else None), \
             patch.object(self.installer, 'restart', side_effect=restart or self.restart), \
             patch.object(self.w, 'step', side_effect=self.step), redirect_stdout(io.StringIO()):
            self.installer.run()

    def test_precheck_all_four_then_install_restart_each_with_backup(self):
        self.run_setup()
        self.assertEqual(self.events[:4], [('inspect', r) for r in range(4)])
        self.assertEqual(self.events[4:-1], [(mode, r) for r in range(4) for mode in ('native-install', 'restart')])
        self.assertEqual(self.events[-1], ('native-check-four', None))
        self.assertIn('--replica 3', (self.w.out / 'ROLLBACK.txt').read_text(encoding='utf-8'))

    def test_missing_hook_in_later_replica_changes_nothing(self):
        self.states[3] = {'state': 'incompatible'}
        with self.assertRaisesRegex(ValueError, '尚未修改'):
            self.run_setup()
        self.assertEqual(self.events, [('inspect', r) for r in range(4)])

    def test_busy_or_unknown_workload_stops_before_installation(self):
        with self.assertRaisesRegex(ValueError, 'busy'):
            self.run_setup(idle=ValueError('busy'))
        self.assertEqual(self.events, [('inspect', r) for r in range(4)])

    def test_loaded_same_adapter_is_idempotent_and_unloaded_one_restarts(self):
        self.states = [{'state': 'installed', 'backup': '/vendor/b' + str(r)} for r in range(4)]
        self.loaded.update((0, 1, 3))
        self.run_setup()
        self.assertEqual([r for mode, r in self.events if mode == 'inspect'], [2])
        self.assertEqual([r for mode, r in self.events if mode == 'native-install'], [2])
        self.assertEqual([r for mode, r in self.events if mode == 'restart'], [2])

    def test_loading_failure_stops_before_next_replica(self):
        def failed(replica):
            self.events.append(('restart_failed', replica))
            raise ValueError('not ready')
        with self.assertRaisesRegex(ValueError, 'not ready'):
            self.run_setup(restart=failed)
        self.assertEqual(self.events[4:], [('native-install', 0), ('restart_failed', 0)])
        self.assertEqual(read_json(self.w.out / 'backups.json'), {'0': '/vendor/dabs_native_backup_0'})

    def test_partial_install_recovers_backup_receipt_before_stopping(self):
        def failed(mode, label, extra=()):
            self.states[0] = {'state': 'incompatible', 'backup': '/vendor/partial-backup'}
            self.events.append(('partial-install', 0))
            raise ValueError('partial install failure')
        with patch.object(self, 'step', side_effect=failed), self.assertRaisesRegex(ValueError, 'partial install'):
            self.run_setup()
        self.assertEqual(read_json(self.w.out / 'backups.json'), {'0': '/vendor/partial-backup'})
        self.assertNotIn('restart', [mode for mode, _ in self.events])

    def test_container_inspection_does_not_persist_env_or_command_secrets(self):
        raw = {'Name': '/wushibo-qwen-small-npu0', 'State': {'Running': True}, 'Id': 'fixture-id',
               'Image': 'fixture-image', 'Config': {'Env': ['TOKEN=fixture-secret'], 'Cmd': ['--api-key=fixture-secret']},
               'NetworkSettings': {'Ports': {'8000/tcp': [{'HostPort': '8001'}]}}}
        def capture(args, input=None):
            if args[0] == 'inspect':
                return json.dumps([raw])
            return json.dumps({'state': 'fresh', 'check': {'compatible': True}}) if args[-1] == setup.INSPECT else ''
        with patch.object(self.installer, 'capture', side_effect=capture):
            self.assertEqual(self.installer.inspect(0)['state'], 'fresh')
        self.assertNotIn('fixture-secret', (self.w.out / 'install-state-0.json').read_text(encoding='utf-8'))
        raw['NetworkSettings']['Ports']['8000/tcp'][0]['HostPort'] = '9001'
        with patch.object(self.installer, 'capture', side_effect=capture) as call, self.assertRaises(ValueError):
            self.installer.inspect(0)
        self.assertEqual(call.call_count, 1)

    def test_restart_identity_test_requires_explicit_option(self):
        self.installer.verify_restart = True
        self.run_setup()
        self.assertEqual(self.events[-3:], [('native-restart-arm', None), ('restart', 0), ('native-restart-check', None)])
        self.assertEqual(self.w.record['next_commands'], ['bash run_native.sh smoke'])

    def test_idle_refuses_unknown_and_nonzero_metrics(self):
        for metrics in ('', 'vllm:num_requests_running 1\nvllm:num_requests_waiting 0',
                        'vllm:num_requests_running 0\nvllm:num_requests_waiting 2'):
            with patch.object(self.installer, 'http', return_value=metrics), self.assertRaises(ValueError):
                self.installer.idle(0)
        with patch.object(self.installer, 'http', return_value='vllm:num_requests_running 0\nvllm:num_requests_waiting 0'):
            self.installer.idle(0)

    def test_restart_targets_only_existing_named_container_and_waits(self):
        with patch.object(self.installer, 'idle') as idle, patch.object(self.installer, 'wait_ready') as wait, \
             patch.object(self.w, 'launch', return_value=0) as launch, redirect_stdout(io.StringIO()):
            self.installer.restart(2)
        self.assertEqual(launch.call_args.args[0], ['sudo', 'docker', 'restart', '--time', '120', 'wushibo-qwen-small-npu2'])
        idle.assert_called_once_with(2)
        wait.assert_called_once_with(2)

    def test_readiness_uses_actual_interfaces_not_adapter_hash(self):
        from test_native_simplification import capability, observation
        good = capability()
        from dabs_ascend.native_timing import FIELDS
        good['timing_fields'] = list(FIELDS)
        for row, expected in (({}, False), (dict(good, adapter_sha256={'old': 'different'}), True), (good, True)):
            with patch.object(self.installer, 'http', side_effect=['', json.dumps(row), json.dumps(observation())]):
                self.assertEqual(bool(self.installer.ready(0)), expected)

    def test_all_live_services_skip_inspection_install_and_restart(self):
        self.loaded.update(range(4))
        self.run_setup(idle=AssertionError('ordinary capability reuse must not idle-gate'))
        self.assertEqual(self.events, [('native-check-four', None)])

    def test_old_working_interface_needs_timing_update_not_hash_match(self):
        from test_native_simplification import capability
        with patch.object(self.installer, 'http', side_effect=['', json.dumps(capability())]):
            self.assertIsNone(self.installer.ready(0))
        self.assertIn('pending_before', self.installer.last_ready_error)

    def test_working_old_admit_interface_requires_queue_and_window_boundary_fix(self):
        from test_native_simplification import capability
        from dabs_ascend.native_timing import FIELDS
        old = capability(); old['timing_fields'] = list(FIELDS)
        del old['control_semantics']
        with patch.object(self.installer, 'http', side_effect=['', json.dumps(old)]):
            self.assertIsNone(self.installer.ready(0))
        self.assertIn('admission_only_preserves_queue', self.installer.last_ready_error)
        self.assertIn('mutation_time_window_check', self.installer.last_ready_error)

    def test_selected_install_touches_only_one_replica(self):
        self.w.replica = 2
        self.run_setup()
        self.assertEqual([r for mode, r in self.events if mode in ('inspect', 'restart')], [2, 2])
        self.assertEqual(self.events[-1][0], 'native-check')

    def test_timeout_does_not_move_on(self):
        with patch.object(self.installer, 'ready', return_value=None), \
             patch.object(setup.time, 'monotonic', side_effect=[0, 901]), self.assertRaises(ValueError):
            self.installer.wait_ready(0)

    def test_staged_inspection_accepts_existing_adapter_without_credentials(self):
        import dabs_native
        target = self.root / 'vendor/dabs_native'
        target.mkdir(parents=True)
        module = SimpleNamespace(check=lambda: {'compatible': True, 'notes': ['different version/hash']})
        with patch.dict(sys.modules, {'dabs_native.install': module}), \
             patch.object(dabs_native, 'install', module, create=True), \
             patch.object(sysconfig, 'get_paths', return_value={'purelib': str(target.parent)}), redirect_stdout(io.StringIO()) as output:
            exec(setup.INSPECT, {})
        self.assertEqual(json.loads(output.getvalue())['state'], 'update')
