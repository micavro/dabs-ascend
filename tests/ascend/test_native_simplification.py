"""CPU checks for capability-only deployment and real receipt-based acceptance."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from dabs_native import install as installer
from dabs_native.api import install_api
from dabs_native.contracts import validate_capabilities, validate_observation
from dabs_ascend.common import write_json
from dabs_ascend.native_client import NativeClient
from dabs_ascend.native_gate import verify_live
from dabs_ascend.native_probe import near_release


def capability():
    return {'identity': {'engine_instance': 'cpu-only', 'boot_id': 'boot', 'clock_namespace_inode': 7},
            'native_execution': True, 'workers': [{'npu_available': True}],
            'timing_policy': __import__('dabs_native.timing_policy', fromlist=['capabilities']).capabilities(),
            'resources': {'step_tokens': 1024, 'block_size': 128, 'server_max_seqs': 10,
                          'context_tokens': 4096, 'physical_blocks': 1000},
            'supported_actions': ['admission_permits', 'unsubmitted_route', 'safe_prefill_queue_order'],
            'control_semantics': {'admission_only_preserves_queue': True, 'mutation_time_window_check': True},
            'host_step_boundary': 'EngineCore.step', 'adapter_sha256': {'older.py': 'different'}}


def observation():
    return {'identity': capability()['identity'], 'snapshot_id': 's', 'step': 1,
            'captured_ns': 1, 'expires_ns': 120000001, 'resources': capability()['resources'],
            'physical': {'total_usable_blocks': 1000, 'active_blocks': 0, 'reclaimable_cached_blocks': 0,
                         'allocator_free_blocks': 1000, 'available_for_new_reservations': 1000},
            'running': [], 'waiting': [], 'skipped_waiting': [], 'cache': [], 'inflight_batches': [],
            'running_count': 0, 'waiting_count': 0}


SOURCES = dict(zip(installer.PATCHED, [
    b'class EngineCore:\n    def __init__(self):\n        pass\n',
    b'def build_app():\n    app = object()\n    return app\n',
    b'class SchedulerOutput:\n    count: int = 0\n',
    b'class NPUWorker:\n    def execute_model(self, scheduler_output):\n        pass\n']))


class SimplificationTests(unittest.TestCase):
    def test_no_credentials_header_and_different_adapter_hash_connects(self):
        requests = []
        def handle(request):
            self.assertNotIn('authorization', request.headers)
            op = json.loads(request.content)['op']; requests.append(op)
            row = {'capabilities': capability(), 'snapshot': observation(),
                   'mapping': {'identity': capability()['identity'], 'mappings': []},
                   'batches': {'identity': capability()['identity'], 'batches': []}}[op]
            return httpx.Response(200, json=row)
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                native = NativeClient(client, ['http://test'])
                with patch('dabs_ascend.native_client.Path', return_value=SimpleNamespace(read_text=lambda: 'boot')), \
                     patch('dabs_ascend.native_client.os.stat', return_value=SimpleNamespace(st_ino=7)):
                    self.assertEqual((await native.inspect())[0]['resources']['step_tokens'], 1024)
        asyncio.run(run())
        self.assertEqual(requests, ['capabilities', 'snapshot', 'mapping', 'batches'])

    def test_missing_route_reports_replica_operation_status(self):
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404, text='Not Found'))) as client:
                with self.assertRaisesRegex(ValueError, 'replica 0.*op=capabilities.*404'):
                    await NativeClient(client, ['http://test']).inspect()
        asyncio.run(run())

    def test_missing_fields_are_named_without_version_allowlist(self):
        row = capability(); del row['resources']['physical_blocks']
        with self.assertRaisesRegex(ValueError, 'resources.physical_blocks'):
            validate_capabilities(row)
        row = observation(); del row['physical']['active_blocks']
        with self.assertRaisesRegex(ValueError, 'physical.active_blocks'):
            validate_observation(row)

    def test_api_control_forwards_to_real_engine_without_authorization(self):
        routes = {}
        core = SimpleNamespace(call_utility_async=AsyncMock(return_value=observation()))
        app = SimpleNamespace(state=SimpleNamespace(engine_client=SimpleNamespace(engine_core=core)),
                              add_api_route=lambda path, fn, **kw: routes.update({path: fn}))
        class Request:
            headers = {}
            async def body(self): return b'{"op":"snapshot"}'
        request = Request(); request.app = app
        with patch.dict('sys.modules', {'fastapi': SimpleNamespace(Request=Request, HTTPException=Exception),
                                        'starlette.responses': SimpleNamespace(StreamingResponse=object)}):
            install_api(app)
        result = asyncio.run(routes['/dabs/v1/control'](request))
        self.assertEqual(result, observation())
        core.call_utility_async.assert_awaited_once_with('dabs_native', {'op': 'snapshot'})

    def test_changed_version_source_install_upgrade_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = {}
            for i, (name, data) in enumerate(SOURCES.items()):
                path = root / ('module' + str(i) + '.py'); path.write_bytes(data + b'# local variant\n')
                paths[name] = path
            target = root / 'dabs_native'; target.mkdir()
            (target / 'old_module.py').write_text('# existing adapter')
            with patch.object(installer, 'installed_path', side_effect=lambda name: paths[name]), \
                 patch.object(installer.metadata, 'version', return_value='site-local-version'), \
                 patch.object(installer.sysconfig, 'get_paths', return_value={'purelib': str(root)}):
                checked = installer.check()
                self.assertTrue(checked['compatible']); self.assertTrue(checked['notes'])
                result = installer.install()
                self.assertTrue(result['installed'])
                self.assertFalse((target / 'old_module.py').exists())
                for name, path in paths.items():
                    self.assertEqual(installer.patch_source(name, path.read_bytes()), path.read_bytes())
                # A harmless vendor edit never requires a hash allowlist reset.
                paths[installer.PATCHED[0]].write_bytes(paths[installer.PATCHED[0]].read_bytes() + b'# extra edit\n')
                second = installer.install()
                self.assertTrue(second['installed'])
                installer.rollback(second['backup'])
                self.assertIn(b'# extra edit', paths[installer.PATCHED[0]].read_bytes())
                installer.rollback(result['backup'])
                self.assertTrue((target / 'old_module.py').exists())
                self.assertEqual(paths[installer.PATCHED[0]].read_bytes(), SOURCES[installer.PATCHED[0]] + b'# local variant\n')

    def test_missing_engine_method_is_precise(self):
        from dabs_native.engine import install_engine
        with self.assertRaisesRegex(ValueError, 'EngineCore.step_fn'):
            install_engine(SimpleNamespace(add_request=lambda: None))

    def test_live_gate_uses_epoch_and_actual_receipt_not_hash_or_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cap = capability()
            receipt = {'identity': cap['identity'], 'action_id': 'a', 'status': 'applied', 'applied': ['x'],
                       'applied_step': 51, 'requested': {'order': ['x']}}
            (out / 'native_events.jsonl').write_text(json.dumps({'replica': 0, 'op': 'apply', 'response': receipt}) + '\n')
            write_json(out / 'native_profile.json', {'identities': [cap['identity']], 'resources': [cap['resources']]})
            native = SimpleNamespace(capabilities=[cap], identities={0: cap['identity']},
                                     inspect=AsyncMock(return_value=[cap]), call=AsyncMock(return_value=receipt))
            local = {'pass': True, 'missing': [], 'not_run': ['native-invalidate', 'native-restart-check']}
            with patch('dabs_ascend.native_gate.evaluate_local', side_effect=lambda paths: copy.deepcopy(local)):
                paths = {'native-calibrate': directory}
                self.assertTrue(asyncio.run(verify_live(native, paths))['live_verified'])
                native.call.return_value = {**receipt, 'applied': []}
                rejected = asyncio.run(verify_live(native, paths))
                self.assertFalse(rejected['pass']); self.assertIn('live action receipt differs', rejected['missing'][0])
                native.identities = {0: {**cap['identity'], 'engine_instance': 'new'}}
                self.assertFalse(asyncio.run(verify_live(native, paths))['pass'])

    def test_site_trigger_waits_for_measured_short_decode_release(self):
        def req(progress, maximum=16, phase='decode'):
            return {'output_tokens': progress, 'max_output_tokens': maximum, 'phase': phase}
        # The failed site snapshot: eight 11/12-token short decodes plus two
        # long decodes. No runnable slot yet; publishing priority is premature.
        observed = {'running': [req(11), req(12)] * 4 + [req(11, 512), req(12, 512)]}
        self.assertFalse(near_release(observed))
        observed['running'][0] = req(14)
        self.assertTrue(near_release(observed))
        for records in ([req(500, 512)], [req(14, phase='prefill')], [req(None)], []):
            self.assertFalse(near_release({'running': records}))
