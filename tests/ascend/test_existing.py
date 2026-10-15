"""Existing-service attachment and host launcher regression tests; no real NPUs."""
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from dabs_ascend.cli import execute_suite, main, make_plan, prepare_args
from dabs_ascend.common import ROOT, read_json, write_json
from dabs_ascend.existing import ExistingServers, deployment_records, preflight, selected_endpoints, validate_manifest
from dabs_ascend.report import make_report


def deployment(devices=(4, 5)):
    return [{"name": f"wushibo-qwen-small-npu{d}", "container_id": f"container-{d}",
             "image_id": "sha256:fixture", "running": True,
             "devices": [{"PathOnHost": f"/dev/davinci{d}", "PathInContainer": "/dev/davinci0"},
                         {"PathOnHost": "/dev/davinci_manager"}],
             "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(8001 + d)}]}}
            for d in devices]


def args_for(manifest, devices="4,5", mode="smoke"):
    return prepare_args([mode, "--devices", devices, "--deployment-manifest", str(manifest)], existing=True)


class ExistingConfigurationTests(unittest.TestCase):
    def test_subset_ports_and_cpu_modes_need_no_manifest(self):
        args = prepare_args(["plan", "--devices", "4,7", "--suite", "smoke"], existing=True)
        self.assertEqual(selected_endpoints(args), ["http://127.0.0.1:8005", "http://127.0.0.1:8008"])
        self.assertEqual(len(make_plan(args)), 4)
        self.assertEqual(args.max_inflight, 2)
        self.assertEqual(prepare_args(["test"], existing=True).devices, list(range(8)))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_args(["check"], existing=True)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_args(["plan", "--devices", "7", "--base-port", "65529"], existing=True)

    def test_deployment_rejects_wrong_or_missing_bindings(self):
        args = args_for(Path("unused"))
        self.assertEqual([r["declared_physical_device"] for r in validate_manifest(deployment(), args)], [4, 5])
        bad_rows = [[], deployment() + [deployment()[0]]]
        for field, value in (("running", False), ("devices", [{"PathOnHost": "/dev/davinci0"}]),
                             ("ports", {"8000/tcp": [{"HostPort": "8001"}]}),
                             ("ports", {"8000/tcp": [{"HostPort": "8005", "HostIp": "192.0.2.1"}]})):
            rows = deployment()
            rows[0][field] = value
            bad_rows.append(rows)
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_manifest(rows, args)

    def test_docker_snapshot_omits_env_and_secret_command_options(self):
        raw = {"Name": "/wushibo-qwen-small-npu4", "Id": "id", "Image": "image",
               "State": {"Running": True}, "HostConfig": {"Devices": []},
               "NetworkSettings": {"Ports": {}}, "Config": {"Env": ["TOKEN=secret-env"],
               "Cmd": ["server", "--api-key", "secret-cli", "--max-num-seqs", "10",
                       "--model=/model/qwen", "--enable-prefix-caching"]}}
        records = deployment_records([raw])
        self.assertNotIn("secret", json.dumps(records))
        self.assertEqual(records[0]["server_options"]["--max-num-seqs"], "10")
        self.assertTrue(records[0]["server_options"]["--enable-prefix-caching"])

    def test_preflight_uses_http_and_client_packages_only(self):
        requests = []
        def handler(request):
            requests.append((request.url.port, request.url.path))
            return httpx.Response(200, json={"data": [{"id": "Qwen2.5-1.5B-Instruct", "max_model_len": 4096}]})
        real_client = httpx.Client
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = out / "deployment.json"
            write_json(manifest, deployment())
            with patch("dabs_ascend.existing.platform.system", return_value="Linux"), \
                 patch("dabs_ascend.existing.validate_model", return_value={"local_metadata": True}), \
                 patch("dabs_ascend.existing.importlib.metadata.version", return_value="test") as versions, \
                 patch("httpx.Client", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)), \
                 patch("dabs_ascend.runtime.capture", side_effect=AssertionError("No NPU probing")):
                facts = preflight(args_for(manifest), out)
            self.assertTrue(facts["passed"])
            self.assertFalse(facts["physical_npu_probe_performed"])
            self.assertEqual({c.args[0] for c in versions.call_args_list}, {"httpx", "numpy"})
            self.assertEqual(requests, [(8005, "/health"), (8005, "/v1/models"),
                                        (8006, "/health"), (8006, "/v1/models")])

    def test_failed_health_or_context_check_retains_failure_evidence(self):
        real_client = httpx.Client
        for status, context in ((503, 4096), (200, 2048)):
            def handler(request):
                return httpx.Response(status, json={"data": [{"id": "Qwen2.5-1.5B-Instruct", "max_model_len": context}]})
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                manifest = out / "deployment.json"
                write_json(manifest, deployment())
                with patch("dabs_ascend.existing.platform.system", return_value="Linux"), \
                     patch("dabs_ascend.existing.validate_model", return_value={}), \
                     patch("httpx.Client", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)), \
                     self.assertRaises(RuntimeError):
                    preflight(args_for(manifest), out)
                self.assertFalse(read_json(out / "environment.json")["passed"])

    def test_report_keeps_http_and_npu_claims_separate_in_all_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_json(out / "environment.json", {"mode": "existing-http-services", "passed": True,
                       "devices": [4, 5], "limits": ["host snapshot only"]})
            write_json(out / "canary_summary.json", [{"passed": True}, {"passed": True}])
            report = make_report(out, [1], [], "failed", "later failure")
            self.assertFalse(report["functional_test_pass"])
            self.assertFalse(report["real_npu_verified"])
            self.assertTrue(report["http_inference_verified"])
            with (out / "test_items.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["test"], report["tests"][1]["test"])
            text = (out / "TEST_REPORT.md").read_text(encoding="utf-8")
            self.assertIn(report["tests"][1]["test"], text)
            self.assertNotIn("physical NPUs perform", text)
            self.assertIn("host snapshot only", text)

    def test_main_check_selects_existing_preflight_without_managed_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new-output"
            def checked(args, target):
                write_json(target / "environment.json", {"passed": True, "mode": "existing-http-services"})
            with patch("dabs_ascend.existing.preflight", side_effect=checked) as external, \
                 patch("dabs_ascend.cli.preflight", side_effect=AssertionError("Managed preflight called")), \
                 patch("dabs_ascend.cli.Servers", side_effect=AssertionError("Managed servers created")), \
                 redirect_stdout(io.StringIO()):
                code = main(["check", "--devices", "4,5", "--deployment-manifest", "unused.json", "--out", str(out)], existing=True)
            self.assertEqual(code, 0)
            self.assertEqual(external.call_count, 1)
            self.assertFalse(read_json(out / "status.json")["real_inference"])


class ExistingLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_propagates_failure_and_cleanup_never_signals_services(self):
        unhealthy = False
        def handler(request):
            return httpx.Response(503 if unhealthy else 200)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = out / "deployment.json"
            write_json(manifest, deployment())
            servers = ExistingServers(args_for(manifest), out)
            servers.monitor_interval = .001
            with patch("os.kill", side_effect=AssertionError("External process signaled")), \
                 patch("os.killpg", create=True, side_effect=AssertionError("External group signaled")), \
                 patch("subprocess.Popen", side_effect=AssertionError("Model process started")):
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    await servers.start(client)
                    servers.check_alive()
                    unhealthy = True
                    async def wait_failure():
                        while servers.failure is None:
                            await asyncio.sleep(.001)
                    await asyncio.wait_for(wait_failure(), 2)
                    with self.assertRaisesRegex(RuntimeError, "health check failed"):
                        servers.check_alive()
                    await servers.aclose()
            self.assertTrue(servers.monitor.done())
            self.assertFalse(read_json(out / "cleanup.json")["external_services_stopped"])

    async def test_existing_suite_failure_cleans_monitor_without_owned_server_cleanup(self):
        real_client = httpx.AsyncClient
        def handler(request):
            return httpx.Response(200 if request.url.path == "/health" else 503)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = out / "deployment.json"
            write_json(manifest, deployment())
            with patch("httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)), \
                 patch("dabs_ascend.cli.Servers", side_effect=AssertionError("Managed model startup")), \
                 patch("os.killpg", create=True, side_effect=AssertionError("External service stopped")), \
                 self.assertRaises(httpx.HTTPStatusError):
                await execute_suite(args_for(manifest), out, [], [], {})
            self.assertFalse(read_json(out / "cleanup.json")["external_services_stopped"])


@unittest.skipUnless(platform.system() == "Linux", "Host Bash/Docker wiring runs in Linux CI")
class HostLauncherTests(unittest.TestCase):
    def run_host(self, mode, entry='run_dabs_existing.sh', devices='4,5', extra=()):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model = folder / "model with spaces"
            model.mkdir()
            if mode not in ("test", "plan"):
                (model / "config.json").write_text("{}")
            log = folder / "docker.jsonl"
            sudo = folder / "sudo"
            sudo.write_text("#!" + sys.executable + "\n" + '''
import json, os, sys
args = sys.argv[1:]
expected = ['wushibo-qwen-small-npu'+d for d in os.environ['FAKE_EXPECTED_DEVICES'].split(',')]
with open(os.environ['FAKE_DOCKER_LOG'], 'a') as f:
    f.write(json.dumps(args)+'\\n')
if args[:3] == ['docker','inspect','--format']:
    assert args[-1] == expected[0]
    print('sha256:fixture')
elif args[:2] == ['docker','inspect']:
    assert args[2:] == expected
    print(json.dumps([{'Name':'/'+name,'Id':'fixture-'+name,'Image':'sha256:fixture',
        'State':{'Running':True},'HostConfig':{'Devices':[]},
        'NetworkSettings':{'Ports':{}},'Config':{'Env':['TOKEN=secret']}}
        for name in args[2:]]))
elif args[:3] == ['docker','image','inspect']:
    pass
elif args[:2] == ['docker','run']:
    assert '--rm' in args and '--device' not in args
else:
    raise SystemExit('unexpected Docker operation')
''', encoding="utf-8")
            sudo.chmod(0o755)
            env = {**os.environ, "PATH": str(folder) + os.pathsep + os.environ["PATH"],
                   'FAKE_EXPECTED_DEVICES': devices or '0,1,2,3', "MODEL_PATH": str(model), "FAKE_DOCKER_LOG": str(log)}
            if devices is None:
                env.pop('DEVICE_IDS', None)
            else:
                env['DEVICE_IDS'] = devices
            result = subprocess.run(["bash", str(ROOT / entry)] + ([mode] if mode else []) + list(extra),
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
        return calls

    def test_r234_defaults_to_four_services_and_full_suite_with_pause_forwarding(self):
        calls = self.run_host(None, entry='run_dabs_r234.sh', devices=None, extra=('--max-conditions', '4'))
        run = calls[-1]
        self.assertEqual(run[run.index('--devices') + 1], '0,1,2,3')
        self.assertEqual(run[run.index('--suite') + 1], 'r234')
        self.assertEqual(run[run.index('--max-conditions') + 1], '4')
        self.assertEqual(run[run.index('tools/run_dabs_existing.py') + 1], 'benchmark')
        self.assertTrue(any(x.startswith('DABS_HOST_PROJECT=') for x in run))

    def test_host_selects_only_requested_cards_and_reuses_first_selected_image(self):
        calls = self.run_host("check")
        run = calls[-1]
        self.assertEqual(run[run.index("--devices") + 1], "4,5")
        self.assertIn("sha256:fixture", run)
        self.assertTrue(any("dst=/models/qwen,readonly" in x for x in run))
        self.assertFalse(any("docker.sock" in x for x in run))
        self.assertEqual(len([c for c in calls if c[:2] == ["docker", "inspect"]]), 2)

    def test_r234_diagnosis_is_explicit_and_pauses_at_eight_by_default(self):
        run = self.run_host('diagnose', entry='run_dabs_r234.sh', devices='0,1,2,3')[-1]
        self.assertEqual(run[run.index('--execution-scope') + 1], 'http-limited')
        self.assertEqual(run[run.index('--max-conditions') + 1], '8')
        run = self.run_host('diagnose', entry='run_dabs_r234.sh', devices='0,1,2,3', extra=('--max-conditions', '4'))[-1]
        self.assertEqual(run.count('--max-conditions'), 1)
        self.assertEqual(run[run.index('--max-conditions') + 1], '4')

    def test_test_and_plan_do_not_require_model_or_all_eight_containers(self):
        for mode in ("test", "plan"):
            with self.subTest(mode=mode):
                calls = self.run_host(mode)
                self.assertEqual(len(calls), 3)
                self.assertNotIn("--deployment-manifest", calls[-1])
                self.assertFalse(any("dst=/models/qwen" in x for x in calls[-1]))
