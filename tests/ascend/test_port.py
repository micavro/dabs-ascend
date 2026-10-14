"""CPU-only contracts and mocked HTTP integration; never evidence of NPU success."""
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import multiprocessing
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import httpx

from dabs_ascend.backend import Backend, Tokenizer, constraint_field
from dabs_ascend.calibration import fit_model
from dabs_ascend.cli import execute_suite, main, make_plan, prepare_args
from dabs_ascend.common import DATASET, ROOT, configuration, read_json, source_paths, verify_original_assets, write_json
from dabs_ascend.report import make_report
from dabs_ascend.runner import run_condition
from dabs_ascend.runtime import Servers, device_ids, environment, ensure_free_ports, server_command, validate_model
from dabs_ascend.scheduler import baseline, plan_dag, planner_ready, snapshot

source_paths()
from semantic19.dataset import load_public
from typed19.answers import bind, parse
from typed19.workflow import compile_packets, instantiate

fixture_spec = importlib.util.spec_from_file_location("dabs_ascend_cpu_fixture", ROOT / "source/spra_workflow_sched_v2_1/tests/synthetic.py")
fixture_module = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)
FakeTokenizer = fixture_module.FakeTokenizer


def api_document(field):
    return {"paths": {"/v1/completions": {"post": {"requestBody": {"content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/CompletionRequest"}}}}}},
        "/tokenize": {}, "/detokenize": {}}, "components": {"schemas": {"CompletionRequest": {
            "properties": {field: {}, "stream_options": {}}}}}}


def model_for(replicas=2):
    rows = [{"replica": r, "prompt_tokens": p, "ttft_text_proxy_ms": 1 + p / 100.,
             "tpot_text_proxy_ms": 1.5} for r in range(replicas) for p in (256, 512, 1024, 2048)]
    return fit_model(rows, replicas)


class ConfigurationTests(unittest.TestCase):
    def test_frozen_original_source_and_dataset_are_unchanged(self):
        result = verify_original_assets()
        self.assertTrue(result["source_check"])
        self.assertIn("pilot.public.jsonl", result["dataset_sha256"])

    def test_unique_devices_and_single_device_mapping(self):
        self.assertEqual(device_ids("0,2,7"), [0, 2, 7])
        for value in ("", "0,0", "0,-1", "0,a", "0,"):
            with self.assertRaises(ValueError):
                device_ids(value)
        env = environment([7])
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "7")
        self.assertEqual(env["NPU_VISIBLE_DEVICES"], "7")
        self.assertEqual(env["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_server_command_preserves_dabs_envelope(self):
        command = server_command(Path("/x/my model"), 18000, .75)
        for flag, value in (("--tensor-parallel-size", "1"), ("--max-model-len", "4096"),
                            ("--max-num-batched-tokens", "4096"), ("--block-size", "16")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn(str(Path("/x/my model")), command)
        self.assertNotIn("--trust-remote-code", command)

    def test_no_silent_model_substitution_or_missing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(path / "config.json", {"model_type": "llama"})
            write_json(path / "tokenizer_config.json", {})
            write_json(path / "tokenizer.json", {})
            (path / "model.safetensors").write_bytes(b"test-only")
            with self.assertRaisesRegex(ValueError, "Qwen"):
                validate_model(path)
            write_json(path / "config.json", {"model_type": "qwen2", "hidden_size": 1536, "num_hidden_layers": 28})
            self.assertFalse(validate_model(path)["weight_contents_hashed"])
            write_json(path / "model.safetensors.index.json", {"weight_map": {"layer": "missing.safetensors"}})
            with self.assertRaisesRegex(ValueError, "shard"):
                validate_model(path)

    def test_occupied_port_not_reused(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            with self.assertRaises(OSError):
                ensure_free_ports(sock.getsockname()[1], 1)

    def test_declared_plans_and_pairing(self):
        args = prepare_args(["smoke", "--devices", "0,1,2,3"])
        plan = make_plan(args)
        self.assertEqual(len(plan), 4)
        self.assertEqual(len({p["seed"] for p in plan}), 1)
        self.assertEqual({p["method"] for p in plan}, {"RR", "P2C", "Affinity", "DAG-Ascend"})
        args = prepare_args(["plan", "--suite", "case", "--devices", "0,1,2,3"])
        self.assertEqual(len(make_plan(args)), 32)
        self.assertTrue(all(p["slice"] == [32, 48] for p in make_plan(args)))
        args = prepare_args(["plan", "--suite", "grid", "--devices", "0,1,2,3"])
        self.assertEqual(len(make_plan(args)), 512)  # Four methods; not historical five-method 640.

    def test_calibration_rejects_invalid_or_short_output(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            fit_model([], 1)
        rows = [{"replica": 0, "prompt_tokens": p, "ttft_text_proxy_ms": 2., "tpot_text_proxy_ms": None}
                for p in (256, 512, 1024, 2048)]
        with self.assertRaisesRegex(ValueError, "short"):
            fit_model(rows, 1)
        model = model_for()
        self.assertFalse(model["nvidia_coefficients_used"])
        self.assertFalse(model["native_phase_model"])
        self.assertTrue(all(c >= 0 for entry in model["models"].values() for c in entry["coef"]))

    def test_unknown_parent_is_preserved_without_gold(self):
        question, used = bind("Who lives in #1?", {0: {"text": '{"answer":"UNKNOWN"}', "finish_reason": "stop"}})
        self.assertIn("UNAVAILABLE_STEP_1", question)
        self.assertIsNone(used["1"]["bound_value"])
        self.assertEqual(parse('{"answer":"x","answer":"y"}')["status"], "format_error")

    def test_no_false_native_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            report = make_report(out, [{"label": "unfinished"}], [], "failed", "test failure")
            self.assertFalse(report["functional_test_pass"])
            self.assertFalse(report["strict_gpu39_reproduction_pass"])
            self.assertFalse(report["real_npu_verified"])
            self.assertEqual(next(r["status"] for r in report["tests"] if r["id"] == "A10"), "UNSUPPORTED")

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "old_result.txt"
            marker.write_text("preserve me", encoding="utf-8")
            self.assertEqual(main(["smoke", "--out", directory]), 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")

    def test_cleanup_targets_only_created_process_groups(self):
        args = prepare_args(["smoke", "--devices", "1,3"])
        with tempfile.TemporaryDirectory() as directory:
            servers = Servers(args, Path(directory))
            servers.processes = [MagicMock(pid=100101, returncode=-15), MagicMock(pid=100102, returncode=-15)]
            with patch("dabs_ascend.runtime.os.killpg", create=True) as killpg, patch("dabs_ascend.runtime.signal.SIGKILL", 9, create=True):
                servers.close()
                self.assertEqual({call.args[0] for call in killpg.call_args_list}, {100101, 100102})
            self.assertEqual(len(read_json(Path(directory) / "cleanup.json")), 2)


class WireTests(unittest.IsolatedAsyncioTestCase):
    def test_api_schema_new_and_old(self):
        self.assertEqual(constraint_field(api_document("guided_json")), "guided_json")
        self.assertEqual(constraint_field(api_document("structured_outputs")), "structured_outputs")
        with self.assertRaises(ValueError):
            constraint_field(api_document("unsupported"))

    async def request(self, field, malformed=None):
        observed = []
        async def handler(request):
            payload = json.loads(request.content)
            observed.append(payload)
            usage = {"prompt_tokens": 3, "completion_tokens": 5}
            if malformed == "tokens":
                usage["prompt_tokens"] = 9
            events = [json.dumps({"choices": [{"index": 0, "text": '{"answer":"Paris"}', "finish_reason": None}]}),
                      json.dumps({"choices": [{"index": 0, "text": "", "finish_reason": "stop"}], "usage": usage})]
            if malformed == "no_usage":
                events = events[:1]
            if malformed == "error":
                events = [json.dumps({"error": {"message": "backend crashed"}})]
            if malformed != "no_done":
                events.append("[DONE]")
            return httpx.Response(200, text="".join("data: " + e + "\n\n" for e in events))
        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                backend = Backend(client, ["http://127.0.0.1:18000"])
                backend.fields = [field]
                node = {"id": "wire-test", "tokens": [1, 2, 3], "max_output_tokens": 96}
                result = await backend.generate(0, node, 7, Path(directory) / "stream.jsonl")
        return result, observed

    async def test_structured_request_and_real_usage(self):
        for field in ("guided_json", "structured_outputs"):
            result, observed = await self.request(field)
            self.assertEqual(result["usage"]["completion_tokens"], 5)
            self.assertIn(field, observed[0])
            self.assertNotIn("min_tokens", observed[0])
            self.assertNotIn("ignore_eos", observed[0])

    async def test_rejects_incomplete_sse_and_usage_drift(self):
        for malformed in ("tokens", "no_done", "no_usage", "error"):
            with self.assertRaises((ValueError, RuntimeError)):
                await self.request("structured_outputs", malformed)

    async def test_replica_schema_mismatch_is_rejected(self):
        async def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "Qwen2.5-1.5B-Instruct"}]})
            return httpx.Response(200, json=api_document("guided_json" if request.url.port == 18000 else "structured_outputs"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = Backend(client, ["http://127.0.0.1:18000", "http://127.0.0.1:18001"])
            with self.assertRaisesRegex(ValueError, "different"):
                await backend.inspect()


class SimulatedBackend:
    """TEST ONLY. Never exposed as a CLI inference backend."""
    def __init__(self, fail=False, delay=.005):
        self.fail, self.delay = fail, delay

    async def generate(self, replica, node, seed, log):
        started = time.monotonic()
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("simulated transport failure")
        now = time.monotonic()
        result = {"text": '{"answer":"UNKNOWN"}', "finish_reason": "stop",
                  "usage": {"prompt_tokens": len(node["tokens"]), "completion_tokens": 8},
                  "ttft_text_proxy_ms": 1., "tpot_text_proxy_ms": .5,
                  "start": started, "finish": now, "request_ms": (now - started) * 1000}
        node["stream"] = result
        node["answer_contract"] = parse(result["text"], "stop")
        return result


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Slow CPU tokenization is expected for this character-token test double.
        asyncio.get_running_loop().slow_callback_duration = 5.

    async def setup_data(self):
        cfg = configuration(2)
        cfg.update(document_tokens_budget=2000, epoch_s=.05, planning_soft_limit_s=.035,
                   ready_coalesce_s=.001, commit_reserve_s=.002, condition_wall_s=10., max_num_seqs=4,
                   arrival_rate=8.)
        rows, _ = load_public(DATASET, "smoke")
        tok = FakeTokenizer()
        packets, _ = await compile_packets(rows, tok, cfg)
        return cfg, tok, packets

    async def test_all_methods_with_real_dag_and_actual_parent_binding(self):
        cfg, tok, packets = await self.setup_data()
        with tempfile.TemporaryDirectory() as directory, ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            await asyncio.get_running_loop().run_in_executor(pool, planner_ready)
            for method in ("RR", "P2C", "Affinity", "DAG-Ascend"):
                spec = {"method": method, "seed": 7, "label": "test-" + method}
                out = Path(directory) / method
                with redirect_stdout(io.StringIO()):
                    result = await run_condition(SimulatedBackend(), tok, packets, model_for(), cfg, spec, out, pool)
                self.assertEqual(result["completed_workflows"], 2)
                self.assertEqual(result["completed_requests"], 7)
                self.assertTrue(result["functional_pass"])
                self.assertFalse(result["strict_gpu39_reproduction"])
                for workflow in result["workflow_rows"]:
                    self.assertAlmostEqual(sum(workflow["critical_path_ms"].values()), workflow["jct_ms"], places=5)
                ns = read_json(out / "nodes.json")
                for n in ns.values():
                    for parent in n["parents"]:
                        self.assertGreaterEqual(n["start"], ns[parent]["finish"])
                    if n["parents"]:
                        self.assertIn("UNAVAILABLE_STEP", n["resolved_question"])
                if method == "DAG-Ascend":
                    self.assertGreater(result["selector_scored_decisions"], 0)

    async def test_transport_failure_keeps_partial_nodes_and_status(self):
        cfg, tok, packets = await self.setup_data()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "failed"
            with self.assertRaisesRegex(RuntimeError, "transport failure"):
                await run_condition(SimulatedBackend(fail=True), tok, packets, model_for(), cfg,
                                    {"method": "RR", "seed": 7}, out, None)
            self.assertEqual(read_json(out / "status.json")["status"], "failed")
            self.assertTrue((out / "nodes.json").is_file())
            self.assertFalse((out / "summary.json").exists())

    async def test_timeout_is_failure_not_success(self):
        cfg, tok, packets = await self.setup_data()
        cfg["condition_wall_s"] = .1
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "timeout"
            with self.assertRaises(asyncio.TimeoutError):
                await run_condition(SimulatedBackend(delay=1), tok, packets, model_for(), cfg,
                                    {"method": "RR", "seed": 7}, out, None)
            self.assertEqual(read_json(out / "status.json")["status"], "failed")

    async def test_snapshot_has_no_fabricated_native_cache(self):
        cfg, tok, packets = await self.setup_data()
        nodes, materialize, _ = await instantiate(packets, tok, cfg, 7)
        now = time.monotonic()
        for key, node in nodes.items():
            node["arrival"] = now
            if not node["parents"]:
                await materialize(key, nodes)
                node.update(state="ready", ready_at=now)
        state = snapshot(nodes, {}, 2, cfg, 1, now)
        self.assertEqual(state["cache_evidence"], [])
        self.assertEqual(state["prefix_certificates"], {})
        self.assertIn("NOT physical", state["capacity_source"])
        result = plan_dag(state, model_for(), cfg, time.monotonic() + 1.)
        self.assertEqual(result["action"]["order"], [])
        self.assertFalse(result["native_dag_equivalent"])
        for evaluation in result["evaluations"]:
            self.assertEqual(evaluation["action"]["order"], [])

    async def test_partial_startup_failure_still_cleans_up(self):
        class FailedServers:
            closed = False
            async def start(self, client):
                raise RuntimeError("second replica failed to start")
            def close(self):
                self.closed = True
        servers = FailedServers()
        with tempfile.TemporaryDirectory() as directory, patch("dabs_ascend.cli.Servers", return_value=servers):
            with self.assertRaisesRegex(RuntimeError, "second replica"):
                await execute_suite(prepare_args(["smoke"]), Path(directory), [], [], {})
        self.assertTrue(servers.closed)

    async def test_expired_planner_result_is_not_applied(self):
        cfg, tok, packets = await self.setup_data()
        cfg["planning_soft_limit_s"] = .002
        def slow_plan(state, model, current_cfg, deadline):
            time.sleep(.07)
            return {"action": baseline(state, current_cfg, "Affinity"), "model_selection": True,
                    "selected_cell": "10", "evaluations": []}
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            with patch("dabs_ascend.runner.plan_dag", slow_plan), redirect_stdout(io.StringIO()):
                result = await run_condition(SimulatedBackend(), tok, packets, model_for(), cfg,
                                             {"method": "DAG-Ascend", "seed": 7}, Path(directory) / "expired", pool)
            self.assertGreater(result["expired_plan_count"], 0)
            self.assertEqual(result["nonreference_actions"], 0)
            self.assertEqual(result["selector_scored_decisions"], 0)


if __name__ == "__main__":
    unittest.main()
