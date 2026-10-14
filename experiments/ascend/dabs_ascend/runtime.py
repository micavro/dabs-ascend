"""Own-process Linux service lifecycle; never changes the installed NPU stack."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

from .common import sha, write_json

MODEL_NAME = "Qwen2.5-1.5B-Instruct"


def device_ids(value):
    parts = str(value).split(",")
    if not parts or any(not p.strip().isdigit() for p in parts):
        raise ValueError("DEVICE_IDS must contain comma-separated nonnegative integers")
    ids = [int(p) for p in parts]
    if len(set(ids)) != len(ids) or not 1 <= len(ids) <= 8:
        raise ValueError("Select 1 to 8 distinct NPU devices")
    return ids


def environment(devices):
    env = os.environ.copy()
    visible = ",".join(map(str, devices))
    env.update(ASCEND_RT_VISIBLE_DEVICES=visible, NPU_VISIBLE_DEVICES=visible,
               VLLM_TARGET_DEVICE="npu", VLLM_USE_V1="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def capture(argv, env=None, timeout=60):
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": -1, "error": str(exc)}


def validate_model(path):
    path = Path(path).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise ValueError(f"Local model config missing: {path}/config.json. Set MODEL_PATH.")
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError("Local model tokenizer_config.json is missing")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Local tokenizer files are missing")
    weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("Local model weight files are missing; runtime downloads are disabled")
    for index in path.glob("*.index.json"):
        value = json.loads(index.read_text(encoding="utf-8"))
        for name in set(value.get("weight_map", {}).values()):
            target = (path / name).resolve()
            if not target.is_relative_to(path) or not target.is_file():
                raise ValueError(f"Model shard missing or outside model directory: {name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    # DABS's original model is Qwen2.5-1.5B; never relabel the MFE Llama model.
    if config.get("model_type") != "qwen2" or config.get("hidden_size") != 1536 or config.get("num_hidden_layers") != 28:
        raise ValueError("Expected Qwen2.5-1.5B-Instruct architecture. MFE's Llama weights cannot be silently substituted.")
    return {"path": str(path), "served_model_name": MODEL_NAME,
            "metadata_sha256": {p.name: sha(p) for p in path.glob("*.json")},
            "weights": [{"name": p.name, "bytes": p.stat().st_size} for p in weights],
            "weight_contents_hashed": False,
            "identity_limit": "Architecture checked; local weight revision must be verified against the original by the operator."}


def server_command(model, port, memory):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model), "--served-model-name", MODEL_NAME,
            "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", "1", "--dtype", "bfloat16",
            "--max-model-len", "4096", "--max-num-batched-tokens", "4096",
            "--max-num-seqs", "64", "--block-size", "16",
            "--gpu-memory-utilization", str(memory), "--enforce-eager",
            "--enable-prefix-caching", "--enable-chunked-prefill",
            "--generation-config", "vllm", "--seed", "20261014"]


def ensure_free_ports(base_port, count):
    sockets = []
    try:
        for port in range(base_port, base_port + count):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", port))
    finally:
        for sock in sockets:
            sock.close()


def active_servers():
    """Only process commands, never environment/credentials. Conservative within container."""
    found = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            argv = (folder / "cmdline").read_bytes().split(b"\0")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        text = b" ".join(argv).decode("utf-8", "replace")
        if "vllm.entrypoints.openai.api_server" in text or (b"serve" in argv and any(b"vllm" in x for x in argv[:2])):
            found.append({"pid": int(folder.name), "kind": "vllm API server"})
    return found


def preflight(args, out):
    facts = {"platform": platform.platform(), "python": sys.version,
             "devices": args.devices, "errors": [], "packages": {}}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers", "httpx", "numpy"):
        try:
            facts["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            facts["packages"][name] = None
            facts["errors"].append(f"Missing {name}; use the existing complete MFE Ascend container")
    if platform.system() != "Linux":
        facts["errors"].append("Real Ascend runs require Linux; use 'test' on this machine")
    if sys.version_info < (3, 10):
        facts["errors"].append("Python >=3.10 is required")
    try:
        facts["model"] = validate_model(args.model)
    except (ValueError, OSError) as exc:
        facts["errors"].append(str(exc))
    if platform.system() == "Linux":
        facts["npu_smi"] = capture(["npu-smi", "info"])
        if facts["npu_smi"]["returncode"] != 0:
            facts["errors"].append("npu-smi info failed; check driver/CANN/container device mapping")
        missing = [f"/dev/davinci{d}" for d in args.devices if not Path(f"/dev/davinci{d}").exists()]
        if missing:
            facts["errors"].append("NPU device nodes missing: " + ", ".join(missing))
        facts["existing_servers"] = active_servers()
        if facts["existing_servers"]:
            facts["errors"].append("Existing vLLM service in this container; use an idle experiment container. No process was stopped.")
        try:
            ensure_free_ports(args.base_port, len(args.devices))
        except OSError as exc:
            facts["errors"].append(f"Ports {args.base_port}..{args.base_port + len(args.devices) - 1} unavailable: {exc}")
    if not facts["errors"]:
        probe = ("import json,torch,torch_npu,vllm,vllm_ascend; "
                 "print('DABS_NPU_PROBE='+json.dumps({'available':torch.npu.is_available(),"
                 "'count':torch.npu.device_count(),'name':torch.npu.get_device_name(0)}))")
        facts["import_probe"] = capture([sys.executable, "-c", probe], environment(args.devices), 120)
        try:
            line = next(x.split("=", 1)[1] for x in facts["import_probe"].get("stdout", "").splitlines() if x.startswith("DABS_NPU_PROBE="))
            state = json.loads(line)
            if facts["import_probe"]["returncode"] or not state["available"] or state["count"] != len(args.devices):
                raise ValueError("NPU visibility does not match selected devices")
            facts["npu_probe"] = state
        except (StopIteration, KeyError, ValueError) as exc:
            facts["errors"].append("torch/torch_npu/vllm/vllm_ascend import or visibility check failed: " + str(exc))
    if not facts["errors"]:
        help_result = capture([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"], environment(args.devices[:1]), 120)
        facts["server_help"] = help_result
        help_text = help_result.get("stdout", "") + help_result.get("stderr", "")
        required = [x for x in server_command(args.model, args.base_port, args.memory) if x.startswith("--")]
        absent = [x for x in required if x not in help_text]
        if help_result["returncode"] or absent:
            facts["errors"].append("Installed server CLI is incompatible: " + ", ".join(absent))
    facts["passed"] = not facts["errors"]
    write_json(out / "environment.json", facts)
    if facts["errors"]:
        raise RuntimeError("; ".join(facts["errors"]))
    return facts


@contextmanager
def device_locks(devices, directory=None):
    import fcntl
    handles = []
    directory = Path(directory) if directory is not None else Path("/tmp")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for device in sorted(devices):
            handle = open(directory / f"dabs-ascend-device-{device}.lock", "a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in handles:
            handle.close()


class Servers:
    def __init__(self, args, out):
        self.args, self.out = args, Path(out)
        self.processes, self.logs = [], []
        self.endpoints = [f"http://127.0.0.1:{args.base_port + i}" for i in range(len(args.devices))]

    async def start(self, client):
        records = []
        # Sequential startup avoids all replicas compiling/loading at once.
        for index, device in enumerate(self.args.devices):
            log = (self.out / f"server_npu{device}.log").open("x", encoding="utf-8")
            self.logs.append(log)
            command = server_command(self.args.model, self.args.base_port + index, self.args.memory)
            proc = subprocess.Popen(command, env=environment([device]), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            self.processes.append(proc)
            records.append({"replica": index, "physical_device": device, "pid": proc.pid,
                            "endpoint": self.endpoints[index], "argv": command,
                            "visible_devices": str(device), "tensor_parallel_size": 1})
            write_json(self.out / "servers.json", records)
            deadline = time.monotonic() + self.args.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"NPU {device} server exited ({proc.returncode}); see {log.name}")
                try:
                    response = await client.get(self.endpoints[index] + "/health", timeout=3)
                    if response.status_code == 200:
                        break
                except Exception as exc:
                    import httpx
                    if not isinstance(exc, httpx.HTTPError):
                        raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"NPU {device} server startup timed out; see {log.name}")
                await asyncio.sleep(1)
            print(f"ASCEND_SERVER_READY replica={index} physical_npu={device}", flush=True)

    def check_alive(self):
        for proc in self.processes:
            if proc.poll() is not None:
                raise RuntimeError(f"Owned vLLM server {proc.pid} exited: {proc.returncode}")

    def close(self):
        results = []
        # Signal only the sessions this object created, including engine children.
        for proc in self.processes:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 20
        for proc in self.processes:
            try:
                proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
            results.append({"pid": proc.pid, "returncode": proc.returncode, "owned_group_stopped": True})
        for log in self.logs:
            log.close()
        write_json(self.out / "cleanup.json", results)
