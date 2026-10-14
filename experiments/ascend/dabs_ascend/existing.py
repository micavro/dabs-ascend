"""Attach to existing HTTP services without acquiring NPU or process ownership."""
from __future__ import annotations

import asyncio
import importlib.metadata
import platform
import re
import sys
from pathlib import Path

from .common import read_json, sha, write_json
from .runtime import MODEL_NAME, validate_model


def container_name(prefix, device):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("Invalid container name prefix")
    return f"{prefix}{device}"


def selected_endpoints(args):
    return [f"http://127.0.0.1:{args.base_port + device}" for device in args.devices]


def deployment_records(rows):
    """Keep only deployment evidence; never persist Docker Env or raw Cmd secrets."""
    allowed = {"--model", "--served-model-name", "--port", "--dtype", "--max-model-len",
               "--max-num-seqs", "--max-num-batched-tokens", "--tensor-parallel-size",
               "--gpu-memory-utilization", "--block-size", "--generation-config"}
    switches = {"--enforce-eager", "--enable-prefix-caching", "--enable-chunked-prefill"}
    result = []
    for row in rows:
        argv = row.get("Config", {}).get("Cmd") or []
        options = {}
        if isinstance(argv, list):
            for i, arg in enumerate(argv):
                flag, separator, value = arg.partition("=")
                if flag in allowed:
                    if separator:
                        options[flag] = value
                    elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                        options[flag] = argv[i + 1]
                elif arg in switches:
                    options[arg] = True
        result.append({"name": row["Name"].lstrip("/"), "container_id": row["Id"],
                       "image_id": row["Image"], "running": row["State"]["Running"],
                       "devices": row["HostConfig"].get("Devices") or [],
                       "ports": row["NetworkSettings"].get("Ports") or {},
                       "server_options": options})
    return result


def validate_manifest(rows, args):
    if not isinstance(rows, list):
        raise ValueError("Deployment manifest must be a list")
    selected = []
    for device, endpoint in zip(args.devices, selected_endpoints(args)):
        name = container_name(args.container_prefix, device)
        matches = [row for row in rows if row.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate deployment container: {name}")
        row = matches[0]
        if row.get("running") is not True:
            raise ValueError(f"Deployment container is not running: {name}")
        if not row.get("container_id") or not row.get("image_id"):
            raise ValueError(f"Deployment identity missing: {name}")
        paths = {item.get("PathOnHost") for item in row.get("devices", []) or []}
        npus = {p for p in paths if p and re.fullmatch(r"/dev/davinci\d+", p)}
        if npus != {f"/dev/davinci{device}"}:
            raise ValueError(f"Single-NPU device mapping mismatch: {name}: {sorted(npus)}")
        bindings = (row.get("ports") or {}).get("8000/tcp") or []
        if not any(str(b.get("HostPort")) == str(args.base_port + device)
                   and b.get("HostIp", "") in ("", "0.0.0.0", "127.0.0.1") for b in bindings):
            raise ValueError(f"Loopback port mapping mismatch: {name}")
        selected.append({**row, "replica": len(selected), "endpoint": endpoint,
                         "declared_physical_device": device, "lifecycle": "external"})
    return selected


def preflight(args, out):
    facts = {"devices": args.devices, "mode": "existing-http-services",
             "platform": platform.platform(), "python": sys.version,
             "errors": [], "packages": {}, "physical_npu_probe_performed": False,
             "limits": ["Physical mappings come from a host Docker snapshot, not a controller NPU probe.",
                        "HTTP cannot prove the loaded weights match the local model byte-for-byte.",
                        "Existing services may share hardware with other work; exclusivity is not verified."]}
    try:
        if platform.system() != "Linux" or sys.version_info < (3, 10):
            raise ValueError("Existing-service runs require Linux and Python >=3.10")
        for name in ("httpx", "numpy"):
            facts["packages"][name] = importlib.metadata.version(name)
        facts["model"] = validate_model(args.model)
        facts["deployment_manifest_sha256"] = sha(args.deployment_manifest)
        facts["deployment"] = validate_manifest(read_json(args.deployment_manifest), args)
        import httpx
        with httpx.Client(timeout=15, trust_env=False) as client:
            for row in facts["deployment"]:
                endpoint = row["endpoint"]
                client.get(endpoint + "/health").raise_for_status()
                response = client.get(endpoint + "/v1/models")
                response.raise_for_status()
                models = [m for m in response.json()["data"] if m.get("id") == MODEL_NAME]
                if len(models) != 1 or models[0].get("max_model_len", 0) < 4096:
                    raise ValueError(f"Served model or context limit mismatch: {endpoint}")
                row["served_model"] = models[0]
    except Exception as exc:
        facts["errors"].append(repr(exc))
    facts["passed"] = not facts["errors"]
    write_json(out / "environment.json", facts)
    write_json(out / "external_adapter.json", {
        "adapter_sha256": sha(Path(__file__)), "endpoints": selected_endpoints(args),
        "starts_model_processes": False, "stops_external_services": False,
        "memory_flag_applies_to_external_services": False,
        "deployment_manifest": str(args.deployment_manifest)})
    if facts["errors"]:
        raise RuntimeError("; ".join(facts["errors"]))
    return facts


class ExistingServers:
    monitor_interval = 5.

    def __init__(self, args, out):
        self.args, self.out = args, Path(out)
        self.endpoints = selected_endpoints(args)
        self.monitor = None
        self.failure = None

    async def start(self, client):
        records = validate_manifest(read_json(self.args.deployment_manifest), self.args)
        write_json(self.out / "servers.json", records)
        for endpoint in self.endpoints:
            response = await client.get(endpoint + "/health", timeout=10)
            response.raise_for_status()
            print("ASCEND_EXTERNAL_READY=" + endpoint, flush=True)

        async def monitor():
            try:
                while True:
                    await asyncio.sleep(self.monitor_interval)
                    for endpoint in self.endpoints:
                        response = await client.get(endpoint + "/health", timeout=10)
                        response.raise_for_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failure = repr(exc)
        self.monitor = asyncio.create_task(monitor())

    def check_alive(self):
        if self.failure:
            raise RuntimeError("Existing service health check failed: " + self.failure)

    async def aclose(self):
        if self.monitor is not None:
            self.monitor.cancel()
            try:
                await self.monitor
            except asyncio.CancelledError:
                pass
        write_json(self.out / "cleanup.json", {"mode": "external", "external_services_stopped": False,
                   "note": "Only controller health monitoring stopped; model services remain running."})
