from __future__ import annotations
import inspect,json,os,time
from pathlib import Path
from .common import make_budget_class
# This patch corrects package identification only. It does not change scheduling.
KNOWN_DEPLOYED_SCHEDULER_SHA256 = "99cfe49101dec65171b98ff24f8e8c098ac217e9a493d10fdacea4d5111f0bcf"
PREFLIGHT_REVISION = "stage18-native-precheck-fix-1"

class NativeContractError(RuntimeError):
    def __init__(self, reason, facts):
        self.reason = reason
        self.facts = facts
        super().__init__(reason + "; imported=" + repr(facts.get("imported_version"))
                         + "; metadata=" + repr(facts.get("distribution_version"))
                         + "; scheduler_sha256=" + str(facts.get("scheduler_source_sha256")))

def _release_063(value):
    import re
    if not isinstance(value, str):
        return False
    # A build suffix does not make a different release acceptable.
    return bool(re.fullmatch(r"0\.6\.3(?:\.post\d+)?(?:\+[A-Za-z0-9][A-Za-z0-9._-]*)?", value))

def _version_decision(facts):
    """A missing generated version file is NOT permission to accept arbitrary code."""
    imported = facts.get("imported_version")
    metadata = facts.get("distribution_version")
    if _release_063(imported):
        if metadata is not None and not _release_063(metadata):
            raise NativeContractError("imported_and_distribution_release_disagree", facts)
        return "release_0.6.3_plus_native_contract"
    if imported == "dev":
        if not _release_063(metadata):
            raise NativeContractError("dev_requires_distribution_metadata_0.6.3", facts)
        if facts.get("scheduler_source_sha256") != KNOWN_DEPLOYED_SCHEDULER_SHA256:
            raise NativeContractError("dev_requires_previously_recorded_scheduler_hash", facts)
        if facts.get("distribution_scheduler_matches_loaded") is not True:
            raise NativeContractError("distribution_does_not_identify_loaded_scheduler", facts)
        return "dev_metadata_0.6.3_and_recorded_scheduler_plus_native_contract"
    raise NativeContractError("unsupported_imported_vllm_release", facts)

def native_facts():
    """Read import/package evidence. No model, engine, scheduler or request is created."""
    import hashlib
    import importlib.metadata as metadata
    import platform
    import vllm
    import vllm.core.scheduler as sm
    from vllm.engine.async_llm_engine import _AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    def file_info(path):
        if not path:
            return {"path": None, "sha256": None}
        p = Path(path).resolve()
        try:
            return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        except OSError as ex:
            return {"path": str(p), "sha256": None, "error": type(ex).__name__ + ": " + str(ex)}
    scheduler = file_info(getattr(sm, "__file__", None))
    facts = {"revision": PREFLIGHT_REVISION,
             "imported_version": str(getattr(vllm, "__version__", "<missing>")),
             "vllm_file": getattr(vllm, "__file__", None),
             "python": platform.python_version(),
             "distribution_version": None,
             "distribution_scheduler_matches_loaded": None,
             "scheduler_file": scheduler["path"],
             "scheduler_source_sha256": scheduler["sha256"],
             "known_scheduler_hash_provenance": "prior project runtime logs, not a claimed full-wheel or upstream attestation",
             "gpu_requests": 0, "model_generations": 0, "changes": 0,
             "gpu_compatibility_proven": False}
    try:
        dist = metadata.distribution("vllm")
        facts["distribution_version"] = dist.version
        dp = Path(dist.locate_file("vllm/core/scheduler.py")).resolve()
        facts["distribution_scheduler_path"] = str(dp)
        facts["distribution_scheduler_matches_loaded"] = (str(dp) == scheduler["path"] and dp.is_file())
    except metadata.PackageNotFoundError:
        facts["distribution_error"] = "PackageNotFoundError"
    except Exception as ex:
        facts["distribution_error"] = type(ex).__name__ + ": " + str(ex)
    funcs = {"Scheduler.schedule": getattr(sm.Scheduler, "schedule", None),
             "Scheduler._get_num_new_tokens": getattr(sm.Scheduler, "_get_num_new_tokens", None),
             "Scheduler.add_seq_group": getattr(sm.Scheduler, "add_seq_group", None),
             "_AsyncLLMEngine.step_async": getattr(_AsyncLLMEngine, "step_async", None),
             "LLMEngine.step": getattr(LLMEngine, "step", None)}
    facts["native_functions"] = {}
    for name, fn in funcs.items():
        item = {"callable": callable(fn), "module": getattr(fn, "__module__", None)}
        if callable(fn):
            try:
                item.update(signature=str(inspect.signature(fn)),
                            parameters=list(inspect.signature(fn).parameters),
                            source_file=inspect.getsourcefile(fn))
            except (TypeError, ValueError, OSError) as ex:
                item["inspection_error"] = str(ex)
        facts["native_functions"][name] = item
    facts["native_async_is_coroutine"] = inspect.iscoroutinefunction(getattr(_AsyncLLMEngine, "step_async", None))
    import sys
    facts["native_module_files"] = {}
    for name in ("vllm.version", "vllm.config", "vllm.sequence", "vllm.engine.async_llm_engine", "vllm.engine.llm_engine"):
        module = sys.modules.get(name)
        facts["native_module_files"][name] = file_info(getattr(module, "__file__", None))
    return facts

def _budget_cpu_contract(Native):
    """Exercise actual installed budget class, locally; no class monkey-patching."""
    context = {"prefill_cap": 256}
    Budget = make_budget_class(Native, lambda: context)
    checks = []
    def require(ok, name):
        if not ok:
            raise RuntimeError("native_budget_contract:" + name)
        checks.append(name)
    b = Budget(4096, 64)
    b.j18_current_prefill = True; b.j18_phases["p"] = True
    require(b.remaining_token_budget() == 256, "prefill_cap")
    require(not b.can_schedule(num_new_tokens=257, num_new_seqs=1), "reject_excess_prefill")
    b.add_num_batched_tokens("p", 256)
    require(b.remaining_token_budget() == 0, "prefill_exhausted")
    b.add_num_batched_tokens("p", 256)
    require(b.j18_prefill_used == 256 and b.num_batched_tokens == 256, "duplicate_not_double_counted")
    b.j18_current_prefill = False; b.j18_phases["d"] = False
    require(b.can_schedule(num_new_tokens=1, num_new_seqs=1), "decode_after_prefill_exhaustion")
    b.add_num_batched_tokens("d", 1)
    require(b.j18_prefill_used == 256 and b.num_batched_tokens == 257, "decode_not_charged_to_prefill")
    b.subtract_num_batched_tokens("p", 256)
    b.j18_current_prefill = True
    require(b.remaining_token_budget() == 256 and b.num_batched_tokens == 1, "subtract_restores_prefill")
    b = Budget(64, 64); b.j18_current_prefill = False; b.add_num_batched_tokens("all", 64)
    require(not b.can_schedule(num_new_tokens=1, num_new_seqs=1), "native_total_ceiling")
    b = Budget(4096, 64); b.add_num_seqs("all", 64)
    require(not b.can_schedule(num_new_tokens=1, num_new_seqs=1), "native_sequence_ceiling")
    context["prefill_cap"] = None; b = Budget(2048, 64); b.j18_current_prefill = True
    require(b.remaining_token_budget() == 2048, "uncapped_native_total")
    return checks

def capabilities():
    facts = native_facts()
    decision = _version_decision(facts)
    import vllm.core.scheduler as sm
    from vllm.engine.async_llm_engine import _AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    for k in ("schedule", "_get_num_new_tokens", "add_seq_group"):
        if not callable(getattr(sm.Scheduler, k, None)):
            raise NativeContractError("missing native " + k, facts)
    if not callable(getattr(_AsyncLLMEngine, "step_async", None)) or not facts["native_async_is_coroutine"]:
        raise NativeContractError("missing asynchronous engine timing hook", facts)
    if not callable(getattr(LLMEngine, "step", None)):
        raise NativeContractError("missing synchronous engine timing hook", facts)
    actual = list(inspect.signature(sm.Scheduler._get_num_new_tokens).parameters)
    if actual != ["self", "seq_group", "status", "enable_chunking", "budget"]:
        raise NativeContractError("native token-count signature changed", facts)
    if sm.Scheduler.schedule.__module__ != "vllm.core.scheduler":
        raise NativeContractError("another scheduler hook already installed", facts)
    if sm.Scheduler._get_num_new_tokens.__module__ != "vllm.core.scheduler":
        raise NativeContractError("another native token hook already installed", facts)
    try:
        checks = _budget_cpu_contract(sm.SchedulingBudget)
    except Exception as ex:
        raise NativeContractError("native budget CPU contract failed: " + repr(ex), facts) from ex
    return {**facts, "version": facts["imported_version"], "token_signature": actual,
            "identity_decision": decision, "budget_contract_checks": checks,
            "precheck_pass": True, "gpu_canary_required": True}

