from __future__ import annotations

import asyncio
import copy
import math
import statistics
import time

from .common import DATASET, append_json, percentile, write_json
from .scheduler import baseline, plan_dag, snapshot


def summarize(nodes, start, finish, decisions, quality):
    groups = {}
    for node in nodes.values():
        groups.setdefault(node["wid"], []).append(node)
    workflows = []
    for wid, group in groups.items():
        arrival = min(n["arrival"] for n in group)
        end = max(n["finish"] for n in group)
        chain = []
        current = max(group, key=lambda n: n["finish"])
        while True:
            chain.append(current)
            if not current["parents"]:
                break
            current = max((nodes[p] for p in current["parents"]), key=lambda n: n["finish"])
        parts = {}
        previous = arrival
        for node in reversed(chain):
            stream = node["stream"]
            stages = {"dependency_release": node["ready_at"] - previous,
                      "input_preparation": node["input_prepare_end"] - node["ready_at"],
                      "queue_and_admission": stream["start"] - node["input_prepare_end"],
                      "http_request_and_generation": stream["finish"] - stream["start"],
                      "client_finalize": node["finish"] - stream["finish"]}
            if any(value < -.000001 for value in stages.values()):
                raise ValueError("Negative critical-path timing component")
            for key, value in stages.items():
                parts[key + "_ms"] = parts.get(key + "_ms", 0.) + value * 1000
            previous = node["finish"]
        if abs(sum(parts.values()) - (end - arrival) * 1000) > .001:
            raise ValueError("Critical-path components do not sum to workflow JCT")
        workflows.append({"wid": wid, "dataset_id": group[0]["dataset_id"], "arrival": arrival,
                          "finish": end, "jct_ms": (end - arrival) * 1000,
                          "critical_path_ms": parts,
                          "scheduled_arrival": group[0]["scheduled_arrival"],
                          "arrival_lag_ms": (arrival - group[0]["scheduled_arrival"]) * 1000})
    latencies = [w["jct_ms"] for w in workflows]
    duration = finish - min(w["arrival"] for w in workflows)
    if duration <= 0:
        raise ValueError("Invalid measured workflow interval")
    output_tokens = sum(n["stream"]["usage"]["completion_tokens"] for n in nodes.values())
    schema_errors = sum(n["answer_contract"]["status"] == "format_error" for n in nodes.values())
    dependency_ok = all(n["start"] >= nodes[p]["finish"] and
                        n.get("parent_response_bindings", {}).get(str(nodes[p]["step_position"] + 1), {}).get("response") == nodes[p]["stream"]["text"]
                        for n in nodes.values() for p in n["parents"])
    return {"completed_workflows": len(groups), "completed_requests": len(nodes),
            "jct_mean_ms": statistics.mean(latencies), "jct_p95_ms": percentile(latencies, .95),
            "ttft_text_proxy_mean_ms": statistics.mean(n["stream"]["ttft_text_proxy_ms"] for n in nodes.values()),
            "tpot_text_proxy_mean_ms": (statistics.mean(values) if (values := [n["stream"]["tpot_text_proxy_ms"] for n in nodes.values()
                                                                            if n["stream"]["tpot_text_proxy_ms"] is not None]) else None),
            "cohort_duration_s": duration, "condition_wall_s": finish - start,
            "workflows_s": len(groups) / duration, "output_tokens_s": output_tokens / duration,
            "output_tokens": output_tokens, "answer_contract_errors": schema_errors,
            "dependency_handoff_pass": dependency_ok, "functional_pass": dependency_ok and schema_errors == 0,
            "scheduler_decisions": len(decisions),
            "selector_scored_decisions": sum(bool(d.get("result", {}).get("model_selection")) for d in decisions),
            "nonreference_actions": sum(d.get("result", {}).get("selected_cell") == "10" for d in decisions),
            "expired_plan_count": sum(d.get("discarded_expired", False) for d in decisions),
            "busy_planner_count": sum(d.get("planner_busy", False) for d in decisions),
            "workflow_rows": workflows, "quality": quality["semantic_quality"],
            "native_kv_hit_rate": None, "native_prefill_decode_ms": None,
            "strict_gpu39_reproduction": False}


async def run_condition(backend, tokenizer, packets, model, cfg, spec, out, executor, alive=lambda: None):
    from typed19.workflow import instantiate
    from typed19.quality import evaluate
    from joint18v2.common import validate_action

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"status": "preparing", "spec": spec})
    nodes = {}
    tasks = set()
    pending = None
    decisions, history, errors = [], {}, []
    try:
        # Historical output-length priors and GPU timing coefficients are not loaded.
        nodes, materialize, proof = await instantiate(packets, tokenizer, cfg, spec["seed"], output_prior=None)
        proof.update(native_cold_guard_unchanged=False, native_cache_observed=False,
                     native_input_readback=False, execution_adapter="ascend-http-v1")
        write_json(out / "input_proof.json", proof)
        write_json(out / "initial_nodes.json", nodes)
        for node in nodes.values():
            node["arrival"] = None
        successors = {k: [] for k in nodes}
        groups = {}
        for key, node in nodes.items():
            groups.setdefault(node["wid"], []).append(key)
            for parent in node["parents"]:
                successors[parent].append(key)

        def spawn(coroutine):
            task = asyncio.create_task(coroutine)
            tasks.add(task)
            def finished(t):
                tasks.discard(t)
                if not t.cancelled() and t.exception() is not None:
                    errors.append(t.exception())
            task.add_done_callback(finished)

        async def prepare(key):
            node = nodes[key]
            node["input_prepare_start"] = time.monotonic()
            await materialize(key, nodes)
            node.update(state="ready", input_prepare_end=time.monotonic())

        def release(keys):
            for key in keys:
                node = nodes[key]
                if node["state"] == "blocked" and not node.get("preparing") and node["arrival"] is not None and all(nodes[p]["state"] == "done" for p in node["parents"]):
                    node.update(preparing=True, ready_at=max([node["arrival"]] + [nodes[p]["finish"] for p in node["parents"]]))
                    spawn(prepare(key))

        async def arrive(keys):
            at = start + nodes[keys[0]]["arrival_offset"]
            await asyncio.sleep(max(0., at - time.monotonic()))
            now = time.monotonic()
            for key in keys:
                nodes[key].update(arrival=now, scheduled_arrival=at)
            release(keys)

        async def execute(key, replica):
            node = nodes[key]
            result = await backend.generate(replica, node, spec["seed"], out / "requests.jsonl")
            node.update(state="done", finish=time.monotonic(), phase="done",
                        output=result["usage"]["completion_tokens"], output_is_estimate=False,
                        output_done=result["usage"]["completion_tokens"])
            locations = history.setdefault(node["prefix_hash"], [])
            if replica not in locations:
                locations.append(replica)
            release(successors[key])

        start = time.monotonic()
        for keys in groups.values():
            spawn(arrive(keys))
        write_json(out / "status.json", {"status": "running", "spec": spec})

        async def loop():
            nonlocal pending
            index = 0
            next_tick = start
            while not all(n["state"] == "done" for n in nodes.values()):
                if errors:
                    raise errors[0]
                alive()
                await asyncio.sleep(max(0., next_tick - time.monotonic()))
                await asyncio.sleep(cfg["ready_coalesce_s"])
                if errors:
                    raise errors[0]
                now = time.monotonic()
                next_tick = start + (math.floor((now - start) / cfg["epoch_s"]) + 1) * cfg["epoch_s"]
                index += 1
                state = snapshot(nodes, history, cfg["replicas"], cfg, index, now)
                if not state["ready"]:
                    if pending is not None and pending.done():
                        pending.result()  # Never swallow a failure from an expired plan.
                        pending = None
                    continue
                base = baseline(state, cfg, spec["method"])
                result = {"action": base, "evaluations": [], "model_selection": False}
                decision = {"index": index, "at": now, "snapshot": state}
                if spec["method"] == "DAG-Ascend":
                    if pending is not None and pending.done():
                        pending.result()
                        pending = None
                    if pending is None:
                        deadline = min(now + cfg["planning_soft_limit_s"], next_tick - cfg["commit_reserve_s"])
                        pending = asyncio.get_running_loop().run_in_executor(executor, plan_dag, state, model, cfg, deadline)
                        try:
                            result = await asyncio.wait_for(asyncio.shield(pending), timeout=max(.001, deadline - time.monotonic()))
                            pending = None
                        except asyncio.TimeoutError:
                            decision["discarded_expired"] = True
                    else:
                        decision["planner_busy"] = True
                # Bound the age of plans even if the event loop was busy.
                if time.monotonic() >= next_tick - cfg["commit_reserve_s"]:
                    decision["discarded_expired"] = True
                    decision["result"] = result
                    decisions.append(decision)
                    append_json(out / "decisions.jsonl", decision)
                    continue
                action = result["action"]
                validate_action(action, state, cfg)
                # Completion callbacks may have run during scoring. No submitted
                # request can be cancelled, migrated or reordered by this port.
                for key in sorted(action["admit"], key=lambda k: (nodes[k]["ready_at"], k)):
                    node = nodes[key]
                    if node["state"] != "ready":
                        raise ValueError("Planned node changed state before submission")
                    replica = action["placement"][key]
                    inflight = sum(n["state"] == "submitted" and n["replica"] == replica for n in nodes.values())
                    if inflight >= cfg["max_num_seqs"]:
                        raise ValueError("Client in-flight admission limit exceeded")
                    node.update(state="submitted", replica=replica, start=time.monotonic(), phase="prefill")
                    spawn(execute(key, replica))
                decision["result"] = result
                decisions.append(decision)
                append_json(out / "decisions.jsonl", decision)

        await asyncio.wait_for(loop(), timeout=cfg["condition_wall_s"])
        if errors:
            raise errors[0]
        finish = time.monotonic()
        if pending is not None:
            await pending
            pending = None
        # Gold is read only after all actual requests, including refusals, finish.
        quality = evaluate(nodes, DATASET, finish - start)
        summary = {"spec": spec, **summarize(nodes, start, finish, decisions, quality)}
        write_json(out / "quality.json", quality)
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {"status": "complete", "functional_pass": summary["functional_pass"]})
        return summary
    except BaseException as exc:
        write_json(out / "status.json", {"status": "failed", "error": repr(exc), "spec": spec})
        raise
    finally:
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*list(tasks), return_exceptions=True)
        if pending is not None:
            # Child search is time bounded; consume any exception and preserve its trace.
            try:
                await pending
            except BaseException as exc:
                append_json(out / "decisions.jsonl", {"event": "pending_plan_failed", "error": repr(exc)})
        if nodes:
            write_json(out / "nodes.json", nodes)
            if "proof" in locals():
                write_json(out / "input_proof.json", proof)
