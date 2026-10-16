from __future__ import annotations

import asyncio
import copy
import math
import statistics
import time

from .common import DATASET, append_json, percentile, write_json
from .scheduler import baseline, plan_dag, snapshot
from .repair import admission_check, aggregate_diagnostics, effect_signature, with_budget


def summarize(nodes, start, finish, decisions, quality, *, fixed=False):
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
    schema_errors = 0 if fixed else sum(n["answer_contract"]["status"] == "format_error" for n in nodes.values())
    dependency_ok = all(n["start"] >= nodes[p]["finish"] and
                        (fixed or n.get("parent_response_bindings", {}).get(str(nodes[p]["step_position"] + 1), {}).get("response") == nodes[p]["stream"]["text"])
                        for n in nodes.values() for p in n["parents"])
    return {"completed_workflows": len(groups), "completed_requests": len(nodes),
            "jct_mean_ms": statistics.mean(latencies), "jct_p95_ms": percentile(latencies, .95),
            "ttft_text_proxy_mean_ms": statistics.mean(n["stream"]["ttft_text_proxy_ms"] for n in nodes.values()),
            'ttft_text_proxy_p95_ms': percentile([n['stream']['ttft_text_proxy_ms'] for n in nodes.values()], .95),
            "tpot_text_proxy_mean_ms": (statistics.mean(values) if (values := [n["stream"]["tpot_text_proxy_ms"] for n in nodes.values()
                                                                            if n["stream"]["tpot_text_proxy_ms"] is not None]) else None),
            "cohort_duration_s": duration, "condition_wall_s": finish - start,
            "workflows_s": len(groups) / duration, "output_tokens_s": output_tokens / duration,
            "output_tokens": output_tokens, "answer_contract_errors": schema_errors,
            "dependency_handoff_pass": dependency_ok, "functional_pass": dependency_ok and schema_errors == 0,
            "scheduler_decisions": sum(not d.get('event') for d in decisions),
            "selector_scored_decisions": sum(bool(d.get("result", {}).get("model_selection")) for d in decisions if not d.get('event')),
            'action_accounting_version': 2,
            'selected_nonreference_actions': sum(d.get('result', {}).get('selected_cell', '00') != '00' for d in decisions),
            "nonreference_actions": sum(d.get('client_applied_nonreference', False) for d in decisions),
            'engine_applied_nonreference_actions': None,
            'http_completed_requests': len(nodes),
            'score_diagnostics': aggregate_diagnostics(decisions),
            "expired_plan_count": sum(d.get("discarded_expired", False) for d in decisions if not d.get('event')),
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
    pending_origin = None
    decisions, history, errors = [], {}, []
    fixed = spec.get('measurement_mode') == 'fixed_token_replay'
    native = cfg.get('execution_scope') == 'native'
    if native:
        import uuid
        backend.native.salt = 'dabs-condition-' + uuid.uuid4().hex
        backend.native.log = out / 'native_events.jsonl'
        await backend.native.drain(cfg['replicas'])
    if fixed:
        from . import r234
    try:
        # Historical output-length priors and GPU timing coefficients are not loaded.
        # Preserve the frozen input protocol's 16-token namespace sanity check.
        # It is not an allocator block size: physical projections/queries/scoring
        # below use cfg.block_size=128. Changing input construction would create
        # a different workload (and its within-condition check assumes 16).
        input_cfg = {**cfg, 'block_size': 16} if native else cfg
        nodes, materialize, proof = await instantiate(packets, tokenizer, input_cfg, spec["seed"], output_prior=None)
        if fixed:
            proof.update(await r234.convert(nodes, tokenizer))
        proof.update(native_cold_guard_unchanged=False, native_cache_observed=False,
                     native_input_readback=False, execution_adapter="ascend-http-v1")
        if native:
            proof.update(native_cache_observed=True, native_input_readback=True, execution_adapter='dabs-native-v023-1',
                         cache_namespace=backend.native.salt, cold_boundary='fresh per-condition engine cache salt',
                         frozen_input_namespace_check_tokens=16, physical_kv_block_tokens=cfg['block_size'])
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
            if not fixed:
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
            if fixed:
                result = await backend.generate(replica, node, spec['seed'], out / 'requests.jsonl', schema=None)
            else:
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

        async def consume_late_plan():
            nonlocal pending, pending_origin
            # Preserve diagnostics from expired searches, never apply their action.
            late = await pending
            record = {'event': 'expired_plan_diagnostic', 'index': pending_origin,
                      'discarded_expired': True, 'client_applied_nonreference': False,
                      'engine_applied': None, 'result': late}
            decisions.append(record)
            append_json(out / 'decisions.jsonl', record)
            pending, pending_origin = None, None

        async def loop():
            nonlocal pending, pending_origin
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
                if native:
                    if index % 5 == 0:
                        await backend.native.harvest(cfg['replicas'])
                    state = await backend.native.observe(nodes, cfg, index, model)
                    state['action_deadline_ns'] = int(next_tick * 1e9)
                    state['action_horizon_ms'] = max(.001, (next_tick - state['now']) * 1000)
                else:
                    state = snapshot(nodes, history, cfg["replicas"], cfg, index, now)
                if not state["ready"]:
                    if pending is not None and pending.done():
                        await consume_late_plan()
                    continue
                base = r234.baseline(state, cfg, spec['method']) if fixed else baseline(state, cfg, spec["method"])
                result = {"action": base, "evaluations": [], "model_selection": False}
                decision = {"index": index, "at": now, "snapshot": state,
                            'engine_accepted': None, 'engine_applied': None,
                            'client_applied_nonreference': False}
                if native:
                    decision['reference_action'] = copy.deepcopy(base)
                if spec["method"] in ("DAG-Ascend", 'Ours'):
                    if pending is not None and pending.done():
                        await consume_late_plan()
                    if pending is None:
                        deadline = min(now + cfg["planning_soft_limit_s"], next_tick - cfg["commit_reserve_s"])
                        planner = r234.plan_ours if fixed else plan_dag
                        pending_origin = index
                        pending = asyncio.get_running_loop().run_in_executor(executor, with_budget, planner, state, model, cfg,
                                                                            max(0., deadline - time.monotonic()))
                        try:
                            result = await asyncio.wait_for(asyncio.shield(pending), timeout=max(.001, deadline - time.monotonic()))
                            pending, pending_origin = None, None
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
                decision['requested_action'] = copy.deepcopy(action)
                reason = admission_check(action, nodes, cfg, state['now'], time.monotonic())
                if native:
                    # Never refresh an old queue permutation under a new snapshot.
                    # A stale selection can only fall back to fresh reference admission.
                    fallback = bool(reason or state.get('native_observation_errors'))
                    if fallback:
                        state = await backend.native.observe(nodes, cfg, index, model)
                        action = r234.baseline(state, cfg, spec['method'])
                    order, receipts = await backend.native.commit(action, state, nodes, fallback=fallback)
                    decision.update(native_receipts=receipts, engine_applied=any(
                        x['receipt']['status'] in ('applied', 'partially_applied') for x in receipts),
                        engine_accepted=any(x['receipt']['accepted'] for x in receipts),
                        native_fallback_reason=reason or state.get('native_observation_errors'),
                        applied_submission_order=order, result=result)
                    # If an engine advanced during scoring, retain the rejection;
                    # make a separately labelled atomic baseline admission next.
                    if not order and action['admit'] and not fallback:
                        fresh_state = await backend.native.observe(nodes, cfg, index, model)
                        base_now = r234.baseline(fresh_state, cfg, spec['method'])
                        order, extra = await backend.native.commit(base_now, fresh_state, nodes, fallback=True)
                        action = base_now
                        decision['native_receipts'] += extra
                        decision['native_fallback_reason'] = 'engine_rejected_selection; fresh atomic baseline admission'
                        decision['applied_submission_order'] = order
                    decision['client_applied_nonreference'] = bool(order and not decision.get('native_fallback_reason')
                        and result.get('selected_cell', '00') != '00' and effect_signature(action, nodes) != effect_signature(base, nodes))
                    for key in order:
                        node = nodes[key]
                        replica = action['placement'][key]
                        node.update(state='submitted', replica=replica, start=time.monotonic(), phase='prefill')
                        spawn(execute(key, replica))
                    decisions.append(decision)
                    append_json(out / 'decisions.jsonl', decision)
                    continue
                if reason:
                    decision['commit_reject_reason'] = reason
                    fresh = snapshot(nodes, history, cfg['replicas'], cfg, index)
                    action = r234.baseline(fresh, cfg, spec['method']) if fixed else baseline(fresh, cfg, spec['method'])
                    validate_action(action, fresh, cfg)
                    rejection = admission_check(action, nodes, cfg, fresh['now'], time.monotonic())
                    if rejection:
                        raise ValueError('Fresh reference admission failed: ' + rejection)
                # Completion callbacks may have run during scoring. No submitted
                # request can be cancelled, migrated or reordered by this port.
                order = r234.submission_order(action, nodes) if fixed else sorted(action['admit'], key=lambda k: (nodes[k]['ready_at'], k))
                decision['client_accepted_action'] = copy.deepcopy(action)
                decision['applied_submission_order'] = order
                decision['client_applied_nonreference'] = bool(
                    not reason and not decision.get('discarded_expired') and order
                    and result.get('selected_cell', '00') != '00'
                    and effect_signature(action, nodes) != effect_signature(base, nodes))
                decision['client_apply_at'] = time.monotonic()
                decision['snapshot_to_client_apply_ms'] = (decision['client_apply_at'] - now) * 1000
                for key in order:
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
            await consume_late_plan()
        # Gold is read only after all actual requests, including refusals, finish.
        quality = ({'semantic_quality': {'final_em': None, 'final_f1': None,
                    'quality_scope': 'nonsemantic_fixed_token_replay', 'applicable': False}}
                   if fixed else evaluate(nodes, DATASET, finish - start))
        summary = {"spec": spec, **summarize(nodes, start, finish, decisions, quality, fixed=fixed)}
        if native:
            await backend.native.drain(cfg['replicas'])
            while await backend.native.harvest(cfg['replicas'], wait_complete=True):
                pass
            applied, routed, queued = 0, 0, 0
            for decision in decisions:
                if decision.get('result', {}).get('selected_cell', '00') == '00' or decision.get('native_fallback_reason'):
                    continue
                route_effect, queue_effect = False, False
                reference = decision['reference_action']
                for item in decision.get('native_receipts', []):
                    receipt = await backend.native.call(item['replica'], 'receipt', action_id=item['receipt']['action_id'])
                    for mapping in receipt.get('engine_admissions', []):
                        key = mapping['node_id']
                        if key not in reference['admit'] or reference['placement'].get(key) != item['replica']:
                            route_effect = True
                    # A permutation that expired before a target was scheduled
                    # is a queue application, but not an executed nonreference order.
                    actual = [r for r in receipt.get('priority_scheduled_order', []) if r in receipt.get('applied', [])]
                    old = [r['engine_request_id'] for r in decision['snapshot']['native_snapshots'][item['replica']]['waiting']]
                    if len(actual) >= 2 and actual != [r for r in old if r in actual]:
                        queue_effect = True
                    append_json(out / 'decisions.jsonl', {'event': 'final_engine_receipt', 'decision_index': decision['index'],
                        'replica': item['replica'], 'receipt': receipt})
                applied += route_effect or queue_effect
                routed += route_effect; queued += queue_effect
            summary.update(execution_scope='native', native_cache_observed=True,
                native_prefill_decode_ms='see raw native_events.jsonl; engine host-step and auxiliary NPU forward events are separate',
                engine_applied_nonreference_actions=applied, engine_nonreference_routing_actions=routed,
                engine_nonreference_queue_actions=queued,
                engine_admitted_requests=sum(bool(n['stream'].get('native_mapping')) for n in nodes.values()))
        if fixed:
            replica_rows = []
            for replica in range(cfg['replicas']):
                requests = [n for n in nodes.values() if n['replica'] == replica]
                events = sorted([(n['start'], 1) for n in requests] + [(n['finish'], -1) for n in requests])
                occupancy = peak = 0
                for _, delta in events:
                    occupancy += delta
                    peak = max(peak, occupancy)
                replica_rows.append({'replica': replica, 'requests': len(requests),
                                     'output_tokens': sum(n['stream']['usage']['completion_tokens'] for n in requests),
                                     'mean_http_inflight': sum(n['finish'] - n['start'] for n in requests) / summary['cohort_duration_s'],
                                     'peak_http_inflight': peak, 'native_kv_occupancy': None})
            summary.update(fixed_work_pass=r234.audit_work(nodes),
                           external_work_fingerprint=proof['external_work_fingerprint'],
                           arrival_offsets=[nodes[keys[0]]['arrival_offset'] for keys in groups.values()],
                           input_tokens=sum(n['stream']['usage']['prompt_tokens'] for n in nodes.values()),
                           v5_candidate_generated=sum(d['result'].get('v5_candidate_generated', 0) for d in decisions),
                           v5_candidate_scored=sum(d['result'].get('v5_candidate_scored', 0) for d in decisions),
                           v5_decisions=sum('v5_revision' in d['result'] for d in decisions),
                           replica_rows=replica_rows,
                           planner_mean_wall_ms=(statistics.mean(values) if (values := [d['result']['wall_ms'] for d in decisions if 'wall_ms' in d['result']]) else None),
                           native_cache_observed=native)
            summary['functional_pass'] &= summary['fixed_work_pass']
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
                await consume_late_plan()
            except BaseException as exc:
                append_json(out / "decisions.jsonl", {"event": "pending_plan_failed", "error": repr(exc)})
        if nodes:
            write_json(out / "nodes.json", nodes)
            if "proof" in locals():
                write_json(out / "input_proof.json", proof)
