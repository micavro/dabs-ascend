"""Single-writer, event-driven admission runner using the existing native API.

No engine queue mutation, installation, model fitting, full V5 or NPU restart.
No result-driven retry. A failed run keeps all raw evidence and stops the suite.
"""
from __future__ import annotations

import asyncio
import copy
import math
import statistics
import time
import uuid
from collections import Counter

from .fast_io import append_json, write_json, distribution, percentile
from .fast_policy import REVISION, admission_only, validate_admissions, worker_choose


def compact_snapshot(state):
    # Transport evidence stays in native_events.jsonl; no request/DAG is removed.
    result = {k: v for k, v in state.items() if k != 'native_snapshots'}
    result['nodes'] = {k: {f: v for f, v in n.items() if f not in ('native_request', 'tokens', 'stream')}
                       for k, n in state['nodes'].items()}
    result['cache_evidence'] = [{k: v for k, v in e.items() if k != 'native'} for e in state.get('cache_evidence', [])]
    return result


async def wait_for_work(event, interval, coalesce):
    try:
        await asyncio.wait_for(event.wait(), interval)
    except asyncio.TimeoutError:
        pass
    event.clear()
    if coalesce:
        await asyncio.sleep(coalesce)


async def atomic_commit(native, action, state, nodes, cfg):
    """Real permits only. No native order claim is made for concurrent HTTPs."""
    # Validate against live client state just before any await. One writer owns
    # all admissions, so completions can only free slots while RPCs are pending.
    live = {**state, 'nodes': nodes, 'ready': [k for k, n in nodes.items() if n['state'] == 'ready']}
    live['reserved_blocks'] = [sum(math.ceil((n['prompt'] + n['max_output_tokens']) / cfg['block_size'])
        for n in nodes.values() if n['state'] == 'submitted' and n.get('replica') == r) for r in range(cfg['replicas'])]
    validate_admissions(action, live, cfg)
    async def one(r):
        keys = [k for k in action['admit'] if action['placement'][k] == r]
        if not keys:
            return [], None
        # No queue/order keys are transmitted. EngineCore checks CURRENT state.
        receipt = await native.call(r, 'admit', admissions=[native.admission(nodes[k]) for k in keys])
        if receipt.get('status') not in ('applied', 'partially_applied', 'rejected', 'expired'):
            raise ValueError('unexpected atomic admission receipt')
        permitted = set(receipt.get('admitted_permits', []))
        requested = {nodes[k]['client_request_id'] for k in keys}
        if permitted - requested:
            raise ValueError('engine returned an unrequested admission permit')
        accepted = []
        for k in keys:
            if nodes[k]['client_request_id'] in permitted:
                nodes[k]['native_action_id'] = receipt['action_id']
                accepted.append(k)
        return accepted, {'replica': r, 'receipt': receipt}
    responses = await asyncio.gather(*(one(r) for r in range(cfg['replicas'])), return_exceptions=True)
    failures = [value for value in responses if isinstance(value, BaseException)]
    if failures:
        # Other replicas may already have issued short-lived permits. Await
        # every response, preserve native logs, and STOP; never retry blindly.
        raise failures[0]
    return [k for keys, _ in responses for k in keys], [row for _, row in responses if row is not None]


def summarize(nodes, spec, decisions, finish, proof, counters):
    groups = {}
    for n in nodes.values():
        groups.setdefault(n['wid'], []).append(n)
    wfrows = []
    for wid, group in groups.items():
        arrival = min(n['arrival'] for n in group)
        end = max(n['finish'] for n in group)
        wfrows.append({'wid': wid, 'jct_ms': (end - arrival) * 1000,
                       'arrival_lag_ms': (arrival - group[0]['scheduled_arrival']) * 1000})
    jcts = [r['jct_ms'] for r in wfrows]
    duration = max(n['finish'] for n in nodes.values()) - min(n['arrival'] for n in nodes.values())
    dependency = all(n['start'] >= nodes[p]['finish'] for n in nodes.values() for p in n['parents'])
    actual_work = all(n['stream']['usage']['prompt_tokens'] == len(n['tokens']) and
                      n['stream']['usage']['completion_tokens'] == n['max_output_tokens'] for n in nodes.values())
    output = sum(n['stream']['usage']['completion_tokens'] for n in nodes.values())
    values = lambda name: [d[name] for d in decisions if d.get(name) is not None]
    return {'spec': spec, 'selector_revision': REVISION, 'completed_workflows': len(groups),
            'completed_requests': len(nodes), 'jct_mean_ms': statistics.mean(jcts), 'jct_p95_ms': percentile(jcts, .95),
            'cohort_duration_s': duration, 'workflows_s': len(groups) / duration, 'output_tokens_s': output / duration,
            'input_tokens': sum(n['stream']['usage']['prompt_tokens'] for n in nodes.values()), 'output_tokens': output,
            'fixed_work_pass': actual_work, 'dependency_handoff_pass': dependency, 'functional_pass': actual_work and dependency,
            'external_work_fingerprint': proof.get('external_work_fingerprint'),
            'arrival_offsets': [group[0]['arrival_offset'] for group in groups.values()],
            'workflow_rows': wfrows, 'native_batch_rollout': False, 'formal_acceptance': False,
            'native_kv_hit_rate': None, 'performance_claim': 'single-scenario screening; no superiority inferred from completion',
            'diagnostics': dict(counters), 'timings_ms': {
                'observe': distribution(values('observe_ms')), 'plan_wait': distribution(values('plan_wait_ms')),
                'commit': distribution(values('commit_ms')),
                'ready_to_submit': distribution([(n['start'] - n['ready_at']) * 1000 for n in nodes.values()]),
                'http_done_to_client_done': distribution([(n['finish'] - n['stream']['finish']) * 1000 for n in nodes.values()])},
            'ttft_text_proxy_mean_ms': statistics.mean(n['stream']['ttft_text_proxy_ms'] for n in nodes.values()),
            'ttft_scope': 'HTTP text proxy, not engine native TTFT'}


async def run_condition(backend, nodes, proof, model, cfg, spec, out, executor, baseline_fn,
                        planner_fn=worker_choose):
    """Nodes are already instantiated/converted with the original fixed work."""
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'spec.json', spec); write_json(out / 'config.json', cfg)
    write_json(out / 'initial_nodes.json', nodes); write_json(out / 'input_proof.json', proof)
    write_json(out / 'status.json', {'status': 'running', 'spec': spec})
    native = backend.native
    native.log = out / 'native_events.jsonl'
    native.salt = 'dabs-fast-' + uuid.uuid4().hex
    native.input_identity_cache, native.query_hints = {}, {}
    native.defer_evidence = True
    await native.drain(cfg['replicas'])
    # Ensure empty decision files also exist for a failed first observation.
    (out / 'decisions.jsonl').touch()
    tasks, errors, decisions = set(), [], []
    counters = Counter()
    event, pending = asyncio.Event(), None
    pending_index = None
    groups, children = {}, {k: [] for k in nodes}
    for k, n in nodes.items():
        n['arrival'] = None
        groups.setdefault(n['wid'], []).append(k)
        for p in n['parents']:
            children[p].append(k)
    start, last_progress = time.monotonic(), time.monotonic()
    last_report, last_harvest = start, start
    current_index = 0

    def spawn(coro):
        task = asyncio.create_task(coro)
        tasks.add(task)
        def finished(t):
            tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                errors.append(t.exception())
            event.set()
        task.add_done_callback(finished)

    def release(keys):
        for k in keys:
            n = nodes[k]
            if n['state'] == 'blocked' and n['arrival'] is not None and all(nodes[p]['state'] == 'done' for p in n['parents']):
                now = time.monotonic()
                n.update(state='ready', ready_at=max([n['arrival']] + [nodes[p]['finish'] for p in n['parents']]),
                         input_prepare_start=now, input_prepare_end=now)
                event.set()

    async def arrive(keys):
        at = start + nodes[keys[0]]['arrival_offset']
        await asyncio.sleep(max(0., at - time.monotonic()))
        now = time.monotonic()
        for k in keys:
            nodes[k].update(arrival=now, scheduled_arrival=at)
        release(keys)

    async def execute(k, replica):
        nonlocal last_progress
        n = nodes[k]
        result = await backend.generate(replica, n, spec['seed'], out / 'requests.jsonl', schema=None)
        now = time.monotonic()
        # Publish immediately after the backend validates SSE and binding.
        n.update(state='done', phase='done', finish=now, client_done_at=now,
                 output=result['usage']['completion_tokens'], output_done=result['usage']['completion_tokens'],
                 output_is_estimate=False, stream=result)
        last_progress = now
        release(children[k])
        event.set()

    async def collect_late():
        nonlocal pending, pending_index
        if pending is not None and pending.done():
            value = pending.result()  # Programming errors are not disguised as fallback.
            append_json(out / 'decisions.jsonl', {'event': 'late_plan_discarded', 'index': pending_index,
                                                 'result': value})
            pending = None; pending_index = None
            counters['late_plans_discarded'] += 1

    async def loop():
        nonlocal pending, pending_index, current_index, last_progress, last_report, last_harvest
        for keys in groups.values():
            spawn(arrive(keys))
        event.set()
        while not all(n['state'] == 'done' for n in nodes.values()):
            await wait_for_work(event, cfg['fast_poll_s'], cfg['ready_coalesce_s'])
            if errors:
                raise errors[0]
            await collect_late()
            ready = [k for k, n in nodes.items() if n['state'] == 'ready']
            counts = [sum(n['state'] == 'submitted' and n.get('replica') == r for n in nodes.values()) for r in range(cfg['replicas'])]
            if time.monotonic() - last_harvest >= cfg.get('fast_harvest_s', 5.):
                # Existing bounded export/ACK, not an unbounded drain in the hot path.
                await native.harvest(cfg['replicas'])
                last_harvest = time.monotonic()
            if not ready or all(c >= cfg['client_max_inflight'] for c in counts):
                counters['no_ready_or_free_slot'] += 1
                continue
            if time.monotonic() - last_progress > cfg.get('native_idle_stall_s', 60.) and not sum(counts):
                raise ValueError('ready work stalled without any active request for 60s')
            current_index += 1
            index, observed = current_index, time.monotonic()
            try:
                state = await asyncio.wait_for(native.observe(nodes, {
                    **cfg, 'native_observation_deadline': observed + cfg['fast_snapshot_age_s']}, index, model),
                    timeout=cfg['fast_observe_timeout_s'])
            except asyncio.TimeoutError:
                counters['observation_timeout'] += 1
                append_json(out / 'decisions.jsonl', {'index': index, 'reason': 'observation_timeout'})
                continue
            observe_ms = (time.monotonic() - observed) * 1000
            if not state['ready']:
                continue
            base = admission_only(baseline_fn(state, cfg, spec['method']))
            validate_admissions(base, state, cfg)
            result = {'action': base, 'selected_cell': '00', 'evaluations': [], 'stop_reason': 'baseline'}
            d = {'index': index, 'reference_action': base, 'observe_ms': observe_ms,
                 'snapshot': compact_snapshot(state),
                 'native_snapshot_refs': [{'replica': r, 'snapshot_id': row['snapshot_id'], 'captured_ns': row['captured_ns']}
                                          for r, row in enumerate(state.get('native_snapshots', []))]}
            captured = [row['captured_ns'] / 1e9 for row in state.get('native_snapshots', [])]
            expiry = min([state['now'] + cfg['fast_snapshot_age_s']] +
                         [row['expires_ns'] / 1e9 for row in state.get('native_snapshots', [])] +
                         [t + cfg['fast_snapshot_age_s'] for t in captured])
            plan_start = time.monotonic()
            deadline = min(plan_start + cfg['fast_plan_budget_s'], expiry - cfg['fast_commit_reserve_s'])
            if spec['method'] == 'Ours' and deadline > plan_start:
                if pending is None:
                    loop_ = asyncio.get_running_loop()
                    pending = loop_.run_in_executor(executor, planner_fn, compact_snapshot(state), model, cfg, base, deadline)
                    pending_index = index
                    counters['planner_started'] += 1
                    try:
                        result = await asyncio.wait_for(asyncio.shield(pending), max(.001, deadline - time.monotonic()))
                        pending = None; pending_index = None
                        if time.monotonic() >= deadline:
                            result = {'action': base, 'selected_cell': '00', 'evaluations': [], 'stop_reason': 'planner_result_late'}
                            counters['planner_timeout'] += 1
                    except asyncio.TimeoutError:
                        result['stop_reason'] = 'planner_timeout_reference'
                        counters['planner_timeout'] += 1
                else:
                    counters['planner_busy_reference'] += 1
                    result['stop_reason'] = 'planner_busy_reference'
            d['plan_wait_ms'] = (time.monotonic() - plan_start) * 1000
            action = result['action']
            counters['stop:' + result.get('stop_reason', 'unspecified')] += 1
            if time.monotonic() >= expiry:
                # Old optimized actions never get a fresh lease. No blind resubmit.
                counters['snapshot_age_deferred'] += 1
                append_json(out / 'decisions.jsonl', {**d, 'result': result, 'reason': 'snapshot_age_deferred'})
                continue
            if state.get('native_observation_errors'):
                counters['observation_unusable_reference'] += 1
                action = base
                result.update(action=base, selected_cell='00', stop_reason='observation_unusable_atomic_reference')
            if not action['admit']:
                counters['no_admissible_request'] += 1
                append_json(out / 'decisions.jsonl', {**d, 'result': result})
                continue
            commit_start = time.monotonic()
            admitted, receipts = await atomic_commit(native, action, state, nodes, cfg)
            d.update(result=result, commit_ms=(time.monotonic() - commit_start) * 1000,
                     receipts=receipts, admitted_nodes=admitted, execution_path='atomic_new_admissions',
                     fallback=result.get('selected_cell', '00') == '00' and spec['method'] == 'Ours')
            # No await between receiving permits and publishing ALL submissions.
            for k in admitted:
                if nodes[k]['state'] != 'ready':
                    raise ValueError('single-writer request state changed before submission')
                r = action['placement'][k]
                nodes[k].update(state='submitted', replica=r, phase='prefill', start=time.monotonic())
                spawn(execute(k, r))
            counters['decisions_committed'] += 1
            counters['requests_admitted'] += len(admitted)
            counters['nonreference_selected'] += result.get('selected_cell', '00') != '00'
            counters['fast_complete_comparisons'] += bool(result.get('model_selection'))
            counters['candidates_generated'] += result.get('candidate_diagnostics', {}).get('distinct_candidates', 0)
            counters['permit_rejected_requests'] += len(action['admit']) - len(admitted)
            if admitted:
                last_progress = time.monotonic()
            decisions.append(d)
            append_json(out / 'decisions.jsonl', d)
            native.flush_evidence()
            if time.monotonic() - last_report >= 5.:
                last_report = time.monotonic()
                print(f"FAST_PROGRESS {spec['label']} done={sum(n['state']=='done' for n in nodes.values())}/{len(nodes)} "
                      f"ready={len(state['ready'])} observe_ms={observe_ms:.1f} plan_ms={d['plan_wait_ms']:.1f} "
                      f"selected={counters['nonreference_selected']}", flush=True)
                write_json(out / 'status.json', {'status': 'running', 'spec': spec, 'counts': dict(counters)})
    try:
        await asyncio.wait_for(loop(), cfg['condition_wall_s'])
        if errors:
            raise errors[0]
        finish = time.monotonic()
        await native.drain(cfg['replicas'])
        while await native.harvest(cfg['replicas'], wait_complete=True):
            pass
        confirmed, confirmed_nodes = 0, 0
        for d in decisions:
            if d['result'].get('selected_cell', '00') == '00':
                continue
            executed = []
            for entry in d['receipts']:
                receipt = await native.call(entry['replica'], 'receipt', action_id=entry['receipt']['action_id'])
                append_json(out / 'decisions.jsonl', {'event': 'terminal_receipt', 'index': d['index'],
                                                     'replica': entry['replica'], 'receipt': receipt})
                for binding in receipt.get('engine_admissions', []):
                    k = binding['node_id']
                    if k not in nodes or nodes[k]['state'] != 'done':
                        continue
                    changed = k not in d['reference_action']['admit'] or d['reference_action']['placement'].get(k) != entry['replica']
                    bound = nodes[k]['stream'].get('native_mapping', {}).get('mappings', [])
                    actual_ids = {m.get('engine_request_id') for m in bound if m.get('action_id') == receipt['action_id']}
                    if changed and binding.get('engine_request_id') in actual_ids:
                        executed.append(k)
            confirmed += bool(executed); confirmed_nodes += len(set(executed))
        counters['nonreference_execution_confirmed'] = confirmed
        counters['changed_admission_or_route_nodes_confirmed'] = confirmed_nodes
        summary = summarize(nodes, spec, decisions, finish, proof, counters)
        write_json(out / 'summary.json', summary)
        write_json(out / 'status.json', {'status': 'completed', 'functional_pass': summary['functional_pass']})
        if not summary['functional_pass']:
            raise ValueError('fixed work or DAG dependency audit failed')
        return summary
    except BaseException as exc:
        write_json(out / 'status.json', {'status': 'failed', 'error': repr(exc), 'spec': spec})
        raise
    finally:
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*list(tasks), return_exceptions=True)
        if pending is not None:
            # The fast worker is cooperatively bounded. Never apply its late result.
            try:
                await asyncio.wait_for(asyncio.shield(pending), 2.)
                await collect_late()
            except BaseException as exc:
                append_json(out / 'decisions.jsonl', {'event': 'pending_plan_not_used', 'reason': repr(exc)})
        native.defer_evidence = False
        native.flush_evidence(durable=True)
        write_json(out / 'nodes.json', nodes)
