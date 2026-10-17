#!/usr/bin/env python3
"""Read-only, standard-library diagnosis of existing Ours shadow timing logs.

No model execution, network, hashes, installation, or writes to the input run.
Joins use the ORIGINAL decision snapshot, never its later fallback observation.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

NO_CANDIDATE = 'no_distinct_executable_candidate'
UNUSABLE = 'native_observation_unusable'


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def seconds(value):
    return value/1e9 if number(value) else None


def rows(path, gaps=None):
    if not path.is_file():
        if gaps is not None:
            gaps['missing:' + path.name] += 1
        return
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('not object')
                yield row
            except (ValueError, TypeError):
                if gaps is not None:
                    gaps['invalid_row:' + path.name] += 1


def object_file(path, gaps):
    if not path.is_file():
        gaps['missing:' + path.name] += 1
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(value, dict):
            return value
    except (ValueError, TypeError):
        pass
    gaps['invalid:' + path.name] += 1
    return {}


def condition_dirs(source):
    source = source.resolve()
    if (source/'decisions.jsonl').is_file():
        return [source]
    candidates = [source/'conditions', source/'r234_ours_timing/conditions']
    found = [p for p in candidates if p.is_dir()]
    if len(found) != 1:
        raise ValueError('--input must be one ours-timing run, its r234_ours_timing child, or one condition directory')
    result = sorted(p for p in found[0].iterdir() if p.is_dir() and 'manonly' not in p.name.lower())
    if not result:
        raise ValueError('No condition directories found')
    return result


def safe_sample(value):
    """Only called on explicit whitelists, never raw requests or snapshots."""
    if isinstance(value, dict):
        return {str(k)[:80]: safe_sample(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_sample(v) for v in value[:6]]
    if isinstance(value, str):
        return value[:120]
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    return value


def snapshot_key(replica, row):
    return replica, row.get('snapshot_id'), row.get('identity', {}).get('engine_instance')


def collect(directory):
    gaps, counts, reasons, stages = Counter(), Counter(), Counter(), Counter()
    funnel, funnel_calls, opportunities, nc_flags = Counter(), Counter(), Counter(), Counter()
    errors, combinations, lifecycle, lifecycle_calls = Counter(), Counter(), Counter(), Counter()
    transit_capture, transit_client = Counter(), Counter()
    samples, sample_counts = [], Counter()
    wanted, states = {}, {}
    def result_summary():
        return dict(counts=counts, reasons=reasons, stages=stages, funnel=funnel, funnel_calls=funnel_calls,
                    opportunities=opportunities, nc_flags=nc_flags, errors=errors, combinations=combinations,
                    lifecycle=lifecycle, lifecycle_calls=lifecycle_calls, gaps=Counter({k: v for k, v in gaps.items() if v}),
                    transit_capture=transit_capture, transit_client=transit_client,
                    samples=samples)
    def sample(kind, index, **fields):
        if sample_counts[kind] < 2:
            samples.append(safe_sample(dict(condition=directory.name, i=index, kind=kind, **fields)))
            sample_counts[kind] += 1

    for row in rows(directory/'decisions.jsonl', gaps):
        if row.get('event') != 'unbounded_planner_timing':
            continue
        result, index = row.get('result', {}), row.get('index')
        reason = result.get('v5_stop_reason', 'reference_or_candidate_returned')
        counts['shadow_returns'] += 1
        reasons[reason] += 1
        t = result.get('shadow_timing', {})
        counts['with_candidates'] += bool(t.get('generated'))
        counts['all_scores_complete'] += bool(t.get('all_scores_complete'))
        if reason not in (NO_CANDIDATE, UNUSABLE):
            continue
        if index in wanted:
            gaps['duplicate_shadow_index'] += 1
        wanted[index] = reason
        if reason == UNUSABLE:
            counts['observation_unusable'] += 1
            reported = sorted(set(result.get('native_fallback') or []))
            if not reported:
                gaps['unusable_without_reported_errors'] += 1
            errors.update(reported)
            combinations['+'.join(reported) or 'missing_errors'] += 1
            continue
        counts['no_candidate'] += 1
        d = result.get('candidate_diagnostics', {})
        o = result.get('opportunity_diagnostics', {})
        if not d:
            stage = 'missing_candidate_diagnostics'
        elif not d.get('shortlist_before_native_dedup'):
            stage = 'empty_before_native_dedup'
        elif not d.get('distinct_executable_candidates') and d.get('no_distinct_native_effect') == d.get('shortlist_before_native_dedup'):
            stage = 'all_removed_as_native_duplicates'
        else:
            stage = 'other_or_inconsistent_funnel'
        stages[stage] += 1
        funnel.update({k: v for k, v in d.items() if number(v)})
        funnel_calls.update(k for k, v in d.items() if number(v) and v > 0)
        for key in ('ready', 'reference_admitted', 'reference_held', 'mutable', 'source_known_ready', 'source_known_admitted', 'source_known_held'):
            if number(o.get(key)):
                opportunities[key] += o[key]
        if not o:
            nc_flags['missing_opportunity_diagnostics'] += 1
        else:
            nc_flags['no_mutable_submitted_nodes'] += o.get('mutable') == 0
            slots = o.get('client_slots_after_reference')
            nc_flags['all_client_slots_full_after_reference'] += bool(slots) and all(v == 0 for v in slots)
            nc_flags['ready_but_no_known_source_cache'] += o.get('ready', 0) > 0 and o.get('source_known_ready') == 0
            nc_flags['known_sources_but_no_pressure_eligible_route'] += o.get('source_known_ready', 0) > 0 and d.get('pressure_cache_eligible_routes') == 0
        nc_flags['admission_change_rejection_present'] += d.get('admission_set_would_change', 0) > 0
        nc_flags['alternatives_skipped_after_rejection_present'] += d.get('alternatives_skipped_after_rejection', 0) > 0
        nc_flags['held_only_route_no_effect_present'] += d.get('held_only_route_no_effect', 0) > 0
        details = [{k: s[k] for k in ('kind', 'origin', 'held_only', 'node', 'source', 'rejected_destination', 'skipped_destination', 'alternative_feasibility', 'moved_nodes') if k in s}
                   for s in result.get('candidate_diagnostic_samples', [])]
        rejected = [{k: s[k] for k in ('lost_admissions', 'gained_admissions') if k in s}
                    for s in result.get('admission_rejection_samples', [])]
        signature = stage + ':' + (details[0].get('kind', '') if details else 'no_sample')
        sample('NC:' + signature, index, funnel=d, opportunity=o, details=details, admission_rejections=rejected)

    # Keep only compact state needed for lifecycle joins. Snapshots for NC rows
    # contribute structural counts, not a replay with changed algorithm inputs.
    for row in rows(directory/'decisions.jsonl'):
        index = row.get('index')
        if row.get('event') or index not in wanted or 'snapshot' not in row:
            continue
        state = row['snapshot']
        if wanted[index] == NO_CANDIDATE:
            ready = set(state.get('ready', []))
            siblings = any(len(ready.intersection(group)) > 1 for key in ready
                           for group in state.get('nodes', {}).get(key, {}).get('compiled_groups', []))
            nc_flags['no_ready_DAG_sibling_group'] += not siblings
            counts['nc_original_snapshots_joined'] += 1
            continue
        states[index] = {'now': state.get('now'), 'errors': state.get('native_observation_errors', []),
                         'refs': state.get('native_snapshot_refs', []),
                         'nodes': {k: {f: n.get(f) for f in ('state', 'replica', 'finish')} for k, n in state.get('nodes', {}).items()},
                         'projection_ms': state.get('native_projection_ms'),
                         'prepared_at': row.get('planning_window', {}).get('prepared_at')}
    counts['obs_original_snapshots_joined'] = len(states)
    gaps['missing_obs_original_snapshot'] += counts['observation_unusable']-len(states)
    gaps['missing_nc_original_snapshot'] += counts['no_candidate']-counts['nc_original_snapshots_joined']
    if not states:
        return result_summary()

    refs = {snapshot_key(ref.get('replica'), ref) for state in states.values() for ref in state['refs']}
    snapshots, lifetimes, mapping_calls = {}, {}, defaultdict(list)
    def merge_mapping(mapping):
        key = mapping.get('node_id')
        if key:
            life = lifetimes.setdefault(key, {})
            for field in ('engine_admitted_ns', 'engine_finished_ns'):
                if number(mapping.get(field)):
                    life[field.removesuffix('_ns')] = seconds(mapping[field])

    for row in rows(directory/'native_events.jsonl', gaps):
        response, replica = row.get('response', {}), row.get('replica')
        if row.get('op') == 'mapping':
            for m in response.get('mappings', []):
                merge_mapping(m)
                if m.get('node_id'):
                    mapping_calls[m['node_id']].append((row.get('client_start'), row.get('client_finish')))
        elif row.get('op') == 'snapshot' and snapshot_key(replica, response) in refs:
            queue = []
            for name in ('running', 'waiting', 'skipped_waiting'):
                for req in response.get(name, []):
                    m = req.get('mapping') or {}
                    merge_mapping(m)
                    queue.append((req.get('engine_request_id'), m.get('node_id')))
            snapshots[snapshot_key(replica, response)] = {
                'captured': seconds(response.get('captured_ns')), 'expires': seconds(response.get('expires_ns')),
                'rpc_start': row.get('client_start'), 'rpc_finish': row.get('client_finish'), 'queue': queue,
                'pending': sorted({r['engine_request_id'] for b in response.get('inflight_batches', [])
                                   for r in b.get('requests', []) if r.get('engine_request_id')})}
    final_nodes = object_file(directory/'nodes.json', gaps)
    for key, node in final_nodes.items():
        if not isinstance(node, dict):
            continue
        life = lifetimes.setdefault(key, {})
        stream = node.get('stream') or {}
        life.update(client_done=node.get('finish'), http_done=stream.get('finish'))
        for m in (stream.get('native_mapping') or {}).get('mappings', []):
            merge_mapping(m)
    del final_nodes
    for row in rows(directory/'requests.jsonl', gaps):
        if row.get('event') == 'response':
            result = row.get('result', {})
            life = lifetimes.setdefault(row.get('node'), {})
            if number(result.get('finish')):
                life['http_done'] = result['finish']
            for m in result.get('native_mapping', {}).get('mappings', []):
                merge_mapping(m)

    for index, state in states.items():
        call_kinds = set()
        def evidence(kind, replica, key, snap):
            lifecycle[kind] += 1
            call_kinds.add(kind)
            life = lifetimes.get(key, {})
            capture, observed = snap.get('captured'), state['now']
            times = {'client_observed': observed, 'window_prepared': state['prepared_at'],
                     'snapshot_rpc_start': snap.get('rpc_start'), 'snapshot_rpc_finish': snap.get('rpc_finish'), **life}
            intervals = mapping_calls.get(key, [])
            chosen = next(((a, b) for a, b in intervals if number(a) and number(b) and number(observed) and a <= observed <= b), None)
            if chosen is None and intervals:
                chosen = intervals[-1]
            if chosen:
                times.update(mapping_rpc_start=chosen[0], mapping_rpc_finish=chosen[1])
            offsets = {k: round((v-capture)*1000, 3) for k, v in times.items() if number(v) and number(capture)}
            sample('OBS:' + kind, index, replica=replica, node=key, ms_from_engine_capture=offsets,
                   reported=state['errors'])

        for ref in state['refs']:
            replica = ref.get('replica')
            snap = snapshots.get(snapshot_key(replica, ref))
            if snap is None:
                gaps['missing_referenced_engine_snapshot'] += 1
                continue
            capture, observed = snap['captured'], state['now']
            if not number(capture) or not number(observed):
                gaps['missing_snapshot_clock'] += 1
                continue
            raw = dict(snap['queue'])
            valid, filtered_done = {}, set()
            for rid, key in raw.items():
                if key is None:
                    gaps['engine_queue_record_without_node_binding'] += 1
                    continue
                node = state['nodes'].get(key)
                if node is None or node['state'] == 'done':
                    filtered_done.add(rid)
                    done = lifetimes.get(key, {}).get('client_done')
                    if number(done) and capture <= done <= observed:
                        kind = 'client_completed_between_capture_and_observe'
                    elif number(done) and done < capture:
                        kind = 'client_done_before_engine_capture'
                    else:
                        kind = 'completed_filter_without_matching_done_timestamp'
                    evidence(kind, replica, key, snap)
                elif node['state'] != 'submitted' or node['replica'] != replica:
                    evidence('engine_client_state_or_replica_mismatch', replica, key, snap)
                else:
                    valid[rid] = key
            for key, node in state['nodes'].items():
                if node['state'] != 'submitted' or node['replica'] != replica or key in valid.values():
                    continue
                life = lifetimes.get(key, {})
                http, done = life.get('http_done'), life.get('client_done')
                admitted, finished = life.get('engine_admitted'), life.get('engine_finished')
                pending_rpc = any(number(a) and number(b) and a <= observed < b for a, b in mapping_calls.get(key, []))
                # Two independent time boundaries: a very short request could
                # be pre-admission at capture AND HTTP-complete by client copy.
                at_capture = ('pre_engine_admission' if number(admitted) and admitted > capture else
                              'post_engine_finish' if number(finished) and finished <= capture else
                              'inside_recorded_lifetime' if number(admitted) and number(finished) and admitted <= capture < finished else
                              'insufficient_lifecycle_evidence')
                transit_capture[at_capture] += 1
                client_phase = ('http_done_mapping_RPC_pending' if pending_rpc else 'http_done_client_finalization_pending') if (
                    number(http) and http <= observed and (not number(done) or observed <= done)) else (
                    'HTTP_not_yet_done' if number(http) and http > observed else 'insufficient_or_inconsistent_finish_evidence')
                transit_client[client_phase] += 1
                if number(http) and http <= observed and (not number(done) or observed <= done):
                    kind = 'http_done_waiting_mapping_RPC' if pending_rpc else 'http_done_waiting_client_finalization'
                elif number(finished) and finished <= capture:
                    kind = 'engine_finished_before_capture_client_still_submitted'
                elif number(admitted) and admitted > capture:
                    kind = 'not_engine_admitted_at_capture'
                elif number(admitted) and number(finished) and admitted <= capture < finished:
                    kind = 'missing_queue_inside_recorded_engine_lifetime'
                else:
                    kind = 'transit_lifecycle_evidence_insufficient'
                evidence(kind, replica, key, snap)
            for rid in snap['pending']:
                if rid not in valid:
                    kind = 'inflight_unmapped_after_completed_filter' if rid in filtered_done else (
                        'inflight_unmapped_after_state_filter' if rid in raw else 'inflight_request_absent_from_snapshot_queues')
                    evidence(kind, replica, raw.get(rid), snap)
            if number(snap['expires']) and snap['expires'] <= observed:
                evidence('snapshot_expired_at_client_observe', replica, None, snap)
            elif number(snap['expires']) and number(state['projection_ms']) and snap['expires'] <= observed+state['projection_ms']/1000:
                evidence('snapshot_expired_during_projection', replica, None, snap)
        if not state['refs']:
            gaps['obs_snapshot_refs_missing'] += 1
        lifecycle_calls.update(call_kinds)
    return result_summary()


def report(source, max_kib=12):
    names = ('counts', 'reasons', 'stages', 'funnel', 'funnel_calls', 'opportunities', 'nc_flags',
             'errors', 'combinations', 'lifecycle', 'lifecycle_calls', 'transit_capture', 'transit_client', 'gaps')
    totals = {name: Counter() for name in names}
    conditions, samples = [], []
    for directory in condition_dirs(source):
        current = collect(directory)
        conditions.append({'condition': directory.name, **current['counts'], 'nc_stages': current['stages'], 'gaps': current['gaps']})
        for name in names:
            totals[name].update(current[name])
        samples.extend(current['samples'])
    dumps = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    lines = ['DABS offline diagnosis v1; read-only; no inference/hash checks. All counts are over raw shadow-return records.',
             'NC_STAGE is exclusive. FUNNEL events are totals; calls count positive occurrence. FLAGS/OBS categories overlap; do not sum them as calls.',
             'TRANSIT_CAPTURE and TRANSIT_CLIENT independently classify the same missing request-cases at different instants; OBS_LIFECYCLE prefers HTTP-completion status.',
             'OBS time offsets are ms from that replica engine snapshot capture; negative=before capture. Missing evidence is not a diagnosis.']
    mapping = [('TOTAL', 'counts'), ('STOP_CALLS', 'reasons'), ('NC_STAGE_CALLS', 'stages'),
               ('NC_FUNNEL_EVENTS', 'funnel'), ('NC_FUNNEL_CALLS', 'funnel_calls'), ('NC_OPPORTUNITY_TOTALS', 'opportunities'),
               ('NC_FLAGS_CALLS', 'nc_flags'), ('OBS_REASON_CALLS', 'errors'), ('OBS_REASON_COMBINATIONS', 'combinations'),
               ('OBS_LIFECYCLE_REQUEST_CASES', 'lifecycle'), ('OBS_LIFECYCLE_CALLS', 'lifecycle_calls'),
               ('TRANSIT_CAPTURE_REQUEST_CASES', 'transit_capture'), ('TRANSIT_CLIENT_REQUEST_CASES', 'transit_client'), ('DATA_GAPS', 'gaps')]
    lines += [label + ' ' + dumps({k: v for k, v in totals[name].items() if v}) for label, name in mapping]
    lines += ['COND ' + dumps(c) for c in conditions]
    selected, used, omitted = [], 0, 0
    budget = max_kib*1024-180
    for line in lines:
        size = len((line+'\n').encode('utf-8'))
        if used+size <= budget:
            selected.append(line); used += size
        else:
            omitted += 1
    # Round-robin kinds/conditions: no-candidate samples cannot crowd out all
    # lifecycle evidence. Numbers above were aggregated before this size bound.
    buckets = defaultdict(list)
    for s in samples:
        buckets[s['kind']].append(s)
    for kind, bucket in buckets.items():
        if len(bucket) > 1:
            second = next((s for s in bucket[1:] if s['condition'] != bucket[0]['condition']), bucket[1])
            buckets[kind] = [bucket[0], second]
    example_lines = []
    for i in range(2):
        for kind in sorted(buckets, key=lambda k: (not k.startswith('OBS:'), k)):
            if i < len(buckets[kind]):
                example_lines.append('SAMPLE ' + dumps(buckets[kind][i]))
    omitted_samples = 0
    for line in example_lines:
        size = len((line+'\n').encode('utf-8'))
        if used+size <= budget:
            selected.append(line); used += size
        else:
            omitted_samples += 1
    selected.append('OUTPUT_LIMIT omitted_summary_lines=%d sample_kind_cap=%d sample_byte_cap=%d; raise --max-kib if needed; raw logs unchanged.' %
                    (omitted, len(samples)-len(example_lines), omitted_samples))
    return '\n'.join(selected)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--max-kib', type=int, default=12, choices=range(6, 65), metavar='6..64')
    args = parser.parse_args(argv)
    try:
        text = report(args.input, args.max_kib)
    except (OSError, ValueError) as exc:
        parser.exit(2, 'Cannot extract: ' + str(exc) + '\n')
    sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
