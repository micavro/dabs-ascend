"""Recompute acceptance from raw engine records; never read a PASS flag."""
import json
from pathlib import Path

from .common import read_json, sha, source_identity
from .native_calibration import batch_features, fit_model
from dabs_native.contracts import engine_epoch, resource_values
from .native_timing import TARGET, reconstruct, collection_floors


def events(directory):
    return [json.loads(line) for line in (Path(directory) / 'native_events.jsonl').read_text(encoding='utf-8').splitlines() if line]


def evaluate_probe(directory, replicas):
    meta = read_json(Path(directory) / 'probe.json')
    rows = events(directory)
    roles = meta['roles']
    def role(name):
        ref = roles.get(name, {})
        found = [e['response'] for e in rows if e['replica'] == ref.get('replica') and
                 ((ref.get('snapshot_id') and e['response'].get('snapshot_id') == ref['snapshot_id'] and e['op'] == 'snapshot') or
                  (ref.get('action_id') and e['response'].get('action_id') == ref['action_id']))]
        return found[-1] if found else {}
    def cache(name):
        return role(name).get('cache', [])
    checks = {}
    admissions = [e['response'] for e in rows if e['op'] == 'admit' and e['response'].get('status') == 'applied']
    checks['admission_only_preserves_queue'] = bool(admissions) and all(
        r.get('queue_control_changed') is False and r.get('queue_application', {}).get('phase') == 'admission_only' and
        'control_before' in r['queue_application'] and
        r['queue_application']['control_before'] == r['queue_application'].get('control_after') and
        set(r['queue_application'].get('queues', {})) == {'running', 'waiting'} and
        all(q.get('reason') == 'unchanged_admission_only' and q.get('before') == q.get('after')
            for q in r['queue_application']['queues'].values()) for r in admissions)
    applied = [e['response'] for e in rows if e['op'] in ('apply', 'admit') and
               e['response'].get('status') in ('applied', 'partially_applied')]
    checks['application_within_window'] = bool(applied) and all(
        type(r.get('mutation_at_ns')) is int and
        r.get('requested', {}).get('not_before_ns', 10**30) <= r['mutation_at_ns'] <
        r.get('requested', {}).get('action_expires_ns', -1) and
        r['mutation_at_ns'] <= r.get('requested', {}).get('expires_ns', -1) for r in applied)
    checks['cold'] = len(cache('cold')) == 1 and cache('cold')[0]['state'] == 'miss'
    checks['hot'] = len(cache('warm')) == 1 and cache('warm')[0]['state'] == 'hit' and cache('warm')[0]['reusable_tokens'] == 1024
    checks['namespace'] = len(cache('namespace')) == 1 and cache('namespace')[0]['state'] == 'miss' and cache('namespace')[0]['salt_digest'] != cache('warm')[0]['salt_digest']
    checks['block_boundary'] = {e['query_id']: e['reusable_tokens'] for e in cache('boundary')} == {'127': 0, '128': 0, '129': 128, '256': 128, '257': 256}
    checks['expired_action'] = role('expired').get('status') == 'expired'
    duplicates = [e['response'] for e in rows if e['op'] == 'apply' and e['response'].get('action_id') == roles.get('duplicate', {}).get('action_id')]
    checks['idempotent_action'] = len(duplicates) == 2 and duplicates[0] == duplicates[1] and duplicates[0]['status'] == 'applied'
    waiting, queue = role('waiting'), role('queue')
    checks['real_waiting'] = waiting.get('waiting_count', 0) >= 2 and len(waiting.get('waiting', [])) >= 2
    checks['engine_queue_applied'] = (queue.get('status') == 'applied' and bool(queue.get('applied')) and
        queue.get('effective_waiting_order') != [r['engine_request_id'] for r in waiting.get('waiting', [])] and
        queue.get('scheduled_step', -1) > queue.get('applied_step', 10**20) and bool(queue.get('scheduled_order')))
    checks['late_action'] = role('late').get('status') in ('expired', 'rejected') and not role('late').get('applied')
    iterations = [b for e in rows if e['op'] == 'batches' for b in e['response']['batches']]
    matching = [b for b in iterations if b.get('queue_application', {}).get('action_id') == queue.get('action_id')]
    checks['finite_window'] = bool(matching) and all(
        (b['queue_application']['phase'] == 'current_action') ==
        (queue['not_before_ns'] <= b['queue_application']['at_ns'] < queue['action_expires_ns']) for b in matching)
    continuations = [b['queue_application'] for b in matching if b['queue_application']['phase'] == 'common_continuation']
    checks['expiry_fifo'] = bool(continuations) and all(q['reason'] == 'fifo_acceptance' for b in continuations for q in b['queues'].values())
    active_scheduled = [rid for b in matching if b['queue_application']['phase'] == 'current_action'
                        for rid in b['scheduled_tokens'] if rid in queue.get('applied', [])]
    checks['priority_reached_execution_within_window'] = bool(active_scheduled)
    seen_order = list(dict.fromkeys(active_scheduled))
    checks['executed_priority_order'] = bool(seen_order) and seen_order == [r for r in queue.get('requested', {}).get('order', []) if r in seen_order]
    dag = meta['dag']
    mapping_rows = [m for e in rows if e['op'] == 'mapping' for m in e['response'].get('mappings', [])]
    checks['fork_join'] = (all(b['start'] >= dag['root']['finish'] for b in dag['branches']) and
        dag['join']['start'] >= max(b['finish'] for b in dag['branches']) and
        all(x['mapping'].get('engine_request_id') and x['mapping'].get('action_id') and x['mapping'] in mapping_rows
            for x in [dag['root'], *dag['branches'], dag['join']]))
    checks['engine_fork_join'] = (all(b['mapping'].get('engine_admitted_ns', -1) >= dag['root']['mapping'].get('engine_finished_ns', 10**30)
        for b in dag['branches']) and dag['join']['mapping'].get('engine_admitted_ns', -1) >=
        max(b['mapping'].get('engine_finished_ns', 10**30) for b in dag['branches']))
    drain = role('drain')
    checks['drain'] = drain.get('running_count') == 0 and drain.get('waiting_count') == 0 and not drain.get('inflight_batches')
    batches = [b for e in rows if e['op'] == 'batches' for b in e['response']['batches'] if b.get('device')]
    checks['async_batch_correlation'] = bool(batches) and all(b['batch_id'] == b['device']['batch_id'] and b['device']['device_event_complete'] for b in batches)
    if replicas == 4:
        checks['cross_replica'] = all(cache('cross-' + str(r)) and cache('cross-' + str(r))[0]['state'] == 'miss' and
            cache('cross-hot-' + str(r))[0]['reusable_tokens'] == 1024 for r in (1, 2, 3))
    checks['replica_count'] = meta['replicas'] == replicas
    return checks


def evaluate_local(paths, *, historical_predictor=False):
    checks, missing, hashes = {}, [], {}
    required = ('native-probe-one', 'native-probe-four', 'native-calibrate')
    optional = ('native-invalidate', 'native-restart-check')
    not_run = [name for name in optional if name not in paths]
    for name in required:
        if name not in paths:
            missing.append('missing evidence: ' + name)
    if missing:
        return {'pass': False, 'missing': missing, 'checks': checks, 'artifacts': hashes}
    try:
        for name, replicas in (('native-probe-one', 1), ('native-probe-four', 4)):
            checks[name] = evaluate_probe(paths[name], replicas)
        if 'native-invalidate' in paths:
            invalid = read_json(Path(paths['native-invalidate']) / 'invalidation.json')
            invalid_raw = [e['response'] for e in events(paths['native-invalidate'])]
            if any(invalid[k] not in invalid_raw for k in ('before', 'reset', 'after', 'receipt')):
                raise ValueError('invalidation summary differs from raw engine records')
            checks['invalidation'] = {'cache_reset_invalidates_evidence': invalid['before']['cache'][0]['state'] == 'hit' and
                invalid['reset']['cache_reset'] and invalid['after']['cache'][0]['state'] == 'miss' and
                invalid['receipt']['status'] in ('expired', 'rejected')}
        if 'native-restart-check' in paths:
            restart = read_json(Path(paths['native-restart-check']) / 'restart.json')
            restart_raw = [e['response'] for e in events(paths['native-restart-check'])]
            if restart['new'] not in restart_raw or restart['receipt'] not in restart_raw:
                raise ValueError('restart summary differs from raw engine records')
            checks['restart'] = {'old_identity_rejected': restart['new']['identity']['engine_instance'] != restart['old_identity']['engine_instance'] and
                restart['new']['identity'].get('previous_engine_instance') == restart['old_identity']['engine_instance'] and
                restart['receipt']['status'] == 'rejected' and restart['receipt']['reason'] == 'engine_restarted'}
        calibration = Path(paths['native-calibrate'])
        model, samples = read_json(calibration / 'native_profile.json'), read_json(calibration / 'calibration_samples.json')
        predictor = model.get('predictor', 'phase6_v1') if historical_predictor else None
        caps = read_json(calibration / 'capabilities.json')
        accounting = model.get('timing_target') == TARGET
        raw = {} if accounting else {(e['replica'], b['batch_id']): batch_features(b) for e in events(calibration) if e['op'] == 'batches'
               for b in e['response']['batches'] if b.get('device')}
        raw_batches = {(e['replica'], b['batch_id']): b for e in events(calibration) if e['op'] == 'batches'
                       for b in e['response']['batches'] if b.get('device')}
        plan_record = read_json(calibration / 'calibration_plan.json')
        declared_plan = plan_record['plan']
        if accounting:
            from .native_calibration import calibration_plan
            expected_plan = calibration_plan(predictor=predictor) if historical_predictor else calibration_plan()
            if plan_record.get('diagnostic_only') is not False or declared_plan != expected_plan:
                raise ValueError('diagnostic or incomplete/nondeclared calibration plan cannot open gate')
            raw_events = events(calibration)
            floors = collection_floors(raw_events, plan_record)
            if set(floors) != set(range(len(caps))):
                raise ValueError('missing replica collection-start snapshot')
            rebuilt, excluded = reconstruct(raw_events, declared_plan, after_steps=floors)
            from .native_prediction import PREDICTOR, check_startup_inputs
            if model.get('predictor') == PREDICTOR:
                check_startup_inputs(rebuilt, raw_events)
            raw = {(s['replica'], s['batch_id']): s for s in rebuilt}
            if excluded or len(samples) != len(raw):
                raise ValueError('excluded or omitted raw calibration batches')
            expected = {(r, g['seed']) for r in range(len(caps)) for g in declared_plan}
            if {(s['replica'], s['group']['seed']) for s in rebuilt} != expected:
                raise ValueError('incomplete calibration groups')
        seen = set()
        for sample in samples:
            key = sample['replica'], sample['batch_id']
            if key in seen or key not in raw or any(sample[k] != raw[key][k] for k in raw[key]):
                raise ValueError('calibration samples changed, duplicated or not backed by native events')
            group = sample['group']
            if group not in declared_plan or sample['split'] != group['split'] or any(
                    not q['mapping']['node_id'].startswith('cal-' + sample['split'] + '-' + str(group['seed']) + '-')
                    for q in raw_batches[key]['requests']):
                raise ValueError('train/holdout group does not match engine request identity')
            seen.add(key)
        recomputed = fit_model(samples, caps, predictor=predictor)
        checks['calibration'] = {'frozen_independent_holdout': recomputed['pass'] and model['models'] == recomputed['models'] and
            model['validation'] == recomputed['validation'] and
            model['timing_target'] == recomputed['timing_target'] and
            model['host_step_equivalence'] == recomputed['host_step_equivalence'] and
            model.get('frozen') is True and not model.get('diagnostic_only', False) and
            model['identities'] == recomputed['identities'] and model['resources'] == recomputed['resources'] and
            all(model.get(k) == recomputed.get(k) for k in ('original_gpu_timing_equivalent',
                'accounting_model', 'prediction_semantics', 'pending_batch_model', 'thresholds_predeclared', 'predictor'))}
        for name in (*required, *(name for name in optional if name in paths)):
            directory = Path(paths[name])
            for path in directory.iterdir():
                if path.suffix in ('.json', '.jsonl'):
                    hashes[str(path.resolve())] = sha(path)
        for section, values in checks.items():
            missing.extend(section + ': ' + name for name, passed in values.items() if passed is not True)
    except (KeyError, ValueError, OSError, TypeError, IndexError) as exc:
        missing.append('invalid/incomplete raw evidence: ' + str(exc))
    return {'pass': not missing, 'checks': checks, 'missing': missing, 'artifacts': hashes,
            'source_sha256': source_identity(), 'paths': paths, 'live_verified': False,
            'not_run': not_run, 'scope': 'ordinary native mechanisms; destructive probes reported separately'}


async def verify_live(native, paths):
    result = evaluate_local(paths)
    if not result['pass']:
        return result
    try:
        model = read_json(Path(paths['native-calibrate']) / 'native_profile.json')
        await check_live(native, paths, model)
    except (ValueError, KeyError, IndexError, OSError) as exc:
        result['missing'].append(str(exc))
    result['pass'] = not result['missing']
    result['live_verified'] = result['pass']
    return result


async def check_live(native, paths, model, *, allow_single=False):
    # Exercise live read interfaces. Source metadata never determines whether
    # a real engine is usable; epoch and effective resources do.
    await native.inspect()
    receipts = {}
    for directory in paths.values():
        for event in events(directory):
            row, replica = event['response'], event['replica']
            if engine_epoch(row.get('identity', {})) != engine_epoch(native.identities[replica]):
                raise ValueError('engine restarted since evidence capture at replica ' + str(replica))
            if event['op'] in ('apply', 'admit', 'receipt') and row.get('action_id'):
                receipts[replica, row['action_id']] = row
    # Re-read actual EngineCore ledger entries instead of trusting PASS or a
    # locally supplied receipt summary. Transitions may have gained later
    # scheduling details, so compare the stable application fields.
    for (replica, aid), recorded in receipts.items():
        live = await native.call(replica, 'receipt', action_id=aid)
        for key in ('action_id', 'status', 'reason', 'requested', 'snapshot_id', 'applied', 'applied_step', 'admitted_permits'):
            if live.get(key) != recorded.get(key):
                raise ValueError('live action receipt differs: ' + aid + '.' + key)
    caps = native.capabilities[:1] if allow_single and len(model['identities']) == 1 else native.capabilities
    if ([engine_epoch(i) for i in model['identities']] != [engine_epoch(c['identity']) for c in caps]
            or [resource_values(r) for r in model['resources']] != [resource_values(c['resources']) for c in caps]):
        raise ValueError('calibrated service identity/configuration changed')
