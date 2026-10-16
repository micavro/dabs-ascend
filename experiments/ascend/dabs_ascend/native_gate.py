"""Recompute acceptance from raw engine-signed records; never read a PASS flag."""
import json
from pathlib import Path

from .common import read_json, sha, source_identity
from .native_calibration import batch_features, fit_model


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
    checks['source_identity'] = meta.get('source_sha256') == source_identity()
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


def evaluate_local(paths):
    checks, missing, hashes = {}, [], {}
    required = ('native-probe-one', 'native-probe-four', 'native-calibrate', 'native-invalidate', 'native-restart-check')
    for name in required:
        if name not in paths:
            missing.append('missing evidence: ' + name)
    if missing:
        return {'pass': False, 'missing': missing, 'checks': checks, 'artifacts': hashes}
    try:
        for name, replicas in (('native-probe-one', 1), ('native-probe-four', 4)):
            checks[name] = evaluate_probe(paths[name], replicas)
        invalid = read_json(Path(paths['native-invalidate']) / 'invalidation.json')
        invalid_raw = [e['response'] for e in events(paths['native-invalidate'])]
        if any(invalid[k] not in invalid_raw for k in ('before', 'reset', 'after', 'receipt')):
            raise ValueError('invalidation summary differs from raw signed records')
        checks['invalidation'] = {'cache_reset_invalidates_evidence': invalid['before']['cache'][0]['state'] == 'hit' and
            invalid['reset']['cache_reset'] and invalid['after']['cache'][0]['state'] == 'miss' and
            invalid['receipt']['status'] in ('expired', 'rejected')}
        restart = read_json(Path(paths['native-restart-check']) / 'restart.json')
        restart_raw = [e['response'] for e in events(paths['native-restart-check'])]
        if restart['new'] not in restart_raw or restart['receipt'] not in restart_raw:
            raise ValueError('restart summary differs from raw signed records')
        checks['restart'] = {'old_identity_rejected': restart['new']['identity']['engine_instance'] != restart['old_identity']['engine_instance'] and
            restart['new']['identity'].get('previous_engine_instance') == restart['old_identity']['engine_instance'] and
            restart['receipt']['status'] == 'rejected' and restart['receipt']['reason'] == 'engine_restarted'}
        calibration = Path(paths['native-calibrate'])
        model, samples = read_json(calibration / 'native_profile.json'), read_json(calibration / 'calibration_samples.json')
        caps = read_json(calibration / 'capabilities.json')
        raw = {(e['replica'], b['batch_id']): batch_features(b) for e in events(calibration) if e['op'] == 'batches'
               for b in e['response']['batches'] if b.get('device')}
        raw_batches = {(e['replica'], b['batch_id']): b for e in events(calibration) if e['op'] == 'batches'
                       for b in e['response']['batches'] if b.get('device')}
        declared_plan = read_json(calibration / 'calibration_plan.json')['plan']
        seen = set()
        for sample in samples:
            key = sample['replica'], sample['batch_id']
            if key in seen or key not in raw or any(sample[k] != raw[key][k] for k in raw[key]):
                raise ValueError('calibration samples changed, duplicated or not backed by signed native events')
            group = sample['group']
            if group not in declared_plan or sample['split'] != group['split'] or any(
                    not q['mapping']['node_id'].startswith('cal-' + sample['split'] + '-' + str(group['seed']) + '-')
                    for q in raw_batches[key]['requests']):
                raise ValueError('train/holdout group does not match engine-signed request identity')
            seen.add(key)
        recomputed = fit_model(samples, caps)
        checks['calibration'] = {'frozen_independent_holdout': recomputed['pass'] and model['models'] == recomputed['models'] and
            model['validation'] == recomputed['validation'] and model['source_sha256'] == source_identity() and
            model['timing_target'] == recomputed['timing_target'] and model['host_step_equivalence'] is True and
            model['raw_samples_sha256'] == sha(calibration / 'calibration_samples.json') and
            model['raw_native_sha256'] == sha(calibration / 'native_events.jsonl')}
        for name in required:
            directory = Path(paths[name])
            if any('signature' not in e['response'] for e in events(directory) if e['op'] != 'verify'):
                raise ValueError('unsigned engine evidence in ' + name)
            for path in directory.iterdir():
                if path.suffix in ('.json', '.jsonl'):
                    hashes[str(path.resolve())] = sha(path)
        for section, values in checks.items():
            missing.extend(section + ': ' + name for name, passed in values.items() if passed is not True)
    except (KeyError, ValueError, OSError, TypeError, IndexError) as exc:
        missing.append('invalid/incomplete raw evidence: ' + str(exc))
    return {'pass': not missing, 'checks': checks, 'missing': missing, 'artifacts': hashes,
            'source_sha256': source_identity(), 'paths': paths, 'live_verified': False}


async def verify_live(native, paths):
    result = evaluate_local(paths)
    if not result['pass']:
        return result
    pending = {r: [] for r in range(len(native.capabilities))}
    sizes = {r: 0 for r in pending}
    async def flush(replica):
        if not pending[replica]:
            return
        checked = await native.call(replica, 'verify', records=pending[replica])
        if checked['valid'] != [True] * len(pending[replica]) or checked['identity'] != native.identities[replica]:
            raise ValueError('engine signature/epoch mismatch at replica ' + str(replica))
        pending[replica], sizes[replica] = [], 0
    try:
        for directory in paths.values():
            for event in events(directory):
                row, replica = event['response'], event['replica']
                if 'signature' not in row:
                    if event['op'] == 'verify':
                        continue
                    raise ValueError('unsigned raw evidence')
                size = len(json.dumps(row).encode('utf-8'))
                if sizes[replica] + size > 2_000_000 or len(pending[replica]) >= 64:
                    await flush(replica)
                pending[replica].append(row)
                sizes[replica] += size
        for replica in pending:
            await flush(replica)
    except ValueError as exc:
        result['missing'].append(str(exc)); result['pass'] = False
        return result
    model = read_json(Path(paths['native-calibrate']) / 'native_profile.json')
    if model['identities'] != [c['identity'] for c in native.capabilities] or model['resources'] != [c['resources'] for c in native.capabilities]:
        result['missing'].append('calibrated service identity/configuration changed')
    result['pass'] = not result['missing']
    result['live_verified'] = result['pass']
    return result
