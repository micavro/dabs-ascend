"""Active mechanism probes. Cache reset and restart checks are separate commands."""
import asyncio
import time
import uuid

from .common import write_json, source_identity


async def probe(backend, tokenizer, out, replicas):
    native = backend.native
    await native.drain(len(backend.endpoints))
    pad = (await tokenizer.encode(' A'))[-1]
    tokens = (await tokenizer.encode('DABS native concrete prefix ')) + [pad] * 1100
    tokens = tokens[:1025]
    native.salt = 'dabs-probe-' + uuid.uuid4().hex
    manifest = {'kind': 'probe', 'replicas': replicas, 'roles': {}, 'identities': native.identities,
                'source_sha256': source_identity()}
    roles = manifest['roles']
    async def snap(name, replica, queries=None):
        row = await native.call(replica, 'snapshot', queries=queries or [])
        roles[name] = {'replica': replica, 'snapshot_id': row['snapshot_id']}
        return row
    query = lambda n, salt=None: {'query_id': str(n), 'tokens': tokens[:n], 'salt': salt or native.salt}
    await snap('cold', 0, [query(1025)])
    async def request(replica, name, prompt=tokens, output=32, prepared=None):
        node = prepared or {'id': name, 'tokens': prompt, 'max_output_tokens': output, 'fixed_workload_replay': True}
        if prepared is None:
            await native.reserve(replica, node)
        result = await backend.generate(replica, node, 341100, out / 'requests.jsonl', schema=None)
        return {'node_id': name, 'start': result['start'], 'finish': result['finish'],
                'mapping': result['native_mapping']['mappings'][0]}
    await request(0, 'warm-' + uuid.uuid4().hex)
    warm = await snap('warm', 0, [query(1025)])
    await snap('namespace', 0, [query(1025, native.salt + '-different')])
    await snap('boundary', 0, [query(n) for n in (127, 128, 129, 256, 257)])
    if replicas == 4:
        for r in range(1, 4):
            await snap('cross-' + str(r), r, [query(1025)])
            await request(r, 'cross-warm-' + str(r))
            await snap('cross-hot-' + str(r), r, [query(1025)])
    # Expiry and duplicate actions exercise actual engine receipts, not status 200.
    action = native.action(warm)
    await asyncio.sleep(.15)
    expired = await native.call(0, 'apply', action=action)
    roles['expired'] = {'replica': 0, 'action_id': expired['action_id']}
    row = await snap('duplicate-base', 0)
    duplicate = native.action(row)
    first = await native.call(0, 'apply', action=duplicate)
    second = await native.call(0, 'apply', action=duplicate)
    roles['duplicate'] = {'replica': 0, 'action_id': first['action_id']}
    # 12 requests explicitly exceed the engine's 10 sequences. This is a probe
    # load, separate from the formal per-replica client limit of two.
    long = (tokens * 4)[:3584]
    waiting_nodes = [{'id': 'waiting-' + uuid.uuid4().hex, 'tokens': long,
                     'max_output_tokens': 16 if j < 10 else 512, 'fixed_workload_replay': True} for j in range(12)]
    # Complete every admission action before testing priority; a later admission
    # action must not supersede the window whose expiration we are measuring.
    await asyncio.gather(*(native.reserve(0, n) for n in waiting_nodes))
    tasks = [asyncio.create_task(request(0, n['id'], prepared=n)) for n in waiting_nodes]
    queue_action, queue_receipt, before = None, None, None
    deadline = time.monotonic() + 25
    try:
        while time.monotonic() < deadline and not all(t.done() for t in tasks):
            if queue_action:
                applied = await native.call(0, 'receipt', action_id=queue_action['action_id'])
                if applied.get('priority_scheduled_order'):
                    break
            before = await native.call(0, 'snapshot')
            waiting = [r for r in before['waiting'] if r['queue_mutable']]
            if len(waiting) >= 2 and (queue_action is None or time.monotonic_ns() >= queue_action['action_expires_ns']):
                order = [r['engine_request_id'] for r in reversed(waiting)]
                queue_action = native.action(before, order=order)
                queue_receipt = await native.call(0, 'apply', action=queue_action)
                if queue_receipt['status'] == 'applied':
                    roles['waiting'] = {'replica': 0, 'snapshot_id': before['snapshot_id']}
                    roles['queue'] = {'replica': 0, 'action_id': queue_receipt['action_id']}
                    # New fixed-window actions may be needed while the engine
                    # is full. Do not extend an old action's expiry in place.
            await asyncio.sleep(.005)
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if queue_action:
        late = {**queue_action, 'action_id': uuid.uuid4().hex}
        late_result = await native.call(0, 'apply', action=late)
        roles['late'] = {'replica': 0, 'action_id': late_result['action_id']}
        await native.call(0, 'receipt', action_id=queue_action['action_id'])
    # Real fork/join requests; dependency times are recorded and checked.
    root = await request(0, 'dag-root')
    branches = await asyncio.gather(request(0, 'dag-left'), request(replicas - 1, 'dag-right'))
    join = await request(replicas - 1, 'dag-join')
    manifest['dag'] = {'root': root, 'branches': branches, 'join': join}
    await snap('drain', 0)
    await native.drain(len(backend.endpoints))
    while await native.harvest(len(backend.endpoints), wait_complete=True):
        pass
    write_json(out / 'probe.json', manifest)
    # Evaluation is centralized in the gate; this file contains no editable PASS.
    return manifest


async def invalidation_probe(native, tokenizer, backend, out):
    await native.drain(len(backend.endpoints))
    tokens = (await tokenizer.encode(' Explicit cache invalidation probe. ')) * 100
    tokens = tokens[:1025]
    node = {'id': uuid.uuid4().hex, 'tokens': tokens, 'max_output_tokens': 16, 'fixed_workload_replay': True}
    await native.reserve(0, node)
    await backend.generate(0, node, 1, out / 'requests.jsonl', schema=None)
    query = {'query_id': 'invalidation', 'tokens': tokens, 'salt': native.salt}
    before = await native.call(0, 'snapshot', queries=[query])
    reset = await native.call(0, 'invalidate-cache', explicit_destructive_probe=True)
    after = await native.call(0, 'snapshot', queries=[query])
    receipt = await native.call(0, 'apply', action=native.action(before))
    result = {'kind': 'invalidation', 'before': before, 'reset': reset, 'after': after, 'receipt': receipt}
    result['pass'] = (before['cache'][0]['state'] == 'hit' and reset['cache_reset']
        and after['cache'][0]['state'] == 'miss' and receipt['status'] in ('expired', 'rejected'))
    write_json(out / 'invalidation.json', result)
    return result
