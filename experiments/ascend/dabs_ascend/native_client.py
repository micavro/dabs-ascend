"""Controller side of the EngineCore protocol (no shadow engine)."""
import asyncio
import copy
import math
import os
from pathlib import Path
import time
import uuid

from dabs_native.contracts import digest, prefix_key, validate_capabilities, engine_epoch, validate_observation
from .common import append_json


class NativeClient:
    def __init__(self, client, endpoints, log=None):
        self.client, self.endpoints = client, endpoints
        self.log, self.identities, self.capabilities = log, {}, []
        self.salt = 'dabs-' + uuid.uuid4().hex
        self.watermarks = {}

    async def call(self, replica, op, **payload):
        started = time.monotonic()
        response = await self.client.post(self.endpoints[replica] + '/dabs/v1/control',
            json={'op': op, **payload})
        if response.is_error:
            raise ValueError(f'replica {replica} /dabs/v1/control op={op}: HTTP {response.status_code}: {response.text[:1000]}')
        row = response.json()
        if self.log:
            append_json(self.log, {'replica': replica, 'op': op, 'request': payload,
                'response': row, 'client_start': started, 'client_finish': time.monotonic()})
            if op == 'batches':
                with open(self.log, 'a', encoding='utf-8') as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
        if not isinstance(row, dict) or not row.get('identity', {}).get('engine_instance'):
            raise ValueError(f'replica {replica} op={op}: missing identity.engine_instance')
        old = self.identities.setdefault(replica, row['identity'])
        if engine_epoch(row['identity']) != engine_epoch(old):
            raise ValueError('engine restart: invalidate state, calibration and acceptance; start a new run')
        return row

    async def inspect(self):
        self.capabilities = await asyncio.gather(*(self.call(r, 'capabilities') for r in range(len(self.endpoints))))
        for replica, row in enumerate(self.capabilities):
            try:
                validate_capabilities(row)
                validate_observation(await self.call(replica, 'snapshot'))
                for op, field, payload in [('mapping', 'mappings', {'client_request_ids': []}), ('batches', 'batches', {'after_step': 0})]:
                    observed = await self.call(replica, op, **payload)
                    if not isinstance(observed.get(field), list):
                        raise ValueError('missing native capability: ' + op + '.' + field)
            except ValueError as exc:
                raise ValueError(f'replica {replica}: {exc}') from exc
        resources = self.capabilities[0]['resources']
        for row in self.capabilities:
            if (row['identity']['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                    or row['identity']['clock_namespace_inode'] != os.stat('/proc/self/ns/time').st_ino):
                raise ValueError('fixed action windows require the same host monotonic clock namespace')
            for key in ('step_tokens', 'block_size', 'server_max_seqs', 'context_tokens'):
                if row['resources'][key] != resources[key]:
                    raise ValueError('heterogeneous effective engine setting: ' + key)
        return self.capabilities

    def action(self, snapshot, admissions=(), order=(), deadline_ns=None):
        return {'action_id': uuid.uuid4().hex, 'identity': snapshot['identity'],
                'snapshot_id': snapshot['snapshot_id'], 'expires_ns': snapshot['expires_ns'],
                'not_before_ns': time.monotonic_ns(),
                'action_expires_ns': deadline_ns if deadline_ns is not None else snapshot['captured_ns'] + 200_000_000,
                'resource_digest': digest(snapshot['resources']), 'admissions': list(admissions), 'order': list(order)}

    def admission(self, node):
        node.setdefault('client_request_id', uuid.uuid4().hex)
        node['cache_salt'] = self.salt
        return {'node_id': node['id'], 'client_request_id': node['client_request_id'],
                'prompt_tokens': len(node['tokens']), 'max_tokens': node['max_output_tokens'],
                'input_digest': prefix_key(node['tokens'], self.salt)}

    async def reserve(self, replica, node):
        """Atomic fresh-state admission used only by probes / explicit fallback."""
        result = await self.call(replica, 'admit', admissions=[self.admission(node)])
        if result['status'] != 'applied':
            raise ValueError('native admission refused: ' + str(result))
        node['native_action_id'] = result['action_id']
        return result

    async def observe(self, nodes, cfg, index, model):
        from .scheduler import snapshot
        from joint18v2.model import predict
        count = cfg['replicas']
        queries = [{'query_id': k, 'tokens': n['tokens'], 'salt': self.salt}
                   for k, n in nodes.items() if n['state'] == 'ready']
        started = time.monotonic()
        raw = await asyncio.gather(*(self.call(r, 'snapshot', queries=queries) for r in range(count)))
        now = time.monotonic()
        state = snapshot(nodes, {}, count, cfg, index, now)
        for key, n in state['nodes'].items():
            n['prefix_hash'] = prefix_key(nodes[key]['tokens'][:n['share']], self.salt)
        state.update(native_snapshots=raw, native_mutable=[], native_queue_groups={}, native_running={}, cache_evidence=[],
                     inflight_batches=[], prefix_history={}, prefix_certificates={}, native_observation_errors=[],
                     capacity_blocks=[], physical_reserved_blocks=[], physical_active_blocks=[],
                     pressure=[], native_pressure=[], state_ages_s=[], native_state_ages_s=[],
                     state_source='EngineCore utility RPC', capacity_source='EngineCore block pool; null block excluded')
        state['resource_units'].update(client_max_inflight=cfg['client_max_inflight'],
            server_max_num_seqs=cfg['max_num_seqs'], physical_kv_capacity_blocks=[x['resources']['physical_blocks'] for x in raw])
        for replica, row in enumerate(raw):
            resource = row['resources']
            if resource != self.capabilities[replica]['resources']:
                raise ValueError('runtime resources changed')
            ttl = (row['expires_ns'] - row['captured_ns']) / 1e9 - (now - started)
            if ttl <= 0:
                state['native_observation_errors'].append('snapshot_expired_in_transit')
            physical = row['physical']
            state['capacity_blocks'].append(physical['total_usable_blocks'])
            state['physical_reserved_blocks'].append(physical['total_usable_blocks'] - physical['available_for_new_reservations'])
            state['physical_active_blocks'].append(physical['active_blocks'])
            state['pressure'].append(row['running_count'] + 2 * row['waiting_count'])
            state['state_ages_s'].append(now - started)
            mapping, groups = {}, {'running_prefill': [], 'waiting': []}
            for group, records in [('running_prefill', row['running']), ('waiting', row['waiting'] + row['skipped_waiting'])]:
                for request in records:
                    binding = request['mapping']
                    if not binding or binding['node_id'] not in nodes:
                        raise ValueError('foreign/unmapped engine work; native experiment requires drained exclusive services')
                    key = binding['node_id']
                    if key not in state['nodes'] or state['nodes'][key]['state'] == 'done':
                        state['native_observation_errors'].append('request_completed_during_observation')
                        continue
                    n = state['nodes'][key]
                    if n['state'] != 'submitted' or n['replica'] != replica:
                        state['native_observation_errors'].append('engine_client_request_state_race')
                        continue
                    mapping[request['engine_request_id']] = key
                    n.update(computed=request['completed_computed_tokens'], output_done=request['output_tokens'],
                        accepted_at=request['accepted_ns'] / 1e9,
                        phase=request['phase'], progress_source='native_completed_progress', native_request=request)
                    if request['phase'] != 'decode':
                        groups[group].append(key)
                    if request['queue_mutable']:
                        state['native_mutable'].append(key)
            state['native_queue_groups'][str(replica)] = groups
            state['native_running'][str(replica)] = [mapping[r['engine_request_id']] for r in row['running'] if r['engine_request_id'] in mapping]
            state['queue_order'][str(replica)] = [mapping[r['engine_request_id']] for r in row['running'] + row['waiting'] + row['skipped_waiting'] if r['engine_request_id'] in mapping]
            # A just-sent or just-completed HTTP request may not be in this engine
            # snapshot. Do not score fabricated phase/progress during that gap.
            if any(n['state'] == 'submitted' and n['replica'] == replica and k not in mapping.values()
                   for k, n in state['nodes'].items()):
                state['native_observation_errors'].append('request_in_API_EngineCore_transit')
            for evidence in row['cache']:
                key = evidence['query_id']
                # Distinguish every concrete input, including its namespace.
                known = ttl > 0 and evidence['state'] != 'unknown'
                shared = state['nodes'][key]['share'] // resource['block_size'] * resource['block_size']
                state['cache_evidence'].append({'prefix_hash': state['nodes'][key]['prefix_hash'], 'replica': replica,
                    'known': known, 'tokens': min(shared, evidence['reusable_tokens']) if known else None,
                    'state': ('active_computed' if evidence['state'] == 'hit' else 'absent') if known else 'unknown',
                    'valid_until_ms': max(0, ttl * 1000), 'native': evidence})
            end = 0.
            for batch in sorted(row['inflight_batches'], key=lambda b: b['step']):
                pre, dec, p, dc, pc = [], [], 0, 0, 0
                for request in batch['requests']:
                    key = mapping.get(request['engine_request_id'])
                    if key is None:
                        state['native_observation_errors'].append('unmapped_inflight_batch')
                        continue
                    if request['prefill_tokens']:
                        pre.append((key, request['prefill_tokens']))
                        p += request['prefill_tokens']; pc = max(pc, request['context_tokens'])
                    if request['decode_tokens']:
                        dec.append(key); dc += request['context_tokens']
                # Predict an entire completion charge for each pending batch.
                # The current interval may be partly elapsed: this is a rollout
                # approximation, NOT a measured duration or guaranteed bound.
                from .native_scoring import check_model_range
                try:
                    if p or dec:
                        check_model_range(model, replica, p, len(dec), dc, pc)
                    duration = predict(model, replica, p, len(dec), dc, pc) if p or dec else .001
                    end = end + duration if end is not None else None
                except ValueError as exc:
                    if not str(exc).startswith('native_calibration_out_of_range:'):
                        raise
                    end = None
                    state['native_observation_errors'].append(str(exc))
                state['inflight_batches'].append({'replica': replica, 'batch_id': batch['batch_id'],
                    'prefill': pre, 'decode': dec, 'remaining_ms': end, 'cap': resource['step_tokens'],
                    'timing_target': model.get('timing_target'),
                    'timing_source': 'frozen completion-charge prediction; serial rollout approximation' if end is not None else 'unknown; outside frozen model support'})
        state['native_pressure'] = list(state['pressure'])
        state['native_state_ages_s'] = list(state['state_ages_s'])
        state['pressure_source'] = 'engine running + 2*waiting'
        state['cache_unknown_count'] = sum(not e['known'] for e in state['cache_evidence'])
        return state

    async def commit(self, action, state, nodes, *, fallback=False):
        async def one(replica, snapshot):
            selected = [k for k in action['admit'] if action['placement'][k] == replica and nodes[k]['state'] == 'ready']
            rank = {k: i for i, k in enumerate(action['order'])}
            selected.sort(key=lambda k: (rank.get(k, len(rank)), nodes[k]['ready_at'], k))
            admissions = [self.admission(nodes[k]) for k in selected]
            by_node = {r['mapping']['node_id']: r['engine_request_id'] for r in snapshot['running'] + snapshot['waiting']
                       if r['mapping'] and r['queue_mutable']}
            order = [by_node[k] for k in action['order'] if k in by_node]
            if not admissions and not order:
                return [], []
            if fallback:
                result = await self.call(replica, 'admit', admissions=admissions)
                result['controller_fallback'] = 'atomic fresh admission; no submitted queue changes'
            else:
                result = await self.call(replica, 'apply', action=self.action(snapshot, admissions, order, state.get('action_deadline_ns')))
            admitted = []
            accepted = set(result.get('admitted_permits', []))
            for key in selected:
                if nodes[key]['client_request_id'] in accepted:
                    nodes[key]['native_action_id'] = result['action_id']
                    admitted.append(key)
            return admitted, [{'replica': replica, 'receipt': result}]
        rows = await asyncio.gather(*(one(r, s) for r, s in enumerate(state['native_snapshots'])))
        return [k for a, _ in rows for k in a], [r for _, receipts in rows for r in receipts]

    async def drain(self, replicas):
        rows = await asyncio.gather(*(self.call(r, 'snapshot') for r in range(replicas)))
        if any(x['running_count'] or x['waiting_count'] or x['inflight_batches'] for x in rows):
            raise ValueError('engine did not drain')
        return rows

    async def harvest(self, replicas, *, wait_complete=False):
        exported = []
        for replica in range(replicas):
            row = await self.call(replica, 'batches', after_step=self.watermarks.get(replica, 0))
            deadline = time.monotonic() + 10
            while wait_complete and any(b['requests'] and not b['device'] for b in row['batches']):
                if time.monotonic() >= deadline:
                    raise ValueError('drained engine has missing NPU event completion; preserve raw trace')
                await asyncio.sleep(.01)
                row = await self.call(replica, 'batches', after_step=self.watermarks.get(replica, 0))
            complete = []
            for batch in row['batches']:
                if batch['requests'] and not batch['device']:
                    break
                complete.append(batch)
            exported.extend((replica, b) for b in complete)
            if complete and self.log:
                # call() has already appended the raw response; acknowledge
                # only the contiguous completed prefix, never an unready event.
                self.watermarks[replica] = complete[-1]['step']
                await self.call(replica, 'ack-trace', through_step=complete[-1]['step'])
        return exported
