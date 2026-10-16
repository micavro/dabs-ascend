"""Real EngineCore control surface; invoked on its existing utility-RPC loop.

All scheduler access occurs on the EngineCore thread. No HTTP-side shadow KV.
Only the reviewed TP1/full-attention/async/eager configuration is supported.
"""
import copy
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import time
from types import MethodType
import uuid

from . import REVISION
from .contracts import Receipts, check_action, digest, fresh, prefix_key, validate_resources, prefill_order


class EngineAdapter:
    def __init__(self, core, installation):
        self.core, self.s = core, core.scheduler
        self.installation = installation
        self.pool = self.s.kv_cache_manager.block_pool
        self.identity = {'engine_instance': uuid.uuid4().hex, 'pid': os.getpid(),
                         'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                         'clock_namespace_inode': os.stat('/proc/self/ns/time').st_ino,
                         'adapter': REVISION, 'installation_digest': digest(installation)}
        epoch_file = Path(__file__).parent / 'engine_epoch.json'
        previous = json.loads(epoch_file.read_text()) if epoch_file.exists() else {}
        self.identity['previous_engine_instance'] = previous.get('engine_instance')
        temporary = epoch_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.identity))
        os.replace(temporary, epoch_file)
        self.signing_key = secrets.token_bytes(32)
        self.step, self.serial = 0, 0
        self.snapshots, self.pending, self.batches = {}, {}, []
        self.receipts, self.permits, self.mappings = Receipts(), {}, {}
        self.client_ids = set()
        self.receipt_for_step = []
        self.accepted_ns, self.active_control = {}, None
        self.engine_ids_by_client, self.batch_index = {}, {}
        self.host_call, self.host_call_serial = None, 0
        self.completed_progress = {}
        self.resources = self.read_resources()

    def signed(self, row):
        if 'transitions' in row and row['transitions'][-1]['status'] != row['status']:
            row['transitions'].append({'status': row['status'], 'at_ns': time.monotonic_ns()})
        row = copy.deepcopy(row)
        row['identity'] = self.identity
        row['signature'] = hmac.new(self.signing_key, digest(row).encode(), 'sha256').hexdigest()
        return row

    def verify(self, row):
        row = copy.deepcopy(row)
        signature = row.pop('signature', '')
        return row.get('identity') == self.identity and hmac.compare_digest(
            signature, hmac.new(self.signing_key, digest(row).encode(), 'sha256').hexdigest())

    def read_resources(self):
        cfg = self.core.vllm_config
        sc, pc, mc = cfg.scheduler_config, cfg.parallel_config, cfg.model_config
        if (pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1
                or not sc.async_scheduling or not sc.enable_chunked_prefill or not mc.enforce_eager
                or cfg.speculative_config is not None or self.s.connector is not None
                or not self.s.kv_cache_manager.enable_caching):
            raise ValueError('unsupported engine configuration: require TP/PP/DP=1, async, eager, chunked APC, no speculative/connector')
        if type(self.s).__module__ != 'vllm.v1.core.sched.async_scheduler' or type(self.s).__name__ != 'AsyncScheduler':
            raise ValueError('unreviewed scheduler implementation')
        if type(self.s.waiting).__name__ != 'FCFSRequestQueue':
            raise ValueError('adapter supports the existing FCFS queue only')
        if len(self.s.kv_cache_manager.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError('hybrid/multiple KV groups not supported')
        spec = self.s.kv_cache_manager.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if type(spec).__name__ != 'FullAttentionSpec':
            raise ValueError('only reviewed full-attention KV reuse is supported')
        block = self.s.block_size
        return validate_resources({'block_size': block, 'step_tokens': self.s.max_num_scheduled_tokens,
            'configured_max_num_batched_tokens': sc.max_num_batched_tokens,
            'long_prefill_token_threshold': sc.long_prefill_token_threshold,
            'server_max_seqs': sc.max_num_seqs, 'context_tokens': mc.max_model_len,
            'physical_blocks': self.pool.num_gpu_blocks - 1, 'null_blocks': 1,
            'hash_block_size': self.pool.hash_block_size, 'async_scheduling': True,
            'scheduler': type(self.s).__module__ + '.' + type(self.s).__name__,
            'kv_manager': type(self.s.kv_cache_manager).__module__ + '.' + type(self.s.kv_cache_manager).__name__})

    def physical(self):
        blocks = [b for b in self.pool.blocks if not b.is_null]
        active = sum(b.ref_cnt > 0 for b in blocks)
        reclaimable = sum(b.ref_cnt == 0 and b.block_hash is not None for b in blocks)
        free = self.pool.get_num_free_blocks()
        if active + free != len(blocks):
            raise ValueError('block pool accounting mismatch')
        # Future growth of active requests is a reservation, not physical use.
        growth = 0
        for request in self.s.requests.values():
            if request.is_finished():
                continue
            ids = self.s.kv_cache_manager.get_blocks(request.request_id).get_block_ids()[0]
            growth += max(0, math.ceil((request.num_prompt_tokens + request.max_tokens) / self.resources['block_size']) - len(ids))
        now = time.monotonic_ns()
        permit_blocks = sum(p['reserved_blocks'] for p in self.permits.values() if p['expires_ns'] >= now)
        return {'total_usable_blocks': len(blocks), 'active_blocks': active,
                'reclaimable_cached_blocks': reclaimable, 'unused_uncached_blocks': free - reclaimable,
                'allocator_free_blocks': free, 'future_growth_reservation_blocks': growth,
                'unconsumed_permit_blocks': permit_blocks,
                'available_for_new_reservations': max(0, free - growth - permit_blocks)}

    def request_row(self, req):
        inflight = sum(b['scheduled_tokens'].get(req.request_id, 0) for b in self.pending.values())
        completed = max(0, req.num_computed_tokens - inflight)
        return {'engine_request_id': req.request_id, 'status': req.status.name,
                'phase': 'prefill' if completed < req.num_prompt_tokens else 'decode',
                'scheduled_computed_tokens': req.num_computed_tokens,
                'completed_computed_tokens': completed, 'inflight_tokens': inflight,
                'output_tokens': req.num_output_tokens, 'output_placeholders': req.num_output_placeholders,
                'prompt_tokens': req.num_prompt_tokens, 'max_output_tokens': req.max_tokens,
                'mapping': self.mappings.get(req.request_id),
                'accepted_ns': self.accepted_ns[req.request_id],
                'queue_mutable': req.num_computed_tokens < req.num_prompt_tokens and inflight == 0
                                 and req.status.name in ('WAITING', 'PREEMPTED', 'RUNNING')}

    def query_prefix(self, query):
        tokens, salt = query['tokens'], query.get('salt')
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= self.resources['context_tokens']
                or any(type(t) is not int or t < 0 for t in tokens)
                or not isinstance(salt, str) or not 1 <= len(salt) <= 160):
            raise ValueError('invalid concrete token prefix or cache salt')
        from vllm.v1.request import Request
        from vllm import SamplingParams
        request = Request('dabs-cache-query', tokens, SamplingParams(max_tokens=1), None,
                          cache_salt=salt, block_hasher=self.core.request_block_hasher)
        # Same hash function, salt and prompt-length-minus-one rule as the engine;
        # call coordinator directly to avoid polluting prefix-cache hit statistics.
        groups, count = self.s.kv_cache_manager.coordinator.find_longest_cache_hit(request.block_hashes, len(tokens) - 1)
        ids = [b.block_id for b in groups[0]]
        writing = set(i for batch in self.pending.values() for i in batch['writing_blocks'])
        unknown = any(i in writing for i in ids)
        return {'query_id': query['query_id'], 'prefix_key': prefix_key(tokens, salt),
                'state': 'unknown' if unknown else 'hit' if count else 'miss',
                'reusable_tokens': None if unknown else count,
                'reason': 'async_KV_write_inflight' if unknown else None,
                'block_ids': ids, 'block_hashes': [str(b.block_hash) for b in groups[0]],
                'lifetime': 'instantaneous observation; revalidate at action application',
                'query_tokens': len(tokens), 'salt_digest': digest(salt)}

    def snapshot(self, payload):
        if len(payload.get('queries', [])) > 256:
            raise ValueError('too many prefix queries')
        now = time.monotonic_ns()
        self.serial += 1
        sid = self.identity['engine_instance'] + ':' + str(self.serial)
        row = {'snapshot_id': sid, 'step': self.step, 'captured_ns': now, 'expires_ns': now + 120_000_000,
               'resources': self.resources, 'physical': self.physical(),
               'running': [self.request_row(r) for r in self.s.running],
               'waiting': [self.request_row(r) for r in self.s.waiting],
               'skipped_waiting': [self.request_row(r) for r in self.s.skipped_waiting],
               'cache': [self.query_prefix(q) for q in payload.get('queries', [])],
               'inflight_batches': list(self.pending.values())}
        row['running_count'], row['waiting_count'] = self.s.get_request_counts()
        signed = self.signed(row)
        # Keep query tokens only in engine memory for apply-time revalidation.
        self.snapshots = {k: v for k, v in self.snapshots.items() if v[0]['expires_ns'] >= now}
        self.snapshots[sid] = (signed, payload.get('queries', []))
        return signed

    def apply(self, action):
        receipt, new = self.receipts.begin(action)
        if not new:
            return self.signed(receipt)
        now = time.monotonic_ns()
        if action.get('identity') != self.identity:
            receipt.update(status='rejected', reason='engine_restarted')
            return self.signed(receipt)
        pair = self.snapshots.get(action.get('snapshot_id'))
        reason = 'unknown_or_expired_snapshot' if pair is None else check_action(action, pair[0], self.identity, now, self.step)
        if reason:
            receipt.update(status='expired' if 'expired' in reason or 'advanced' in reason else 'rejected', reason=reason)
            return self.signed(receipt)
        snapshot, queries = pair
        observed = {r['engine_request_id']: r for r in snapshot['running'] + snapshot['waiting']}
        for rid in action.get('order', []):
            req = self.s.requests.get(rid)
            old = observed.get(rid)
            if old is None or req is None:
                receipt.update(status='rejected', reason='request_left_snapshot_queue')
                return self.signed(receipt)
            current = self.request_row(req)
            if (current['status'] != old['status'] or
                    current['scheduled_computed_tokens'] != old['scheduled_computed_tokens']):
                receipt.update(status='expired', reason='request_started_since_snapshot')
                return self.signed(receipt)
        if ([r.request_id for r in self.s.waiting] != [r['engine_request_id'] for r in snapshot['waiting']]
                or [r.request_id for r in self.s.running] != [r['engine_request_id'] for r in snapshot['running']]):
            receipt.update(status='rejected', reason='queue_changed_since_snapshot')
            return self.signed(receipt)
        if any(self.query_prefix(q) != old for q, old in zip(queries, snapshot['cache'])):
            receipt.update(status='rejected', reason='cache_evidence_invalidated')
            return self.signed(receipt)
        if self.read_resources() != snapshot['resources']:
            receipt.update(status='rejected', reason='resources_changed')
            return self.signed(receipt)
        admissions = action.get('admissions', [])
        available = self.physical()['available_for_new_reservations']
        needed = 0
        for spec in admissions:
            cid = spec['client_request_id']
            if cid in self.client_ids or cid in self.permits:
                receipt.update(status='rejected', reason='duplicate_admission')
                return self.signed(receipt)
            total = spec['prompt_tokens'] + spec['max_tokens']
            if not 0 < total <= self.resources['context_tokens'] or spec['max_tokens'] <= 0:
                receipt.update(status='rejected', reason='invalid_request_budget')
                return self.signed(receipt)
            needed += math.ceil(total / self.resources['block_size'])
        if len({s['client_request_id'] for s in admissions}) != len(admissions) or needed > available:
            receipt.update(status='rejected', reason='physical_reservation_unavailable')
            return self.signed(receipt)
        receipt['status'] = 'accepted'
        receipt['transitions'].append({'status': 'accepted', 'at_ns': time.monotonic_ns()})
        mutable = {r.request_id for r in list(self.s.waiting) + list(self.s.running) if self.request_row(r)['queue_mutable']}
        eligible = [rid for rid in action.get('order', []) if rid in mutable]
        for rid in action.get('order', []):
            if rid not in mutable:
                receipt['rejected'][rid] = 'not_waiting_or_prefill_safe_boundary'
        previous_control = self.active_control
        self.active_control = {'action_id': action['action_id'], 'order': eligible,
            'clients': [s['client_request_id'] for s in admissions],
            'not_before_ns': action['not_before_ns'], 'action_expires_ns': action['action_expires_ns']}
        queue_application = self.reorder_boundary(now)
        age_rejected = {rid for q in queue_application['queues'].values() if q['reason'] == 'age_fifo'
                        for rid in q['before'] if rid in eligible}
        receipt['rejected'].update({rid: 'prefill_age_guard_5s' for rid in age_rejected})
        eligible = [rid for rid in eligible if rid not in age_rejected]
        self.active_control['order'] = eligible
        if not eligible and not admissions:
            self.active_control = previous_control
        for spec in admissions:
            # The action was validated/applied within the 120 ms snapshot TTL.
            # Its reserved resources get a separate bounded delivery lease;
            # this cannot refresh a stale queue action or start decode twice.
            self.permits[spec['client_request_id']] = {**spec, 'expires_ns': time.monotonic_ns() + 2_000_000_000,
                'action_id': action['action_id'], 'reserved_blocks': math.ceil((spec['prompt_tokens'] + spec['max_tokens']) / self.resources['block_size'])}
        receipt.update(status='partially_applied' if receipt['rejected'] else 'applied',
            accepted=eligible + [s['client_request_id'] for s in admissions], applied=eligible,
            admitted_permits=[s['client_request_id'] for s in admissions],
            admission_delivery_lease_ms=2000,
            applied_step=self.step, applied_at_ns=time.monotonic_ns(),
            application_boundary='EngineCore utility loop between schedule calls',
            queue_application=queue_application,
            not_before_ns=action['not_before_ns'], action_expires_ns=action['action_expires_ns'],
            effective_waiting_order=[r.request_id for r in self.s.waiting],
            effective_running_order=[r.request_id for r in self.s.running])
        if receipt['rejected'] and not eligible and not admissions:
            receipt['status'] = 'rejected'
        self.receipt_for_step.append(action['action_id'])
        return self.signed(receipt)

    def add_request(self, original, request, *args, **kwargs):
        self.accepted_ns.setdefault(request.request_id, time.monotonic_ns())
        extra = (request.sampling_params.extra_args or {}) if request.sampling_params else {}
        binding = extra.get('dabs_binding')
        if binding:
            if not isinstance(binding, dict) or any(not isinstance(binding.get(k), str) or not 1 <= len(binding[k]) <= 160
                    for k in ('client_request_id', 'node_id', 'server_request_id')):
                request.abort_immediately = True
                return original(request, *args, **kwargs)
            cid = binding['client_request_id']
            self.engine_ids_by_client[cid] = request.request_id
            permit = self.permits.get(cid)
            if permit is None or permit['expires_ns'] < time.monotonic_ns() or cid in self.client_ids:
                request.abort_immediately = True
                self.mappings[request.request_id] = {**binding, 'engine_request_id': request.request_id,
                    'rejected': 'missing/expired/used native admission permit'}
                return original(request, *args, **kwargs)
            if (permit['node_id'] != binding['node_id'] or permit['prompt_tokens'] != request.num_prompt_tokens
                    or permit['max_tokens'] != request.max_tokens
                    or permit['input_digest'] != prefix_key(request.prompt_token_ids, request.cache_salt)):
                request.abort_immediately = True
                self.mappings[request.request_id] = {**binding, 'engine_request_id': request.request_id,
                    'rejected': 'request does not match native admission'}
                return original(request, *args, **kwargs)
            self.permits.pop(cid)
            self.client_ids.add(cid)
            self.mappings[request.request_id] = {**binding, 'engine_request_id': request.request_id,
                'action_id': permit['action_id'], 'engine_admitted_ns': time.monotonic_ns()}
            self.receipts.rows[permit['action_id']].setdefault('engine_admissions', []).append(self.mappings[request.request_id])
        return original(request, *args, **kwargs)

    def reorder_boundary(self, now=None):
        now = time.monotonic_ns() if now is None else now
        control = self.active_control
        active = bool(control and control['not_before_ns'] <= now < control['action_expires_ns'])
        priority = list(control['order']) if active else []
        if active:
            priority += [self.engine_ids_by_client[c] for c in control['clients'] if c in self.engine_ids_by_client]
        records = {'phase': 'current_action' if active else 'common_continuation',
                   'action_id': control['action_id'] if control else None,
                   'action_expires_ns': control['action_expires_ns'] if control else None,
                   'at_ns': now, 'queues': {}}
        for name in ('running', 'waiting'):
            queue = getattr(self.s, name)
            before = list(queue)
            after, reason = prefill_order(before, priority, self.accepted_ns, now,
                lambda r: self.request_row(r)['queue_mutable'])
            if name == 'running':
                queue[:] = after
            else:
                queue.clear(); queue.extend(after)
            records['queues'][name] = {'reason': reason,
                'before': [r.request_id for r in before], 'after': [r.request_id for r in after]}
        return records

    def host_step(self, original):
        """Time the actual step call; retain both sides of async pipelining."""
        self.host_call_serial += 1
        call = {'call_id': self.identity['engine_instance'] + '/host/' + str(self.host_call_serial),
                'begin_ns': time.monotonic_ns(), 'scheduled_batch_ids': [], 'completed_batch_ids': [],
                'boundary': original.__name__, 'gpu_kernel_ms': None}
        self.host_call = call
        try:
            result = original()
        except BaseException as exc:
            call['error'] = type(exc).__name__
            raise
        finally:
            call['end_ns'] = time.monotonic_ns()
            call['engine_host_ms'] = (call['end_ns'] - call['begin_ns']) / 1e6
            self.host_call = None
        # A call can submit B while completing A. Do not assign B's dispatch
        # elapsed time to A, or silently call overlapping host time equivalent.
        related = set(call['scheduled_batch_ids']) | set(call['completed_batch_ids'])
        for bid in related:
            batch = self.batch_index[bid]
            batch.setdefault('host_calls', []).append(copy.deepcopy(call))
            if batch['batch_id'] in call['completed_batch_ids']:
                batch['engine_host_ms'] = call['engine_host_ms']
                batch['host_step_equivalent'] = (call['scheduled_batch_ids'] == [batch['batch_id']]
                    and call['completed_batch_ids'] == [batch['batch_id']] and 'error' not in call)
                batch['host_batch_span_ms'] = (call['end_ns'] - batch['host_begin_ns']) / 1e6
        return result

    def schedule(self, original):
        started = time.monotonic_ns()
        queue_application = self.reorder_boundary(started)
        computed_before = {rid: r.num_computed_tokens for rid, r in self.s.requests.items()}
        output = original()
        self.step += 1
        bid = self.identity['engine_instance'] + '/step/' + str(self.step)
        counts = dict(output.num_scheduled_tokens)
        rows, writing = [], []
        for rid, count in counts.items():
            req = self.s.requests[rid]
            end = req.num_computed_tokens
            begin = end - count
            prefill = min(count, max(0, req.num_prompt_tokens - begin))
            cache_skip = max(0, begin - computed_before[rid]) if prefill else 0
            rows.append({'engine_request_id': rid, 'tokens': count, 'prefill_tokens': prefill,
                         'logical_prefill_tokens': prefill + cache_skip, 'cache_input_skipped_tokens': cache_skip,
                         'decode_tokens': count - prefill, 'context_tokens': computed_before[rid],
                         'context_after_cache': begin,
                         'prompt_tokens': req.num_prompt_tokens, 'mapping': self.mappings.get(rid)})
            blocks = self.s.kv_cache_manager.get_blocks(rid).get_block_ids()[0]
            writing.extend(blocks[begin // self.resources['block_size']:math.ceil(end / self.resources['block_size'])])
        batch = {'batch_id': bid, 'step': self.step, 'scheduled_tokens': counts, 'requests': rows,
                 'queue_application': queue_application,
                 'host_begin_ns': self.host_call['begin_ns'] if self.host_call else started,
                 'writing_blocks': sorted(set(writing)), 'host_schedule_ms': (time.monotonic_ns() - started) / 1e6,
                 'scheduled_ns': started}
        output.dabs_batch_id, output.dabs_batch_meta = bid, batch
        self.pending[bid] = batch
        self.batch_index[bid] = batch
        if self.host_call is not None:
            self.host_call['scheduled_batch_ids'].append(bid)
        remaining_receipts = []
        for aid in self.receipt_for_step:
            receipt = self.receipts.rows[aid]
            targets = set(receipt['applied']) | {m['engine_request_id'] for m in receipt.get('engine_admissions', [])}
            seen = receipt.setdefault('first_scheduled_order', [])
            matched = [rid for rid in counts if rid in targets and rid not in seen]
            if matched:
                seen.extend(matched)
                receipt.setdefault('scheduled_step', self.step)
                receipt['scheduled_order'] = list(seen)
                in_window = (receipt['not_before_ns'] <= started < receipt['action_expires_ns']
                             and self.active_control and self.active_control['action_id'] == aid)
                receipt.setdefault('target_steps', []).append({'step': self.step, 'order': matched,
                    'at_ns': started, 'priority_window_active': bool(in_window)})
                if in_window:
                    receipt.setdefault('priority_scheduled_order', []).extend(matched)
            live_permits = any(p['action_id'] == aid and p['expires_ns'] >= time.monotonic_ns() for p in self.permits.values())
            if (targets - set(seen)) & set(self.s.requests) or live_permits:
                remaining_receipts.append(aid)
        self.receipt_for_step = remaining_receipts
        return output

    def update(self, original, output, model_output):
        before = set(self.s.requests)
        result = original(output, model_output)
        for rid in before - set(self.s.requests):
            if rid in self.mappings:
                self.mappings[rid]['engine_finished_ns'] = time.monotonic_ns()
        batch = self.pending.pop(output.dabs_batch_id)
        batch = {**batch, 'engine_output_observed_ns': time.monotonic_ns(), 'engine_output_completed': True}
        self.batches.append(batch)
        self.batch_index[batch['batch_id']] = batch
        if self.host_call is not None:
            self.host_call['completed_batch_ids'].append(batch['batch_id'])
        if len(self.batches) > 200000:
            raise ValueError('native trace capacity exhausted; evidence cannot be silently dropped')
        return result

    def dispatch(self, payload):
        op = payload['op']
        if op == 'snapshot':
            return self.snapshot(payload)
        if op == 'apply':
            return self.apply(payload['action'])
        if op == 'admit':
            # Deliberately narrow fallback: validate a NEW admission against an
            # atomic current snapshot. It cannot carry a stale queue permutation.
            row = self.snapshot({})
            return self.apply({'action_id': uuid.uuid4().hex, 'identity': self.identity,
                'snapshot_id': row['snapshot_id'], 'expires_ns': row['expires_ns'],
                'not_before_ns': row['captured_ns'], 'action_expires_ns': row['captured_ns'] + 200_000_000,
                'resource_digest': digest(row['resources']), 'order': [],
                'admissions': payload['admissions'], 'fallback': 'atomic_admission_only'})
        if op == 'receipt':
            return self.signed(self.receipts.rows[payload['action_id']])
        if op == 'mapping':
            return self.signed({'mappings': [self.mappings[self.engine_ids_by_client[cid]]
                for cid in payload['client_request_ids'] if cid in self.engine_ids_by_client]})
        if op == 'verify':
            return {'valid': [self.verify(row) for row in payload['records']], 'identity': self.identity}
        if op == 'capabilities':
            workers = self.core.collective_rpc('dabs_worker_info')
            return self.signed({'resources': self.resources, 'workers': workers,
                'adapter_sha256': self.installation['adapter_sha256'],
                'vendor_patched_sha256': self.installation['patched_sha256'],
                'host_step_boundary': 'EngineCore.step_with_batch_queue' if self.core.batch_queue_size > 1 else 'EngineCore.step',
                'batch_queue_size': self.core.batch_queue_size,
                'supported_actions': ['admission_permits', 'unsubmitted_route', 'safe_prefill_queue_order'],
                'unsupported_actions': ['migrate_submitted', 'decode_reorder', 'budget_change', 'preempt'],
                'native_execution': len(workers) == 1 and all(w['npu_available'] for w in workers)})
        if op == 'batches':
            workers = self.core.collective_rpc('dabs_timing_read', args=(payload.get('after_step', 0),))
            device = {r['batch_id']: r for w in workers for r in w}
            return self.signed({'batches': [{**b, 'device': device.get(b['batch_id'])} for b in self.batches
                                           if b['step'] > payload.get('after_step', 0)][:128]})
        if op == 'ack-trace':
            # Controller sends this only AFTER persisting the signed export.
            step = payload['through_step']
            if type(step) is not int or step < 0 or step > self.step:
                raise ValueError('invalid trace acknowledgement')
            self.core.collective_rpc('dabs_timing_ack', args=(step,))
            for b in self.batches:
                if b['step'] <= step:
                    self.batch_index.pop(b['batch_id'], None)
            self.batches[:] = [b for b in self.batches if b['step'] > step]
            return self.signed({'acknowledged_step': step})
        if op == 'invalidate-cache':
            if payload.get('explicit_destructive_probe') is not True or self.s.requests or self.pending:
                raise ValueError('cache reset requires explicit probe and fully drained engine')
            ok = self.s.reset_prefix_cache()
            self.snapshots.clear()
            return self.signed({'cache_reset': bool(ok), 'at_step': self.step})
        raise ValueError('unsupported native operation')


def install_engine(core):
    from .install import check
    verified = check(allow_patched=True)
    if not verified['compatible']:
        raise RuntimeError('DABS source/version mismatch: ' + str(verified['errors']))
    adapter = EngineAdapter(core, verified['installation'])
    original_schedule, original_update, original_add = core.scheduler.schedule, core.scheduler.update_from_output, core.add_request
    core.scheduler.schedule = lambda: adapter.schedule(original_schedule)
    core.scheduler.update_from_output = lambda output, model_output: adapter.update(original_update, output, model_output)
    core.add_request = lambda request, *a, **kw: adapter.add_request(original_add, request, *a, **kw)
    def native_rpc(self, payload):
        return adapter.dispatch(payload)
    core.dabs_native = MethodType(native_rpc, core)
    core._dabs_adapter = adapter
    original_step = core.step_fn
    core.step_fn = lambda: adapter.host_step(original_step)
