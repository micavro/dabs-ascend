"""Native engine HOST-step calibration; NPU forward events are auxiliary only."""
import asyncio
import math
import random
import uuid

from .common import write_json, source_identity, sha

PHASES = ('prefill_only', 'decode_only', 'mixed')


def batch_features(batch):
    from core18v2.service import features, phase
    rows = batch['requests']
    p = sum(r['prefill_tokens'] for r in rows)
    d = sum(r['decode_tokens'] for r in rows)
    dc = sum(r['context_tokens'] for r in rows if r['decode_tokens'])
    pc = max((r['context_tokens'] for r in rows if r['prefill_tokens']), default=0)
    device = batch['device']
    if not device or not device['device_event_complete'] or device['batch_id'] != batch['batch_id']:
        raise ValueError('missing/misassociated completed NPU event')
    calls = [c for c in batch.get('host_calls', []) if batch['batch_id'] in c['completed_batch_ids']]
    if len(calls) != 1 or calls[0]['engine_host_ms'] != batch.get('engine_host_ms'):
        raise ValueError('missing/misassociated completed engine host-step')
    call = calls[0]
    equivalent = (call['scheduled_batch_ids'] == [batch['batch_id']] and call['completed_batch_ids'] == [batch['batch_id']])
    if batch.get('host_step_equivalent') != equivalent or not math.isfinite(batch['engine_host_ms']) or batch['engine_host_ms'] < 0:
        raise ValueError('invalid host timing equivalence or elapsed time')
    return {'phase': phase(p, d), 'x': features(p, d, dc, pc), 'p': p, 'd': d, 'dc': dc, 'pc': pc,
            'logical_prefill_tokens': sum(r['logical_prefill_tokens'] for r in rows),
            'cache_input_skipped_tokens': sum(r['cache_input_skipped_tokens'] for r in rows),
            'y': batch['engine_host_ms'], 'host_step_equivalent': equivalent,
            'host_call_id': call['call_id'], 'npu_forward_ms_auxiliary': device['npu_forward_ms'],
            'host_batch_span_ms_approximation': batch['host_batch_span_ms'], 'batch_id': batch['batch_id']}


def fit_model(samples, capabilities):
    import numpy as np
    from .common import percentile
    models, validation, failures = {}, {}, []
    ambiguous = sum(s.get('host_step_equivalent') is not True for s in samples)
    eligible = [s for s in samples if s.get('host_step_equivalent') is True]
    for replica, capability in enumerate(capabilities):
        phases, checks = {}, {}
        for phase in PHASES:
            train = [s for s in eligible if s['replica'] == replica and s['phase'] == phase and s['split'] == 'train']
            holdout = [s for s in eligible if s['replica'] == replica and s['phase'] == phase and s['split'] == 'holdout']
            if len(train) < 20 or len(holdout) < 20:
                failures.append(f'r{replica}/{phase}: require >=20 independent train and holdout batches with one batch scheduled AND completed in the same host step call; async overlapping calls excluded')
                continue
            x, y = np.asarray([s['x'] for s in train]), np.asarray([s['y'] for s in train])
            # Nonnegative least squares by deterministic coordinate descent.
            coef = np.zeros(6)
            for _ in range(2000):
                old = coef.copy()
                for j in range(6):
                    denom = float(x[:, j] @ x[:, j])
                    if denom:
                        coef[j] = max(0., float(x[:, j] @ (y - x @ coef + x[:, j] * coef[j])) / denom)
                if float(np.max(np.abs(old - coef))) < 1e-10:
                    break
            pred = [max(.1, float(np.asarray(s['x']) @ coef)) for s in holdout]
            errors = [abs(p - s['y']) / max(.1, s['y']) for p, s in zip(pred, holdout)]
            rmse = math.sqrt(sum((p - s['y']) ** 2 for p, s in zip(pred, holdout)) / len(holdout))
            mean = sum(s['y'] for s in holdout) / len(holdout)
            passed = percentile(errors, .95) <= .35 and rmse <= .25 * max(.1, mean)
            checks[phase] = {'train_batches': len(train), 'holdout_batches': len(holdout),
                'p95_relative_error': percentile(errors, .95), 'rmse_ms': rmse, 'holdout_mean_ms': mean,
                'pass': passed, 'range': {k: [min(s[k] for s in train), max(s[k] for s in train)] for k in ('p', 'd', 'dc', 'pc')}}
            phases[phase] = {'coef': coef.tolist()}
            if not passed:
                failures.append(f'r{replica}/{phase}: frozen holdout error thresholds exceeded')
        combined = [s for s in eligible if s['replica'] == replica and s['split'] == 'train']
        if (not combined or max(s['p'] for s in combined) < capability['resources']['step_tokens'] * .875
                or max(s['pc'] for s in combined) < 3072 or max(s['dc'] for s in combined) < 3584
                or max(s['d'] for s in combined) < 2):
            failures.append(f'r{replica}: actual batch coverage does not cover the fixed R234/client-inflight=2 workload')
        models[str(replica)], validation[str(replica)] = {'phases': phases}, checks
    return {'models': models, 'validation': validation, 'failures': failures, 'pass': not failures,
            'timing_target': 'engine_host_ms: actual EngineCore step entry to return, only unambiguous same-batch calls',
            'excluded_ambiguous_host_batches': ambiguous, 'eligible_host_batches': len(eligible),
            'host_step_equivalence': not failures,
            'overhead': 'host monotonic_ns brackets plus two auxiliary NPU events and nonblocking queries; instrumentation is included, not subtracted',
            'thresholds_predeclared': {'minimum_batches_per_split_phase_replica': 20, 'p95_relative_error': .35, 'rmse_over_mean': .25},
            'pending_batch_model': 'conservative serial pending host-step intervals, not a physical NPU simulator',
            'identities': [c['identity'] for c in capabilities], 'resources': [c['resources'] for c in capabilities],
            'source_sha256': source_identity(), 'frozen': not failures}


async def run_calibration(backend, tokenizer, out):
    native = backend.native
    await native.drain(len(backend.endpoints))
    base = await tokenizer.encode(' Native DABS calibration sample. ')
    pads = await tokenizer.encode(' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z')
    cases = [(128, 1, 16, False), (129, 1, 16, False), (1024, 1, 32, False),
             (2048, 1, 48, False), (3584, 1, 64, False), (3584, 1, 128, False),
             (3584, 4, 16, False), (3584, 4, 128, False), (3584, 10, 128, False),
             (3584, 12, 128, False), (3584, 4, 128, True), (1024, 4, 64, False)]
    plan = [{'split': split, 'length': length, 'concurrency': concurrency, 'output': output, 'hot': hot,
             'seed': 271800 + offset * 10000 + i * 100 + concurrency}
            for offset, split in enumerate(('train', 'holdout'))
            for i, (length, concurrency, output, hot) in enumerate(cases)]
    write_json(out / 'calibration_plan.json', {'plan': plan, 'fit_after_collection': True,
        'formal_experiment_data_used': False, 'timing_target': 'engine host step; overlapping call/batch associations excluded',
        'thresholds': '>=20 batches/split/phase/replica; p95 relative error<=.35, RMSE/mean<=.25'})
    samples = []
    for replica in range(len(backend.endpoints)):
        for group in plan:
            native.salt = 'dabs-calibration-' + uuid.uuid4().hex
            rng = random.Random(group['seed'])
            nodes = []
            for j in range(group['concurrency']):
                tokens = (base + [rng.choice(pads) for _ in range(group['length'])])[:group['length']]
                nodes.append({'id': 'cal-' + group['split'] + '-' + str(group['seed']) + '-' + uuid.uuid4().hex, 'tokens': tokens, 'max_output_tokens': group['output'],
                              'fixed_workload_replay': True})
            if group['hot']:
                for n in nodes:
                    warm = {**n, 'id': 'warm-' + n['id'], 'max_output_tokens': 16}
                    await native.reserve(replica, warm)
                    await backend.generate(replica, warm, group['seed'], out / 'requests.jsonl', schema=None)
                while await native.harvest(len(backend.endpoints), wait_complete=True):
                    pass
            async def one(j, node):
                # Stagger later arrivals to observe real mixed prefill/decode.
                await asyncio.sleep(.015 * j)
                await native.reserve(replica, node)
                return await backend.generate(replica, node, group['seed'], out / 'requests.jsonl', schema=None)
            await asyncio.gather(*(one(j, n) for j, n in enumerate(nodes)))
            await native.drain(len(backend.endpoints))
            while True:
                rows = await native.harvest(len(backend.endpoints), wait_complete=True)
                if not rows:
                    break
                ids = {n['id'] for n in nodes}
                for r, batch in rows:
                    if r == replica and batch['requests'] and all(q['mapping'] and q['mapping']['node_id'] in ids for q in batch['requests']):
                        samples.append({**batch_features(batch), 'replica': r, 'split': group['split'], 'group': group})
            write_json(out / 'calibration_samples.json', samples)
            print('NATIVE_CALIBRATION', replica, group, 'samples', len(samples), flush=True)
    model = fit_model(samples, native.capabilities)
    model['raw_samples_sha256'] = sha(out / 'calibration_samples.json')
    model['raw_native_sha256'] = sha(out / 'native_events.jsonl')
    write_json(out / 'native_profile.json', model)
    return model
