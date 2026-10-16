"""Native engine HOST-step calibration; NPU forward events are auxiliary only."""
import asyncio
import math
import random
import uuid

from .common import write_json, source_identity, sha
from .native_timing import TARGET, audit, read_events, reconstruct, require_timing, collection_floors
from .native_prediction import PREDICTOR, regression_features, check_startup_inputs

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
    return {'phase': phase(p, d), 'x': list(features(p, d, dc, pc)), 'p': p, 'd': d, 'dc': dc, 'pc': pc,
            'logical_prefill_tokens': sum(r['logical_prefill_tokens'] for r in rows),
            'cache_input_skipped_tokens': sum(r['cache_input_skipped_tokens'] for r in rows),
            'y': batch['engine_host_ms'], 'host_step_equivalent': equivalent,
            'host_call_id': call['call_id'], 'npu_forward_ms_auxiliary': device['npu_forward_ms'],
            'host_batch_span_ms_approximation': batch['host_batch_span_ms'], 'batch_id': batch['batch_id']}


def fit_model(samples, capabilities, *, predictor=None):
    import numpy as np
    from .common import percentile
    models, validation, failures = {}, {}, []
    accounting = any(s.get('timing_target') == TARGET for s in samples)
    predictor = predictor or (PREDICTOR if accounting else 'phase6_v1')
    if predictor not in (PREDICTOR, 'phase6_v1'):
        raise ValueError('unknown predictor')
    if accounting and any(s.get('timing_target') != TARGET for s in samples):
        raise ValueError('mixed timing targets')
    eligible = [s for s in samples if (s.get('accounting_valid') is True if accounting else s.get('host_step_equivalent') is True)]
    ambiguous = len(samples) - len(eligible)
    seen, group_splits = set(), {}
    for s in eligible:
        key = s['replica'], s['batch_id']
        if key in seen or not math.isfinite(s['y']) or s['y'] <= 0:
            raise ValueError('duplicate batch or invalid timing sample')
        seen.add(key)
        if s['split'] not in ('train', 'holdout') or s['group']['split'] != s['split']:
            raise ValueError('invalid train/holdout assignment')
        key = s['replica'], s['group']['seed']
        if key in group_splits and group_splits[key] != s['split']:
            raise ValueError('train/holdout group leakage')
        group_splits[key] = s['split']
    for replica, capability in enumerate(capabilities):
        phases, checks = {}, {}
        for phase in PHASES:
            train = [s for s in eligible if s['replica'] == replica and s['phase'] == phase and s['split'] == 'train']
            holdout = [s for s in eligible if s['replica'] == replica and s['phase'] == phase and s['split'] == 'holdout']
            checks[phase] = {'train_batches': len(train), 'holdout_batches': len(holdout), 'pass': False,
                'train_groups': len({s['group']['seed'] for s in train}),
                'holdout_groups': len({s['group']['seed'] for s in holdout})}
            if len(train) < 20 or len(holdout) < 20:
                failures.append(f'r{replica}/{phase}: require >=20 batches in EACH independent train/holdout split; actual train={len(train)}, holdout={len(holdout)}; target={TARGET if accounting else "same-batch host call"}')
                continue
            x, y = np.asarray([regression_features(s, predictor) for s in train]), np.asarray([s['y'] for s in train])
            # Nonnegative least squares by deterministic coordinate descent.
            coef = np.zeros(x.shape[1])
            for _ in range(2000):
                old = coef.copy()
                for j in range(len(coef)):
                    denom = float(x[:, j] @ x[:, j])
                    if denom:
                        coef[j] = max(0., float(x[:, j] @ (y - x @ coef + x[:, j] * coef[j])) / denom)
                if float(np.max(np.abs(old - coef))) < 1e-10:
                    break
            pred = [max(.1, float(np.asarray(regression_features(s, predictor)) @ coef)) for s in holdout]
            errors = [abs(p - s['y']) / max(.1, s['y']) for p, s in zip(pred, holdout)]
            rmse = math.sqrt(sum((p - s['y']) ** 2 for p, s in zip(pred, holdout)) / len(holdout))
            mean = sum(s['y'] for s in holdout) / len(holdout)
            passed = percentile(errors, .95) <= .35 and rmse <= .25 * max(.1, mean)
            checks[phase].update({
                'p95_relative_error': percentile(errors, .95), 'rmse_ms': rmse, 'holdout_mean_ms': mean,
                'pass': passed, 'range': {k: [min(s[k] for s in train), max(s[k] for s in train)] for k in ('p', 'd', 'dc', 'pc')}})
            phases[phase] = {'coef': coef[:6].tolist()}
            if predictor == PREDICTOR and phase == 'prefill_only':
                regimes = sorted({s['regime'] for s in train})
                phases[phase].update(startup_extra_ms=float(coef[6]), observed_regimes=regimes)
                checks[phase]['regimes'] = {}
                for regime in ('startup', 'pipeline'):
                    selected = [(s, p) for s, p in zip(holdout, pred) if s['regime'] == regime]
                    checks[phase]['regimes'][regime] = {
                        'train_batches': sum(s['regime'] == regime for s in train),
                        'holdout_batches': len(selected),
                        'actual_mean_ms': sum(s['y'] for s, _ in selected) / len(selected) if selected else None,
                        'predicted_mean_ms': sum(p for _, p in selected) / len(selected) if selected else None}
                    if selected and regime not in regimes:
                        passed = checks[phase]['pass'] = False
                        failures.append(f'r{replica}/{phase}: holdout has untrained regime {regime}')
            if not passed:
                failures.append(f'r{replica}/{phase}: frozen holdout error thresholds exceeded')
        combined = [s for s in eligible if s['replica'] == replica and s['split'] == 'train']
        if (not combined or max(s['p'] for s in combined) < capability['resources']['step_tokens'] * .875
                or max(s['pc'] for s in combined) < 3072 or max(s['dc'] for s in combined) < 3584
                or max(s['d'] for s in combined) < 2):
            failures.append(f'r{replica}: actual batch coverage does not cover the fixed R234/client-inflight=2 workload')
        models[str(replica)], validation[str(replica)] = {'phases': phases}, checks
    return {'models': models, 'validation': validation, 'failures': failures, 'pass': not failures, 'predictor': predictor,
            'timing_target': TARGET if accounting else 'engine_host_ms: actual EngineCore step entry to return, only unambiguous same-batch calls',
            'excluded_ambiguous_host_batches': ambiguous, 'eligible_host_batches': len(eligible),
            'host_step_equivalence': False if accounting else not failures,
            'accounting_model': accounting, 'original_gpu_timing_equivalent': False if accounting else not failures,
            'prediction_semantics': 'additive nonoverlapping host completion charges; throughput proxy including startup, not per-batch latency or device execution' if accounting else 'same-batch host step',
            'overhead': 'host monotonic_ns brackets plus two auxiliary NPU events and nonblocking queries; instrumentation is included, not subtracted',
            'thresholds_predeclared': {'minimum_batches_per_split_phase_replica': 20, 'p95_relative_error': .35, 'rmse_over_mean': .25},
            'pending_batch_model': 'serial predicted completion charges; no conservative bound claimed for a partly completed interval' if accounting else 'conservative serial pending host-step intervals, not a physical NPU simulator',
            'identities': [c['identity'] for c in capabilities], 'resources': [c['resources'] for c in capabilities],
            'source_sha256': source_identity(), 'frozen': not failures}


def calibration_plan(diagnostic=False, *, predictor=PREDICTOR):
    # Fixed in advance; no adapting cases/thresholds to holdout outcomes.
    cases = [(128, 1, 16, False), (129, 1, 16, False), (1024, 1, 32, False),
             (2048, 1, 48, False), (3584, 1, 64, False), (3584, 1, 128, False),
             (3584, 2, 64, False), (3584, 2, 128, False),
             (3584, 4, 16, False), (3584, 4, 128, False), (3584, 10, 128, False),
             (3584, 12, 128, False), (3584, 4, 128, True), (1024, 4, 64, False)]
    if diagnostic:
        cases = [(3584, 1, 16, False), (3584, 2, 32, False)]
    repeats = 1  # Fixed budget; insufficient phase coverage is reported, never padded.
    return [{'split': split, 'length': length, 'concurrency': concurrency, 'output': output,
             'hot': hot, 'repeat': repeat, 'seed': (271800 if predictor == 'phase6_v1' else 2271800) + offset * 100000 + repeat * 10000 + i * 100 + concurrency}
            for offset, split in enumerate(('train', 'holdout'))
            for repeat in range(repeats)
            for i, (length, concurrency, output, hot) in enumerate(cases)]


def rebuild(directory):
    from .common import read_json
    rows = list(read_events(directory / 'native_events.jsonl'))
    record = read_json(directory / 'calibration_plan.json')
    samples, excluded = reconstruct(rows, record['plan'], after_steps=collection_floors(rows, record))
    write_json(directory / 'timing_audit.json', audit(rows))
    write_json(directory / 'calibration_samples.json', samples)
    write_json(directory / 'calibration_excluded.json', excluded)
    check_startup_inputs(samples, rows)
    return samples, excluded


def refit_existing(directory, out):
    """Retrospective diagnosis only: previously examined holdout cannot certify v2."""
    from .common import read_json
    rows = list(read_events(directory / 'native_events.jsonl'))
    record = read_json(directory / 'calibration_plan.json')
    samples, excluded = reconstruct(rows, record['plan'], after_steps=collection_floors(rows, record))
    write_json(out / 'timing_audit.json', audit(rows))
    write_json(out / 'calibration_excluded.json', excluded)
    write_json(out / 'calibration_samples.json', samples)
    if excluded or not samples:
        raise ValueError('raw association failed; inspect calibration_excluded.json')
    check_startup_inputs(samples, rows)
    caps = read_json(directory / 'capabilities.json')
    old = fit_model(samples, caps, predictor='phase6_v1')
    model = fit_model(samples, caps)
    retrospective_pass = model['pass']
    model.update(frozen=False, diagnostic_only=True, retrospective_validation=True,
                 acceptance_granted=False, fresh_holdout_required=True, source_directory=str(directory))
    write_json(out / 'native_profile.json', model)
    comparison = {'previous': old['validation'], 'startup_aware': model['validation'],
                  'models': model['models'], 'failures': model['failures'], 'predictor': model['predictor'],
                  'retrospective_error_check_pass': retrospective_pass, 'frozen': False,
                  'fresh_holdout_required': True, 'acceptance_granted': False}
    write_json(out / 'refit_comparison.json', comparison)
    return comparison


async def run_calibration(backend, tokenizer, out, *, diagnostic=False):
    native = backend.native
    for capability in native.capabilities:
        require_timing(capability)
    starting = await native.drain(len(backend.endpoints))
    base = await tokenizer.encode(' Native DABS calibration sample. ')
    pads = await tokenizer.encode(' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z')
    plan = calibration_plan(diagnostic)
    write_json(out / 'calibration_plan.json', {'plan': plan, 'fit_after_collection': True,
        'collection_start': {str(r): {'snapshot_id': s['snapshot_id'], 'step': s['step']} for r, s in enumerate(starting)},
        'formal_experiment_data_used': False, 'timing_target': TARGET, 'diagnostic_only': diagnostic,
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
                if j:
                    # Observe decode before later arrivals; a 15 ms sleep did not
                    # reliably produce mixed batches on the real ~50 ms engine.
                    deadline = asyncio.get_running_loop().time() + 30
                    while True:
                        state = await native.call(replica, 'snapshot')
                        if any(r['phase'] == 'decode' for r in state['running']):
                            break
                        if asyncio.get_running_loop().time() >= deadline:
                            raise ValueError('calibration mixed-load trigger: no observed decode within 30s')
                        await asyncio.sleep(.02)
                    await asyncio.sleep(.015 * (j - 1))
                await native.reserve(replica, node)
                return await backend.generate(replica, node, group['seed'], out / 'requests.jsonl', schema=None)
            await asyncio.gather(*(one(j, n) for j, n in enumerate(nodes)))
            await native.drain(len(backend.endpoints))
            while True:
                rows = await native.harvest(len(backend.endpoints), wait_complete=True)
                if not rows:
                    break
            samples, excluded = rebuild(out)
            partial = fit_model(samples, native.capabilities)
            write_json(out / 'calibration_progress.json', {'validation': partial['validation'], 'excluded': excluded,
                'samples': len(samples), 'diagnostic_only': diagnostic})
            print('NATIVE_CALIBRATION', replica, group, 'eligible', len(samples), 'excluded', len(excluded), flush=True)
            if excluded or not samples:
                partial['pass'] = partial['frozen'] = False
                partial['failures'] += ['timing association failed early; inspect timing_audit.json/calibration_excluded.json']
                write_json(out / 'native_profile.json', partial)
                return partial
    model = fit_model(samples, native.capabilities)
    if diagnostic:
        model['diagnostic_only'] = True
        model['association_pass'] = bool(samples) and not excluded
        model['frozen'] = False
        # Diagnostic success NEVER implies a calibrated model or acceptance.
        write_json(out / 'native_profile.json', model)
        return {'pass': model['association_pass'], 'association_pass': model['association_pass'],
                'calibration_pass': model['pass'], 'diagnostic_only': True, 'frozen': False,
                'eligible_host_batches': len(samples), 'validation': model['validation'], 'failures': model['failures']}
    model['raw_samples_sha256'] = sha(out / 'calibration_samples.json')
    model['raw_native_sha256'] = sha(out / 'native_events.jsonl')
    write_json(out / 'native_profile.json', model)
    return model
