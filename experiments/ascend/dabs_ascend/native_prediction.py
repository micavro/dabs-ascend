"""Controller-only startup-aware prediction; no changes to frozen algorithms."""
import math

PREDICTOR = 'phase6_prefill_startup_v2'


def regression_features(sample, predictor):
    values = list(sample['x'])
    if predictor == PREDICTOR and sample['phase'] == 'prefill_only':
        if sample.get('regime') not in ('startup', 'pipeline'):
            raise ValueError('missing measured startup/pipeline regime')
        values.append(float(sample['regime'] == 'startup'))
    return values


def startup_extra(model, replica, p, d, startup):
    if model.get('predictor') != PREDICTOR or not p or d:
        return 0.
    if type(startup) is not bool:
        raise ValueError('native_calibration_out_of_range:unknown prefill startup state')
    row = model['models'][str(replica)]['phases']['prefill_only']
    regime = 'startup' if startup else 'pipeline'
    if regime not in row['observed_regimes']:
        raise ValueError('native_calibration_out_of_range:unobserved prefill regime:' + regime)
    value = row['startup_extra_ms']
    if not math.isfinite(value) or value < 0:
        raise ValueError('invalid startup coefficient')
    return value if startup else 0.


def predict(model, replica, p, d, dc, pc, *, startup=None):
    from core18v2.service import coefficients, features, phase
    c = coefficients(model, replica, phase(p, d))
    # Add the startup term before the floor, just as during regression.
    return max(.1, sum(a * b for a, b in zip(c, features(p, d, dc, pc))) +
               startup_extra(model, replica, p, d, startup))


def pending_startup(batch):
    """Decision-time input only: the submission call is already in a snapshot."""
    bid = batch['batch_id']
    calls = [c for c in batch.get('host_calls', []) if bid in c.get('scheduled_batch_ids', [])]
    if len(calls) != 1 or not isinstance(calls[0].get('pending_before'), list):
        return None
    return not calls[0]['pending_before']


def check_startup_inputs(samples, rows):
    """Ensure training labels can be obtained at submission, without future timing."""
    from .native_timing import index_events
    batches, _ = index_events(rows)
    for sample in samples:
        if sample['phase'] != 'prefill_only':
            continue
        measured = pending_startup(batches[sample['replica'], sample['batch_id']])
        if measured is None or measured != (sample['regime'] == 'startup'):
            raise ValueError('prefill startup classification is not reproducible from submission pending state: ' + sample['batch_id'])
