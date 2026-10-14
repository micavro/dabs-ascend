"""Independent logical next-batch check for the existing n=1/APC envelope.

Only fix the verified tail rule. No actual scheduled tokens are used to predict.
Unknown allocation/queue differences remain failures, not tolerated percentages.
The trace's free_blocks_before is known to be sampled AFTER schedule; not used.
"""
from __future__ import annotations
import copy
CHECKER_REVISION = 'native-tail-capacity-check-1'

def integer(x, name, low=0):
    if type(x) is not int or x < low:
        raise ValueError(f'{name}: expected integer >= {low}, got {x!r}')
    return x

def prefill_chunk(remaining, available, block=16):
    remaining = integer(remaining, 'remaining')
    available = integer(available, 'available')
    block = integer(block, 'block', 1)
    # vLLM 0.6.3 _get_num_new_tokens: only budget-induced truncation aligns.
    return remaining if remaining <= available else (available // block) * block

def predict_next(ordered_before, total, cap, block=16, max_seqs=64, *, legacy=False):
    integer(total, 'total', 1); integer(max_seqs, 'max_seqs', 1); integer(block, 'block', 1)
    if total % block: raise ValueError('unaligned total')
    if cap is not None:
        integer(cap, 'cap', block)
        if cap % block: raise ValueError('unaligned cap')
    active = list(ordered_before['running']); wait = list(ordered_before['waiting'])
    ids = [x['request_id'] for x in active + wait]
    if len(ids) != len(set(ids)): raise ValueError('duplicate request in before queues')
    dec = [x for x in active if not x['prefill']][:max_seqs]
    pre = [x for x in active if x['prefill']] + wait
    left = total - len(dec)
    if left < 0: raise ValueError('decode exceeds total')
    room = min(left, cap if cap is not None else left)
    out = {x['request_id']: 1 for x in dec}
    for x in pre:
        if len(out) >= max_seqs or room <= 0: break
        prompt = integer(x['prompt'], 'prompt', 1)
        computed = integer(x['computed'], 'computed')
        rem = prompt - computed
        if rem <= 0: raise ValueError('prefill has no positive remaining prompt')
        take = min(rem, room // block * block) if legacy else prefill_chunk(rem, room, block)
        if take <= 0: break
        out[x['request_id']] = take; room -= take
    return out

def compare_iteration(row):
    ident = {k: row.get(k) for k in ('replica', 'iteration', 'pid', 'case', 'plan_id')}
    base = {'checker_revision': CHECKER_REVISION, **ident,
            'prediction_uses_actual_schedule': False,
            'free_blocks_before_used_as_preallocation_evidence': False}
    errors = []
    try:
        predicted = predict_next(row['ordered_before'], row['total'], row['prefill_cap'])
        old = predict_next(row['ordered_before'], row['total'], row['prefill_cap'], legacy=True)
        scheduled = row['scheduled']
        actual = {a['request_id']: integer(a['tokens'], 'scheduled.tokens', 1) for a in scheduled}
        if len(actual) != len(scheduled): errors.append('duplicate_scheduled_request')
        p = sum(a['tokens'] for a in scheduled if a['prefill'])
        d = sum(a['tokens'] for a in scheduled if not a['prefill'])
        if p + d > row['total']: errors.append('total_exceeded')
        if row['prefill_cap'] is not None and p > row['prefill_cap']: errors.append('prefill_cap_exceeded')
        if p != row['prefill_logical_tokens'] or d != row['decode_tokens']: errors.append('logical_totals_differ')
        before = {x['request_id']: x for q in ('running', 'waiting') for x in row['ordered_before'][q]}
        for a in scheduled:
            b = before.get(a['request_id'])
            if b is None: errors.append('scheduled_request_missing_from_before'); continue
            if bool(b['prefill']) != bool(a['prefill']): errors.append('phase_mismatch')
            if a['prefill'] and a['tokens'] > b['prompt'] - b['computed']: errors.append('chunk_exceeds_remaining')
            if not a['prefill'] and a['tokens'] != 1: errors.append('not_single_step_decode')
        supported = not row.get('preemptions', 0) and not row['ordered_before'].get('swapped')
        if errors: classification = 'invariant_violation'
        elif not supported: classification = 'unsupported_preemption_or_swap'
        elif predicted != actual: classification = 'unresolved_prediction_mismatch'
        elif old != actual: classification = 'matched_tail_rule_correction'
        else: classification = 'matched_unchanged'
        match = not errors and supported and predicted == actual
        return {**base, 'predicted': predicted, 'actual': actual, 'legacy_predicted': old,
                'legacy_match': old == actual, 'match': match, 'classification': classification,
                'preemptions': row.get('preemptions'), 'errors': errors,
                'mismatch_scope': None if match else 'not auto-accepted; inspect saved native record'}
    except (KeyError, TypeError, ValueError) as exc:
        return {**base, 'predicted': None, 'actual': None, 'legacy_match': False, 'match': False,
                'classification': 'insufficient_or_malformed_evidence', 'errors': [str(exc)],
                'mismatch_scope': 'missing evidence never interpreted as a match'}
