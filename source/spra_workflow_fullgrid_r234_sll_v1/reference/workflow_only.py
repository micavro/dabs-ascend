"""Fixed-budget validation, copied from the tested D0 remeasurement helper."""
def validate_fixed_summary(summary, count=None):
    if summary.get('condition_valid') is not True:
        raise ValueError('condition not valid')
    if summary.get('budget_invariants') is not True:
        raise ValueError('missing/failed native budget invariants')
    if summary.get('prefill_caps_applied') != [4096]:
        raise ValueError('native cap differs from fixed 4096 or has missing evidence')
    if summary.get('budget_switches_total') != 0:
        raise ValueError('unexpected native budget switch')
    matches, checked = summary.get('native_next_batch_matches'), summary.get('native_next_batch_checked')
    if type(checked) is not int or checked <= 0 or matches != checked:
        raise ValueError('native batch-check mismatch/missing evidence; preserve raw records')
    if count is not None and (summary.get('requests') != count or summary.get('request_correlation') != f'{count}/{count}'):
        raise ValueError('request count/correlation mismatch')
