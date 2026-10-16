"""Predeclared R234 phase support witnesses, checked on independent splits.

These are minimum coverage requirements, NOT invented regression bounds.
Runtime support remains the actual per-phase training min/max. More extreme
states still fail explicitly. No formal-experiment outputs train this model.
"""
CONTRACT = 'r234_phase_tail_prefix_v1'


def coverage_check(samples, replica, phase, step_tokens):
    requirements = {
        'prefill_only': {'p': (1, step_tokens * .875), 'pc': (0, 3584)},
        'mixed': {'p': (1, step_tokens * .875), 'pc': (0, 3584),
                  'd': (1, 2), 'dc': (3584, 7424)},
        'decode_only': {'d': (1, 2), 'dc': (3584, 7424)},
    }[phase]
    checks = {}
    for split in ('train', 'holdout'):
        rows = [s for s in samples if s['replica'] == replica and
                s['phase'] == phase and s['split'] == split]
        actual = {k: [min(s[k] for s in rows), max(s[k] for s in rows)]
                  for k in requirements} if rows else {}
        missing = [k for k, (lo, hi) in requirements.items()
                   if k not in actual or actual[k][0] > lo or actual[k][1] < hi]
        checks[split] = {'actual': actual, 'missing': missing, 'pass': not missing}
    return {'contract': CONTRACT, 'required': {k: list(v) for k, v in requirements.items()}, 'splits': checks,
            'pass': all(c['pass'] for c in checks.values())}
