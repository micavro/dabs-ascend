"""Repair inherited D0 proposal admission intent, without changing resource limits.

The reference has already been projected. A copy of its admitted subset is NOT
an unprojected candidate: resources() resets resource_held on every invocation.
Restore the complete reference intent before original schedule_alternative()
constructs its bounded-admission intent, then let the original resource projector
and the unchanged age validator act on that candidate. Do not patch validation.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
import semantic19.selector as semantic

REVISION = 'wf-admission-intent-repair-1'
_ORIGINAL = semantic.schedule_alternative


def schedule_alternative(snapshot, model, cfg, base):
    held = base.get('resource_held', [])
    if not held:
        # Preserve the original low-pressure proposal path and action fields.
        return _ORIGINAL(snapshot, model, cfg, base)
    ready = set(snapshot['ready'])
    admitted = set(base['admit'])
    blocked = set(held)
    if (admitted & blocked or admitted | blocked != ready
            or len(base['admit']) != len(admitted) or len(held) != len(blocked)):
        raise ValueError('reference admission partition is incomplete or overlapping')
    restored = copy.deepcopy(base)
    restored['admit'] = list(snapshot['ready'])
    restored['resource_held'] = []
    # Original bounded-admission logic is applied AFTER restoration, not bypassed.
    candidate = _ORIGINAL(snapshot, model, cfg, restored)
    candidate['admission_projection_repair'] = {
        'revision': REVISION,
        'inherited_resource_held': list(held),
        'candidate_resource_held': list(candidate.get('resource_held', [])),
        'reference_intent_restored_before_bounded_rule': True,
        'age_validator_changed': False,
        'resource_limits_changed': False,
        'budget_changed': False,
    }
    return candidate


@contextmanager
def binding():
    """The experiment uses a single isolated planning process per session.

    Restore globals even on failure. No deployed source file or engine process is
    patched. The original D choose() reads this global; Lite imports our wrapper.
    """
    saved = semantic.schedule_alternative
    semantic.schedule_alternative = schedule_alternative
    try:
        yield
    finally:
        semantic.schedule_alternative = saved
