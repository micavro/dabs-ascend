"""One Ascend runtime timing contract shared by controller and EngineCore."""

CONTRACT = 'ascend-1500ms-lifecycle-v1'
EPOCH_S = 1.5
SNAPSHOT_MAX_AGE_S = 1.5
PLANNING_SOFT_LIMIT_S = 1.2
COMMIT_RESERVE_S = .025
SNAPSHOT_TTL_NS = 1_500_000_000
ACTION_WINDOW_NS = 1_500_000_000


def configuration():
    return dict(epoch_s=EPOCH_S, snapshot_max_age_s=SNAPSHOT_MAX_AGE_S,
                planning_soft_limit_s=PLANNING_SOFT_LIMIT_S,
                commit_reserve_s=COMMIT_RESERVE_S, action_horizon_ms=EPOCH_S * 1000,
                continuation_period_ms=EPOCH_S * 1000,
                cache_evidence_max_age_s=SNAPSHOT_MAX_AGE_S)


def capabilities():
    return dict(contract=CONTRACT, snapshot_ttl_ns=SNAPSHOT_TTL_NS,
                action_window_ns=ACTION_WINDOW_NS, completion_bindings=True)


def require_timing_policy(row):
    actual = row.get('timing_policy', {})
    if any(actual.get(k) != v for k, v in capabilities().items()):
        raise ValueError('native timing/lifecycle capability mismatch; run bash run_native.sh install '
                         'to update and restart the existing services (1500/1500/1200 ms)')
