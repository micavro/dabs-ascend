"""Finite-window acceptance evidence; never a policy optimizer or performance gate.

The window starts at the first validated nonempty admission, not condition setup.
Its end remains the already selected absolute decision boundary (<=200 ms).
The running native hook is deliberately unchanged by this client-only repair.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from .batchcheck import binding_evidence
from .common import VERSION, load

REVISION = 'stage18v2-finite-admission-fix-1'


def finite_window_evidence(trigger, controls, flushes, iterations, nodes):
    """Require actual work before and after the SAME one-shot control expires.

    Bootstrap current-action frames, unrelated plan IDs, and idle frames cannot
    substitute for exercising the 256 cap. A completed test with no pulse is
    reported as unexercised, never silently passed.
    """
    checks = {}
    result = {'schema': REVISION, 'trigger': copy.deepcopy(trigger),
              'performance_claim': False, 'window_extension_permitted': False,
              'scope': 'one-shot admission-triggered client canary; native observations required'}
    checks['triggered_on_nonempty_admission'] = bool(trigger and trigger.get('admit'))
    if not checks['triggered_on_nonempty_admission']:
        return dict(result, passed=False, checks=checks,
                    failure_reasons=['no_nonempty_admission_trigger'])

    record = trigger['control']
    pid = record['plan_id']
    start = record['snapshot_ns']
    effective_not_before = record['not_before_ns']
    expiry = record['action_expires_ns']
    refcap = int(trigger['continuation_cap'])
    checks['window_positive_and_at_most_200ms'] = (
        0 < expiry - start <= 200_000_000 and start <= effective_not_before < expiry)
    checks['no_pre_admission_setup_pulse'] = all(
        nodes[k].get('ready_at') is not None
        and int(nodes[k]['ready_at'] * 1e9) <= start + 10_000
        for k in trigger['admit'])
    pulse_records = [p for p in controls if p.get('caps') == [256] * len(record['caps'])]
    checks['exactly_one_pulse_publication'] = len(pulse_records) == 1 and pulse_records[0]['plan_id'] == pid

    seq = [x for x in iterations if x.get('plan_id') == pid]
    def instant(x): return x.get('effective_ns', x['begin_ns'])
    before = [x for x in seq if effective_not_before <= instant(x) < expiry]
    after = [x for x in seq if instant(x) >= expiry]
    prework = [x for x in before if x.get('scheduled')]
    postwork = [x for x in after if x.get('scheduled')]
    checks['pulse_has_scheduled_prefill'] = any(x.get('prefill_logical_tokens', 0) > 0 for x in prework)
    checks['pulse_logically_binds_prefill'] = any(binding_evidence(x)['binding_candidate'] for x in prework)
    checks['continuation_has_scheduled_work'] = bool(postwork)
    checks['pulse_cap_and_phase_match'] = bool(before) and all(
        x.get('plan_reason') == 'lease_ok' and x.get('execution_phase') == 'current_action'
        and x.get('prefill_cap') == 256 for x in before)
    checks['expired_cap_and_phase_match'] = bool(after) and all(
        x.get('plan_reason') == 'lease_ok' and x.get('execution_phase') == 'common_continuation'
        and x.get('prefill_cap') == refcap for x in after)
    checks['backend_deadline_unchanged'] = bool(seq) and all(
        x.get('action_expires_ns') == expiry and x.get('snapshot_ns') == start for x in seq)

    sent = [p for p in flushes if p.get('plan_id') == pid]
    heartbeat = [p for p in sent if p.get('cause') == 'heartbeat' and p['issued_ns'] >= expiry]
    checks['heartbeat_keeps_original_deadline'] = bool(sent) and all(
        p['action_expires_ns'] == expiry and p['snapshot_ns'] == start
        and p['not_before_ns'] == effective_not_before for p in sent)
    checks['transport_heartbeat_continues_after_expiry'] = bool(heartbeat)
    checks['work_continues_after_heartbeat'] = bool(heartbeat) and any(
        instant(x) >= min(p['issued_ns'] for p in heartbeat) for x in postwork)

    result.update(passed=all(checks.values()), checks=checks,
                  failure_reasons=[k for k, v in checks.items() if not v],
                  plan_id=pid, window_ms=(expiry-start)/1e6,
                  pulse_work_iterations=len(prework), continuation_work_iterations=len(postwork),
                  pulse_prefill_tokens=sum(x.get('prefill_logical_tokens', 0) for x in prework),
                  pulse_binding_iterations=sum(binding_evidence(x)['binding_candidate'] for x in prework),
                  heartbeat_flushes_after_expiry=len(heartbeat),
                  first_effective_delay_ms=(min(instant(x) for x in prework)-start)/1e6 if prework else None,
                  immutable_deadline_ns=expiry)
    return result


def reuse_cache_chain(run_path, initial, cfg):
    """Reuse only a successful CACHE_CHAIN subcondition on the same live pod0.

    A previously FAILED overall canary is NOT rewritten. Its passed cache-only
    subcondition is explicitly cited by the new canary's result. A different
    process, hook, launch ID, or relevant configuration requires a new full run.
    """
    run = Path(run_path).resolve(strict=True)
    folder = run / 'CACHE_CHAIN'
    paths = [run/'environment.json', run/'canary_result.json', run/'cleanup.json',
             folder/'manifest.json', folder/'summary.json', folder/'status.json',
             folder/'cache_chain_evidence.json']
    contents = {str(p.relative_to(run)): load(p) for p in paths}
    env = contents['environment.json']
    old, now = env[0], initial[0]
    keys = ('pod', 'pid', 'boot_id', 'launch_id', 'hook_source_sha256', 'recovery_revision')
    if len(env) != 1 or len(initial) != 1 or any(old.get(k) is None or old.get(k) != now.get(k) for k in keys):
        raise ValueError('cache reuse requires the SAME pod0 process, launch and hook; run canary without --reuse-cache-chain')
    top = contents['canary_result.json']
    if top.get('version') != VERSION or top.get('hook_hash') != now['hook_source_sha256']:
        raise ValueError('cache evidence version/hook mismatch')
    if contents['cleanup.json'].get('backend_drained') is not True:
        raise ValueError('prior canary did not drain')
    if contents['CACHE_CHAIN/status.json'].get('execution') != 'complete':
        raise ValueError('prior CACHE_CHAIN did not complete')
    previous_cfg = contents['CACHE_CHAIN/manifest.json']['config']
    for k in ('model', 'replicas', 'workflow_count', 'hard_total', 'base_cap',
              'max_model_len', 'block_size', 'max_num_seqs', 'share_tokens', 'input_profile', 'topology'):
        if previous_cfg.get(k) != cfg.get(k):
            raise ValueError('cache reuse configuration mismatch: '+k)
    summary = contents['CACHE_CHAIN/summary.json']
    checks = [summary.get('condition_valid') is True,
              summary.get('budget_invariants') is True,
              summary.get('control_time_match') is True,
              summary.get('request_correlation') == '6/6',
              summary.get('requests') == 6, summary.get('workflows') == 1,
              summary.get('output_tokens') == 576,
              summary.get('native_next_batch_checked', 0) > 0,
              summary.get('native_next_batch_matches') == summary.get('native_next_batch_checked'),
              summary.get('root_to_branch_cache_links_verified', 0) >= 4,
              sum(bool(x.get('root_to_branch_evidence_pass')) for x in contents['CACHE_CHAIN/cache_chain_evidence.json']) >= 4]
    if not all(checks):
        raise ValueError('prior CACHE_CHAIN does not satisfy the existing acceptance contract')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return copy.deepcopy(summary), {
        'source_run': str(run), 'source_condition': str(folder),
        'source_overall_canary_pass': top.get('pass'),
        'cache_subcondition_passed': True, 'same_live_engine': True,
        'source_sha256': hashes, 'source_files_modified': False,
        'new_cache_chain_generation_requests': 0,
        'only_failed_lifetime_probe_will_be_rerun': True}
