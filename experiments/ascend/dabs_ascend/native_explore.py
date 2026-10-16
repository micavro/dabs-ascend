"""Explicit exploratory runs: tolerate measured fit error, never certify it."""
import copy
from pathlib import Path

from .common import read_json
from .native_calibration import fit_model
from .native_gate import evaluate_local, check_live
from .native_timing import TARGET
from dabs_native.contracts import resource_values


def evaluate(paths):
    gate = evaluate_local(paths)
    # evaluate_local still reconstructs every sample and every probe from raw events.
    allowed = ['calibration: frozen_independent_holdout']
    if gate['missing'] not in ([], allowed):
        raise ValueError('exploration still requires raw mechanism/timing evidence: ' + str(gate['missing']))
    directory = Path(paths['native-calibrate'])
    recorded = read_json(directory / 'native_profile.json')
    caps = read_json(directory / 'capabilities.json')
    model = fit_model(read_json(directory / 'calibration_samples.json'), caps)
    if len(caps) not in (1, 4) or model['timing_target'] != TARGET:
        raise ValueError('exploration requires full async calibration of replica 0 or all four replicas')
    for key in ('models', 'validation', 'identities', 'resources', 'timing_target',
                'host_step_equivalence', 'thresholds_predeclared', 'accounting_model'):
        if recorded.get(key) != model.get(key):
            raise ValueError('calibration record differs from raw refit: ' + key)
    if any(not reason.endswith(': frozen holdout error thresholds exceeded') for reason in model['failures']):
        raise ValueError('only prediction error may be deferred: ' + str(model['failures']))
    gate.update(pass_=False, live_verified=False, exploratory=True,
                exploratory_allowed=True, calibration_error_deferred=model['failures'],
                calibration_accuracy_pass=model['pass'], formal_acceptance=False,
                model_transfer_from_replica_0=len(caps) == 1)
    gate['pass'] = gate.pop('pass_')
    return gate, model


async def verify(native, paths):
    gate, model = evaluate(paths)
    await check_live(native, paths, model, allow_single=True)
    original = copy.deepcopy(model)
    if len(model['identities']) == 1:
        if any(resource_values(c['resources']) != resource_values(model['resources'][0])
               for c in native.capabilities):
            raise ValueError('single-replica exploratory model requires identical effective resources on all services')
        model['models'] = {str(r): copy.deepcopy(original['models']['0']) for r in range(4)}
        # Retain only actually measured validation. Never invent r1/r2/r3 measurements.
        model['prediction_validation_map'] = {str(r): '0' for r in range(4)}
        model['identities'] = [c['identity'] for c in native.capabilities]
        model['resources'] = [c['resources'] for c in native.capabilities]
    model.update(frozen=False, exploratory=True, formal_acceptance=False,
                 calibration_source_identities=original['identities'],
                 model_transfer_from_replica_0=len(original['identities']) == 1)
    gate.update(mechanisms_live_verified=True, scope='exploratory native execution; prediction accuracy NOT accepted')
    return gate, model
