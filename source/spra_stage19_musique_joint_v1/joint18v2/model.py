"""One model contract for proposal, rollout and measured-batch validation.

Phase coefficients are selected on archived execution data BEFORE a new suite.
The artifact is immutable during all methods. It contains no future test outcomes.
"""
from core18v2.service import features, predict, phase, coefficients
