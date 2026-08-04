"""Research package for constrained multi-slope IS-NMF decay estimation."""

from .is_objective import gaussian_variance_nll, is_divergence
from .model import (
    exponential_atoms,
    exponential_variance,
    rate_to_t60,
    t60_to_rate,
)
from .sage import FixedDictionarySAGEResult, fixed_dictionary_sage
from .synth import sample_complex_gaussian, sample_power

__all__ = [
    "FixedDictionarySAGEResult",
    "exponential_atoms",
    "exponential_variance",
    "fixed_dictionary_sage",
    "gaussian_variance_nll",
    "is_divergence",
    "rate_to_t60",
    "sample_complex_gaussian",
    "sample_power",
    "t60_to_rate",
]
