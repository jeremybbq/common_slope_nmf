"""SAGE updates for exponential complex-Gaussian variance models."""

from .amplitudes import AmplitudeSAGEResult, amplitude_sage
from .decay import DecaySAGEResult, DecaySAGEUpdateDiagnostics, decay_sage
from .init import DecaySAGEInit, init_decay_sage
from .rates import update_rates_bisection, update_rates_newton
from .weighted import pseudo_decay_sage

__all__ = [
    "AmplitudeSAGEResult",
    "DecaySAGEInit",
    "DecaySAGEResult",
    "DecaySAGEUpdateDiagnostics",
    "amplitude_sage",
    "decay_sage",
    "init_decay_sage",
    "pseudo_decay_sage",
    "update_rates_bisection",
    "update_rates_newton",
]
