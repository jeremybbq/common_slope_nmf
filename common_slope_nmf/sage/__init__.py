"""SAGE updates for exponential complex-Gaussian variance models."""

from .amplitudes import AmplitudeFit, fit_amplitudes
from .decay import DecayFit, fit_decay
from .init import init_decay
from .rates import update_rates_newton

__all__ = [
    "AmplitudeFit",
    "DecayFit",
    "fit_amplitudes",
    "fit_decay",
    "init_decay",
    "update_rates_newton",
]
