"""Contribution-weighted CW-SAGE decay updates."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .decay import DecaySAGEResult, _decay_sage_impl

def cw_decay_sage(
    observed_power: ArrayLike,
    times_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    *,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
    component_weight_power: float = 1.0,
    initial_amplitudes: ArrayLike | None = None,
    initial_noise_floor: ArrayLike | None = None,
    estimate_noise_floor: bool = True,
    max_iter: int = 2_000,
    tol: float = 1e-6,
    decay_tol: float | None = None,
    min_amplitude: float = np.finfo(np.float64).tiny,
    rate_method: Literal["newton", "bisection"] = "newton",
    rate_max_iter: int | None = None,
    rate_tol: float | None = None,
    diagnostic_interval: int | None = None,
) -> DecaySAGEResult:
    """Run contribution-weighted CW-SAGE decay updates.

    Input and output shapes and units match :func:`decay_sage`. Before each
    decay-component M-step, this variant evaluates the current Wiener strength
    ``rho_k = v_k / V`` and freezes ``w_rfnk = rho_rfnk ** p``, where ``p`` is
    the non-negative ``component_weight_power``. For fixed posterior power
    ``S`` and candidate energy-decay rate ``lambda``, it profiles

    ``M_rf = sum_n w_rfn S_rfn exp(lambda * time_n)`` and
    ``a_rf = M_rf / sum_n w_rfn``.

    The weighted time sums, gradient, and Hessian use the same frozen weights.
    The time-invariant noise component, when enabled, retains the ordinary
    unweighted SAGE update. Setting ``component_weight_power=0`` recovers the
    ordinary decay-component updates.

    Notes
    -----
    This is an experimental weighted objective, not an exact SAGE auxiliary
    function for the observed-data likelihood. Consequently its observed IS
    objective history is monitored but is not guaranteed to be monotone.
    ``decay_tol`` optionally stops when decay estimates stabilize and is
    combined with the observed-objective ``tol`` using an OR rule.
    """

    return _decay_sage_impl(
        observed_power,
        times_s,
        initial_rates_per_s,
        rate_bounds_per_s=rate_bounds_per_s,
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=estimate_noise_floor,
        max_iter=max_iter,
        tol=tol,
        decay_tol=decay_tol,
        min_amplitude=min_amplitude,
        rate_method=rate_method,
        rate_max_iter=rate_max_iter,
        rate_tol=rate_tol,
        diagnostic_interval=diagnostic_interval,
        component_weight_power=component_weight_power,
    )
