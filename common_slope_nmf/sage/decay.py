"""Original and contribution-weighted decay SAGE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..loss import is_divergence
from ._util import (
    _check_controls,
    _has_converged,
    _positive_array,
    _rate_bound_pair,
    _real_array,
)
from .rates import update_rates_newton


@dataclass(frozen=True)
class DecayFit:
    """Result of joint SAGE decay-rate and amplitude estimation.

    Attributes
    ----------
    rates_per_s
        Estimated energy-decay rates in inverse seconds, shape ``(F, K)``.
    amplitudes
        Estimated unit-origin variance amplitudes, shape ``(R, F, K)``.
    noise_floor
        Estimated time-invariant variance floors, shape ``(R, F)``. This is identically zero when floor estimation is disabled.
    loss_history
        Summed IS divergence before the first iteration and after every complete SAGE iteration, shape ``(n_iter + 1,)``.
    rate_history_per_s
        Decay rates before the first iteration and after every complete SAGE iteration, shape ``(n_iter + 1, F, K)``, in inverse seconds.
    converged
        Whether the relative loss-decrease stopping rule was met.
    """

    rates_per_s: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    noise_floor: NDArray[np.float64]
    loss_history: NDArray[np.float64]
    rate_history_per_s: NDArray[np.float64]
    converged: bool


def fit_decay(
    observed_power: ArrayLike,
    frame_time_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    *,
    rate_bounds_per_s: tuple[float, float],
    initial_amplitudes: ArrayLike,
    initial_noise_floor: ArrayLike | None = None,
    estimate_noise_floor: bool = True,
    max_iter: int = 2_000,
    tol: float = 1e-6,
    min_amplitude: float = np.finfo(np.float64).tiny,
    rate_max_iter: int = 24,
    rate_tol: float = 1e-12,
    component_weight_power: float | None = None,
) -> DecayFit:
    """Estimate exponential decay rates, amplitudes, and a variance floor.

    Parameters
    ----------
    observed_power
        Positive observed energy/power, shape ``(R, F, N)``.
    frame_time_s
        Non-negative elapsed frame times in seconds, shape ``(N,)``. At least two distinct times are required. Frame zero is the decay origin.
    initial_rates_per_s
        Positive initial energy-decay rates, shape ``(F, K)``.
    rate_bounds_per_s
        Pair ``(lower, upper)`` of positive rate bounds in inverse seconds.
    initial_amplitudes
        Positive unit-origin variance amplitudes, shape ``(R, F, K)``.
    initial_noise_floor
        Positive variance floors, shape ``(R, F)``. Required when ``estimate_noise_floor`` is true; ignored otherwise.
    estimate_noise_floor
        Whether to include and update a time-invariant variance component.
    max_iter
        Maximum number of complete component iterations.
    tol
        Non-negative relative loss-decrease tolerance.
    min_amplitude
        Positive lower bound in variance units after amplitude/floor updates.
    rate_max_iter
        Maximum inner Newton iterations per component update.
    rate_tol
        Positive relative inner rate tolerance.
    component_weight_power
        Optional non-negative contribution-weight exponent ``p``. ``None`` uses original unweighted SAGE. A supplied ``p`` freezes ``w = rho ** p`` from the current Wiener strength ``rho_k = v_k / V`` before each decay-component M-step. ``p = 0`` recovers the original decay-component updates. The time-invariant noise component, when enabled, retains the original unweighted SAGE update.

    Returns
    -------
    DecayFit
        Estimated rates in inverse seconds, amplitudes and floors in variance units, and per-iteration loss/rate histories. Fitted variance is ``exponential_variance(frame_time_s, rates_per_s, amplitudes, noise_floor)``. The number of completed iterations is ``len(loss_history) - 1``.

    Notes
    -----
    The outer component loop is intentional. Each posterior is conditioned on the variance containing all earlier updates from the same iteration, which is the defining SAGE schedule. For each component, calculations are vectorized over all RIRs, frequencies, and frames; scalar profile-rate searches for all frequencies are also performed together with masked array operations.

    A supplied ``component_weight_power`` is an experimental weighted loss, not an exact SAGE auxiliary function for the observed-data likelihood. Consequently its observed IS loss history is monitored but is not guaranteed to be monotone.
    """

    power = _positive_array("observed_power", observed_power, ndim=3)
    times = _real_array("frame_time_s", frame_time_s)
    if times.ndim != 1:
        raise ValueError("frame_time_s must have one dimension.")
    if np.any(times < 0.0) or np.ptp(times) <= 0.0:
        raise ValueError("frame_time_s must be non-negative and contain variation.")
    if power.shape[2] != times.size:
        raise ValueError(
            "observed_power and frame_time_s must have the same frame count."
        )
    rates = _positive_array("initial_rates_per_s", initial_rates_per_s, ndim=2).copy()
    n_rirs, n_frequencies, _ = power.shape
    if rates.shape[0] != n_frequencies:
        raise ValueError("initial_rates_per_s must have shape (F, K) matching power.")
    K = rates.shape[1]
    if K == 0:
        raise ValueError("at least one decay component is required.")
    _check_controls(max_iter, tol, min_amplitude)
    if isinstance(rate_max_iter, bool) or not np.isscalar(rate_max_iter):
        raise TypeError("rate_max_iter must be an integer.")
    rate_max_iter = int(rate_max_iter)
    if rate_max_iter <= 0:
        raise ValueError("rate_max_iter must be positive.")
    if not np.isscalar(rate_tol) or isinstance(rate_tol, bool):
        raise TypeError("rate_tol must be a real scalar.")
    rate_tol = float(rate_tol)
    if not np.isfinite(rate_tol) or rate_tol <= 0.0:
        raise ValueError("rate_tol must be finite and positive.")
    if component_weight_power is not None:
        if isinstance(component_weight_power, bool) or not np.isscalar(
            component_weight_power
        ):
            raise TypeError("component_weight_power must be a real scalar.")
        component_weight_power = float(component_weight_power)
        if not np.isfinite(component_weight_power) or component_weight_power < 0.0:
            raise ValueError("component_weight_power must be finite and non-negative.")
    lower, upper = _rate_bound_pair(rate_bounds_per_s)
    if upper * np.max(times) >= -np.log(np.finfo(np.float64).tiny):
        raise ValueError(
            "upper rate bound underflows exponential features over frame_time_s."
        )
    if np.any((rates < lower) | (rates > upper)):
        raise ValueError("initial_rates_per_s must lie within rate bounds.")

    amplitude_shape = (n_rirs, n_frequencies, K)
    amplitudes = _positive_array(
        "initial_amplitudes", initial_amplitudes, ndim=3
    ).copy()
    if amplitudes.shape != amplitude_shape:
        raise ValueError(f"initial_amplitudes must have shape {amplitude_shape}.")
    amplitudes = np.maximum(amplitudes, min_amplitude)

    floor_shape = (n_rirs, n_frequencies)
    if estimate_noise_floor:
        if initial_noise_floor is None:
            raise ValueError(
                "initial_noise_floor is required when estimate_noise_floor is true."
            )
        noise_floor = _positive_array(
            "initial_noise_floor", initial_noise_floor, ndim=2
        ).copy()
        if noise_floor.shape != floor_shape:
            raise ValueError(f"initial_noise_floor must have shape {floor_shape}.")
        noise_floor = np.maximum(noise_floor, min_amplitude)
    else:
        noise_floor = np.zeros(floor_shape, dtype=np.float64)

    features = np.exp(-rates[:, :, np.newaxis] * times)
    variance = np.einsum("rfk,fkn->rfn", amplitudes, features)
    variance += noise_floor[:, :, np.newaxis]
    loss_history = [float(is_divergence(power, variance))]
    rate_history = [rates.copy()]
    converged = False

    for _ in range(1, int(max_iter) + 1):
        for k in range(K):
            feature = features[:, k, :]
            component = (
                amplitudes[:, :, k, np.newaxis]
                * feature[np.newaxis, :, :]
            )
            residual = np.maximum(variance - component, 0.0)
            gain = component / variance
            posterior_power = gain * (gain * power + residual)
            if component_weight_power is None:
                surrogate_weights = None
            else:
                surrogate_weights = np.power(gain, component_weight_power)

            updated_rates, updated_amplitudes = update_rates_newton(
                posterior_power,
                times,
                rates[:, k],
                (lower, upper),
                surrogate_weights=surrogate_weights,
                max_iter=rate_max_iter,
                tol=rate_tol,
            )
            rates[:, k] = updated_rates
            amplitudes[:, :, k] = np.maximum(
                updated_amplitudes, min_amplitude
            )
            features[:, k, :] = np.exp(
                -updated_rates[:, np.newaxis] * times
            )
            variance = (
                residual
                + amplitudes[:, :, k, np.newaxis]
                * features[np.newaxis, :, k, :]
            )

        if estimate_noise_floor:
            floor_component = noise_floor[:, :, np.newaxis]
            residual = np.maximum(variance - floor_component, 0.0)
            gain = floor_component / variance
            posterior_power = gain * (gain * power + residual)
            noise_floor = np.maximum(np.mean(posterior_power, axis=2), min_amplitude)
            variance = residual + noise_floor[:, :, np.newaxis]

        variance = np.einsum("rfk,fkn->rfn", amplitudes, features)
        variance += noise_floor[:, :, np.newaxis]
        loss_history.append(float(is_divergence(power, variance)))
        rate_history.append(rates.copy())
        if _has_converged(
            loss_history,
            tol,
            require_monotone=component_weight_power is None,
        ):
            converged = True
            break

    return DecayFit(
        rates_per_s=rates.copy(),
        amplitudes=amplitudes.copy(),
        noise_floor=noise_floor.copy(),
        loss_history=np.asarray(loss_history, dtype=np.float64),
        rate_history_per_s=np.asarray(rate_history, dtype=np.float64),
        converged=converged,
    )
