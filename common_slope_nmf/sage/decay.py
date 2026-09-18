"""Ordinary and shared decay-SAGE implementation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..loss import is_divergence
from ._util import (
    _broadcast_rate_bound,
    _check_controls,
    _has_converged,
    _positive_array,
    _real_array,
)
from .rates import update_rates_bisection, update_rates_newton

@dataclass(frozen=True)
class DecaySAGEUpdateDiagnostics:
    """Intermediate quantities from selected complete SAGE sweeps.

    Attributes
    ----------
    sweep_indices
        One-based outer sweep index for every recorded state, shape ``(D,)``.
    scaled_total_error
        Post-sweep scaled total error ``epsilon = Y / V - 1``, shape
        ``(D, R, F, N)``.
    component_weight
        Post-sweep Wiener-style weights ``rho_k = v_k / V``, shape
        ``(D, R, F, K + 1, N)``. The component axis contains the ``K`` decay
        components followed by the constant noise component. It sums to one
        along that axis when floor estimation is enabled.
    profile_moment
        Weighted moments
        ``M_rfk(lambda_new) = sum_n w_rfnk S_rfnk exp(lambda_new tau_n)``
        used during the component updates in that sweep, shape
        ``(D, R, F, K)``.
    profile_weight_sum
        Corresponding fixed M-step weight sums ``sum_n w_rfnk``, shape
        ``(D, R, F, K)``. Thus the profiled amplitude before application of
        ``min_amplitude`` is ``profile_moment / profile_weight_sum``.

    Notes
    -----
    Error and component weights are evaluated after the complete sequential
    sweep so every weight in a saved state has the same denominator. Profile
    moments retain the actual Gauss--Seidel values used while updating each
    component. Full histories can be large; use ``diagnostic_interval`` in
    :func:`decay_sage` to select saved sweeps.
    """

    sweep_indices: NDArray[np.int64]
    scaled_total_error: NDArray[np.float64]
    component_weight: NDArray[np.float64]
    profile_moment: NDArray[np.float64]
    profile_weight_sum: NDArray[np.float64]


@dataclass(frozen=True)
class DecaySAGEResult:
    """Result of joint SAGE decay-rate and amplitude estimation.

    Attributes
    ----------
    rates_per_s
        Estimated energy-decay rates in inverse seconds, shape ``(F, K)``.
    amplitudes
        Estimated unit-origin variance amplitudes, shape ``(R, F, K)``.
    noise_floor
        Estimated time-invariant variance floors, shape ``(R, F)``. This is
        identically zero when floor estimation is disabled.
    variance
        Fitted total variance, shape ``(R, F, N)``.
    objective_history
        Summed IS divergence before the first sweep and after every complete
        SAGE sweep, shape ``(n_iter + 1,)``.
    rate_history_per_s
        Decay rates before the first sweep and after every complete SAGE
        sweep, shape ``(n_iter + 1, F, K)``, in inverse seconds.
    update_diagnostics
        Selected per-component intermediate quantities, or ``None`` when
        diagnostic collection is disabled.
    n_iter
        Number of completed component sweeps.
    converged
        Whether the relative objective-decrease rule or an enabled outer
        decay-stability rule was met.
    """

    rates_per_s: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    noise_floor: NDArray[np.float64]
    variance: NDArray[np.float64]
    objective_history: NDArray[np.float64]
    rate_history_per_s: NDArray[np.float64]
    update_diagnostics: DecaySAGEUpdateDiagnostics | None
    n_iter: int
    converged: bool

def _decay_sage_impl(
    observed_power: ArrayLike,
    times_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    *,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
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
    component_weight_power: float | None = None,
) -> DecaySAGEResult:
    """Implement exact or contribution-weighted decay SAGE.

    Parameters
    ----------
    observed_power
        Positive observed energy/power, shape ``(R, F, N)``.
    times_s
        Non-negative elapsed frame times in seconds, shape ``(N,)``. At least
        two distinct times are required.
    initial_rates_per_s
        Positive initial energy-decay rates, shape ``(F, K)``.
    rate_bounds_per_s
        Pair ``(lower, upper)`` of positive rates in inverse seconds. Each
        value must broadcast to ``(F, K)`` and lower must be below upper.
    initial_amplitudes
        Optional positive unit-origin variance amplitudes, shape ``(R,F,K)``.
    initial_noise_floor
        Optional positive variance floors, shape ``(R,F)``. Used only when
        ``estimate_noise_floor`` is true.
    estimate_noise_floor
        Whether to include and update a time-invariant variance component.
    max_iter
        Maximum number of complete component sweeps.
    tol
        Non-negative relative objective-decrease tolerance.
    decay_tol
        Optional non-negative outer decay-stability tolerance. After each
        complete sweep, the maximum absolute log-ratio between the new and
        previous decay rates is evaluated. This equals the corresponding
        absolute log-ratio for energy-``T60`` and approximates relative
        fractional change near convergence. The fit stops when either this
        condition or the observed-objective condition is met. ``None`` uses
        only the objective condition.
    min_amplitude
        Positive lower bound in variance units after amplitude/floor updates.
    rate_method
        Profile-rate solver: safeguarded ``"newton"`` (default) or bracketed
        ``"bisection"``.
    rate_max_iter
        Maximum inner rate iterations. Defaults to 24 for Newton and 48 for
        bisection.
    rate_tol
        Positive relative inner rate tolerance. Defaults to ``1e-12`` for
        both methods.
    diagnostic_interval
        Optional positive number of outer sweeps between saved per-component
        diagnostics. The first sweep is always recorded when enabled, and
        ``1`` records every component update; ``None`` disables collection.
        Each saved error or component-weight field has shape ``(R,F,N)``, so
        choose the interval with memory use in mind.

    Returns
    -------
    DecaySAGEResult
        Estimated rates in inverse seconds, amplitudes and floors in variance
        units, fitted variance, objective history, and stopping diagnostics.

    Notes
    -----
    The outer component loop is intentional. Each posterior is conditioned on
    the variance containing all earlier updates from the same sweep, which is
    the defining SAGE schedule. For each component, calculations are vectorized
    over all RIRs, frequencies, and frames; scalar profile-rate searches for all
    frequencies are also performed together with masked array operations.
    """

    power = _positive_array("observed_power", observed_power, ndim=3)
    times = _real_array("times_s", times_s)
    if times.ndim != 1:
        raise ValueError("times_s must have one dimension.")
    if np.any(times < 0.0) or np.ptp(times) <= 0.0:
        raise ValueError("times_s must be non-negative and contain variation.")
    if power.shape[2] != times.size:
        raise ValueError("observed_power and times_s must have the same frame count.")
    rates = _positive_array("initial_rates_per_s", initial_rates_per_s, ndim=2).copy()
    n_rirs, n_frequencies, _ = power.shape
    if rates.shape[0] != n_frequencies:
        raise ValueError("initial_rates_per_s must have shape (F, K) matching power.")
    n_components = rates.shape[1]
    if n_components == 0:
        raise ValueError("at least one decay component is required.")
    _check_controls(max_iter, tol, min_amplitude)
    if decay_tol is not None:
        if not np.isscalar(decay_tol) or isinstance(decay_tol, bool):
            raise TypeError("decay_tol must be a real scalar or None.")
        decay_tol = float(decay_tol)
        if not np.isfinite(decay_tol) or decay_tol < 0.0:
            raise ValueError("decay_tol must be finite and non-negative.")
    if rate_method not in {"newton", "bisection"}:
        raise ValueError("rate_method must be 'newton' or 'bisection'.")
    if rate_max_iter is None:
        rate_max_iter = 24 if rate_method == "newton" else 48
    if rate_tol is None:
        rate_tol = 1e-12
    if diagnostic_interval is not None:
        if isinstance(diagnostic_interval, bool) or not isinstance(
            diagnostic_interval, Integral
        ):
            raise TypeError("diagnostic_interval must be an integer or None.")
        if diagnostic_interval <= 0:
            raise ValueError("diagnostic_interval must be positive.")
    if component_weight_power is not None:
        if isinstance(component_weight_power, bool) or not np.isscalar(
            component_weight_power
        ):
            raise TypeError("component_weight_power must be a real scalar.")
        component_weight_power = float(component_weight_power)
        if not np.isfinite(component_weight_power) or component_weight_power < 0.0:
            raise ValueError("component_weight_power must be finite and non-negative.")
    if len(rate_bounds_per_s) != 2:
        raise ValueError("rate_bounds_per_s must contain (lower, upper).")

    lower = _broadcast_rate_bound("lower rate bound", rate_bounds_per_s[0], rates.shape)
    upper = _broadcast_rate_bound("upper rate bound", rate_bounds_per_s[1], rates.shape)
    if np.any(lower >= upper):
        raise ValueError("every lower rate bound must be below its upper bound.")
    maximum_exponent = np.max(upper * np.max(times))
    if maximum_exponent >= -np.log(np.finfo(np.float64).tiny):
        raise ValueError("upper rate bounds underflow exponential atoms over times_s.")
    if np.any((rates < lower) | (rates > upper)):
        raise ValueError("initial_rates_per_s must lie within rate bounds.")

    amplitude_shape = (n_rirs, n_frequencies, n_components)
    if initial_amplitudes is None:
        initial_share = n_components + int(estimate_noise_floor)
        amplitudes = np.mean(power, axis=2, keepdims=True)
        amplitudes = np.broadcast_to(amplitudes / initial_share, amplitude_shape).copy()
    else:
        amplitudes = _positive_array(
            "initial_amplitudes", initial_amplitudes, ndim=3
        ).copy()
        if amplitudes.shape != amplitude_shape:
            raise ValueError(f"initial_amplitudes must have shape {amplitude_shape}.")
    amplitudes = np.maximum(amplitudes, min_amplitude)

    floor_shape = (n_rirs, n_frequencies)
    if estimate_noise_floor:
        if initial_noise_floor is None:
            noise_floor = np.mean(power, axis=2) / (n_components + 1)
        else:
            noise_floor = _positive_array(
                "initial_noise_floor", initial_noise_floor, ndim=2
            ).copy()
            if noise_floor.shape != floor_shape:
                raise ValueError(f"initial_noise_floor must have shape {floor_shape}.")
        noise_floor = np.maximum(noise_floor, min_amplitude)
    else:
        noise_floor = np.zeros(floor_shape, dtype=np.float64)

    atoms = np.exp(-rates[:, :, np.newaxis] * times)
    variance = np.einsum("rfk,fkn->rfn", amplitudes, atoms)
    variance += noise_floor[:, :, np.newaxis]
    history = [float(is_divergence(power, variance))]
    rate_history = [rates.copy()]
    diagnostic_sweeps: list[int] = []
    diagnostic_errors: list[NDArray[np.float64]] = []
    diagnostic_weights: list[NDArray[np.float64]] = []
    diagnostic_moments: list[NDArray[np.float64]] = []
    diagnostic_profile_weight_sums: list[NDArray[np.float64]] = []
    converged = False
    rate_updater = (
        update_rates_newton if rate_method == "newton" else update_rates_bisection
    )

    for sweep_index in range(1, int(max_iter) + 1):
        record_diagnostics = diagnostic_interval is not None and (
            sweep_index == 1 or sweep_index % diagnostic_interval == 0
        )
        sweep_moments: list[NDArray[np.float64]] = []
        sweep_profile_weight_sums: list[NDArray[np.float64]] = []
        for component_index in range(n_components):
            atom = atoms[:, component_index, :]
            component = (
                amplitudes[:, :, component_index, np.newaxis] * atom[np.newaxis, :, :]
            )
            residual = np.maximum(variance - component, 0.0)
            gain = component / variance
            posterior_power = gain * (gain * power + residual)
            if component_weight_power is None:
                surrogate_weights = None
                profile_weight_sums = np.full(
                    posterior_power.shape[:2], times.size, dtype=np.float64
                )
            else:
                surrogate_weights = np.power(gain, component_weight_power)
                profile_weight_sums = np.sum(surrogate_weights, axis=2)

            updated_rates, updated_amplitudes = rate_updater(
                posterior_power,
                times,
                rates[:, component_index],
                (
                    lower[:, component_index],
                    upper[:, component_index],
                ),
                surrogate_weights=surrogate_weights,
                max_iter=rate_max_iter,
                tol=rate_tol,
            )
            rates[:, component_index] = updated_rates
            amplitudes[:, :, component_index] = np.maximum(
                updated_amplitudes, min_amplitude
            )
            if record_diagnostics:
                sweep_moments.append((profile_weight_sums * updated_amplitudes).copy())
                sweep_profile_weight_sums.append(profile_weight_sums.copy())
            atoms[:, component_index, :] = np.exp(-updated_rates[:, np.newaxis] * times)
            variance = (
                residual
                + amplitudes[:, :, component_index, np.newaxis]
                * atoms[np.newaxis, :, component_index, :]
            )

        if estimate_noise_floor:
            floor_component = noise_floor[:, :, np.newaxis]
            residual = np.maximum(variance - floor_component, 0.0)
            gain = floor_component / variance
            posterior_power = gain * (gain * power + residual)
            noise_floor = np.maximum(np.mean(posterior_power, axis=2), min_amplitude)
            variance = residual + noise_floor[:, :, np.newaxis]

        variance = np.einsum("rfk,fkn->rfn", amplitudes, atoms)
        variance += noise_floor[:, :, np.newaxis]
        if record_diagnostics:
            decay_components = (
                amplitudes[:, :, :, np.newaxis] * atoms[np.newaxis, :, :, :]
            )
            decay_weights = decay_components / variance[:, :, np.newaxis, :]
            floor_weight = (
                noise_floor[:, :, np.newaxis, np.newaxis]
                / variance[:, :, np.newaxis, :]
            )
            diagnostic_sweeps.append(sweep_index)
            diagnostic_errors.append((power / variance - 1.0).copy())
            diagnostic_weights.append(
                np.concatenate((decay_weights, floor_weight), axis=2)
            )
            diagnostic_moments.append(np.stack(sweep_moments, axis=2))
            diagnostic_profile_weight_sums.append(
                np.stack(sweep_profile_weight_sums, axis=2)
            )
        history.append(float(is_divergence(power, variance)))
        rate_history.append(rates.copy())
        objective_converged = _has_converged(
            history,
            tol,
            require_monotone=component_weight_power is None,
        )
        decay_converged = decay_tol is not None and np.max(
            np.abs(np.log(rate_history[-1] / rate_history[-2]))
        ) <= decay_tol
        if objective_converged or decay_converged:
            converged = True
            break

    if diagnostic_interval is None:
        update_diagnostics = None
    else:
        if diagnostic_errors:
            scaled_total_error = np.stack(diagnostic_errors)
            component_weight = np.stack(diagnostic_weights)
            profile_moment = np.stack(diagnostic_moments)
            profile_weight_sum = np.stack(diagnostic_profile_weight_sums)
        else:
            scaled_total_error = np.empty((0, *power.shape), dtype=np.float64)
            component_weight = np.empty(
                (0, n_rirs, n_frequencies, n_components + 1, times.size),
                dtype=np.float64,
            )
            profile_moment = np.empty(
                (0, n_rirs, n_frequencies, n_components),
                dtype=np.float64,
            )
            profile_weight_sum = np.empty(
                (0, n_rirs, n_frequencies, n_components),
                dtype=np.float64,
            )
        update_diagnostics = DecaySAGEUpdateDiagnostics(
            sweep_indices=np.asarray(diagnostic_sweeps, dtype=np.int64),
            scaled_total_error=scaled_total_error,
            component_weight=component_weight,
            profile_moment=profile_moment,
            profile_weight_sum=profile_weight_sum,
        )

    return DecaySAGEResult(
        rates_per_s=rates.copy(),
        amplitudes=amplitudes.copy(),
        noise_floor=noise_floor.copy(),
        variance=variance.copy(),
        objective_history=np.asarray(history, dtype=np.float64),
        rate_history_per_s=np.asarray(rate_history, dtype=np.float64),
        update_diagnostics=update_diagnostics,
        n_iter=len(history) - 1,
        converged=converged,
    )


def decay_sage(
    observed_power: ArrayLike,
    times_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    *,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
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
    """Estimate exponential decay rates, amplitudes, and a variance floor.

    Parameters have the following shapes: positive observed power ``(R,F,N)``,
    non-negative frame times in seconds ``(N,)``, initial energy-decay rates
    in inverse seconds ``(F,K)``, optional unit-origin variance amplitudes
    ``(R,F,K)``, and an optional time-invariant variance floor ``(R,F)``.
    ``rate_bounds_per_s`` contains lower and upper bounds broadcastable to
    ``(F,K)``. The returned :class:`DecaySAGEResult` includes those fitted
    quantities, total variance ``(R,F,N)``, and per-sweep objective/rate
    histories.

    The components use the ordinary unweighted SAGE surrogate and are updated
    sequentially. ``rate_method`` selects safeguarded Newton (the default) or
    bracketed bisection for the profiled decay update. ``diagnostic_interval``
    optionally records the first and every requested complete sweep.
    ``decay_tol`` optionally stops when the decay estimates stabilize; this is
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
        component_weight_power=None,
    )
