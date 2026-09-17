"""Profiled decay-rate updates for SAGE component M-steps."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ._util import _broadcast_rate_bound, _positive_array, _real_array

def _profile_statistics(
    posterior_power: NDArray[np.float64],
    surrogate_weights: NDArray[np.float64],
    times_s: NDArray[np.float64],
    rates_per_s: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    positive_posterior = np.maximum(posterior_power, np.finfo(np.float64).tiny)
    with np.errstate(divide="ignore"):
        log_surrogate_weights = np.log(surrogate_weights)
    log_weights = (
        np.log(positive_posterior)
        + log_surrogate_weights
        + rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )
    log_moments = logsumexp(log_weights, axis=2)
    normalized = np.exp(log_weights - log_moments[:, :, np.newaxis])
    weighted_time = np.sum(normalized * times_s, axis=2)
    weighted_time_squared = np.sum(normalized * times_s**2, axis=2)
    weight_sums = np.sum(surrogate_weights, axis=2)
    weighted_time_sums = np.sum(surrogate_weights * times_s, axis=2)
    gradient = np.sum(weight_sums * weighted_time - weighted_time_sums, axis=0)
    weighted_variance = weighted_time_squared - weighted_time**2
    hessian = np.sum(weight_sums * np.maximum(weighted_variance, 0.0), axis=0)
    return gradient, hessian, log_moments


def _validate_profile_update(
    posterior_power: ArrayLike,
    times_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
    surrogate_weights: ArrayLike | None,
    *,
    max_iter: int,
    tol: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    posterior = _positive_array("posterior_power", posterior_power, ndim=3)
    times = _real_array("times_s", times_s)
    if times.ndim != 1:
        raise ValueError("times_s must have one dimension.")
    if np.any(times < 0.0) or np.ptp(times) <= 0.0:
        raise ValueError("times_s must be non-negative and contain variation.")
    if posterior.shape[2] != times.size:
        raise ValueError("posterior_power and times_s must have the same frame count.")
    if surrogate_weights is None:
        weights = np.ones_like(posterior)
    else:
        supplied_weights = _real_array("surrogate_weights", surrogate_weights)
        if np.any(supplied_weights < 0.0):
            raise ValueError("surrogate_weights must be non-negative.")
        try:
            weights = np.broadcast_to(supplied_weights, posterior.shape).astype(
                np.float64, copy=True
            )
        except ValueError as exc:
            raise ValueError(
                "surrogate_weights must broadcast to posterior_power shape "
                f"{posterior.shape}."
            ) from exc
        if np.any(np.sum(weights, axis=2) <= 0.0):
            raise ValueError(
                "surrogate_weights must have positive frame sums for every (R, F) row."
            )
    rates = _positive_array("initial_rates_per_s", initial_rates_per_s, ndim=1).copy()
    if rates.shape != (posterior.shape[1],):
        raise ValueError("initial_rates_per_s must have shape (F,).")
    if len(rate_bounds_per_s) != 2:
        raise ValueError("rate_bounds_per_s must contain (lower, upper).")
    lower = _broadcast_rate_bound("lower rate bound", rate_bounds_per_s[0], rates.shape)
    upper = _broadcast_rate_bound("upper rate bound", rate_bounds_per_s[1], rates.shape)
    if np.any(lower >= upper):
        raise ValueError("every lower rate bound must be below its upper bound.")
    maximum_exponent = np.max(upper) * np.max(times)
    if maximum_exponent >= -np.log(np.finfo(np.float64).tiny):
        raise ValueError("upper rate bounds underflow exponential atoms over times_s.")
    if np.any((rates < lower) | (rates > upper)):
        raise ValueError("initial_rates_per_s must lie within rate bounds.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive.")
    return posterior, weights, times, rates, lower, upper


def _bounded_profile_setup(
    posterior_power: NDArray[np.float64],
    surrogate_weights: NDArray[np.float64],
    times_s: NDArray[np.float64],
    lower_per_s: NDArray[np.float64],
    upper_per_s: NDArray[np.float64],
) -> tuple[
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    lower_gradient, _, _ = _profile_statistics(
        posterior_power, surrogate_weights, times_s, lower_per_s
    )
    upper_gradient, _, _ = _profile_statistics(
        posterior_power, surrogate_weights, times_s, upper_per_s
    )
    interior = (lower_gradient < 0.0) & (upper_gradient > 0.0)
    rates = np.where(lower_gradient >= 0.0, lower_per_s, upper_per_s)
    return interior, rates, lower_per_s.copy(), upper_per_s.copy()


def _profile_amplitudes(
    posterior_power: NDArray[np.float64],
    surrogate_weights: NDArray[np.float64],
    times_s: NDArray[np.float64],
    rates_per_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    _, _, log_moments = _profile_statistics(
        posterior_power, surrogate_weights, times_s, rates_per_s
    )
    log_weight_sums = np.log(np.sum(surrogate_weights, axis=2))
    return np.exp(log_moments - log_weight_sums)


def update_rates_bisection(
    posterior_power: ArrayLike,
    times_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
    *,
    surrogate_weights: ArrayLike | None = None,
    max_iter: int = 48,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimize profiled component surrogates by bracketed bisection.

    Parameters
    ----------
    posterior_power
        Positive SAGE posterior component powers, shape ``(R, F, N)``, in
        variance units.
    times_s
        Non-negative elapsed frame times in seconds, shape ``(N,)``.
    initial_rates_per_s
        Positive current decay rates in inverse seconds, shape ``(F,)``. They
        are validated for API consistency but bisection uses the full bracket.
    rate_bounds_per_s
        Positive lower and upper rate bounds, each broadcastable to ``(F,)``.
    surrogate_weights
        Optional non-negative fixed M-step weights, broadcastable to
        ``(R,F,N)``, with a positive frame sum for every ``(R,F)`` row.
        ``None`` uses unit weights and recovers ordinary SAGE.
    max_iter
        Maximum number of interval halvings.
    tol
        Relative bracket-width tolerance in rate units.

    Returns
    -------
    rates_per_s, amplitudes
        Profile-optimal bounded rates, shape ``(F,)``, and their exact
        unit-origin variance amplitudes, shape ``(R, F)``.
    """

    posterior, weights, times, _, lower, upper = _validate_profile_update(
        posterior_power,
        times_s,
        initial_rates_per_s,
        rate_bounds_per_s,
        surrogate_weights,
        max_iter=max_iter,
        tol=tol,
    )
    interior, rates, bracket_lower, bracket_upper = _bounded_profile_setup(
        posterior, weights, times, lower, upper
    )

    for _ in range(int(max_iter)):
        midpoint = 0.5 * (bracket_lower + bracket_upper)
        midpoint_gradient, _, _ = _profile_statistics(
            posterior, weights, times, midpoint
        )
        move_lower = midpoint_gradient < 0.0
        bracket_lower = np.where(interior & move_lower, midpoint, bracket_lower)
        bracket_upper = np.where(interior & ~move_lower, midpoint, bracket_upper)
        relative_width = (bracket_upper - bracket_lower) / np.maximum(
            1.0, np.abs(midpoint)
        )
        if np.all(~interior | (relative_width <= tol)):
            break

    rates = np.where(interior, 0.5 * (bracket_lower + bracket_upper), rates)
    amplitudes = _profile_amplitudes(posterior, weights, times, rates)
    return rates, amplitudes


def update_rates_newton(
    posterior_power: ArrayLike,
    times_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
    *,
    surrogate_weights: ArrayLike | None = None,
    max_iter: int = 24,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimize profiled component surrogates by safeguarded Newton steps.

    Parameters
    ----------
    posterior_power
        Positive SAGE posterior component powers, shape ``(R, F, N)``, in
        variance units.
    times_s
        Non-negative elapsed frame times in seconds, shape ``(N,)``.
    initial_rates_per_s
        Positive current decay rates in inverse seconds, shape ``(F,)``.
    rate_bounds_per_s
        Positive lower and upper rate bounds, each broadcastable to ``(F,)``.
    surrogate_weights
        Optional non-negative fixed M-step weights, broadcastable to
        ``(R,F,N)``, with a positive frame sum for every ``(R,F)`` row.
        ``None`` uses unit weights and recovers ordinary SAGE.
    max_iter
        Maximum number of safeguarded Newton iterations.
    tol
        Relative rate-step and bracket-width tolerance.

    Returns
    -------
    rates_per_s, amplitudes
        Profile-optimal bounded rates, shape ``(F,)``, and their exact
        unit-origin variance amplitudes, shape ``(R, F)``.

    Notes
    -----
    Each iteration proposes ``rate - gradient / hessian`` using the analytic
    profile Hessian. A proposal outside its valid bracket, or one with unusable
    curvature, is replaced by the bracket midpoint.
    """

    posterior, weights, times, initial, lower, upper = _validate_profile_update(
        posterior_power,
        times_s,
        initial_rates_per_s,
        rate_bounds_per_s,
        surrogate_weights,
        max_iter=max_iter,
        tol=tol,
    )
    interior, boundary_rates, bracket_lower, bracket_upper = _bounded_profile_setup(
        posterior, weights, times, lower, upper
    )
    rates = np.where(interior, initial, boundary_rates)
    active = interior.copy()

    for _ in range(int(max_iter)):
        gradient, hessian, _ = _profile_statistics(posterior, weights, times, rates)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton_step = -gradient / hessian
        step_converged = (
            active
            & np.isfinite(newton_step)
            & (np.abs(newton_step) <= tol * np.maximum(1.0, np.abs(rates)))
        )
        searching = active & ~step_converged
        bracket_lower = np.where(searching & (gradient < 0.0), rates, bracket_lower)
        bracket_upper = np.where(searching & (gradient >= 0.0), rates, bracket_upper)

        midpoint = 0.5 * (bracket_lower + bracket_upper)
        proposal = rates + newton_step
        usable_newton = (
            searching
            & np.isfinite(proposal)
            & (hessian > np.finfo(np.float64).eps)
            & (proposal > bracket_lower)
            & (proposal < bracket_upper)
        )
        next_rates = np.where(usable_newton, proposal, midpoint)
        relative_width = (bracket_upper - bracket_lower) / np.maximum(
            1.0, np.abs(next_rates)
        )
        width_converged = searching & (relative_width <= tol)
        rates = np.where(searching, next_rates, rates)
        active &= ~(step_converged | width_converged)
        if not np.any(active):
            break

    amplitudes = _profile_amplitudes(posterior, weights, times, rates)
    return rates, amplitudes
