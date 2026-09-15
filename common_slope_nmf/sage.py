"""SAGE updates for exponential complex-Gaussian variance models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .is_objective import is_divergence
from .preprocess import DecayFit, fit_coarse_decay, head_power, tail_power


@dataclass(frozen=True)
class AmplitudeSAGEResult:
    """Result of SAGE amplitude updates for supplied temporal atoms.

    Attributes
    ----------
    amplitudes
        Estimated positive variance amplitudes, shape ``(R, Q)``.
    variance
        Fitted variance ``amplitudes @ atoms``, shape ``(R, N)``.
    objective_history
        Summed IS divergence before the first sweep and after every complete
        SAGE sweep, shape ``(n_iter + 1,)``.
    n_iter
        Number of completed component sweeps.
    converged
        Whether the relative objective-decrease stopping rule was met.
    """

    amplitudes: NDArray[np.float64]
    variance: NDArray[np.float64]
    objective_history: NDArray[np.float64]
    n_iter: int
    converged: bool


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


@dataclass(frozen=True)
class DecaySAGEInit:
    """Data-derived starting point for :func:`decay_sage` variants.

    Attributes
    ----------
    rates_per_s
        Equal-per-component initial energy-decay rates, shape ``(F,K)``, in
        inverse seconds.
    amplitudes
        Equal-split initial variance amplitudes, shape ``(R,F,K)``.
    noise_floor
        Tail-average variance-floor initialization, shape ``(R,F)``.
    fit
        Unclipped pooled log-power decay fit and its frame diagnostics.
    """

    rates_per_s: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    noise_floor: NDArray[np.float64]
    fit: DecayFit


def _real_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values.")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_array(
    name: str, values: ArrayLike, ndim: int | None = None
) -> NDArray[np.float64]:
    array = _real_array(name, values)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return array


def _check_controls(max_iter: int, tol: float, min_amplitude: float) -> None:
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and non-negative.")
    if not np.isfinite(min_amplitude) or min_amplitude <= 0.0:
        raise ValueError("min_amplitude must be finite and positive.")


def _has_converged(
    history: list[float], tol: float, *, require_monotone: bool = True
) -> bool:
    previous, objective = history[-2:]
    decrease = previous - objective
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, previous)
    if decrease < -roundoff:
        if require_monotone:
            raise RuntimeError(
                "IS objective increased beyond floating-point tolerance."
            )
        return False
    return decrease <= tol * max(1.0, previous)


def _initial_amplitudes(
    observed_power: NDArray[np.float64],
    atoms: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_components = atoms.shape[0]
    mean_power = np.mean(observed_power, axis=1, keepdims=True)
    mean_atoms = np.mean(atoms, axis=1, keepdims=True).T
    return mean_power / (n_components * mean_atoms)


def _amplitude_sage_sweep(
    observed_power: NDArray[np.float64],
    atoms: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    min_amplitude: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one complete sequential amplitude-SAGE sweep."""

    updated = amplitudes.copy()
    variance = updated @ atoms
    for component_index in range(atoms.shape[0]):
        atom = atoms[component_index]
        component = updated[:, component_index, np.newaxis] * atom
        residual = np.maximum(variance - component, 0.0)
        gain = component / variance
        posterior_power = gain * (gain * observed_power + residual)
        updated[:, component_index] = np.maximum(
            np.mean(posterior_power / atom, axis=1), min_amplitude
        )
        variance = residual + updated[:, component_index, np.newaxis] * atom

    variance = updated @ atoms
    return updated, variance


def amplitude_sage(
    observed_power: ArrayLike,
    atoms: ArrayLike,
    initial_amplitudes: ArrayLike | None = None,
    *,
    max_iter: int = 2_000,
    tol: float = 1e-10,
    min_amplitude: float = np.finfo(np.float64).tiny,
) -> AmplitudeSAGEResult:
    """Estimate amplitudes for supplied positive temporal atoms.

    Parameters
    ----------
    observed_power
        Strictly positive observed energy/power, shape ``(R, N)``.
    atoms
        Supplied strictly positive unitless temporal atoms, shape ``(Q, N)``.
        A row of ones can represent a time-invariant variance floor.
    initial_amplitudes
        Optional positive variance amplitudes, shape ``(R, Q)``. If omitted,
        mean observed power is divided equally among the supplied atoms.
    max_iter
        Maximum number of complete component sweeps.
    tol
        Non-negative relative objective-decrease tolerance.
    min_amplitude
        Positive lower bound in variance units after every update.

    Returns
    -------
    AmplitudeSAGEResult
        Estimated amplitudes and variance, objective history, iteration count,
        and convergence flag.

    Notes
    -----
    Components are updated sequentially because SAGE immediately inserts each
    new component into the total variance. Operations within one component are
    vectorized over rows and frames. A simultaneous component-axis update would
    be an EM/Jacobi-style algorithm rather than this SAGE/Gauss--Seidel sweep.
    """

    power = _positive_array("observed_power", observed_power, ndim=2)
    temporal_atoms = _positive_array("atoms", atoms, ndim=2)
    if power.shape[1] != temporal_atoms.shape[1]:
        raise ValueError("observed_power and atoms must have the same frame count.")
    _check_controls(max_iter, tol, min_amplitude)

    expected_shape = (power.shape[0], temporal_atoms.shape[0])
    if initial_amplitudes is None:
        amplitudes = _initial_amplitudes(power, temporal_atoms)
    else:
        amplitudes = _positive_array(
            "initial_amplitudes", initial_amplitudes, ndim=2
        ).copy()
        if amplitudes.shape != expected_shape:
            raise ValueError(f"initial_amplitudes must have shape {expected_shape}.")

    amplitudes = np.maximum(amplitudes, min_amplitude)
    variance = amplitudes @ temporal_atoms
    history = [float(is_divergence(power, variance))]
    converged = False

    for _ in range(int(max_iter)):
        amplitudes, variance = _amplitude_sage_sweep(
            power, temporal_atoms, amplitudes, min_amplitude
        )
        history.append(float(is_divergence(power, variance)))
        if _has_converged(history, tol):
            converged = True
            break

    return AmplitudeSAGEResult(
        amplitudes=amplitudes.copy(),
        variance=variance.copy(),
        objective_history=np.asarray(history, dtype=np.float64),
        n_iter=len(history) - 1,
        converged=converged,
    )


def _broadcast_rate_bound(
    name: str, values: ArrayLike, shape: tuple[int, ...]
) -> NDArray[np.float64]:
    bound = _positive_array(name, values)
    try:
        return np.broadcast_to(bound, shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to shape {shape}.") from exc


def init_decay_sage(
    observed_power: ArrayLike,
    times_s: ArrayLike,
    n_components: int,
    *,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
    n_head_frames: int = 8,
    n_tail_frames: int = 8,
    floor_margin_db: float | None = 6.0,
    min_regression_frames: int = 3,
    min_amplitude: float = np.finfo(np.float64).tiny,
) -> DecaySAGEInit:
    """Construct a simple data-derived decay-SAGE starting point.

    Parameters
    ----------
    observed_power
        Positive observed energy/power, shape ``(R,F,N)``.
    times_s
        Non-negative elapsed frame times in seconds, shape ``(N,)``.
    n_components
        Number of decay components ``K``.
    rate_bounds_per_s
        Positive lower and upper energy-decay rates in inverse seconds, each
        broadcastable to ``(F,K)``. The single coarse rate is clipped to the
        intersection of all component bounds at each frequency.
    n_head_frames
        Number of leading frames averaged for the total initial amplitude.
    n_tail_frames
        Number of trailing frames averaged for the initial noise floor and
        excluded from the pooled log-power regression.
    floor_margin_db
        Optional non-negative power margin above the mean initialized floor
        used to select common regression frames. ``None`` uses every non-tail
        frame.
    min_regression_frames
        Minimum common regression frames required at each frequency.
    min_amplitude
        Positive lower bound in variance units for amplitudes and floors.

    Returns
    -------
    DecaySAGEInit
        Initial rates ``(F,K)``, amplitudes ``(R,F,K)``, floor ``(R,F)``, and
        the pooled coarse-fit diagnostics.

    Notes
    -----
    The leading-frame mean is split equally across components without
    subtracting the initialized floor. Every component at a frequency starts
    from the same clipped coarse decay rate.
    """

    power = _positive_array("observed_power", observed_power, ndim=3)
    times = _real_array("times_s", times_s)
    if times.ndim != 1 or times.shape[0] != power.shape[2]:
        raise ValueError("times_s must have shape (N,) matching observed_power.")
    if np.any(times < 0.0) or np.ptp(times) <= 0.0:
        raise ValueError("times_s must be non-negative and contain variation.")
    if isinstance(n_components, bool) or not isinstance(n_components, Integral):
        raise TypeError("n_components must be an integer.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if not np.isfinite(min_amplitude) or min_amplitude <= 0.0:
        raise ValueError("min_amplitude must be finite and positive.")
    if len(rate_bounds_per_s) != 2:
        raise ValueError("rate_bounds_per_s must contain (lower, upper).")

    n_rirs, n_frequencies, _ = power.shape
    rate_shape = (n_frequencies, int(n_components))
    lower = _broadcast_rate_bound("lower rate bound", rate_bounds_per_s[0], rate_shape)
    upper = _broadcast_rate_bound("upper rate bound", rate_bounds_per_s[1], rate_shape)
    if np.any(lower >= upper):
        raise ValueError("every lower rate bound must be below its upper bound.")
    common_lower = np.max(lower, axis=1)
    common_upper = np.min(upper, axis=1)
    if np.any(common_lower >= common_upper):
        raise ValueError(
            "component rate bounds must have a common interval at each "
            "frequency for equal-rate initialization."
        )

    noise_floor = np.maximum(tail_power(power, n_tail_frames), min_amplitude)
    fit = fit_coarse_decay(
        power,
        times,
        noise_floor=noise_floor,
        n_tail_frames=n_tail_frames,
        floor_margin_db=floor_margin_db,
        min_frames=min_regression_frames,
    )
    coarse_rate = np.clip(fit.rate_per_s, common_lower, common_upper)
    rates_per_s = np.repeat(coarse_rate[:, np.newaxis], int(n_components), axis=1)
    total_amplitude = head_power(power, n_head_frames)
    amplitudes = np.repeat(
        (total_amplitude / n_components)[:, :, np.newaxis],
        int(n_components),
        axis=2,
    )
    amplitudes = np.maximum(amplitudes, min_amplitude)
    if amplitudes.shape != (n_rirs, n_frequencies, int(n_components)):
        raise RuntimeError("internal amplitude initialization shape mismatch.")
    return DecaySAGEInit(
        rates_per_s=rates_per_s,
        amplitudes=amplitudes,
        noise_floor=noise_floor,
        fit=fit,
    )


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
