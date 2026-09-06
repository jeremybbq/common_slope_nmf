"""Safeguarded SQUAREM acceleration for ordinary SAGE fixed-point maps."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .is_objective import is_divergence
from .sage import (
    _amplitude_sage_sweep,
    _broadcast_rate_bound,
    _check_controls,
    _initial_amplitudes,
    _positive_array,
    _real_array,
    decay_sage,
)


@dataclass(frozen=True)
class AmplitudeSQUAREMResult:
    """Result of safeguarded SQUAREM-accelerated amplitude SAGE.

    Attributes
    ----------
    amplitudes
        Estimated positive variance amplitudes, shape ``(R, Q)``.
    variance
        Fitted variance ``amplitudes @ atoms``, shape ``(R, N)``.
    objective_history
        Summed IS divergence at the initial state and after each retained
        accelerated or fallback state, shape ``(H,)``.
    sweep_evaluation_history
        Cumulative complete SAGE-sweep evaluations corresponding to
        ``objective_history``, shape ``(H,)``. The initial state is zero.
    fixed_point_residual_history
        Maximum per-RIR relative Euclidean residual of one complete SAGE
        sweep, shape ``(D,)``. A residual is recorded whenever the first
        sweep of a SQUAREM cycle is evaluated.
    n_sweep_evaluations
        Number of complete SAGE-sweep map evaluations. This is the relevant
        work count when comparing against :func:`amplitude_sage`.
    n_accepted_extrapolations
        Total number of per-RIR extrapolated blocks accepted after a
        stabilizing SAGE sweep.
    n_rejected_extrapolations
        Total number of attempted per-RIR extrapolated blocks rejected for
        infeasibility, non-finite values, or an increased rowwise IS
        objective.
    converged
        Whether both the objective-change and fixed-point-residual stopping
        conditions were met within the sweep-evaluation budget.
    """

    amplitudes: NDArray[np.float64]
    variance: NDArray[np.float64]
    objective_history: NDArray[np.float64]
    sweep_evaluation_history: NDArray[np.int64]
    fixed_point_residual_history: NDArray[np.float64]
    n_sweep_evaluations: int
    n_accepted_extrapolations: int
    n_rejected_extrapolations: int
    converged: bool


@dataclass(frozen=True)
class DecaySQUAREMResult:
    """Result of safeguarded SQUAREM-accelerated decay SAGE.

    Attributes
    ----------
    rates_per_s
        Estimated energy-decay rates in inverse seconds, shape ``(F, K)``.
    amplitudes
        Estimated unit-origin variance amplitudes, shape ``(R, F, K)``.
    noise_floor
        Estimated time-invariant variance floors, shape ``(R, F)``.
    variance
        Fitted total variance, shape ``(R, F, N)``.
    objective_history
        Summed original IS divergence at the initial state and after every
        retained accelerated or fallback state, shape ``(H,)``.
    sweep_evaluation_history
        Cumulative complete decay-SAGE sweep-map evaluations corresponding
        to ``objective_history``, shape ``(H,)``. The initial state is zero.
    rate_history_per_s
        Rates corresponding to ``objective_history``, shape ``(H, F, K)``,
        in inverse seconds.
    fixed_point_residual_history
        Maximum dimensionless per-frequency residual of one complete decay
        SAGE sweep, shape ``(D,)``.
    n_sweep_evaluations
        Number of complete decay-SAGE sweep-map evaluations.
    n_accepted_extrapolations
        Total number of accepted per-frequency extrapolated blocks.
    n_rejected_extrapolations
        Total number of attempted per-frequency extrapolated blocks rejected
        by feasibility or original-objective safeguards.
    converged
        Whether both stopping conditions were met within the sweep budget.
    """

    rates_per_s: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    noise_floor: NDArray[np.float64]
    variance: NDArray[np.float64]
    objective_history: NDArray[np.float64]
    sweep_evaluation_history: NDArray[np.int64]
    rate_history_per_s: NDArray[np.float64]
    fixed_point_residual_history: NDArray[np.float64]
    n_sweep_evaluations: int
    n_accepted_extrapolations: int
    n_rejected_extrapolations: int
    converged: bool


def _rowwise_is_objective(
    observed_power: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = is_divergence(observed_power, variance, reduction="none")
    return np.sum(values, axis=1)


def _relative_row_residual(
    current: NDArray[np.float64], updated: NDArray[np.float64]
) -> NDArray[np.float64]:
    difference_norm = np.linalg.norm(updated - current, axis=1)
    current_norm = np.linalg.norm(current, axis=1)
    return difference_norm / np.maximum(1.0, current_norm)


def amplitude_squarem(
    observed_power: ArrayLike,
    atoms: ArrayLike,
    initial_amplitudes: ArrayLike | None = None,
    *,
    max_sweep_evaluations: int = 2_000,
    objective_tol: float = 1e-10,
    fixed_point_tol: float = 1e-8,
    min_amplitude: float = np.finfo(np.float64).tiny,
    initial_step_max: float = 1.0,
    step_factor: float = 4.0,
    step_max_limit: float = 1_000.0,
) -> AmplitudeSQUAREMResult:
    """Estimate supplied-atom amplitudes with safeguarded SQUAREM-S3.

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
    max_sweep_evaluations
        Maximum number of complete sequential SAGE-sweep evaluations. A
        SQUAREM cycle uses two sweeps without extrapolation and three when a
        stabilizing extrapolation sweep is attempted.
    objective_tol
        Non-negative relative rowwise IS-objective decrease required for
        convergence together with ``fixed_point_tol``.
    fixed_point_tol
        Non-negative tolerance for the maximum per-RIR relative Euclidean
        residual of one complete SAGE sweep.
    min_amplitude
        Positive lower bound in variance units after SAGE updates and in
        feasibility checks for extrapolated amplitudes.
    initial_step_max
        Initial upper bound for the dimensionless SQUAREM step length. Values
        below one are invalid because a step of one equals two SAGE sweeps.
    step_factor
        Factor greater than one used to expand or contract the adaptive
        per-RIR maximum step length.
    step_max_limit
        Finite upper bound on every dimensionless SQUAREM step length.

    Returns
    -------
    AmplitudeSQUAREMResult
        Estimated amplitudes and variance, original IS-objective and
        fixed-point-residual histories, SAGE-sweep work count, safeguard
        counts, and convergence flag.

    Notes
    -----
    One fixed-point evaluation is one complete Gauss--Seidel/SAGE component
    sweep. The S3 step length is computed independently for every RIR. An
    extrapolated row is retained only after a stabilizing SAGE sweep and only
    when its original rowwise IS objective does not increase. Otherwise the
    row falls back to the valid two-sweep SAGE state. The IS criterion and all
    component E/M updates are unchanged.
    """

    power = _positive_array("observed_power", observed_power, ndim=2)
    temporal_atoms = _positive_array("atoms", atoms, ndim=2)
    if power.shape[1] != temporal_atoms.shape[1]:
        raise ValueError("observed_power and atoms must have the same frame count.")
    if isinstance(max_sweep_evaluations, bool) or not isinstance(
        max_sweep_evaluations, Integral
    ):
        raise TypeError("max_sweep_evaluations must be an integer.")
    if max_sweep_evaluations <= 0:
        raise ValueError("max_sweep_evaluations must be positive.")
    _check_controls(1, objective_tol, min_amplitude)
    if not np.isfinite(fixed_point_tol) or fixed_point_tol < 0.0:
        raise ValueError("fixed_point_tol must be finite and non-negative.")
    if not np.isfinite(initial_step_max) or initial_step_max < 1.0:
        raise ValueError("initial_step_max must be finite and at least one.")
    if not np.isfinite(step_factor) or step_factor <= 1.0:
        raise ValueError("step_factor must be finite and greater than one.")
    if not np.isfinite(step_max_limit) or step_max_limit < initial_step_max:
        raise ValueError("step_max_limit must be finite and at least initial_step_max.")

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
    objective_rows = _rowwise_is_objective(power, variance)
    objective_history = [float(np.sum(objective_rows))]
    sweep_evaluation_history = [0]
    residual_history: list[float] = []
    step_max = np.full(power.shape[0], initial_step_max, dtype=np.float64)
    n_sweep_evaluations = 0
    n_accepted = 0
    n_rejected = 0
    converged = False

    while n_sweep_evaluations < int(max_sweep_evaluations):
        state_0 = amplitudes
        objective_0 = objective_rows
        state_1, variance_1 = _amplitude_sage_sweep(
            power, temporal_atoms, state_0, min_amplitude
        )
        n_sweep_evaluations += 1
        residual_rows = _relative_row_residual(state_0, state_1)
        residual_history.append(float(np.max(residual_rows)))
        objective_1 = _rowwise_is_objective(power, variance_1)
        relative_decrease = (objective_0 - objective_1) / np.maximum(1.0, objective_0)
        if np.all(relative_decrease <= objective_tol) and np.all(
            residual_rows <= fixed_point_tol
        ):
            amplitudes = state_1
            variance = variance_1
            objective_rows = objective_1
            objective_history.append(float(np.sum(objective_rows)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            converged = True
            break

        if n_sweep_evaluations >= int(max_sweep_evaluations):
            amplitudes = state_1
            variance = variance_1
            objective_rows = objective_1
            objective_history.append(float(np.sum(objective_rows)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            break

        state_2, variance_2 = _amplitude_sage_sweep(
            power, temporal_atoms, state_1, min_amplitude
        )
        n_sweep_evaluations += 1
        first_difference = state_1 - state_0
        second_difference = state_2 - 2.0 * state_1 + state_0
        first_norm = np.linalg.norm(first_difference, axis=1)
        second_norm = np.linalg.norm(second_difference, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_step = first_norm / second_norm
        raw_step = np.where(np.isfinite(raw_step), raw_step, step_max)
        step = np.clip(raw_step, 1.0, step_max)
        attempted = step > 1.0 + 16.0 * np.finfo(np.float64).eps

        if not np.any(attempted):
            amplitudes = state_2
            variance = variance_2
            objective_rows = _rowwise_is_objective(power, variance)
            hit_cap = raw_step >= step_max
            step_max = np.where(
                hit_cap,
                np.minimum(step_max_limit, step_factor * step_max),
                step_max,
            )
            objective_history.append(float(np.sum(objective_rows)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            continue

        proposal = (
            state_0
            + 2.0 * step[:, np.newaxis] * first_difference
            + step[:, np.newaxis] ** 2 * second_difference
        )
        feasible = (
            attempted
            & np.all(np.isfinite(proposal), axis=1)
            & np.all(proposal >= min_amplitude, axis=1)
        )
        stabilization_input = state_2.copy()
        stabilization_input[feasible] = proposal[feasible]

        if n_sweep_evaluations >= int(max_sweep_evaluations):
            amplitudes = state_2
            variance = variance_2
            objective_rows = _rowwise_is_objective(power, variance)
            n_rejected += int(np.count_nonzero(attempted))
            objective_history.append(float(np.sum(objective_rows)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            break

        stabilized, stabilized_variance = _amplitude_sage_sweep(
            power, temporal_atoms, stabilization_input, min_amplitude
        )
        n_sweep_evaluations += 1
        stabilized_objective = _rowwise_is_objective(power, stabilized_variance)
        roundoff = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, objective_0)
        accepted = feasible & (stabilized_objective <= objective_0 + roundoff)
        rejected = attempted & ~accepted
        n_accepted += int(np.count_nonzero(accepted))
        n_rejected += int(np.count_nonzero(rejected))

        amplitudes = state_2.copy()
        amplitudes[accepted] = stabilized[accepted]
        variance = variance_2.copy()
        variance[accepted] = stabilized_variance[accepted]
        objective_rows = _rowwise_is_objective(power, variance)
        objective_history.append(float(np.sum(objective_rows)))
        sweep_evaluation_history.append(n_sweep_evaluations)

        hit_cap = raw_step >= step_max
        step_max = np.where(
            accepted & hit_cap,
            np.minimum(step_max_limit, step_factor * step_max),
            step_max,
        )
        step_max = np.where(
            rejected & hit_cap,
            np.maximum(1.0, step_max / step_factor),
            step_max,
        )

    return AmplitudeSQUAREMResult(
        amplitudes=amplitudes.copy(),
        variance=variance.copy(),
        objective_history=np.asarray(objective_history, dtype=np.float64),
        sweep_evaluation_history=np.asarray(
            sweep_evaluation_history, dtype=np.int64
        ),
        fixed_point_residual_history=np.asarray(residual_history, dtype=np.float64),
        n_sweep_evaluations=n_sweep_evaluations,
        n_accepted_extrapolations=n_accepted,
        n_rejected_extrapolations=n_rejected,
        converged=converged,
    )


def _frequency_is_objective(
    observed_power: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = is_divergence(observed_power, variance, reduction="none")
    return np.sum(values, axis=(0, 2))


def _scaled_decay_difference_norm(
    rate_difference: NDArray[np.float64],
    amplitude_difference: NDArray[np.float64],
    floor_difference: NDArray[np.float64],
    rate_scale_per_s: NDArray[np.float64],
    power_scale: NDArray[np.float64],
    *,
    estimate_noise_floor: bool,
) -> NDArray[np.float64]:
    squared_norm = np.sum((rate_difference / rate_scale_per_s) ** 2, axis=1)
    squared_norm += np.sum(
        (amplitude_difference / power_scale[:, :, np.newaxis]) ** 2,
        axis=(0, 2),
    )
    if estimate_noise_floor:
        squared_norm += np.sum((floor_difference / power_scale) ** 2, axis=0)
    return np.sqrt(squared_norm)


def decay_squarem(
    observed_power: ArrayLike,
    times_s: ArrayLike,
    initial_rates_per_s: ArrayLike,
    *,
    rate_bounds_per_s: tuple[ArrayLike, ArrayLike],
    initial_amplitudes: ArrayLike | None = None,
    initial_noise_floor: ArrayLike | None = None,
    estimate_noise_floor: bool = True,
    max_sweep_evaluations: int = 2_000,
    objective_tol: float = 1e-6,
    fixed_point_tol: float = 1e-6,
    min_amplitude: float = np.finfo(np.float64).tiny,
    rate_method: Literal["newton", "bisection"] = "newton",
    rate_max_iter: int | None = None,
    rate_tol: float | None = None,
    initial_step_max: float = 1.0,
    step_factor: float = 4.0,
    step_max_limit: float = 1_000.0,
) -> DecaySQUAREMResult:
    """Estimate decay parameters with safeguarded full-sweep SQUAREM-S3.

    Parameters
    ----------
    observed_power
        Positive observed energy/power, shape ``(R,F,N)``.
    times_s
        Non-negative elapsed frame times in seconds, shape ``(N,)``.
    initial_rates_per_s
        Positive initial energy-decay rates, shape ``(F,K)``, in inverse
        seconds.
    rate_bounds_per_s
        Positive lower and upper rate bounds in inverse seconds, each
        broadcastable to ``(F,K)``.
    initial_amplitudes
        Optional positive unit-origin variance amplitudes, shape ``(R,F,K)``.
    initial_noise_floor
        Optional positive variance floors, shape ``(R,F)``.
    estimate_noise_floor
        Whether a time-invariant variance component is fitted.
    max_sweep_evaluations
        Maximum number of complete ordinary decay-SAGE sweep evaluations.
    objective_tol
        Non-negative relative per-frequency IS-objective decrease required
        for convergence together with ``fixed_point_tol``.
    fixed_point_tol
        Non-negative tolerance for the maximum dimensionless per-frequency
        residual of one complete SAGE sweep.
    min_amplitude
        Positive amplitude and floor bound in variance units.
    rate_method, rate_max_iter, rate_tol
        Inner profiled-rate solver and its controls, as in :func:`decay_sage`.
    initial_step_max
        Initial maximum dimensionless SQUAREM step; at least one.
    step_factor
        Factor greater than one for adaptive step-bound changes.
    step_max_limit
        Finite upper bound on every SQUAREM step length.

    Returns
    -------
    DecaySQUAREMResult
        Rates, amplitudes, floor, variance, original-objective and rate
        histories, fixed-point residuals, work counts, and convergence state.

    Notes
    -----
    One fixed-point map evaluation is one complete ordinary, unweighted
    decay-SAGE sweep. S3 steps and safeguards are independent by frequency;
    each frequency block contains all of its shared rates, RIR-dependent
    amplitudes, and floors. Step norms divide rates by their fixed bound
    widths and amplitudes/floors by initial mean-power scales, making the norm
    dimensionless. Extrapolation remains in physical coordinates. A
    stabilized block is accepted only if it is feasible and does not increase
    that frequency's original IS objective.
    """

    power = _positive_array("observed_power", observed_power, ndim=3)
    times = _real_array("times_s", times_s)
    if times.ndim != 1 or times.shape[0] != power.shape[2]:
        raise ValueError("times_s must have shape (N,) matching observed_power.")
    if np.any(times < 0.0) or np.ptp(times) <= 0.0:
        raise ValueError("times_s must be non-negative and contain variation.")
    rates = _positive_array("initial_rates_per_s", initial_rates_per_s, ndim=2).copy()
    n_rirs, n_frequencies, _ = power.shape
    if rates.shape[0] != n_frequencies or rates.shape[1] == 0:
        raise ValueError(
            "initial_rates_per_s must have shape (F,K) with at least one component."
        )
    if isinstance(max_sweep_evaluations, bool) or not isinstance(
        max_sweep_evaluations, Integral
    ):
        raise TypeError("max_sweep_evaluations must be an integer.")
    if max_sweep_evaluations <= 0:
        raise ValueError("max_sweep_evaluations must be positive.")
    _check_controls(1, objective_tol, min_amplitude)
    if not np.isfinite(fixed_point_tol) or fixed_point_tol < 0.0:
        raise ValueError("fixed_point_tol must be finite and non-negative.")
    if not np.isfinite(initial_step_max) or initial_step_max < 1.0:
        raise ValueError("initial_step_max must be finite and at least one.")
    if not np.isfinite(step_factor) or step_factor <= 1.0:
        raise ValueError("step_factor must be finite and greater than one.")
    if not np.isfinite(step_max_limit) or step_max_limit < initial_step_max:
        raise ValueError("step_max_limit must be finite and at least initial_step_max.")
    if rate_method not in {"newton", "bisection"}:
        raise ValueError("rate_method must be 'newton' or 'bisection'.")
    if len(rate_bounds_per_s) != 2:
        raise ValueError("rate_bounds_per_s must contain (lower, upper).")

    lower = _broadcast_rate_bound("lower rate bound", rate_bounds_per_s[0], rates.shape)
    upper = _broadcast_rate_bound("upper rate bound", rate_bounds_per_s[1], rates.shape)
    if np.any(lower >= upper):
        raise ValueError("every lower rate bound must be below its upper bound.")
    if np.any((rates < lower) | (rates > upper)):
        raise ValueError("initial_rates_per_s must lie within rate bounds.")
    if np.max(upper) * np.max(times) >= -np.log(np.finfo(np.float64).tiny):
        raise ValueError("upper rate bounds underflow exponential atoms over times_s.")

    n_components = rates.shape[1]
    amplitude_shape = (n_rirs, n_frequencies, n_components)
    if initial_amplitudes is None:
        initial_share = n_components + int(estimate_noise_floor)
        amplitudes = np.broadcast_to(
            np.mean(power, axis=2, keepdims=True) / initial_share,
            amplitude_shape,
        ).copy()
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

    def sweep(
        sweep_rates: NDArray[np.float64],
        sweep_amplitudes: NDArray[np.float64],
        sweep_floor: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        result = decay_sage(
            power,
            times,
            sweep_rates,
            rate_bounds_per_s=(lower, upper),
            initial_amplitudes=sweep_amplitudes,
            initial_noise_floor=(sweep_floor if estimate_noise_floor else None),
            estimate_noise_floor=estimate_noise_floor,
            max_iter=1,
            tol=0.0,
            min_amplitude=min_amplitude,
            rate_method=rate_method,
            rate_max_iter=rate_max_iter,
            rate_tol=rate_tol,
        )
        return (
            result.rates_per_s,
            result.amplitudes,
            result.noise_floor,
            result.variance,
        )

    atoms = np.exp(-rates[:, :, np.newaxis] * times)
    variance = np.einsum("rfk,fkn->rfn", amplitudes, atoms)
    variance += noise_floor[:, :, np.newaxis]
    objective_frequency = _frequency_is_objective(power, variance)
    objective_history = [float(np.sum(objective_frequency))]
    sweep_evaluation_history = [0]
    rate_history = [rates.copy()]
    residual_history: list[float] = []
    rate_scale = upper - lower
    power_scale = np.maximum(np.mean(power, axis=2), min_amplitude)
    step_max = np.full(n_frequencies, initial_step_max, dtype=np.float64)
    n_sweep_evaluations = 0
    n_accepted = 0
    n_rejected = 0
    converged = False

    while n_sweep_evaluations < int(max_sweep_evaluations):
        rates_0, amplitudes_0, floor_0 = rates, amplitudes, noise_floor
        objective_0 = objective_frequency
        rates_1, amplitudes_1, floor_1, variance_1 = sweep(
            rates_0, amplitudes_0, floor_0
        )
        n_sweep_evaluations += 1
        residual_frequency = _scaled_decay_difference_norm(
            rates_1 - rates_0,
            amplitudes_1 - amplitudes_0,
            floor_1 - floor_0,
            rate_scale,
            power_scale,
            estimate_noise_floor=estimate_noise_floor,
        )
        residual_history.append(float(np.max(residual_frequency)))
        objective_1 = _frequency_is_objective(power, variance_1)
        relative_decrease = (objective_0 - objective_1) / np.maximum(1.0, objective_0)
        if np.all(relative_decrease <= objective_tol) and np.all(
            residual_frequency <= fixed_point_tol
        ):
            rates, amplitudes, noise_floor, variance = (
                rates_1,
                amplitudes_1,
                floor_1,
                variance_1,
            )
            objective_frequency = objective_1
            objective_history.append(float(np.sum(objective_frequency)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            rate_history.append(rates.copy())
            converged = True
            break
        if n_sweep_evaluations >= int(max_sweep_evaluations):
            rates, amplitudes, noise_floor, variance = (
                rates_1,
                amplitudes_1,
                floor_1,
                variance_1,
            )
            objective_frequency = objective_1
            objective_history.append(float(np.sum(objective_frequency)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            rate_history.append(rates.copy())
            break

        rates_2, amplitudes_2, floor_2, variance_2 = sweep(
            rates_1, amplitudes_1, floor_1
        )
        n_sweep_evaluations += 1
        rate_first = rates_1 - rates_0
        amplitude_first = amplitudes_1 - amplitudes_0
        floor_first = floor_1 - floor_0
        rate_second = rates_2 - 2.0 * rates_1 + rates_0
        amplitude_second = amplitudes_2 - 2.0 * amplitudes_1 + amplitudes_0
        floor_second = floor_2 - 2.0 * floor_1 + floor_0
        first_norm = _scaled_decay_difference_norm(
            rate_first,
            amplitude_first,
            floor_first,
            rate_scale,
            power_scale,
            estimate_noise_floor=estimate_noise_floor,
        )
        second_norm = _scaled_decay_difference_norm(
            rate_second,
            amplitude_second,
            floor_second,
            rate_scale,
            power_scale,
            estimate_noise_floor=estimate_noise_floor,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_step = first_norm / second_norm
        raw_step = np.where(np.isfinite(raw_step), raw_step, step_max)
        step = np.clip(raw_step, 1.0, step_max)
        attempted = step > 1.0 + 16.0 * np.finfo(np.float64).eps

        if not np.any(attempted):
            rates, amplitudes, noise_floor, variance = (
                rates_2,
                amplitudes_2,
                floor_2,
                variance_2,
            )
            objective_frequency = _frequency_is_objective(power, variance)
            hit_cap = raw_step >= step_max
            step_max = np.where(
                hit_cap,
                np.minimum(step_max_limit, step_factor * step_max),
                step_max,
            )
            objective_history.append(float(np.sum(objective_frequency)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            rate_history.append(rates.copy())
            continue

        rate_proposal = (
            rates_0
            + 2.0 * step[:, np.newaxis] * rate_first
            + step[:, np.newaxis] ** 2 * rate_second
        )
        amplitude_step = step[np.newaxis, :, np.newaxis]
        amplitude_proposal = (
            amplitudes_0
            + 2.0 * amplitude_step * amplitude_first
            + amplitude_step**2 * amplitude_second
        )
        floor_step = step[np.newaxis, :]
        floor_proposal = (
            floor_0 + 2.0 * floor_step * floor_first + floor_step**2 * floor_second
        )
        feasible = (
            attempted
            & np.all(np.isfinite(rate_proposal), axis=1)
            & np.all(rate_proposal >= lower, axis=1)
            & np.all(rate_proposal <= upper, axis=1)
            & np.all(np.isfinite(amplitude_proposal), axis=(0, 2))
            & np.all(amplitude_proposal >= min_amplitude, axis=(0, 2))
        )
        if estimate_noise_floor:
            feasible &= np.all(np.isfinite(floor_proposal), axis=0)
            feasible &= np.all(floor_proposal >= min_amplitude, axis=0)

        stabilization_rates = rates_2.copy()
        stabilization_amplitudes = amplitudes_2.copy()
        stabilization_floor = floor_2.copy()
        stabilization_rates[feasible] = rate_proposal[feasible]
        stabilization_amplitudes[:, feasible, :] = amplitude_proposal[:, feasible, :]
        if estimate_noise_floor:
            stabilization_floor[:, feasible] = floor_proposal[:, feasible]

        if n_sweep_evaluations >= int(max_sweep_evaluations):
            rates, amplitudes, noise_floor, variance = (
                rates_2,
                amplitudes_2,
                floor_2,
                variance_2,
            )
            objective_frequency = _frequency_is_objective(power, variance)
            n_rejected += int(np.count_nonzero(attempted))
            objective_history.append(float(np.sum(objective_frequency)))
            sweep_evaluation_history.append(n_sweep_evaluations)
            rate_history.append(rates.copy())
            break

        (
            stabilized_rates,
            stabilized_amplitudes,
            stabilized_floor,
            stabilized_variance,
        ) = sweep(
            stabilization_rates,
            stabilization_amplitudes,
            stabilization_floor,
        )
        n_sweep_evaluations += 1
        stabilized_objective = _frequency_is_objective(power, stabilized_variance)
        roundoff = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, objective_0)
        accepted = feasible & (stabilized_objective <= objective_0 + roundoff)
        rejected = attempted & ~accepted
        n_accepted += int(np.count_nonzero(accepted))
        n_rejected += int(np.count_nonzero(rejected))

        rates = rates_2.copy()
        rates[accepted] = stabilized_rates[accepted]
        amplitudes = amplitudes_2.copy()
        amplitudes[:, accepted, :] = stabilized_amplitudes[:, accepted, :]
        noise_floor = floor_2.copy()
        noise_floor[:, accepted] = stabilized_floor[:, accepted]
        variance = variance_2.copy()
        variance[:, accepted, :] = stabilized_variance[:, accepted, :]
        objective_frequency = _frequency_is_objective(power, variance)
        objective_history.append(float(np.sum(objective_frequency)))
        sweep_evaluation_history.append(n_sweep_evaluations)
        rate_history.append(rates.copy())

        hit_cap = raw_step >= step_max
        step_max = np.where(
            accepted & hit_cap,
            np.minimum(step_max_limit, step_factor * step_max),
            step_max,
        )
        step_max = np.where(
            rejected & hit_cap,
            np.maximum(1.0, step_max / step_factor),
            step_max,
        )

    return DecaySQUAREMResult(
        rates_per_s=rates.copy(),
        amplitudes=amplitudes.copy(),
        noise_floor=noise_floor.copy(),
        variance=variance.copy(),
        objective_history=np.asarray(objective_history, dtype=np.float64),
        sweep_evaluation_history=np.asarray(
            sweep_evaluation_history, dtype=np.int64
        ),
        rate_history_per_s=np.asarray(rate_history, dtype=np.float64),
        fixed_point_residual_history=np.asarray(residual_history, dtype=np.float64),
        n_sweep_evaluations=n_sweep_evaluations,
        n_accepted_extrapolations=n_accepted,
        n_rejected_extrapolations=n_rejected,
        converged=converged,
    )
