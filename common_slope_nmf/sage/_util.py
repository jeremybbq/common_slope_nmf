"""SAGE control helpers."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from ..util import _finite_real_array as _real_array, _positive_array


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
    previous, loss = history[-2:]
    decrease = previous - loss
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, previous)
    if decrease < -roundoff:
        if require_monotone:
            raise RuntimeError(
                "IS loss increased beyond floating-point tolerance."
            )
        return False
    return decrease <= tol * max(1.0, previous)


def _initial_amplitudes(
    observed_power: NDArray[np.float64],
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    K = features.shape[0]
    mean_power = np.mean(observed_power, axis=1, keepdims=True)
    mean_features = np.mean(features, axis=1, keepdims=True).T
    return mean_power / (K * mean_features)


def _rate_bound_pair(bounds: tuple[float, float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("rate_bounds_per_s must contain (lower, upper).")
    lower, upper = bounds
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not np.isscalar(lower)
        or not np.isscalar(upper)
    ):
        raise TypeError("rate bounds must be real scalars.")
    lower_per_s = float(lower)
    upper_per_s = float(upper)
    if not np.isfinite(lower_per_s) or lower_per_s <= 0.0:
        raise ValueError("lower rate bound must be finite and positive.")
    if not np.isfinite(upper_per_s) or upper_per_s <= 0.0:
        raise ValueError("upper rate bound must be finite and positive.")
    if lower_per_s >= upper_per_s:
        raise ValueError("every lower rate bound must be below its upper bound.")
    return lower_per_s, upper_per_s
