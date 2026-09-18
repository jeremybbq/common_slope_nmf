"""Itakura--Saito criterion for complex-Gaussian variance models."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Reduction = Literal["none", "sum", "mean"]


def _real_finite_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values.")

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _matching_power_and_variance(
    observed_power: ArrayLike,
    model_variance: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    power = _real_finite_array("observed_power", observed_power)
    variance = _real_finite_array("model_variance", model_variance)

    if np.any(power <= 0.0):
        raise ValueError("observed_power must contain only positive values.")
    if np.any(variance <= 0.0):
        raise ValueError("model_variance must contain only positive values.")
    if power.shape != variance.shape:
        raise ValueError("observed_power and model_variance must have the same shape.")
    return power, variance


def _reduce(
    values: NDArray[np.float64], reduction: Reduction
) -> float | NDArray[np.float64]:
    if reduction == "none":
        return values
    if reduction == "sum":
        return float(np.sum(values))
    if reduction == "mean":
        return float(np.mean(values))
    raise ValueError("reduction must be 'none', 'sum', or 'mean'.")


def is_divergence(
    observed_power: ArrayLike,
    model_variance: ArrayLike,
    *,
    reduction: Reduction = "sum",
) -> float | NDArray[np.float64]:
    """Evaluate the Itakura--Saito divergence ``d_IS(Y | V)``.

    Parameters
    ----------
    observed_power
        Strictly positive observed energy/power ``Y``.
    model_variance
        Strictly positive model variance ``V`` with the same shape and energy/power units as ``observed_power``.
    reduction
        ``"sum"`` or ``"mean"`` returns a scalar; ``"none"`` returns the elementwise divergence with the same shape as ``Y`` and ``V``.

    Returns
    -------
    float or ndarray
        ``Y / V - log(Y / V) - 1`` after the requested reduction. The divergence is dimensionless and invariant to a common positive scale applied to ``Y`` and ``V``.
    """

    power, variance = _matching_power_and_variance(observed_power, model_variance)
    log_ratio = np.log(power) - np.log(variance)
    values = np.expm1(log_ratio) - log_ratio
    return _reduce(values, reduction)
