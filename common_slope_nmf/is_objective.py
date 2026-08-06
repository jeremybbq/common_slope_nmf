"""Itakura--Saito criteria for complex-Gaussian variance models."""

from __future__ import annotations

from typing import Literal, overload

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


def _broadcast_power_and_variance(
    observed_power: ArrayLike,
    model_variance: ArrayLike,
    *,
    require_positive_power: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    power = _real_finite_array("observed_power", observed_power)
    variance = _real_finite_array("model_variance", model_variance)

    if require_positive_power and np.any(power <= 0.0):
        raise ValueError("observed_power must contain only positive values.")
    if not require_positive_power and np.any(power < 0.0):
        raise ValueError("observed_power must contain only non-negative values.")
    if np.any(variance <= 0.0):
        raise ValueError("model_variance must contain only positive values.")

    try:
        return np.broadcast_arrays(power, variance)
    except ValueError as exc:
        raise ValueError(
            "observed_power and model_variance must have "
            "broadcast-compatible shapes."
        ) from exc


@overload
def _reduce(
    values: NDArray[np.float64], reduction: Literal["none"]
) -> NDArray[np.float64]:
    ...


@overload
def _reduce(
    values: NDArray[np.float64], reduction: Literal["sum", "mean"]
) -> float:
    ...


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
        Strictly positive observed energy/power ``Y``, with arbitrary shape.
    model_variance
        Strictly positive model variance ``V``, broadcastable with
        ``observed_power`` and expressed in the same energy/power units.
    reduction
        ``"sum"`` or ``"mean"`` returns a scalar; ``"none"`` returns the
        elementwise divergence with the broadcast shape.

    Returns
    -------
    float or ndarray
        ``Y / V - log(Y / V) - 1`` after the requested reduction. The
        divergence is dimensionless and invariant to a common positive scale
        applied to ``Y`` and ``V``.
    """

    power, variance = _broadcast_power_and_variance(
        observed_power, model_variance, require_positive_power=True
    )
    ratio = power / variance
    offset = ratio - 1.0
    values = offset - np.log1p(offset)
    return _reduce(values, reduction)


def gaussian_variance_nll(
    observed_power: ArrayLike,
    model_variance: ArrayLike,
    *,
    reduction: Reduction = "sum",
) -> float | NDArray[np.float64]:
    """Evaluate the model-dependent complex-Gaussian variance criterion.

    Parameters
    ----------
    observed_power
        Non-negative observed energy/power ``Y``, with arbitrary shape.
    model_variance
        Strictly positive model variance ``V``, broadcastable with
        ``observed_power`` and expressed in the same energy/power units.
    reduction
        ``"sum"`` or ``"mean"`` returns a scalar; ``"none"`` returns the
        elementwise criterion with the broadcast shape.

    Returns
    -------
    float or ndarray
        ``log(V) + Y / V`` after the requested reduction. Terms independent
        of ``V`` are omitted. Unlike IS divergence, this value depends on the
        numerical units of the variance.
    """

    power, variance = _broadcast_power_and_variance(
        observed_power, model_variance, require_positive_power=False
    )
    values = np.log(variance) + power / variance
    return _reduce(values, reduction)
