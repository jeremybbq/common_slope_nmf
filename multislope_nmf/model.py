"""Physical parameter transforms and exponential variance construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_T60_RATE_PRODUCT = 6.0 * np.log(10.0)


def _finite_real_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values.")

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = _finite_real_array(name, values)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return array


def _nonnegative_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = _finite_real_array(name, values)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values.")
    return array


def t60_to_rate(t60_s: ArrayLike) -> float | NDArray[np.float64]:
    """Convert energy-decay ``T60`` values in seconds to rates in ``s^-1``.

    Parameters
    ----------
    t60_s
        Positive scalar or array of energy/power ``T60`` values in seconds.

    Returns
    -------
    float or ndarray
        Decay rates with the same shape as ``t60_s``. A scalar input returns a
        scalar. The convention is ``rate = 6 log(10) / T60``.
    """

    t60 = _positive_array("t60_s", t60_s)
    rate = _T60_RATE_PRODUCT / t60
    return float(rate) if rate.ndim == 0 else rate


def rate_to_t60(rate_per_s: ArrayLike) -> float | NDArray[np.float64]:
    """Convert energy-decay rates in ``s^-1`` to ``T60`` values in seconds.

    Parameters
    ----------
    rate_per_s
        Positive scalar or array of exponential energy-decay rates in
        inverse seconds.

    Returns
    -------
    float or ndarray
        Energy/power ``T60`` values with the same shape as ``rate_per_s``.
        A scalar input returns a scalar.
    """

    rate = _positive_array("rate_per_s", rate_per_s)
    t60 = _T60_RATE_PRODUCT / rate
    return float(t60) if t60.ndim == 0 else t60


def exponential_atoms(
    times_s: ArrayLike, rate_per_s: ArrayLike
) -> NDArray[np.float64]:
    """Construct unit-origin exponential energy-decay atoms.

    Parameters
    ----------
    times_s
        One-dimensional elapsed times in seconds, shape ``(N,)``. Values must
        be finite and non-negative.
    rate_per_s
        Positive decay rates in inverse seconds, shape ``(..., K)``. A scalar
        is interpreted as one component.

    Returns
    -------
    ndarray
        Exponential atoms ``exp(-rate * time)``, shape ``(..., K, N)``. Every
        atom equals one when its elapsed time is zero.
    """

    times = _nonnegative_array("times_s", times_s)
    if times.ndim != 1:
        raise ValueError("times_s must be one-dimensional.")

    rates = np.atleast_1d(_positive_array("rate_per_s", rate_per_s))
    return np.exp(-rates[..., np.newaxis] * times)


def exponential_variance(
    times_s: ArrayLike,
    rate_per_s: ArrayLike,
    amplitudes: ArrayLike,
    noise_floor: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Evaluate a non-negative sum of exponential variance components.

    Parameters
    ----------
    times_s
        Elapsed times in seconds, shape ``(N,)``.
    rate_per_s
        Positive shared rates in inverse seconds, shape ``(..., K)``. For the
        full model this is commonly ``(F, K)``.
    amplitudes
        Non-negative energy/variance amplitudes at ``times_s == 0``, with a
        shape broadcastable with ``rate_per_s``. This is commonly
        ``(R, F, K)``.
    noise_floor
        Non-negative time-invariant variance, broadcastable to the parameter
        batch shape after the component axis is removed. This is commonly
        ``(R, F)``.

    Returns
    -------
    ndarray
        Total variance, shape ``broadcast(rate_per_s, amplitudes)[:-1] + (N,)``.
        A scalar rate and scalar amplitude produce shape ``(N,)``.
    """

    times = _nonnegative_array("times_s", times_s)
    if times.ndim != 1:
        raise ValueError("times_s must be one-dimensional.")

    rates = np.atleast_1d(_positive_array("rate_per_s", rate_per_s))
    weights = np.atleast_1d(_nonnegative_array("amplitudes", amplitudes))

    try:
        rates, weights = np.broadcast_arrays(rates, weights)
    except ValueError as exc:
        raise ValueError(
            "rate_per_s and amplitudes must have broadcast-compatible shapes."
        ) from exc

    components = weights[..., np.newaxis] * np.exp(
        -rates[..., np.newaxis] * times
    )
    variance = np.sum(components, axis=-2)

    floor = _nonnegative_array("noise_floor", noise_floor)
    try:
        floor = np.broadcast_to(floor, variance.shape[:-1])
    except ValueError as exc:
        raise ValueError(
            "noise_floor must broadcast to the parameter batch shape."
        ) from exc

    return variance + floor[..., np.newaxis]
