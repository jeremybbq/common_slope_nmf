"""Physical parameter transforms and exponential variance construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .util import _nonnegative_array, _positive_array

_T60_RATE_PRODUCT = 6.0 * np.log(10.0)


def t60_to_rate(t60_s: ArrayLike) -> float | NDArray[np.float64]:
    """Convert energy-decay ``T60`` values in seconds to rates in ``s^-1``.

    Parameters
    ----------
    t60_s
        Positive scalar or array of energy/power ``T60`` values in seconds.

    Returns
    -------
    float or ndarray
        Decay rates with the same shape as ``t60_s``. A scalar input returns a scalar. The convention is ``rate = 6 log(10) / T60``.
    """

    t60 = _positive_array("t60_s", t60_s)
    rate = _T60_RATE_PRODUCT / t60
    return float(rate) if rate.ndim == 0 else rate


def rate_to_t60(rate_per_s: ArrayLike) -> float | NDArray[np.float64]:
    """Convert energy-decay rates in ``s^-1`` to ``T60`` values in seconds.

    Parameters
    ----------
    rate_per_s
        Positive scalar or array of exponential energy-decay rates in inverse seconds.

    Returns
    -------
    float or ndarray
        Energy/power ``T60`` values with the same shape as ``rate_per_s``. A scalar input returns a scalar.
    """

    rate = _positive_array("rate_per_s", rate_per_s)
    t60 = _T60_RATE_PRODUCT / rate
    return float(t60) if t60.ndim == 0 else t60


def exponential_features(
    frame_time_s: ArrayLike, rate_per_s: ArrayLike
) -> NDArray[np.float64]:
    """Construct unit-origin exponential energy-decay features.

    Parameters
    ----------
    frame_time_s
        One-dimensional time vector of STFT frame elapsed time in seconds, shape ``(N,)``. Values must be finite and non-negative. Frame zero is the decay origin.
    rate_per_s
        Positive decay rates in inverse seconds, shape ``(..., K)``. A scalar is interpreted as one component.

    Returns
    -------
    ndarray
        Exponential features ``exp(-rate * time)``, shape ``(..., K, N)``. Every feature equals one at frame time zero.
    """

    frame_time_s = _nonnegative_array("frame_time_s", frame_time_s)
    if frame_time_s.ndim != 1:
        raise ValueError("frame_time_s must be one-dimensional.")

    rates = np.atleast_1d(_positive_array("rate_per_s", rate_per_s))
    return np.exp(-rates[..., np.newaxis] * frame_time_s)


def exponential_variance(
    frame_time_s: ArrayLike,
    rate_per_s: ArrayLike,
    amplitudes: ArrayLike,
    noise_floor: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Evaluate a non-negative sum of exponential variance components.

    Parameters
    ----------
    frame_time_s
        Time vector of STFT frame elapsed time in seconds, shape ``(N,)``. Frame zero is the decay origin.
    rate_per_s
        Positive shared rates in inverse seconds, shape ``(..., K)``. For the full model this is commonly ``(F, K)``.
    amplitudes
        Non-negative energy/variance amplitudes at ``frame_time_s == 0``, with a shape broadcastable with ``rate_per_s``. This is commonly ``(R, F, K)``.
    noise_floor
        Non-negative time-invariant variance, broadcastable to the parameter batch shape after the component axis is removed. This is commonly ``(R, F)``.

    Returns
    -------
    ndarray
        Total variance, shape ``broadcast(rate_per_s, amplitudes)[:-1] + (N,)``. A scalar rate and scalar amplitude produce shape ``(N,)``.
    """

    frame_time_s = _nonnegative_array("frame_time_s", frame_time_s)
    if frame_time_s.ndim != 1:
        raise ValueError("frame_time_s must be one-dimensional.")

    rates = np.atleast_1d(_positive_array("rate_per_s", rate_per_s))
    weights = np.atleast_1d(_nonnegative_array("amplitudes", amplitudes))

    try:
        rates, weights = np.broadcast_arrays(rates, weights)
    except ValueError as exc:
        raise ValueError(
            "rate_per_s and amplitudes must have broadcast-compatible shapes."
        ) from exc

    components = weights[..., np.newaxis] * np.exp(
        -rates[..., np.newaxis] * frame_time_s
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
