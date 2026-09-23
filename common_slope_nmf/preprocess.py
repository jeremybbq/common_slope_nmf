"""Data-derived summaries used to initialize estimators."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _power_array(observed_power: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(observed_power)
    if np.iscomplexobj(raw):
        raise ValueError("observed_power must contain real values.")
    power = np.asarray(observed_power, dtype=np.float64)
    if power.ndim != 3:
        raise ValueError("observed_power must have shape (R,F,N).")
    if power.size == 0 or not np.all(np.isfinite(power)):
        raise ValueError("observed_power must contain finite values.")
    if np.any(power <= 0.0):
        raise ValueError("observed_power must contain positive values.")
    return power


def _positive_frame_count(name: str, value: int, n_frames: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    if value > n_frames:
        raise ValueError(f"{name} must not exceed the available frame count.")
    return int(value)


def head_power(
    observed_power: ArrayLike, n_frames: int = 8
) -> NDArray[np.float64]:
    """Average the first frames of an observed power array.

    Parameters
    ----------
    observed_power
        Positive observed power, shape ``(R,F,N)``.
    n_frames
        Number of leading frames to average.

    Returns
    -------
    ndarray
        Mean leading-frame power in variance units, shape ``(R,F)``.
    """

    power = _power_array(observed_power)
    count = _positive_frame_count("n_frames", n_frames, power.shape[2])
    return np.mean(power[:, :, :count], axis=2)


def tail_power(
    observed_power: ArrayLike, n_frames: int = 8
) -> NDArray[np.float64]:
    """Average the last frames as a time-invariant floor estimate.

    Parameters
    ----------
    observed_power
        Positive observed power, shape ``(R,F,N)``.
    n_frames
        Number of trailing frames to average.

    Returns
    -------
    ndarray
        Mean trailing-frame power in variance units, shape ``(R,F)``.

    Notes
    -----
    Residual long-decay energy can bias this initialization upward. The value is intended as an estimator starting point rather than an unbiased floor estimate under arbitrary observation lengths.
    """

    power = _power_array(observed_power)
    count = _positive_frame_count("n_frames", n_frames, power.shape[2])
    return np.mean(power[:, :, -count:], axis=2)


def fit_linear_decay(
    observed_power: ArrayLike,
    frame_time_s: ArrayLike,
    *,
    noise_floor: ArrayLike | None = None,
    n_tail_frames: int = 0,
    floor_margin_db: float | None = None,
    min_frames: int = 3,
) -> NDArray[np.float64]:
    """Fit one pooled log-power decay rate per frequency.

    Parameters
    ----------
    observed_power
        Positive observed power, shape ``(R,F,N)``.
    frame_time_s
        Time vector of STFT frame elapsed time in seconds, shape ``(N,)``. Frame zero is the decay origin.
    noise_floor
        Optional positive floor estimate, shape ``(R,F)``. It is used only to choose a common regression frame mask and is not subtracted from the instantaneous powers.
    n_tail_frames
        Number of final frames excluded from every regression.
    floor_margin_db
        Optional non-negative power margin above the across-RIR mean floor. Frames below this margin are excluded. Requires ``noise_floor``.
    min_frames
        Minimum number of common frames required at every frequency.

    Returns
    -------
    ndarray
        Positive energy-decay rates in inverse seconds, shape ``(F,)``.

    Notes
    -----
    A single line is fit to all RIR log powers at each frequency. Different RIR intercepts do not bias the slope because every RIR uses the same frame times: each intercept multiplies a centered-time vector whose sum is zero.
    """

    power = _power_array(observed_power)
    frame_time_s = np.asarray(frame_time_s, dtype=np.float64)
    if frame_time_s.ndim != 1 or frame_time_s.shape[0] != power.shape[2]:
        raise ValueError("frame_time_s must have shape (N,) matching observed_power.")
    if not np.all(np.isfinite(frame_time_s)) or np.any(frame_time_s < 0.0):
        raise ValueError("frame_time_s must contain finite non-negative values.")
    if np.ptp(frame_time_s) <= 0.0:
        raise ValueError("frame_time_s must contain variation.")
    if isinstance(n_tail_frames, bool) or not isinstance(n_tail_frames, Integral):
        raise TypeError("n_tail_frames must be an integer.")
    if n_tail_frames < 0 or n_tail_frames >= frame_time_s.size:
        raise ValueError(
            "n_tail_frames must be non-negative and below the frame count."
        )
    min_frames = _positive_frame_count("min_frames", min_frames, frame_time_s.size)
    if min_frames < 2:
        raise ValueError("min_frames must be at least 2.")
    if floor_margin_db is not None:
        if not np.isfinite(floor_margin_db) or floor_margin_db < 0.0:
            raise ValueError("floor_margin_db must be finite and non-negative.")
        if noise_floor is None:
            raise ValueError("floor_margin_db requires noise_floor.")

    n_rirs, n_frequencies, n_frames = power.shape
    frame_mask = np.ones((n_frequencies, n_frames), dtype=np.bool_)
    if n_tail_frames:
        frame_mask[:, -n_tail_frames:] = False
    if noise_floor is not None:
        floor = np.asarray(noise_floor, dtype=np.float64)
        if floor.shape != (n_rirs, n_frequencies):
            raise ValueError("noise_floor must have shape (R,F).")
        if not np.all(np.isfinite(floor)) or np.any(floor <= 0.0):
            raise ValueError("noise_floor must contain finite positive values.")
        if floor_margin_db is not None:
            floor_ratio = 10.0 ** (floor_margin_db / 10.0)
            mean_power = np.mean(power, axis=0)
            mean_floor = np.mean(floor, axis=0)
            frame_mask &= mean_power > mean_floor[:, np.newaxis] * floor_ratio

    rate_per_s = np.empty(n_frequencies, dtype=np.float64)
    for frequency_index in range(n_frequencies):
        mask = frame_mask[frequency_index]
        if np.count_nonzero(mask) < min_frames:
            raise ValueError(
                "too few regression frames at frequency index "
                f"{frequency_index}."
            )
        selected_frame_time_s = frame_time_s[mask]
        centered_times = selected_frame_time_s - np.mean(selected_frame_time_s)
        denominator = n_rirs * np.sum(centered_times**2)
        if denominator <= 0.0:
            raise ValueError("selected regression times must contain variation.")
        log_power = np.log(power[:, frequency_index, :][:, mask])
        slope = np.sum(log_power * centered_times) / denominator
        if not np.isfinite(slope) or slope >= 0.0:
            raise ValueError(
                "pooled log-power regression did not produce a positive "
                f"decay rate at frequency index {frequency_index}."
            )
        rate_per_s[frequency_index] = -slope

    return rate_per_s
