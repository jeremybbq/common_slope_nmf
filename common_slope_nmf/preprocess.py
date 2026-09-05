"""Data-derived summaries used to initialize decay estimators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DecayFit:
    """Result of a pooled coarse log-power decay fit.

    Attributes
    ----------
    rate_per_s
        Fitted positive energy-decay rates in inverse seconds, shape ``(F,)``.
    r_squared
        Within-RIR coefficient of determination, shape ``(F,)``. This is a
        regression diagnostic, not a goodness-of-fit test for the multislope
        model.
    frame_mask
        Common frames used across all RIRs at each frequency, shape ``(F,N)``.
    """

    rate_per_s: NDArray[np.float64]
    r_squared: NDArray[np.float64]
    frame_mask: NDArray[np.bool_]


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
    Residual long-decay energy can bias this initialization upward. The value
    is intended as an estimator starting point rather than an unbiased floor
    estimate under arbitrary observation lengths.
    """

    power = _power_array(observed_power)
    count = _positive_frame_count("n_frames", n_frames, power.shape[2])
    return np.mean(power[:, :, -count:], axis=2)


def fit_coarse_decay(
    observed_power: ArrayLike,
    times_s: ArrayLike,
    *,
    noise_floor: ArrayLike | None = None,
    n_tail_frames: int = 0,
    floor_margin_db: float | None = None,
    min_frames: int = 3,
) -> DecayFit:
    """Fit one pooled log-power decay rate per frequency.

    Parameters
    ----------
    observed_power
        Positive observed power, shape ``(R,F,N)``.
    times_s
        Non-negative elapsed times in seconds, shape ``(N,)``.
    noise_floor
        Optional positive floor estimate, shape ``(R,F)``. It is used only to
        choose a common regression frame mask and is not subtracted from the
        instantaneous powers.
    n_tail_frames
        Number of final frames excluded from every regression.
    floor_margin_db
        Optional non-negative power margin above the across-RIR mean floor.
        Frames below this margin are excluded. Requires ``noise_floor``.
    min_frames
        Minimum number of common frames required at every frequency.

    Returns
    -------
    DecayFit
        Positive coarse rates in inverse seconds and regression diagnostics.

    Notes
    -----
    A single line is fit to all RIR log powers at each frequency. Different
    RIR intercepts do not bias the slope because every RIR uses the same time
    samples: each intercept multiplies a centered-time vector whose sum is
    zero. The returned ``r_squared`` removes each RIR mean before evaluating
    residuals so amplitude differences do not dominate that diagnostic.
    """

    power = _power_array(observed_power)
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] != power.shape[2]:
        raise ValueError("times_s must have shape (N,) matching observed_power.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times_s must contain finite non-negative values.")
    if np.ptp(times) <= 0.0:
        raise ValueError("times_s must contain variation.")
    if isinstance(n_tail_frames, bool) or not isinstance(n_tail_frames, Integral):
        raise TypeError("n_tail_frames must be an integer.")
    if n_tail_frames < 0 or n_tail_frames >= times.size:
        raise ValueError(
            "n_tail_frames must be non-negative and below the frame count."
        )
    min_frames = _positive_frame_count("min_frames", min_frames, times.size)
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
    r_squared = np.empty(n_frequencies, dtype=np.float64)
    for frequency_index in range(n_frequencies):
        mask = frame_mask[frequency_index]
        if np.count_nonzero(mask) < min_frames:
            raise ValueError(
                "too few regression frames at frequency index "
                f"{frequency_index}."
            )
        selected_times = times[mask]
        centered_times = selected_times - np.mean(selected_times)
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

        centered_log_power = log_power - np.mean(
            log_power, axis=1, keepdims=True
        )
        residual = centered_log_power - slope * centered_times
        residual_sum = np.sum(residual**2)
        total_sum = np.sum(centered_log_power**2)
        r_squared[frequency_index] = (
            1.0 - residual_sum / total_sum if total_sum > 0.0 else np.nan
        )

    return DecayFit(
        rate_per_s=rate_per_s,
        r_squared=r_squared,
        frame_mask=frame_mask,
    )
