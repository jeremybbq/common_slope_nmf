"""Data-derived starting points for decay SAGE."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..preprocess import fit_linear_decay, head_power, tail_power
from ._util import _positive_array, _real_array


def init_decay(
    observed_power: ArrayLike,
    frame_time_s: ArrayLike,
    *,
    n_head_frames: int = 8,
    n_tail_frames: int = 8,
    floor_margin_db: float | None = 6.0,
    min_regression_frames: int = 3,
    min_amplitude: float = np.finfo(np.float64).tiny,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Construct a simple data-derived decay starting point.

    Parameters
    ----------
    observed_power
        Positive observed energy/power, shape ``(R,F,N)``.
    frame_time_s
        Non-negative elapsed frame times in seconds, shape ``(N,)``. Frame zero is the decay origin.
    n_head_frames
        Number of leading frames averaged for the total initial amplitude.
    n_tail_frames
        Number of trailing frames averaged for the initial noise floor and excluded from the pooled log-power regression.
    floor_margin_db
        Optional non-negative power margin above the mean initialized floor used to select common regression frames. ``None`` uses every non-tail frame.
    min_regression_frames
        Minimum common regression frames required at each frequency.
    min_amplitude
        Positive lower bound in variance units for amplitudes and floors.

    Returns
    -------
    rate_per_s, amplitudes, noise_floor
        Unclipped pooled linear-fit rate ``(F,)``, leading-frame total amplitude ``(R,F)``, and tail-average floor ``(R,F)``. Before :func:`fit_decay`, repeat the rate along the component axis and split the amplitude equally across ``K``.

    Notes
    -----
    The leading-frame mean is the total unit-origin amplitude; it is not split and the initialized floor is not subtracted.
    """

    power = _positive_array("observed_power", observed_power, ndim=3)
    times = _real_array("frame_time_s", frame_time_s)
    if times.ndim != 1 or times.shape[0] != power.shape[2]:
        raise ValueError("frame_time_s must have shape (N,) matching observed_power.")
    if np.any(times < 0.0) or np.ptp(times) <= 0.0:
        raise ValueError("frame_time_s must be non-negative and contain variation.")
    if not np.isfinite(min_amplitude) or min_amplitude <= 0.0:
        raise ValueError("min_amplitude must be finite and positive.")

    noise_floor = np.maximum(tail_power(power, n_tail_frames), min_amplitude)
    rate_per_s = fit_linear_decay(
        power,
        times,
        noise_floor=noise_floor,
        n_tail_frames=n_tail_frames,
        floor_margin_db=floor_margin_db,
        min_frames=min_regression_frames,
    )
    amplitudes = np.maximum(head_power(power, n_head_frames), min_amplitude)
    return rate_per_s, amplitudes, noise_floor
