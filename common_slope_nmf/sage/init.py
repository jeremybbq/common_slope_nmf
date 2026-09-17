"""Data-derived starting points for decay SAGE."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..preprocess import DecayFit, fit_coarse_decay, head_power, tail_power
from ._util import _broadcast_rate_bound, _positive_array, _real_array


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
