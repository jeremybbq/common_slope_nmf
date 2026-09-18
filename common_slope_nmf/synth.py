"""Synthetic observations for the complex-Gaussian variance model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import exponential_variance, t60_to_rate


@dataclass(frozen=True)
class SynthData:
    """Synthetic data from the common-slope additive complex-Gaussian variance model.

    Attributes
    ----------
    frame_time_s
        Time vector of STFT frame elapsed time in seconds, shape ``(N,)``. Frame time zero is the decay origin.
    t60_s
        Shared energy-decay times in seconds, shape ``(F,K)``.
    rates_per_s
        Shared energy-decay rates in inverse seconds, shape ``(F,K)``.
    amplitudes
        Unit-origin variance amplitudes, shape ``(R,F,K)``. Their component sum equals the requested total amplitude for every ``(R,F)``.
    noise_floor
        Time-invariant variance floors, shape ``(R,F)``.
    variance
        Exact coefficient variances, shape ``(R,F,N)``.
    coefficients
        Circular complex-Gaussian coefficients, shape ``(R,F,N)``.
    observed_power
        Instantaneous powers ``abs(coefficients)**2``, shape ``(R,F,N)``.
    """

    frame_time_s: NDArray[np.float64]
    t60_s: NDArray[np.float64]
    rates_per_s: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    noise_floor: NDArray[np.float64]
    variance: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    observed_power: NDArray[np.float64]


def _positive_variance(variance: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(variance)
    if np.iscomplexobj(raw):
        raise ValueError("variance must contain real values.")

    values = np.asarray(variance, dtype=np.float64)
    if values.size == 0:
        raise ValueError("variance must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("variance must contain only finite values.")
    if np.any(values <= 0.0):
        raise ValueError("variance must contain only positive values.")
    return values


def sample_complex_gaussian(
    variance: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Draw zero-mean circular complex-Gaussian coefficients.

    Parameters
    ----------
    variance
        Strictly positive coefficient variances, with arbitrary shape. Units are energy/power, and ``E[abs(X)**2] = variance``.
    rng
        NumPy random generator. A fresh default generator is created when omitted; pass a seeded generator for reproducible experiments.

    Returns
    -------
    ndarray
        Complex coefficients with the same shape as ``variance``.
    """

    values = _positive_variance(variance)

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    scale = np.sqrt(values / 2.0)
    real = rng.standard_normal(values.shape)
    imaginary = rng.standard_normal(values.shape)
    return scale * (real + 1j * imaginary)


def sample_power(
    variance: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Draw instantaneous powers from the complex-Gaussian variance model.

    Parameters
    ----------
    variance
        Strictly positive target variances with arbitrary shape.
    rng
        NumPy random generator. Pass a seeded generator for reproducibility.

    Returns
    -------
    ndarray
        Instantaneous powers ``abs(X)**2`` with the same shape as ``variance``. Elementwise, ``power / variance`` follows a unit-mean exponential distribution.
    """

    return np.abs(sample_complex_gaussian(variance, rng=rng)) ** 2


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _generator(
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")
    return rng


def sample_t60(
    n_frequencies: int,
    n_components: int,
    t60_range_s: tuple[float, float],
    *,
    min_separation_s: float = 0.0,
    rng: np.random.Generator | None = None,
    max_attempts: int = 10_000,
) -> NDArray[np.float64]:
    """Draw sorted shared energy-decay times from a uniform range.

    Parameters
    ----------
    n_frequencies
        Number of independent frequency bins ``F``.
    n_components
        Number of decay components ``K`` per frequency.
    t60_range_s
        Strictly increasing ``(lower, upper)`` energy-``T60`` range in seconds.
    min_separation_s
        Optional minimum adjacent separation in seconds after sorting.
    rng
        NumPy random generator. Pass a seeded generator for reproducibility.
    max_attempts
        Maximum rejection draws per frequency when enforcing separation.

    Returns
    -------
    ndarray
        Sorted decay times in seconds, shape ``(F,K)``.
    """

    n_frequencies = _positive_int("n_frequencies", n_frequencies)
    n_components = _positive_int("n_components", n_components)
    max_attempts = _positive_int("max_attempts", max_attempts)
    bounds = np.asarray(t60_range_s, dtype=np.float64)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError("t60_range_s must contain two finite values.")
    lower_s, upper_s = bounds
    if lower_s <= 0.0 or lower_s >= upper_s:
        raise ValueError("t60_range_s must be positive and strictly increasing.")
    if not np.isfinite(min_separation_s) or min_separation_s < 0.0:
        raise ValueError("min_separation_s must be finite and non-negative.")
    if (n_components - 1) * min_separation_s >= upper_s - lower_s:
        raise ValueError(
            "min_separation_s leaves no interior support for all components."
        )

    generator = _generator(rng)
    if min_separation_s == 0.0 or n_components == 1:
        return np.sort(
            generator.uniform(
                lower_s,
                upper_s,
                size=(n_frequencies, n_components),
            ),
            axis=1,
        )

    t60_s = np.empty((n_frequencies, n_components), dtype=np.float64)
    for frequency_index in range(n_frequencies):
        for _ in range(max_attempts):
            candidate = np.sort(
                generator.uniform(lower_s, upper_s, size=n_components)
            )
            if np.all(np.diff(candidate) >= min_separation_s):
                t60_s[frequency_index] = candidate
                break
        else:
            raise RuntimeError(
                "could not sample separated decay times within max_attempts."
            )
    return t60_s


def sample_simplex_amplitudes(
    n_rirs: int,
    n_frequencies: int,
    n_components: int,
    *,
    concentration: float,
    total_amplitude: float = 1.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Draw corner-concentrated amplitudes on a fixed-sum simplex.

    Parameters
    ----------
    n_rirs
        Number of RIRs ``R``.
    n_frequencies
        Number of frequency bins ``F``.
    n_components
        Number of decay components ``K``.
    concentration
        Positive symmetric Dirichlet concentration. Values below one favor the ``K`` simplex corners, one is uniform, and values above one favor equal component shares.
    total_amplitude
        Positive scalar total unit-origin variance, shared by every ``(R,F)``. The default is one, or 0 dB in power units.
    rng
        NumPy random generator. Pass a seeded generator for reproducibility.

    Returns
    -------
    ndarray
        Positive variance amplitudes, shape ``(R,F,K)``, whose last axis sums to ``total_amplitude``.
    """

    n_rirs = _positive_int("n_rirs", n_rirs)
    n_frequencies = _positive_int("n_frequencies", n_frequencies)
    n_components = _positive_int("n_components", n_components)
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("concentration must be finite and positive.")
    if np.ndim(total_amplitude) != 0:
        raise ValueError("total_amplitude must be a scalar.")
    if not np.isfinite(total_amplitude) or float(total_amplitude) <= 0.0:
        raise ValueError("total_amplitude must be finite and positive.")

    generator = _generator(rng)
    shares = generator.dirichlet(
        np.full(n_components, concentration, dtype=np.float64),
        size=(n_rirs, n_frequencies),
    )
    return shares * float(total_amplitude)


def sample_stft_power(
    frame_time_s: ArrayLike,
    n_rirs: int,
    n_frequencies: int,
    n_components: int,
    *,
    t60_range_s: tuple[float, float],
    amplitude_concentration: float,
    total_amplitude: float = 1.0,
    noise_mean_db: float,
    noise_std_db: float,
    min_t60_separation_s: float = 0.0,
    rng: np.random.Generator | None = None,
) -> SynthData:
    """Generate one exact-model multislope complex-Gaussian dataset.

    Parameters
    ----------
    frame_time_s
        Time vector of STFT frame elapsed time in seconds, shape ``(N,)``. Frame zero is the decay origin.
    n_rirs, n_frequencies, n_components
        Model dimensions ``R``, ``F``, and ``K``.
    t60_range_s
        Uniform sampling range for energy-``T60`` values in seconds.
    amplitude_concentration
        Symmetric Dirichlet concentration for fixed-sum amplitudes.
    total_amplitude
        Positive scalar total unit-origin variance, shared by every ``(R,F)``.
    noise_mean_db, noise_std_db
        Mean and non-negative standard deviation of Gaussian floor levels in power dB. Linear floors use ``10**(level_db / 10)``.
    min_t60_separation_s
        Optional minimum adjacent ``T60`` separation in seconds.
    rng
        NumPy random generator controlling every draw.

    Returns
    -------
    SynthData
        Sampled parameters, exact variances, coefficients, and powers with explicit array shapes and units.
    """

    frame_time_s = np.asarray(frame_time_s, dtype=np.float64)
    if frame_time_s.ndim != 1 or frame_time_s.size < 2:
        raise ValueError("frame_time_s must be one-dimensional with at least 2 values.")
    if not np.all(np.isfinite(frame_time_s)) or np.any(frame_time_s < 0.0):
        raise ValueError("frame_time_s must contain finite non-negative values.")
    if np.ptp(frame_time_s) <= 0.0:
        raise ValueError("frame_time_s must contain variation.")
    if not np.isfinite(noise_mean_db):
        raise ValueError("noise_mean_db must be finite.")
    if not np.isfinite(noise_std_db) or noise_std_db < 0.0:
        raise ValueError("noise_std_db must be finite and non-negative.")

    n_rirs = _positive_int("n_rirs", n_rirs)
    n_frequencies = _positive_int("n_frequencies", n_frequencies)
    n_components = _positive_int("n_components", n_components)
    generator = _generator(rng)
    t60_s = sample_t60(
        n_frequencies,
        n_components,
        t60_range_s,
        min_separation_s=min_t60_separation_s,
        rng=generator,
    )
    rates_per_s = np.asarray(t60_to_rate(t60_s), dtype=np.float64)
    amplitudes = sample_simplex_amplitudes(
        n_rirs,
        n_frequencies,
        n_components,
        concentration=amplitude_concentration,
        total_amplitude=total_amplitude,
        rng=generator,
    )
    noise_level_db = generator.normal(
        noise_mean_db, noise_std_db, size=(n_rirs, n_frequencies)
    )
    noise_floor = 10.0 ** (noise_level_db / 10.0)
    variance = exponential_variance(
        frame_time_s,
        rates_per_s,
        amplitudes,
        noise_floor=noise_floor,
    )
    coefficients = sample_complex_gaussian(variance, rng=generator)
    observed_power = np.abs(coefficients) ** 2
    return SynthData(
        frame_time_s=frame_time_s.copy(),
        t60_s=t60_s,
        rates_per_s=rates_per_s,
        amplitudes=amplitudes,
        noise_floor=noise_floor,
        variance=variance,
        coefficients=coefficients,
        observed_power=observed_power,
    )
