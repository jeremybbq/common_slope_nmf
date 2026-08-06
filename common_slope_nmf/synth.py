"""Synthetic observations for the complex-Gaussian variance model."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


def _output_shape(
    variance_shape: tuple[int, ...], n_realizations: int | None
) -> tuple[int, ...]:
    if n_realizations is None:
        return variance_shape
    if isinstance(n_realizations, bool) or not isinstance(n_realizations, Integral):
        raise TypeError("n_realizations must be an integer or None.")
    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive.")
    return (int(n_realizations),) + variance_shape


def sample_complex_gaussian(
    variance: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
    n_realizations: int | None = None,
) -> NDArray[np.complex128]:
    """Draw zero-mean circular complex-Gaussian coefficients.

    Parameters
    ----------
    variance
        Strictly positive coefficient variances, with arbitrary shape. Units
        are energy/power, and ``E[abs(X)**2] = variance``.
    rng
        NumPy random generator. A fresh default generator is created when
        omitted; pass a seeded generator for reproducible experiments.
    n_realizations
        Optional number of independent draws for every variance entry.

    Returns
    -------
    ndarray
        Complex coefficients. The shape equals ``variance.shape`` when
        ``n_realizations`` is omitted and
        ``(n_realizations,) + variance.shape`` otherwise.
    """

    values = _positive_variance(variance)
    shape = _output_shape(values.shape, n_realizations)

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    scale = np.sqrt(values / 2.0)
    real = rng.standard_normal(shape)
    imaginary = rng.standard_normal(shape)
    return scale * (real + 1j * imaginary)


def sample_power(
    variance: ArrayLike,
    *,
    rng: np.random.Generator | None = None,
    n_realizations: int | None = None,
) -> NDArray[np.float64]:
    """Draw instantaneous powers from the complex-Gaussian variance model.

    Parameters
    ----------
    variance
        Strictly positive target variances with arbitrary shape.
    rng
        NumPy random generator. Pass a seeded generator for reproducibility.
    n_realizations
        Optional number of independent draws for every variance entry.

    Returns
    -------
    ndarray
        Instantaneous powers ``abs(X)**2`` with the same output shape as
        :func:`sample_complex_gaussian`. Elementwise, ``power / variance``
        follows a unit-mean exponential distribution.
    """

    coefficients = sample_complex_gaussian(
        variance, rng=rng, n_realizations=n_realizations
    )
    return np.abs(coefficients) ** 2
