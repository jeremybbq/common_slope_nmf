"""SAGE amplitude updates for supplied exponential features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..loss import is_divergence
from ._util import (
    _check_controls,
    _has_converged,
    _initial_amplitudes,
    _positive_array,
)


@dataclass(frozen=True)
class AmplitudeFit:
    """Result of SAGE amplitude updates for supplied exponential features.

    Attributes
    ----------
    amplitudes
        Estimated positive variance amplitudes, shape ``(R, K)``.
    loss_history
        Summed IS divergence before the first iteration and after every complete SAGE iteration, shape ``(n_iter + 1,)``.
    converged
        Whether the relative loss-decrease stopping rule was met.
    """

    amplitudes: NDArray[np.float64]
    loss_history: NDArray[np.float64]
    converged: bool


def _amplitude_iteration(
    observed_power: NDArray[np.float64],
    features: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    min_amplitude: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one complete sequential amplitude-SAGE iteration."""

    updated = amplitudes.copy()
    variance = updated @ features
    for k in range(features.shape[0]):
        feature = features[k]
        component = updated[:, k, np.newaxis] * feature
        residual = np.maximum(variance - component, 0.0)
        gain = component / variance
        posterior_power = gain * (gain * observed_power + residual)
        updated[:, k] = np.maximum(
            np.mean(posterior_power / feature, axis=1), min_amplitude
        )
        variance = residual + updated[:, k, np.newaxis] * feature

    variance = updated @ features
    return updated, variance


def fit_amplitudes(
    observed_power: ArrayLike,
    features: ArrayLike,
    initial_amplitudes: ArrayLike | None = None,
    *,
    max_iter: int = 2_000,
    tol: float = 1e-10,
    min_amplitude: float = np.finfo(np.float64).tiny,
) -> AmplitudeFit:
    """Estimate amplitudes for supplied positive exponential features.

    Parameters
    ----------
    observed_power
        Strictly positive observed energy/power, shape ``(R, N)``.
    features
        Supplied strictly positive unitless temporal features, shape ``(K, N)``. Decay rows are exponential features; a row of ones can represent a time-invariant variance floor.
    initial_amplitudes
        Optional positive variance amplitudes, shape ``(R, K)``. If omitted, mean observed power is divided equally among the supplied features.
    max_iter
        Maximum number of complete component iterations.
    tol
        Non-negative relative loss-decrease tolerance.
    min_amplitude
        Positive lower bound in variance units after every update.

    Returns
    -------
    AmplitudeFit
        Estimated amplitudes, loss history, and convergence flag. Fitted variance is ``amplitudes @ features``. The number of completed iterations is ``len(loss_history) - 1``.

    Notes
    -----
    Components are updated sequentially because SAGE immediately inserts each new component into the total variance. Operations within one component are vectorized over rows and frames. A simultaneous component-axis update would be an EM/Jacobi-style algorithm rather than this SAGE/Gauss--Seidel iteration.
    """

    power = _positive_array("observed_power", observed_power, ndim=2)
    features = _positive_array("features", features, ndim=2)
    if power.shape[1] != features.shape[1]:
        raise ValueError("observed_power and features must have the same frame count.")
    _check_controls(max_iter, tol, min_amplitude)

    expected_shape = (power.shape[0], features.shape[0])
    if initial_amplitudes is None:
        amplitudes = _initial_amplitudes(power, features)
    else:
        amplitudes = _positive_array(
            "initial_amplitudes", initial_amplitudes, ndim=2
        ).copy()
        if amplitudes.shape != expected_shape:
            raise ValueError(f"initial_amplitudes must have shape {expected_shape}.")

    amplitudes = np.maximum(amplitudes, min_amplitude)
    variance = amplitudes @ features
    loss_history = [float(is_divergence(power, variance))]
    converged = False

    for _ in range(int(max_iter)):
        amplitudes, variance = _amplitude_iteration(
            power, features, amplitudes, min_amplitude
        )
        loss_history.append(float(is_divergence(power, variance)))
        if _has_converged(loss_history, tol):
            converged = True
            break

    return AmplitudeFit(
        amplitudes=amplitudes.copy(),
        loss_history=np.asarray(loss_history, dtype=np.float64),
        converged=converged,
    )
