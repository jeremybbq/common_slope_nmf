"""Supplied-atom SAGE amplitude updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..is_objective import is_divergence
from ._util import (
    _check_controls,
    _has_converged,
    _initial_amplitudes,
    _positive_array,
)

@dataclass(frozen=True)
class AmplitudeSAGEResult:
    """Result of SAGE amplitude updates for supplied temporal atoms.

    Attributes
    ----------
    amplitudes
        Estimated positive variance amplitudes, shape ``(R, Q)``.
    variance
        Fitted variance ``amplitudes @ atoms``, shape ``(R, N)``.
    objective_history
        Summed IS divergence before the first sweep and after every complete
        SAGE sweep, shape ``(n_iter + 1,)``.
    n_iter
        Number of completed component sweeps.
    converged
        Whether the relative objective-decrease stopping rule was met.
    """

    amplitudes: NDArray[np.float64]
    variance: NDArray[np.float64]
    objective_history: NDArray[np.float64]
    n_iter: int
    converged: bool

def _amplitude_sage_sweep(
    observed_power: NDArray[np.float64],
    atoms: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    min_amplitude: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one complete sequential amplitude-SAGE sweep."""

    updated = amplitudes.copy()
    variance = updated @ atoms
    for component_index in range(atoms.shape[0]):
        atom = atoms[component_index]
        component = updated[:, component_index, np.newaxis] * atom
        residual = np.maximum(variance - component, 0.0)
        gain = component / variance
        posterior_power = gain * (gain * observed_power + residual)
        updated[:, component_index] = np.maximum(
            np.mean(posterior_power / atom, axis=1), min_amplitude
        )
        variance = residual + updated[:, component_index, np.newaxis] * atom

    variance = updated @ atoms
    return updated, variance


def amplitude_sage(
    observed_power: ArrayLike,
    atoms: ArrayLike,
    initial_amplitudes: ArrayLike | None = None,
    *,
    max_iter: int = 2_000,
    tol: float = 1e-10,
    min_amplitude: float = np.finfo(np.float64).tiny,
) -> AmplitudeSAGEResult:
    """Estimate amplitudes for supplied positive temporal atoms.

    Parameters
    ----------
    observed_power
        Strictly positive observed energy/power, shape ``(R, N)``.
    atoms
        Supplied strictly positive unitless temporal atoms, shape ``(Q, N)``.
        A row of ones can represent a time-invariant variance floor.
    initial_amplitudes
        Optional positive variance amplitudes, shape ``(R, Q)``. If omitted,
        mean observed power is divided equally among the supplied atoms.
    max_iter
        Maximum number of complete component sweeps.
    tol
        Non-negative relative objective-decrease tolerance.
    min_amplitude
        Positive lower bound in variance units after every update.

    Returns
    -------
    AmplitudeSAGEResult
        Estimated amplitudes and variance, objective history, iteration count,
        and convergence flag.

    Notes
    -----
    Components are updated sequentially because SAGE immediately inserts each
    new component into the total variance. Operations within one component are
    vectorized over rows and frames. A simultaneous component-axis update would
    be an EM/Jacobi-style algorithm rather than this SAGE/Gauss--Seidel sweep.
    """

    power = _positive_array("observed_power", observed_power, ndim=2)
    temporal_atoms = _positive_array("atoms", atoms, ndim=2)
    if power.shape[1] != temporal_atoms.shape[1]:
        raise ValueError("observed_power and atoms must have the same frame count.")
    _check_controls(max_iter, tol, min_amplitude)

    expected_shape = (power.shape[0], temporal_atoms.shape[0])
    if initial_amplitudes is None:
        amplitudes = _initial_amplitudes(power, temporal_atoms)
    else:
        amplitudes = _positive_array(
            "initial_amplitudes", initial_amplitudes, ndim=2
        ).copy()
        if amplitudes.shape != expected_shape:
            raise ValueError(f"initial_amplitudes must have shape {expected_shape}.")

    amplitudes = np.maximum(amplitudes, min_amplitude)
    variance = amplitudes @ temporal_atoms
    history = [float(is_divergence(power, variance))]
    converged = False

    for _ in range(int(max_iter)):
        amplitudes, variance = _amplitude_sage_sweep(
            power, temporal_atoms, amplitudes, min_amplitude
        )
        history.append(float(is_divergence(power, variance)))
        if _has_converged(history, tol):
            converged = True
            break

    return AmplitudeSAGEResult(
        amplitudes=amplitudes.copy(),
        variance=variance.copy(),
        objective_history=np.asarray(history, dtype=np.float64),
        n_iter=len(history) - 1,
        converged=converged,
    )
