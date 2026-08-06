"""SAGE amplitude estimation for a fixed positive variance dictionary."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .is_objective import is_divergence


@dataclass(frozen=True)
class FixedDictionarySAGEResult:
    """Result of fixed-dictionary IS-SAGE amplitude estimation.

    Attributes
    ----------
    amplitudes
        Estimated non-negative variance amplitudes, shape ``(R, Q)``.
    variance
        Fitted variance ``amplitudes @ dictionary``, shape ``(R, N)``.
    objective_history
        Summed IS divergence before the first sweep and after every complete
        SAGE sweep, shape ``(n_iter + 1,)``.
    n_iter
        Number of completed SAGE sweeps.
    converged
        Whether the relative objective-decrease stopping rule was met.
    """

    amplitudes: NDArray[np.float64]
    variance: NDArray[np.float64]
    objective_history: NDArray[np.float64]
    n_iter: int
    converged: bool


def _positive_matrix(name: str, values: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values.")

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have two dimensions.")
    if matrix.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(matrix <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return matrix


def _initial_amplitudes(
    observed_power: NDArray[np.float64],
    dictionary: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_components = dictionary.shape[0]
    mean_power = np.mean(observed_power, axis=1, keepdims=True)
    mean_atoms = np.mean(dictionary, axis=1, keepdims=True).T
    return mean_power / (n_components * mean_atoms)


def fixed_dictionary_sage(
    observed_power: ArrayLike,
    dictionary: ArrayLike,
    initial_amplitudes: ArrayLike | None = None,
    *,
    max_iter: int = 2_000,
    tol: float = 1e-10,
    min_amplitude: float = np.finfo(np.float64).tiny,
) -> FixedDictionarySAGEResult:
    """Fit non-negative amplitudes for fixed positive variance atoms.

    Parameters
    ----------
    observed_power
        Strictly positive observed energy/power, shape ``(R, N)``, for ``R``
        independent RIRs and ``N`` time frames.
    dictionary
        Fixed strictly positive unitless variance atoms, shape ``(Q, N)``.
        Exponential atoms may be augmented with a row of ones to estimate a
        time-invariant noise floor.
    initial_amplitudes
        Optional strictly positive variance amplitudes, shape ``(R, Q)``.
        Positive values avoid zero-locking. If omitted, the observed mean
        power is divided equally among the atoms.
    max_iter
        Maximum number of complete component sweeps.
    tol
        Non-negative relative tolerance. Convergence is declared when the
        objective decrease over a sweep is no larger than
        ``tol * max(1, previous_objective)``.
    min_amplitude
        Strictly positive lower bound applied after each component update.

    Returns
    -------
    FixedDictionarySAGEResult
        Estimated amplitudes and variance, the summed IS-objective history,
        iteration count, and convergence flag.

    Notes
    -----
    This is the fixed-dictionary componentwise SAGE update associated with
    the complex-Gaussian variance model. The dictionary is fixed throughout;
    this function does not estimate or relocate decay rates.
    """

    power = _positive_matrix("observed_power", observed_power)
    atoms = _positive_matrix("dictionary", dictionary)
    if power.shape[1] != atoms.shape[1]:
        raise ValueError(
            "observed_power and dictionary must have the same frame count."
        )

    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and non-negative.")
    if not np.isfinite(min_amplitude) or min_amplitude <= 0.0:
        raise ValueError("min_amplitude must be finite and positive.")

    if initial_amplitudes is None:
        amplitudes = _initial_amplitudes(power, atoms)
    else:
        amplitudes = _positive_matrix(
            "initial_amplitudes", initial_amplitudes
        ).copy()
        expected_shape = (power.shape[0], atoms.shape[0])
        if amplitudes.shape != expected_shape:
            raise ValueError(
                f"initial_amplitudes must have shape {expected_shape}."
            )

    amplitudes = np.maximum(amplitudes, min_amplitude)
    variance = amplitudes @ atoms
    history = [float(is_divergence(power, variance))]
    converged = False

    for _ in range(int(max_iter)):
        for component_index in range(atoms.shape[0]):
            atom = atoms[component_index]
            component = amplitudes[:, component_index, np.newaxis] * atom
            residual = np.maximum(variance - component, 0.0)
            gain = component / variance
            posterior_power = gain * (gain * power + residual)
            updated = np.mean(posterior_power / atom, axis=1)
            amplitudes[:, component_index] = np.maximum(
                updated, min_amplitude
            )
            variance = (
                residual
                + amplitudes[:, component_index, np.newaxis] * atom
            )

        variance = amplitudes @ atoms
        objective = float(is_divergence(power, variance))
        previous = history[-1]
        history.append(objective)
        decrease = previous - objective
        roundoff = 32.0 * np.finfo(np.float64).eps * max(1.0, previous)
        if decrease < -roundoff:
            raise RuntimeError(
                "IS objective increased beyond floating-point tolerance."
            )
        if decrease <= tol * max(1.0, previous):
            converged = True
            break

    return FixedDictionarySAGEResult(
        amplitudes=amplitudes.copy(),
        variance=variance.copy(),
        objective_history=np.asarray(history, dtype=np.float64),
        n_iter=len(history) - 1,
        converged=converged,
    )
