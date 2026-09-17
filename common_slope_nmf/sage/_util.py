"""Shared validation helpers for SAGE amplitude and decay updates."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _real_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values.")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_array(
    name: str, values: ArrayLike, ndim: int | None = None
) -> NDArray[np.float64]:
    array = _real_array(name, values)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return array


def _check_controls(max_iter: int, tol: float, min_amplitude: float) -> None:
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and non-negative.")
    if not np.isfinite(min_amplitude) or min_amplitude <= 0.0:
        raise ValueError("min_amplitude must be finite and positive.")


def _has_converged(
    history: list[float], tol: float, *, require_monotone: bool = True
) -> bool:
    previous, objective = history[-2:]
    decrease = previous - objective
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, previous)
    if decrease < -roundoff:
        if require_monotone:
            raise RuntimeError(
                "IS objective increased beyond floating-point tolerance."
            )
        return False
    return decrease <= tol * max(1.0, previous)


def _broadcast_rate_bound(
    name: str, values: ArrayLike, shape: tuple[int, ...]
) -> NDArray[np.float64]:
    bound = _positive_array(name, values)
    try:
        return np.broadcast_to(bound, shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to shape {shape}.") from exc
