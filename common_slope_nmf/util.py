"""Shared finite-array checks used by the model, loss, and SAGE helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_real_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
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
    array = _finite_real_array(name, values)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return array


def _nonnegative_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = _finite_real_array(name, values)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values.")
    return array
