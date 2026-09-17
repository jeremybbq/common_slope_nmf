"""SOFA SingleRoomDRIR dataset readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._hdf5 import decode_hdf5_text, h5py_module

@dataclass(frozen=True)
class SOFADatasetInfo:
    """Layout metadata for a SOFA ``SingleRoomDRIR`` dataset.

    Attributes
    ----------
    shape
        ``Data.IR`` shape ``(M,R,N)``: measurements, receiver channels, and
        time samples.
    n_measurements, n_receivers, n_samples
        Sizes of the three ``Data.IR`` axes.
    sample_rate_hz
        Sampling rate in samples per second.
    convention
        Decoded value of the root ``SOFAConventions`` attribute.
    """

    shape: tuple[int, int, int]
    n_measurements: int
    n_receivers: int
    n_samples: int
    sample_rate_hz: float
    convention: str


def inspect_sofa_dataset(path: str | Path) -> SOFADatasetInfo:
    """Inspect a SOFA room-impulse-response file.

    Parameters
    ----------
    path
        Path to a SOFA HDF5 file containing ``Data.IR`` with axes
        ``(measurement, receiver, sample)``.

    Returns
    -------
    SOFADatasetInfo
        Array dimensions, sample rate in hertz, and SOFA convention.
    """

    h5py = h5py_module()
    with h5py.File(Path(path), "r") as stream:
        if "Data.IR" not in stream or "Data.SamplingRate" not in stream:
            raise ValueError(
                "SOFA dataset must contain 'Data.IR' and 'Data.SamplingRate'."
            )
        shape = tuple(int(value) for value in stream["Data.IR"].shape)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("SOFA Data.IR must have non-empty shape (M,R,N).")
        sampling_rate = np.asarray(stream["Data.SamplingRate"], dtype=np.float64)
        if sampling_rate.size != 1:
            raise ValueError("SOFA Data.SamplingRate must contain one value.")
        sample_rate_hz = float(sampling_rate.reshape(-1)[0])
        convention = decode_hdf5_text(stream.attrs.get("SOFAConventions", ""))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("SOFA sampling rate must be finite and positive.")
    return SOFADatasetInfo(
        shape=shape,
        n_measurements=shape[0],
        n_receivers=shape[1],
        n_samples=shape[2],
        sample_rate_hz=sample_rate_hz,
        convention=convention,
    )


def load_sofa_channel(
    path: str | Path,
    *,
    channel_index: int = 0,
    measurement_indices: ArrayLike | None = None,
    time_chunk_samples: int = 8192,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    SOFADatasetInfo,
]:
    """Load one receiver channel from selected SOFA measurements.

    Parameters
    ----------
    path
        Path to a SOFA file with ``Data.IR`` shape ``(M,R,N)``.
    channel_index
        Zero-based receiver-channel index. For an ACN spherical-harmonic RIR,
        channel zero is the omnidirectional channel.
    measurement_indices
        Optional strictly increasing measurement indices, shape ``(Q,)``.
        ``None`` loads all measurements.
    time_chunk_samples
        Positive number of samples read per HDF5 block.

    Returns
    -------
    rirs, listener_positions_m, source_positions_m, info
        RIR samples ``(Q,N)`` in their stored amplitude units, listener and
        source Cartesian positions ``(Q,3)`` in metres, and file metadata.
    """

    info = inspect_sofa_dataset(path)
    if isinstance(channel_index, bool) or not isinstance(channel_index, int):
        raise TypeError("channel_index must be an integer.")
    if not 0 <= channel_index < info.n_receivers:
        raise ValueError("channel_index is outside the available SOFA receivers.")
    if isinstance(time_chunk_samples, bool) or not isinstance(time_chunk_samples, int):
        raise TypeError("time_chunk_samples must be an integer.")
    if time_chunk_samples <= 0:
        raise ValueError("time_chunk_samples must be positive.")

    if measurement_indices is None:
        selected = np.arange(info.n_measurements, dtype=np.int64)
    else:
        selected = np.asarray(measurement_indices)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError(
                "measurement_indices must be a one-dimensional integer array."
            )
        selected = selected.astype(np.int64, copy=False)
        if selected.size == 0:
            raise ValueError("measurement_indices must not be empty.")
        if np.any((selected < 0) | (selected >= info.n_measurements)):
            raise ValueError("measurement_indices contains an out-of-range index.")
        if np.any(np.diff(selected) <= 0):
            raise ValueError("measurement_indices must be strictly increasing.")

    h5py = h5py_module()
    rirs = np.empty((selected.size, info.n_samples), dtype=np.float64)
    with h5py.File(Path(path), "r") as stream:
        source = stream["Data.IR"]
        for start in range(0, info.n_samples, time_chunk_samples):
            stop = min(start + time_chunk_samples, info.n_samples)
            rirs[:, start:stop] = source[selected, channel_index, start:stop]

        positions: list[NDArray[np.float64]] = []
        for name in ("ListenerPosition", "SourcePosition"):
            if name not in stream:
                raise ValueError(f"SOFA dataset must contain '{name}'.")
            values = np.asarray(stream[name], dtype=np.float64)
            if (
                values.ndim != 2
                or values.shape[0]
                not in (
                    1,
                    info.n_measurements,
                )
                or values.shape[1] < 3
            ):
                raise ValueError(f"SOFA {name} must have shape (1,3) or (M,3).")
            if values.shape[0] == 1:
                values = np.repeat(values, info.n_measurements, axis=0)
            positions.append(values[selected, :3].copy())
    return rirs, positions[0], positions[1], info
