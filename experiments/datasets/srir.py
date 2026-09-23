"""MATLAB v7.3 SRIR dataset readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._hdf5 import h5py_module

@dataclass(frozen=True)
class SRIRDatasetInfo:
    """Layout metadata for the public coupled-room MATLAB dataset.

    Attributes
    ----------
    shape
        On-disk ``srirDataset/srirs`` array shape.
    time_axis, receiver_axis, channel_axis
        Axis indices in the on-disk HDF5 array.
    n_samples, n_receivers, n_channels
        Number of time samples, receiver positions, and response channels.
    sample_rate_hz
        Sampling rate in samples per second.
    """

    shape: tuple[int, int, int]
    time_axis: int
    receiver_axis: int
    channel_axis: int
    n_samples: int
    n_receivers: int
    n_channels: int
    sample_rate_hz: float


def inspect_srir_dataset(path: str | Path) -> SRIRDatasetInfo:
    """Inspect the public three-coupled-room MATLAB v7.3 dataset.

    Parameters
    ----------
    path
        Path to ``srirs.mat``.

    Returns
    -------
    SRIRDatasetInfo
        Detected time, receiver, and channel axes and the sampling rate in
        hertz. The longest array axis is treated as time and the shortest as
        channel, matching the published dataset and its LINEX loader.
    """

    h5py = h5py_module()
    with h5py.File(Path(path), "r") as stream:
        if "srirDataset" not in stream:
            raise ValueError("dataset must contain the group 'srirDataset'.")
        group = stream["srirDataset"]
        if "srirs" not in group:
            raise ValueError("srirDataset must contain the array 'srirs'.")
        shape = tuple(int(value) for value in group["srirs"].shape)
        if len(shape) != 3 or len(set(shape)) != 3:
            raise ValueError(
                "srirDataset/srirs must be a 3-D array with uniquely "
                "identifiable time, receiver, and channel axes."
            )
        time_axis = int(np.argmax(shape))
        channel_axis = int(np.argmin(shape))
        receiver_axis = next(
            axis for axis in range(3) if axis not in (time_axis, channel_axis)
        )
        sample_rate_hz = (
            float(np.asarray(group["fs"]).squeeze()) if "fs" in group else 48_000.0
        )
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("dataset sampling rate must be finite and positive.")
    return SRIRDatasetInfo(
        shape=shape,
        time_axis=time_axis,
        receiver_axis=receiver_axis,
        channel_axis=channel_axis,
        n_samples=shape[time_axis],
        n_receivers=shape[receiver_axis],
        n_channels=shape[channel_axis],
        sample_rate_hz=sample_rate_hz,
    )


def load_srir_channel(
    path: str | Path,
    *,
    channel_index: int = 0,
    receiver_indices: ArrayLike | None = None,
    time_chunk_samples: int = 8192,
) -> tuple[NDArray[np.float64], NDArray[np.float64], SRIRDatasetInfo]:
    """Load one response channel from selected coupled-room receivers.

    Parameters
    ----------
    path
        Path to the MATLAB v7.3 ``srirs.mat`` dataset.
    channel_index
        Zero-based response-channel index. Channel zero is the published
        omnidirectional response used by the LINEX comparison.
    receiver_indices
        Optional increasing receiver indices, shape ``(R,)``. ``None`` loads
        every receiver.
    time_chunk_samples
        Positive number of source-rate samples read per HDF5 block.

    Returns
    -------
    rirs, receiver_positions_m, info
        RIR samples with shape ``(R,L)``, receiver ``x,y`` positions in metres
        with shape ``(R,2)``, and detected dataset metadata. RIR samples are
        returned as ``float64`` without normalization.
    """

    info = inspect_srir_dataset(path)
    if isinstance(channel_index, bool) or not isinstance(channel_index, int):
        raise TypeError("channel_index must be an integer.")
    if not 0 <= channel_index < info.n_channels:
        raise ValueError("channel_index is outside the available channels.")
    if isinstance(time_chunk_samples, bool) or not isinstance(time_chunk_samples, int):
        raise TypeError("time_chunk_samples must be an integer.")
    if time_chunk_samples <= 0:
        raise ValueError("time_chunk_samples must be positive.")

    if receiver_indices is None:
        selected = np.arange(info.n_receivers, dtype=np.int64)
    else:
        selected = np.asarray(receiver_indices)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError(
                "receiver_indices must be a one-dimensional integer array."
            )
        selected = selected.astype(np.int64, copy=False)
        if selected.size == 0:
            raise ValueError("receiver_indices must not be empty.")
        if np.any((selected < 0) | (selected >= info.n_receivers)):
            raise ValueError("receiver_indices contains an out-of-range index.")
        if np.any(np.diff(selected) <= 0):
            raise ValueError("receiver_indices must be strictly increasing.")

    h5py = h5py_module()
    rirs = np.empty((selected.size, info.n_samples), dtype=np.float64)
    with h5py.File(Path(path), "r") as stream:
        group = stream["srirDataset"]
        source = group["srirs"]
        remaining_axes = [axis for axis in range(3) if axis != info.channel_axis]
        receiver_axis_after_selection = remaining_axes.index(info.receiver_axis)
        time_axis_after_selection = remaining_axes.index(info.time_axis)
        for start in range(0, info.n_samples, time_chunk_samples):
            stop = min(start + time_chunk_samples, info.n_samples)
            source_index: list[int | slice] = [slice(None)] * 3
            source_index[info.channel_axis] = channel_index
            source_index[info.time_axis] = slice(start, stop)
            block = np.moveaxis(
                np.asarray(source[tuple(source_index)]),
                (receiver_axis_after_selection, time_axis_after_selection),
                (0, 1),
            )
            rirs[:, start:stop] = block[selected]

        if "rcvPos" not in group:
            positions = np.full((info.n_receivers, 2), np.nan)
        else:
            positions = np.asarray(group["rcvPos"], dtype=np.float64)
            if positions.ndim != 2:
                raise ValueError("srirDataset/rcvPos must be a two-dimensional array.")
            if positions.shape[0] in (2, 3):
                positions = positions.T
            if positions.shape[0] != info.n_receivers or positions.shape[1] < 2:
                raise ValueError("srirDataset/rcvPos does not match the receiver axis.")
        selected_positions = positions[selected, :2].copy()
    return rirs, selected_positions, info
