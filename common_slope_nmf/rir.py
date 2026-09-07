"""Real-RIR loading and STFT-power preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly, stft


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


@dataclass(frozen=True)
class STFTPower:
    """Selected complex-STFT powers for a batch of RIRs.

    Attributes
    ----------
    observed_power
        Positive squared STFT magnitudes, shape ``(R,F,N)``, in the
        squared units of the input RIR samples.
    frequencies_hz
        Selected one-sided FFT frequencies, shape ``(F,)``, in hertz.
    times_s
        Elapsed frame times with frame zero as the decay origin, shape
        ``(N,)``, in seconds.
    sample_rate_hz
        Analysis sampling rate in samples per second.
    """

    observed_power: NDArray[np.float64]
    frequencies_hz: NDArray[np.float64]
    times_s: NDArray[np.float64]
    sample_rate_hz: float


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Reading MATLAB v7.3 SRIR data requires h5py; install the "
            "project's examples extra."
        ) from exc
    return h5py


def _decode_hdf5_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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

    h5py = _h5py()
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
        convention = _decode_hdf5_text(stream.attrs.get("SOFAConventions", ""))
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

    h5py = _h5py()
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

    h5py = _h5py()
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

    h5py = _h5py()
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


def global_energy_onset(
    rirs: ArrayLike,
    sample_rate_hz: float,
    *,
    threshold_db: float = -40.0,
    smoothing_duration_s: float = 1e-3,
) -> tuple[int, NDArray[np.float64]]:
    """Estimate one global onset from pooled receiver energy.

    Parameters
    ----------
    rirs
        Real RIR samples, shape ``(R,L)``.
    sample_rate_hz
        Positive source sampling rate in samples per second.
    threshold_db
        Non-positive power threshold relative to the peak smoothed pooled
        energy, in decibels.
    smoothing_duration_s
        Positive moving-average duration in seconds.

    Returns
    -------
    onset_sample, pooled_energy
        First source-rate sample at or above the relative threshold and the
        smoothed mean-square energy across receivers, shape ``(L,)``. The same
        onset is applied to every RIR; individual responses are not aligned.
    """

    samples = np.asarray(rirs, dtype=np.float64)
    if samples.ndim != 2 or samples.size == 0:
        raise ValueError("rirs must have non-empty shape (R,L).")
    if not np.all(np.isfinite(samples)):
        raise ValueError("rirs must contain finite values.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive.")
    if not np.isfinite(threshold_db) or threshold_db > 0.0:
        raise ValueError("threshold_db must be finite and non-positive.")
    if not np.isfinite(smoothing_duration_s) or smoothing_duration_s <= 0.0:
        raise ValueError("smoothing_duration_s must be finite and positive.")

    pooled_energy = np.mean(samples**2, axis=0)
    smoothing_samples = max(1, int(round(smoothing_duration_s * sample_rate_hz)))
    if smoothing_samples > 1:
        kernel = np.full(smoothing_samples, 1.0 / smoothing_samples)
        pooled_energy = np.convolve(pooled_energy, kernel, mode="same")
    peak = float(np.max(pooled_energy))
    if peak <= 0.0:
        raise ValueError("rirs must contain non-zero energy.")
    threshold = peak * 10.0 ** (threshold_db / 10.0)
    onset_sample = int(np.flatnonzero(pooled_energy >= threshold)[0])
    return onset_sample, pooled_energy


def resample_rirs(
    rirs: ArrayLike,
    source_rate_hz: float,
    target_rate_hz: float,
) -> NDArray[np.float64]:
    """Polyphase-resample real RIRs along their sample axis.

    Parameters
    ----------
    rirs
        Real RIR samples, shape ``(R,L)``.
    source_rate_hz, target_rate_hz
        Positive sampling rates in samples per second.

    Returns
    -------
    ndarray
        Resampled RIRs, shape ``(R,L_new)``, in unchanged amplitude units.
    """

    samples = np.asarray(rirs, dtype=np.float64)
    if samples.ndim != 2 or samples.size == 0:
        raise ValueError("rirs must have non-empty shape (R,L).")
    for name, value in (
        ("source_rate_hz", source_rate_hz),
        ("target_rate_hz", target_rate_hz),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    ratio = Fraction(float(target_rate_hz) / float(source_rate_hz)).limit_denominator(
        10_000
    )
    return np.asarray(
        resample_poly(samples, ratio.numerator, ratio.denominator, axis=1),
        dtype=np.float64,
    )


def rir_stft_power(
    rirs: ArrayLike,
    sample_rate_hz: float,
    *,
    frame_size_samples: int = 256,
    hop_size_samples: int = 128,
    fft_size_samples: int = 384,
    minimum_frequency_hz: float = 62.5,
    maximum_frequency_hz: float = 8_000.0,
) -> STFTPower:
    """Compute selected one-sided Hann-STFT powers from real RIRs.

    Parameters
    ----------
    rirs
        Real RIR samples beginning at a common decay origin, shape ``(R,L)``.
    sample_rate_hz
        Positive analysis sampling rate in samples per second.
    frame_size_samples, hop_size_samples, fft_size_samples
        Positive window, hop, and FFT sizes in samples. The FFT size must not
        be smaller than the frame size.
    minimum_frequency_hz, maximum_frequency_hz
        Inclusive selected frequency interval in hertz.

    Returns
    -------
    STFTPower
        Positive squared complex-STFT magnitudes ``(R,F,N)``, selected
        frequencies in hertz, and elapsed frame times in seconds. Boundary
        extension and end padding are disabled, and frame zero is assigned
        elapsed time zero in the decay model.
    """

    samples = np.asarray(rirs, dtype=np.float64)
    if samples.ndim != 2 or samples.size == 0:
        raise ValueError("rirs must have non-empty shape (R,L).")
    if not np.all(np.isfinite(samples)):
        raise ValueError("rirs must contain finite values.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive.")
    for name, value in (
        ("frame_size_samples", frame_size_samples),
        ("hop_size_samples", hop_size_samples),
        ("fft_size_samples", fft_size_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if fft_size_samples < frame_size_samples:
        raise ValueError("fft_size_samples must not be below frame_size_samples.")
    if hop_size_samples > frame_size_samples:
        raise ValueError("hop_size_samples must not exceed frame_size_samples.")
    nyquist_hz = sample_rate_hz / 2.0
    if (
        not np.isfinite(minimum_frequency_hz)
        or not np.isfinite(maximum_frequency_hz)
        or minimum_frequency_hz < 0.0
        or maximum_frequency_hz > nyquist_hz
        or minimum_frequency_hz > maximum_frequency_hz
    ):
        raise ValueError("frequency limits must define an interval within Nyquist.")
    if samples.shape[1] < frame_size_samples:
        raise ValueError("rirs must contain at least one complete STFT frame.")

    frequencies_hz, _, coefficients = stft(
        samples,
        fs=float(sample_rate_hz),
        window="hann",
        nperseg=frame_size_samples,
        noverlap=frame_size_samples - hop_size_samples,
        nfft=fft_size_samples,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=1,
        scaling="spectrum",
    )
    selected = (frequencies_hz >= minimum_frequency_hz) & (
        frequencies_hz <= maximum_frequency_hz
    )
    if not np.any(selected):
        raise ValueError("frequency interval does not contain an FFT bin.")
    observed_power = np.abs(coefficients[:, selected, :]) ** 2
    positive_scale = np.maximum(
        np.max(observed_power, axis=2, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    observed_power = np.maximum(
        observed_power,
        positive_scale * np.finfo(np.float64).eps ** 2,
    )
    n_frames = observed_power.shape[2]
    times_s = (
        np.arange(n_frames, dtype=np.float64) * hop_size_samples / float(sample_rate_hz)
    )
    return STFTPower(
        observed_power=np.asarray(observed_power, dtype=np.float64),
        frequencies_hz=np.asarray(frequencies_hz[selected], dtype=np.float64),
        times_s=times_s,
        sample_rate_hz=float(sample_rate_hz),
    )


def select_stft_frames(
    transformed: STFTPower,
    *,
    discard_initial_frames: int,
    discard_final_frames: int,
) -> STFTPower:
    """Discard leading and trailing STFT frames and reset elapsed time.

    Parameters
    ----------
    transformed
        STFT powers with shape ``(R,F,N)`` and matching frequency/time axes.
    discard_initial_frames, discard_final_frames
        Non-negative counts removed from the start and end of the frame axis.

    Returns
    -------
    STFTPower
        Selected powers ``(R,F,N_new)``. The first retained frame is assigned
        elapsed time zero, so fitted amplitudes refer to that common origin.
    """

    for name, value in (
        ("discard_initial_frames", discard_initial_frames),
        ("discard_final_frames", discard_final_frames),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    power = np.asarray(transformed.observed_power, dtype=np.float64)
    frequencies = np.asarray(transformed.frequencies_hz, dtype=np.float64)
    times = np.asarray(transformed.times_s, dtype=np.float64)
    if power.ndim != 3 or power.shape[1:] != (frequencies.size, times.size):
        raise ValueError("transformed STFT arrays have inconsistent shapes.")
    stop = times.size - discard_final_frames
    if discard_initial_frames >= stop:
        raise ValueError("frame discards must leave at least one frame.")
    selected_times = times[discard_initial_frames:stop].copy()
    selected_times -= selected_times[0]
    return STFTPower(
        observed_power=power[:, :, discard_initial_frames:stop].copy(),
        frequencies_hz=frequencies.copy(),
        times_s=selected_times,
        sample_rate_hz=float(transformed.sample_rate_hz),
    )
