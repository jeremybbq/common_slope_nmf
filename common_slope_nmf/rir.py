"""Array-in STFT-power preprocessing for already-read RIR samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import stft

@dataclass(frozen=True)
class STFTPower:
    """Selected complex-STFT powers for a batch of RIRs.

    Attributes
    ----------
    observed_power
        Positive squared STFT magnitudes, shape ``(R,F,N)``, in the squared units of the input RIR samples.
    frequencies_hz
        Selected one-sided FFT frequencies, shape ``(F,)``, in hertz.
    frame_time_s
        Time vector of STFT frame elapsed time in seconds, shape ``(N,)``. Frame zero is the decay origin.
    sample_rate_hz
        Analysis sampling rate in samples per second.
    """

    observed_power: NDArray[np.float64]
    frequencies_hz: NDArray[np.float64]
    frame_time_s: NDArray[np.float64]
    sample_rate_hz: float

def global_energy_onset(
    rirs: ArrayLike,
    sample_rate_hz: float,
) -> int:
    """Placeholder for one shared onset sample applied to every RIR.

    Not implemented. Callers currently supply RIRs that already begin at a common decay origin.
    """

    raise NotImplementedError("global_energy_onset is a placeholder.")


def rir_stft_power(
    rirs: ArrayLike,
    sample_rate_hz: float,
    *,
    frame_size_samples: int = 512,
    hop_size_samples: int = 256,
    fft_size_samples: int = 512,
    minimum_frequency_hz: float = 125.0,
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
        Positive window, hop, and FFT sizes in samples. The FFT size must not be smaller than the frame size.
    minimum_frequency_hz, maximum_frequency_hz
        Inclusive selected frequency interval in hertz.

    Returns
    -------
    STFTPower
        Positive squared complex-STFT magnitudes ``(R,F,N)``, selected frequencies in hertz, and STFT frame elapsed times in seconds. Boundary extension and end padding are disabled, and frame zero is assigned elapsed time zero in the decay model.
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
    frame_time_s = (
        np.arange(n_frames, dtype=np.float64) * hop_size_samples / float(sample_rate_hz)
    ) # NOTE: this time vector marks the start of each frame, not the center of each frame.
    return STFTPower(
        observed_power=np.asarray(observed_power, dtype=np.float64),
        frequencies_hz=np.asarray(frequencies_hz[selected], dtype=np.float64),
        frame_time_s=frame_time_s,
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
        Selected powers ``(R,F,N_new)``. The first retained frame is assigned elapsed time zero, so fitted amplitudes refer to that common origin.
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
    frame_time_s = np.asarray(transformed.frame_time_s, dtype=np.float64)
    if power.ndim != 3 or power.shape[1:] != (frequencies.size, frame_time_s.size):
        raise ValueError("transformed STFT arrays have inconsistent shapes.")
    stop = frame_time_s.size - discard_final_frames
    if discard_initial_frames >= stop:
        raise ValueError("frame discards must leave at least one frame.")
    selected_frame_time_s = frame_time_s[discard_initial_frames:stop].copy()
    selected_frame_time_s -= selected_frame_time_s[0] # NOTE: the time before chosen onset is not considered, the first used frame is assigned elapsed time zero.
    return STFTPower(
        observed_power=power[:, :, discard_initial_frames:stop].copy(),
        frequencies_hz=frequencies.copy(),
        frame_time_s=selected_frame_time_s,
        sample_rate_hz=float(transformed.sample_rate_hz),
    )
