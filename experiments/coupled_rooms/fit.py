"""Fit weighted no-floor pseudo-SAGE to the three-coupled-room SRIR data.

The experiment uses one global onset, begins the decay 50 ms later, resamples
the omnidirectional responses to 24 kHz, and computes 256-sample Hann STFTs
with a 128-sample hop and 384-point FFT. This produces the exact frequency
grid 62.5, 125, ..., 8000 Hz. A 100-receiver four-frequency pilot gates the
full fit over all receivers and all 128 selected bins.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    fit_coarse_decay,
    global_energy_onset,
    pseudo_decay_sage,
    rate_to_t60,
    resample_rirs,
    rir_stft_power,
    t60_to_rate,
)
from experiments.datasets import load_srir_channel
from experiments._run_output import create_run_output_dir

DATASET_PATH = Path("data/srirs.mat")
CACHE_PATH = Path("data/coupled_rooms_stft_power.npy")
CACHE_METADATA_PATH = Path("data/coupled_rooms_stft_metadata.npz")
INITIAL_T60_S = np.array([0.73, 1.43, 3.48], dtype=np.float64)
PUBLISHED_OCTAVE_FREQUENCIES_HZ = np.array(
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
)
PUBLISHED_OCTAVE_T60_S = np.array(
    [
        [1.425, 1.675, 2.025],
        [0.725, 1.375, 3.425],
        [0.775, 1.625, 3.825],
        [0.775, 1.525, 3.925],
        [0.725, 1.625, 3.725],
        [0.675, 1.575, 3.325],
        [0.825, 1.475, 2.175],
        [0.525, 0.925, 1.225],
    ],
    dtype=np.float64,
)
PILOT_FREQUENCIES_HZ = np.array([250.0, 500.0, 1000.0, 2000.0])
N_COMPONENTS = 3
TARGET_SAMPLE_RATE_HZ = 24_000.0
FRAME_SIZE_SAMPLES = 256
HOP_SIZE_SAMPLES = 128
FFT_SIZE_SAMPLES = 384
MINIMUM_FREQUENCY_HZ = 62.5
MAXIMUM_FREQUENCY_HZ = 8_000.0
ONSET_THRESHOLD_DB = -40.0
DECAY_DELAY_S = 0.050
T60_BOUNDS_S = (0.2, 6.0)
COMPONENT_WEIGHT_POWER = 1.0


@dataclass(frozen=True)
class FrequencyFit:
    """One independently stopped frequency-bin fit."""

    frequency_index: int
    frequency_hz: float
    initial_t60_s: np.ndarray
    t60_s: np.ndarray
    amplitudes: np.ndarray
    objective_history: np.ndarray
    t60_history_s: np.ndarray
    n_iter: int
    converged: bool


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument(
        "--cache-metadata", type=Path, default=CACHE_METADATA_PATH
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--pilot-rirs", type=int, default=100)
    parser.add_argument("--pilot-max-iter", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--initialization",
        choices=("distinct", "equal-log-linear"),
        default="distinct",
        help=(
            "Use the fixed distinct triplet or repeat one pooled log-power "
            "decay estimate across all components."
        ),
    )
    parser.add_argument(
        "--highest-bins",
        type=int,
        default=0,
        help="For the full stage, fit only this many highest-frequency bins.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "pilot", "full"),
        default="all",
        help="Run the complete gated experiment or one resumable stage.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse an existing STFT-power cache after validating metadata.",
    )
    args = parser.parse_args()
    for name in ("pilot_rirs", "pilot_max_iter", "max_iter", "jobs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    if args.highest_bins < 0:
        parser.error("--highest-bins must be non-negative")
    return args


def _last_nonzero_sample(rirs: np.ndarray) -> int:
    active = np.flatnonzero(np.any(rirs != 0.0, axis=0))
    if active.size == 0:
        raise ValueError("the selected RIR channel contains no energy.")
    return int(active[-1])


def initial_decay_rates(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    initialization: str,
) -> np.ndarray:
    """Return experiment decay-rate starts in inverse seconds, shape ``(F,3)``.

    ``"distinct"`` repeats the configured three-time triplet over frequency.
    ``"equal-log-linear"`` fits one pooled log-power rate per frequency over
    the complete no-floor decay window and repeats it across all components.
    """

    n_frequencies = observed_power.shape[1]
    if initialization == "distinct":
        configured = np.asarray(t60_to_rate(INITIAL_T60_S))
        return np.repeat(configured[np.newaxis, :], n_frequencies, axis=0)
    if initialization == "equal-log-linear":
        coarse_fit = fit_coarse_decay(observed_power, times_s)
        coarse_rate = np.clip(
            coarse_fit.rate_per_s,
            t60_to_rate(T60_BOUNDS_S[1]),
            t60_to_rate(T60_BOUNDS_S[0]),
        )
        return np.repeat(coarse_rate[:, np.newaxis], N_COMPONENTS, axis=1)
    raise ValueError(f"unknown initialization {initialization!r}.")


def prepare_power_cache(
    dataset_path: Path,
    cache_path: Path,
    metadata_path: Path,
) -> None:
    """Load, globally crop, resample, transform, and cache all RIR powers."""

    print(f"loading omnidirectional SRIRs from {dataset_path}", flush=True)
    rirs, receiver_positions_m, info = load_srir_channel(
        dataset_path, channel_index=0
    )
    onset_sample, pooled_energy = global_energy_onset(
        rirs,
        info.sample_rate_hz,
        threshold_db=ONSET_THRESHOLD_DB,
    )
    delay_samples = int(round(DECAY_DELAY_S * info.sample_rate_hz))
    decay_start_sample = onset_sample + delay_samples
    final_sample = _last_nonzero_sample(rirs) + 1
    if final_sample - decay_start_sample < FRAME_SIZE_SAMPLES:
        raise ValueError("global onset and delay leave no complete STFT frame.")
    print(
        f"global onset={onset_sample / info.sample_rate_hz:.6f} s; "
        f"decay origin={decay_start_sample / info.sample_rate_hz:.6f} s; "
        f"source-rate end={final_sample / info.sample_rate_hz:.6f} s",
        flush=True,
    )
    selected_rirs = rirs[:, decay_start_sample:final_sample]
    del rirs
    resampled = resample_rirs(
        selected_rirs, info.sample_rate_hz, TARGET_SAMPLE_RATE_HZ
    )
    del selected_rirs
    transformed = rir_stft_power(
        resampled,
        TARGET_SAMPLE_RATE_HZ,
        frame_size_samples=FRAME_SIZE_SAMPLES,
        hop_size_samples=HOP_SIZE_SAMPLES,
        fft_size_samples=FFT_SIZE_SAMPLES,
        minimum_frequency_hz=MINIMUM_FREQUENCY_HZ,
        maximum_frequency_hz=MAXIMUM_FREQUENCY_HZ,
    )
    del resampled
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, transformed.observed_power, allow_pickle=False)
    peak = float(np.max(pooled_energy))
    np.savez_compressed(
        metadata_path,
        frequencies_hz=transformed.frequencies_hz,
        times_s=transformed.times_s,
        receiver_positions_m=receiver_positions_m,
        source_sample_rate_hz=np.asarray(info.sample_rate_hz),
        target_sample_rate_hz=np.asarray(TARGET_SAMPLE_RATE_HZ),
        source_shape=np.asarray(info.shape, dtype=np.int64),
        channel_index=np.asarray(0),
        global_onset_sample=np.asarray(onset_sample),
        global_onset_threshold_db=np.asarray(ONSET_THRESHOLD_DB),
        decay_delay_s=np.asarray(DECAY_DELAY_S),
        decay_start_sample=np.asarray(decay_start_sample),
        source_final_sample=np.asarray(final_sample),
        frame_size_samples=np.asarray(FRAME_SIZE_SAMPLES),
        hop_size_samples=np.asarray(HOP_SIZE_SAMPLES),
        fft_size_samples=np.asarray(FFT_SIZE_SAMPLES),
        pooled_energy_db=10.0
        * np.log10(np.maximum(pooled_energy / peak, np.finfo(float).tiny)),
    )
    print(
        f"cached power {transformed.observed_power.shape} at {cache_path}",
        flush=True,
    )


def validate_power_cache(cache_path: Path, metadata_path: Path) -> None:
    """Reject a missing or configuration-incompatible preprocessing cache."""

    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "STFT cache is missing; run with --stage prepare or omit --reuse-cache."
        )
    power = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_path, allow_pickle=False) as metadata:
        frequencies_hz = metadata["frequencies_hz"]
        times_s = metadata["times_s"]
        checks = {
            "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "frame_size_samples": FRAME_SIZE_SAMPLES,
            "hop_size_samples": HOP_SIZE_SAMPLES,
            "fft_size_samples": FFT_SIZE_SAMPLES,
            "decay_delay_s": DECAY_DELAY_S,
        }
        for name, expected in checks.items():
            if not np.isclose(float(metadata[name]), expected):
                raise ValueError(f"cached {name} does not match this experiment.")
    if power.ndim != 3 or power.shape[1:] != (
        frequencies_hz.size,
        times_s.size,
    ):
        raise ValueError("STFT cache shape does not match its metadata.")
    expected_frequencies = np.arange(1, 129, dtype=np.float64) * 62.5
    if not np.array_equal(frequencies_hz, expected_frequencies):
        raise ValueError("cached frequency grid is not 62.5:62.5:8000 Hz.")


def _fit_one_frequency(
    cache_path: str,
    metadata_path: str,
    frequency_index: int,
    receiver_indices: np.ndarray | None,
    max_iter: int,
    tol: float,
    initialization: str,
    result_path: str,
) -> str:
    power_cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_path, allow_pickle=False) as metadata:
        frequencies_hz = metadata["frequencies_hz"]
        times_s = metadata["times_s"]
    if receiver_indices is None:
        observed_power = np.asarray(
            power_cache[:, frequency_index : frequency_index + 1, :],
            dtype=np.float64,
        )
    else:
        observed_power = np.asarray(
            power_cache[receiver_indices, frequency_index, :], dtype=np.float64
        )[:, np.newaxis, :]

    initial_rates = initial_decay_rates(
        observed_power, times_s, initialization
    )
    initial_t60_s = np.asarray(rate_to_t60(initial_rates[0]))
    head_power = np.mean(observed_power[:, :, :8], axis=2)
    initial_amplitudes = np.repeat(
        (head_power / N_COMPONENTS)[:, :, np.newaxis],
        N_COMPONENTS,
        axis=2,
    )
    result = pseudo_decay_sage(
        observed_power,
        times_s,
        initial_rates,
        rate_bounds_per_s=(
            t60_to_rate(T60_BOUNDS_S[1]),
            t60_to_rate(T60_BOUNDS_S[0]),
        ),
        component_weight_power=COMPONENT_WEIGHT_POWER,
        initial_amplitudes=initial_amplitudes,
        estimate_noise_floor=False,
        max_iter=max_iter,
        tol=tol,
        rate_method="newton",
    )
    labeled_t60_s = np.asarray(rate_to_t60(result.rates_per_s[0]))
    order = np.argsort(labeled_t60_s)
    t60_s = labeled_t60_s[order]
    amplitudes = result.amplitudes[:, 0, order]
    t60_history_s = np.sort(
        np.asarray(rate_to_t60(result.rate_history_per_s[:, 0, :])), axis=1
    )
    np.savez_compressed(
        result_path,
        frequency_index=np.asarray(frequency_index),
        frequency_hz=np.asarray(frequencies_hz[frequency_index]),
        initial_t60_s=initial_t60_s,
        t60_s=t60_s,
        amplitudes=amplitudes,
        objective_history=result.objective_history,
        t60_history_s=t60_history_s,
        n_iter=np.asarray(result.n_iter),
        converged=np.asarray(result.converged),
    )
    return result_path


def _read_frequency_fit(path: str | Path) -> FrequencyFit:
    with np.load(path, allow_pickle=False) as result:
        return FrequencyFit(
            frequency_index=int(result["frequency_index"]),
            frequency_hz=float(result["frequency_hz"]),
            initial_t60_s=(
                result["initial_t60_s"].copy()
                if "initial_t60_s" in result
                else np.full(N_COMPONENTS, np.nan)
            ),
            t60_s=result["t60_s"].copy(),
            amplitudes=result["amplitudes"].copy(),
            objective_history=result["objective_history"].copy(),
            t60_history_s=result["t60_history_s"].copy(),
            n_iter=int(result["n_iter"]),
            converged=bool(result["converged"]),
        )


def run_frequency_fits(
    cache_path: Path,
    metadata_path: Path,
    frequency_indices: np.ndarray,
    receiver_indices: np.ndarray | None,
    max_iter: int,
    tol: float,
    jobs: int,
    initialization: str,
    result_dir: Path,
) -> list[FrequencyFit]:
    """Run independently stopped pseudo-SAGE fits, optionally in parallel."""

    result_dir.mkdir(parents=True, exist_ok=False)
    fits: list[FrequencyFit] = []
    with ProcessPoolExecutor(max_workers=min(jobs, frequency_indices.size)) as pool:
        futures = {}
        for frequency_index in frequency_indices:
            result_path = result_dir / f"frequency_{frequency_index:03d}.npz"
            future = pool.submit(
                _fit_one_frequency,
                str(cache_path),
                str(metadata_path),
                int(frequency_index),
                receiver_indices,
                max_iter,
                tol,
                initialization,
                str(result_path),
            )
            futures[future] = int(frequency_index)
        for completed_count, future in enumerate(as_completed(futures), start=1):
            fit = _read_frequency_fit(future.result())
            fits.append(fit)
            print(
                f"fit {completed_count:03d}/{frequency_indices.size:03d}: "
                f"{fit.frequency_hz:7.1f} Hz, sweeps={fit.n_iter:4d}, "
                f"converged={fit.converged}, T60={np.round(fit.t60_s, 3)} s",
                flush=True,
            )
    return sorted(fits, key=lambda fit: fit.frequency_index)


def pilot_is_healthy(fits: list[FrequencyFit]) -> bool:
    """Return whether pilot fits are finite and improve their IS objective."""

    if not fits:
        return False
    finite = all(
        np.all(np.isfinite(fit.t60_s))
        and np.all(np.isfinite(fit.objective_history))
        for fit in fits
    )
    objective_healthy = all(
        fit.objective_history[-1] <= 1.10 * fit.objective_history[0]
        for fit in fits
    )
    return finite and objective_healthy


def _stack_fits(
    fits: list[FrequencyFit], max_iter: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    frequencies_hz = np.array([fit.frequency_hz for fit in fits])
    initial_t60_s = np.stack([fit.initial_t60_s for fit in fits])
    t60_s = np.stack([fit.t60_s for fit in fits])
    amplitudes = np.stack([fit.amplitudes for fit in fits], axis=1)
    objective_history = np.full((len(fits), max_iter + 1), np.nan)
    t60_history_s = np.full(
        (len(fits), max_iter + 1, N_COMPONENTS), np.nan
    )
    for index, fit in enumerate(fits):
        length = fit.n_iter + 1
        objective_history[index, :length] = fit.objective_history
        t60_history_s[index, :length] = fit.t60_history_s
    return (
        frequencies_hz,
        initial_t60_s,
        t60_s,
        amplitudes,
        objective_history,
        t60_history_s,
    )


def save_summary(
    fits: list[FrequencyFit],
    max_iter: int,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save tabular, numerical, and graphical summaries for fitted bins."""

    (
        frequencies_hz,
        initial_t60_s,
        t60_s,
        amplitudes,
        objective,
        t60_history,
    ) = _stack_fits(fits, max_iter)
    n_iter = np.array([fit.n_iter for fit in fits], dtype=np.int64)
    converged = np.array([fit.converged for fit in fits], dtype=bool)
    final_objective = np.array([fit.objective_history[-1] for fit in fits])
    initial_objective = np.array([fit.objective_history[0] for fit in fits])
    np.savez_compressed(
        output_dir / f"{prefix}_results.npz",
        frequencies_hz=frequencies_hz,
        fitted_initial_t60_s=initial_t60_s,
        estimated_t60_s=t60_s,
        estimated_amplitudes=amplitudes,
        objective_history=objective,
        t60_history_s=t60_history,
        n_iter=n_iter,
        converged=converged,
        distinct_initial_t60_s=INITIAL_T60_S,
        published_octave_frequencies_hz=PUBLISHED_OCTAVE_FREQUENCIES_HZ,
        published_octave_t60_s=PUBLISHED_OCTAVE_T60_S,
        t60_bounds_s=np.asarray(T60_BOUNDS_S),
        component_weight_power=np.asarray(COMPONENT_WEIGHT_POWER),
        estimate_noise_floor=np.asarray(False),
    )

    with (output_dir / f"{prefix}_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frequency_hz",
                "initial_t60_1_s",
                "initial_t60_2_s",
                "initial_t60_3_s",
                "short_t60_s",
                "middle_t60_s",
                "long_t60_s",
                "sweeps",
                "converged",
                "initial_is",
                "final_is",
                "final_over_initial_is",
            ]
        )
        for index, fit in enumerate(fits):
            writer.writerow(
                [
                    fit.frequency_hz,
                    *fit.initial_t60_s,
                    *fit.t60_s,
                    fit.n_iter,
                    fit.converged,
                    initial_objective[index],
                    final_objective[index],
                    final_objective[index] / initial_objective[index],
                ]
            )

def _pilot_frequency_indices(frequencies_hz: np.ndarray) -> np.ndarray:
    indices = [
        int(np.flatnonzero(frequencies_hz == frequency_hz)[0])
        for frequency_hz in PILOT_FREQUENCIES_HZ
    ]
    return np.asarray(indices, dtype=np.int64)


def main() -> None:
    """Prepare the data, gate on the pilot, then run the full experiment."""

    args = _arguments()
    if args.stage in ("all", "prepare") and not args.reuse_cache:
        prepare_power_cache(args.dataset, args.cache, args.cache_metadata)
    validate_power_cache(args.cache, args.cache_metadata)
    if args.stage == "prepare":
        return

    output_dir = create_run_output_dir(args.output_root)
    power = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    with np.load(args.cache_metadata, allow_pickle=False) as metadata:
        frequencies_hz = metadata["frequencies_hz"]
    pilot_receiver_count = min(args.pilot_rirs, power.shape[0])
    pilot_receivers = np.unique(
        np.linspace(0, power.shape[0] - 1, pilot_receiver_count).astype(np.int64)
    )

    if args.stage in ("all", "pilot"):
        pilot_fits = run_frequency_fits(
            args.cache,
            args.cache_metadata,
            _pilot_frequency_indices(frequencies_hz),
            pilot_receivers,
            args.pilot_max_iter,
            args.tol,
            args.jobs,
            args.initialization,
            output_dir / "pilot_frequency_results",
        )
        save_summary(pilot_fits, args.pilot_max_iter, output_dir, "pilot")
        healthy = pilot_is_healthy(pilot_fits)
        print(f"pilot numerical-health gate: {healthy}", flush=True)
        if not healthy:
            raise RuntimeError(
                "pilot produced non-finite values or a >10% objective increase; "
                "the full fit was not started."
            )
    if args.stage == "pilot":
        print(f"pilot outputs: {output_dir}", flush=True)
        return

    if args.highest_bins > frequencies_hz.size:
        raise ValueError("--highest-bins exceeds the cached frequency count.")
    first_frequency = (
        frequencies_hz.size - args.highest_bins if args.highest_bins else 0
    )
    frequency_indices = np.arange(
        first_frequency, frequencies_hz.size, dtype=np.int64
    )
    full_fits = run_frequency_fits(
        args.cache,
        args.cache_metadata,
        frequency_indices,
        None,
        args.max_iter,
        args.tol,
        args.jobs,
        args.initialization,
        output_dir / "full_frequency_results",
    )
    save_summary(full_fits, args.max_iter, output_dir, "full")
    print(f"full outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
