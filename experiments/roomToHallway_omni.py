"""Fit contribution-weighted CW-SAGE to the raw room-to-hallway omni RIRs.

The four v1.3 Meeting Room to Hallway SOFA files are pooled for common decay
rates.  Each measurement, source condition, and frequency retains independent
decay amplitudes and a time-invariant noise floor.  The analysis uses the ACN
channel-zero omnidirectional response, a 512-sample Hann window, 256-sample
hop, 512-point FFT, and no boundary padding.  It discards the first 10 and last
20 of the 280 complete frames, leaving 250 frames and resetting the first
retained frame to decay time zero.

The default pilot compares K=1,2,3 and two deterministic initializations at six
representative FFT bins.  A full fit is a separate explicit stage so pilot
results can be reviewed first.  Contribution-weighted CW-SAGE is experimental and its
observed IS objective is monitored rather than assumed to be monotone.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    head_power,
    init_decay_sage,
    inspect_sofa_dataset,
    load_sofa_channel,
    cw_decay_sage,
    rate_to_t60,
    rir_stft_power,
    select_stft_frames,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir


DATASET_DIR = Path.home() / "Data" / "Coupled_Room_Transition" / "v1.3"
CONDITION_FILES = (
    (
        "room_los",
        "Room Transition RIRs_Meeting Room to Hallway_"
        "Source in Room_With Line of Sight.sofa",
    ),
    (
        "room_no_los",
        "Room Transition RIRs_Meeting Room to Hallway_"
        "Source in Room_No Line of Sight.sofa",
    ),
    (
        "hallway_los",
        "Room Transition RIRs_Meeting Room to Hallway_"
        "Source in Hallway_With Line of Sight.sofa",
    ),
    (
        "hallway_no_los",
        "Room Transition RIRs_Meeting Room to Hallway_"
        "Source in Hallway_No Line of Sight.sofa",
    ),
)
CACHE_PATH = Path("data/roomToHallway_omni_power.npy")
CACHE_METADATA_PATH = Path("data/roomToHallway_omni_metadata.npz")

SAMPLE_RATE_HZ = 48_000.0
FRAME_SIZE_SAMPLES = 512
HOP_SIZE_SAMPLES = 256
FFT_SIZE_SAMPLES = 512
DISCARD_INITIAL_FRAMES = 10
DISCARD_FINAL_FRAMES = 20
MINIMUM_FREQUENCY_HZ = 187.5
MAXIMUM_FREQUENCY_HZ = 8_000.0
PILOT_TARGET_FREQUENCIES_HZ = np.array(
    [250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0]
)
T60_BOUNDS_S = (0.15, 3.0)
COMPONENT_WEIGHT_POWER = 1.0
RHO_TARGET_FREQUENCY_HZ = 2_000.0
N_HEAD_FRAMES = 8
N_TAIL_FRAMES = 8
FLOOR_MARGIN_DB = 6.0
SHORT_SLOPE_RGB = np.array([67.0, 133.0, 190.0]) / 255.0  # #4385BE
LONG_SLOPE_RGB = np.array([255.0, 155.0, 16.0]) / 255.0  # #FF9B10
WEAK_MIXTURE_RGB = np.zeros(3, dtype=np.float64)  # #000000
FLOOR_RHO_RGB = np.full(3, 0.4, dtype=np.float64)  # #2E2E2E
STRONG_MIXTURE_RGB = np.array([230.0, 228.0, 217.0]) / 255.0  # #E6E4D9


@dataclass(frozen=True)
class FrequencyFit:
    """One independent frequency-bin CW-SAGE result.

    Amplitudes have shape ``(R,K)`` in variance units, the noise floor has
    shape ``(R,)`` in variance units, and all decay times are energy ``T60``
    values in seconds.
    """

    frequency_index: int
    frequency_hz: float
    n_components: int
    initialization: str
    coarse_t60_s: float
    coarse_r_squared: float
    initial_t60_s: np.ndarray
    estimated_t60_s: np.ndarray
    estimated_amplitudes: np.ndarray
    estimated_noise_floor: np.ndarray
    objective_history: np.ndarray
    t60_history_s: np.ndarray
    n_iter: int
    converged: bool


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--cache-metadata", type=Path, default=CACHE_METADATA_PATH)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--stage",
        choices=("prepare", "pilot", "full"),
        default="pilot",
        help="The full stage is deliberately separate from pilot review.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute the STFT cache even when compatible files exist.",
    )
    parser.add_argument("--pilot-rirs-per-condition", type=int, default=25)
    parser.add_argument("--pilot-components", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--full-components", nargs="+", type=int, default=[2])
    parser.add_argument(
        "--initializations",
        nargs="+",
        choices=("equal-log-linear", "log-spaced"),
        default=["equal-log-linear", "log-spaced"],
    )
    parser.add_argument("--pilot-max-iter", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--rho-frequency-hz",
        type=float,
        default=RHO_TARGET_FREQUENCY_HZ,
        help="Nearest fitted frequency used for the final rho space-time map.",
    )
    parser.add_argument(
        "--highest-bins",
        type=int,
        default=0,
        help="Fit only this many highest bins during the full stage.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for name in (
        "pilot_rirs_per_condition",
        "pilot_max_iter",
        "max_iter",
        "jobs",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("pilot_components", "full_components"):
        values = getattr(args, name)
        if not values or any(value <= 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} must contain positive integers")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    if not np.isfinite(args.rho_frequency_hz) or args.rho_frequency_hz <= 0.0:
        parser.error("--rho-frequency-hz must be finite and positive")
    if args.highest_bins < 0:
        parser.error("--highest-bins must be non-negative")
    return args


def _pyplot():
    """Import pyplot with a writable, process-safe cache directory."""

    import os
    import tempfile

    config_dir = Path(tempfile.gettempdir()) / "common_slope_nmf_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Serif",
            "mathtext.it": "Liberation Serif:italic",
            "mathtext.bf": "Liberation Serif:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _dataset_paths(dataset_dir: Path) -> list[tuple[str, Path]]:
    paths = [(name, dataset_dir / filename) for name, filename in CONDITION_FILES]
    missing = [str(path) for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing room-transition SOFA file(s): " + ", ".join(missing)
        )
    return paths


def prepare_power_cache(
    dataset_dir: Path,
    cache_path: Path,
    metadata_path: Path,
) -> None:
    """Transform all four raw SOFA conditions and write a float64 power cache.

    The cache has shape ``(404,84,250)`` for the published v1.3 files.  More
    generally, its axes are RIR, selected FFT frequency, and retained frame.
    """

    paths = _dataset_paths(dataset_dir)
    infos = [inspect_sofa_dataset(path) for _, path in paths]
    if any(not np.isclose(info.sample_rate_hz, SAMPLE_RATE_HZ) for info in infos):
        raise ValueError("every SOFA file must use the configured 48 kHz rate.")
    if any(info.n_samples != infos[0].n_samples for info in infos):
        raise ValueError("all SOFA files must have the same RIR duration.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_name(cache_path.name + ".partial")
    total_rirs = sum(info.n_measurements for info in infos)
    cache = None
    frequencies_hz = None
    times_s = None
    listener_positions: list[np.ndarray] = []
    source_positions: list[np.ndarray] = []
    condition_indices: list[np.ndarray] = []
    measurement_indices: list[np.ndarray] = []
    offset = 0

    for condition_index, ((condition_name, path), info) in enumerate(
        zip(paths, infos, strict=True)
    ):
        print(f"preparing {condition_name}: {path.name}", flush=True)
        rirs, listener, source, loaded_info = load_sofa_channel(path, channel_index=0)
        if loaded_info != info:
            raise RuntimeError("SOFA metadata changed while preparing the cache.")
        transformed = rir_stft_power(
            rirs,
            info.sample_rate_hz,
            frame_size_samples=FRAME_SIZE_SAMPLES,
            hop_size_samples=HOP_SIZE_SAMPLES,
            fft_size_samples=FFT_SIZE_SAMPLES,
            minimum_frequency_hz=MINIMUM_FREQUENCY_HZ,
            maximum_frequency_hz=MAXIMUM_FREQUENCY_HZ,
        )
        original_frame_count = transformed.times_s.size
        transformed = select_stft_frames(
            transformed,
            discard_initial_frames=DISCARD_INITIAL_FRAMES,
            discard_final_frames=DISCARD_FINAL_FRAMES,
        )
        if cache is None:
            frequencies_hz = transformed.frequencies_hz.copy()
            times_s = transformed.times_s.copy()
            cache = np.lib.format.open_memmap(
                temporary_cache,
                mode="w+",
                dtype=np.float64,
                shape=(total_rirs, frequencies_hz.size, times_s.size),
            )
        elif not np.array_equal(
            frequencies_hz, transformed.frequencies_hz
        ) or not np.array_equal(times_s, transformed.times_s):
            raise ValueError("SOFA files produced inconsistent STFT axes.")
        next_offset = offset + info.n_measurements
        cache[offset:next_offset] = transformed.observed_power
        listener_positions.append(listener)
        source_positions.append(source)
        condition_indices.append(
            np.full(info.n_measurements, condition_index, dtype=np.int64)
        )
        measurement_indices.append(np.arange(info.n_measurements, dtype=np.int64))
        offset = next_offset
        del rirs, transformed

    if cache is None or frequencies_hz is None or times_s is None:
        raise RuntimeError("no SOFA data were prepared.")
    cache.flush()
    del cache
    temporary_cache.replace(cache_path)

    first_start_s = DISCARD_INITIAL_FRAMES * HOP_SIZE_SAMPLES / SAMPLE_RATE_HZ
    first_center_s = (
        DISCARD_INITIAL_FRAMES * HOP_SIZE_SAMPLES + FRAME_SIZE_SAMPLES / 2
    ) / SAMPLE_RATE_HZ
    np.savez_compressed(
        metadata_path,
        frequencies_hz=frequencies_hz,
        times_s=times_s,
        condition_names=np.asarray([name for name, _ in paths], dtype="U32"),
        condition_filenames=np.asarray([path.name for _, path in paths], dtype="U160"),
        rir_condition_index=np.concatenate(condition_indices),
        measurement_index=np.concatenate(measurement_indices),
        listener_positions_m=np.concatenate(listener_positions),
        source_positions_m=np.concatenate(source_positions),
        sample_rate_hz=np.asarray(SAMPLE_RATE_HZ),
        channel_index=np.asarray(0),
        frame_size_samples=np.asarray(FRAME_SIZE_SAMPLES),
        hop_size_samples=np.asarray(HOP_SIZE_SAMPLES),
        fft_size_samples=np.asarray(FFT_SIZE_SAMPLES),
        boundary_padding=np.asarray(False),
        original_frame_count=np.asarray(original_frame_count),
        discard_initial_frames=np.asarray(DISCARD_INITIAL_FRAMES),
        discard_final_frames=np.asarray(DISCARD_FINAL_FRAMES),
        first_retained_frame_start_s=np.asarray(first_start_s),
        first_retained_frame_center_s=np.asarray(first_center_s),
        t60_bounds_s=np.asarray(T60_BOUNDS_S),
        component_weight_power=np.asarray(COMPONENT_WEIGHT_POWER),
    )
    print(
        f"cached power {(total_rirs, frequencies_hz.size, times_s.size)} at {cache_path}",
        flush=True,
    )


def validate_power_cache(cache_path: Path, metadata_path: Path) -> None:
    """Reject missing or configuration-incompatible room-transition caches."""

    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("room-transition STFT cache or metadata is missing.")
    power = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_path, allow_pickle=False) as metadata:
        frequencies_hz = np.asarray(metadata["frequencies_hz"])
        times_s = np.asarray(metadata["times_s"])
        checks = {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frame_size_samples": FRAME_SIZE_SAMPLES,
            "hop_size_samples": HOP_SIZE_SAMPLES,
            "fft_size_samples": FFT_SIZE_SAMPLES,
            "discard_initial_frames": DISCARD_INITIAL_FRAMES,
            "discard_final_frames": DISCARD_FINAL_FRAMES,
        }
        for name, expected in checks.items():
            if not np.isclose(float(metadata[name]), expected):
                raise ValueError(f"cached {name} does not match this experiment.")
        condition_index = np.asarray(metadata["rir_condition_index"])
    expected_frequencies = np.fft.rfftfreq(FFT_SIZE_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)
    expected_frequencies = expected_frequencies[
        (expected_frequencies >= MINIMUM_FREQUENCY_HZ)
        & (expected_frequencies <= MAXIMUM_FREQUENCY_HZ)
    ]
    if not np.array_equal(frequencies_hz, expected_frequencies):
        raise ValueError("cached frequency grid is not 187.5:93.75:7968.75 Hz.")
    if power.ndim != 3 or power.shape[1:] != (
        frequencies_hz.size,
        times_s.size,
    ):
        raise ValueError("STFT power shape does not match cached axes.")
    if condition_index.shape != (power.shape[0],):
        raise ValueError("cached condition indices do not match the RIR axis.")


def initial_parameters(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    n_components: int,
    initialization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Construct deterministic rate, equal-amplitude, and floor starts.

    Parameters
    ----------
    observed_power
        Positive variance observations, shape ``(R,1,N)``.
    times_s
        Retained elapsed frame times, shape ``(N,)``, in seconds.
    n_components
        Positive number of decay slopes.
    initialization
        ``"equal-log-linear"`` repeats the pooled rate. ``"log-spaced"``
        places energy T60 values geometrically over a factor-four interval
        centred on the pooled T60 before clipping to the experiment bounds.

    Returns
    -------
    rates_per_s, amplitudes, noise_floor, coarse_t60_s, coarse_r_squared
        Rates ``(1,K)`` in inverse seconds, equal amplitudes ``(R,1,K)`` and
        floor ``(R,1)`` in variance units, and pooled-fit diagnostics.
    """

    initialized = init_decay_sage(
        observed_power,
        times_s,
        n_components,
        rate_bounds_per_s=(
            t60_to_rate(T60_BOUNDS_S[1]),
            t60_to_rate(T60_BOUNDS_S[0]),
        ),
        n_head_frames=N_HEAD_FRAMES,
        n_tail_frames=N_TAIL_FRAMES,
        floor_margin_db=FLOOR_MARGIN_DB,
    )
    coarse_t60 = np.asarray(rate_to_t60(initialized.fit.rate_per_s))
    if initialization == "equal-log-linear":
        initial_t60 = np.repeat(coarse_t60[:, np.newaxis], n_components, axis=1)
    elif initialization == "log-spaced":
        factors = np.power(2.0, np.linspace(-1.0, 1.0, n_components))
        initial_t60 = coarse_t60[:, np.newaxis] * factors
    else:
        raise ValueError(f"unknown initialization {initialization!r}.")
    initial_t60 = np.clip(initial_t60, *T60_BOUNDS_S)
    rates = np.asarray(t60_to_rate(initial_t60), dtype=np.float64)

    floor = initialized.noise_floor
    initial_signal_power = np.maximum(
        head_power(observed_power, N_HEAD_FRAMES) - floor,
        np.finfo(np.float64).tiny,
    )
    amplitudes = np.repeat(
        (initial_signal_power / n_components)[:, :, np.newaxis],
        n_components,
        axis=2,
    )
    return (
        rates,
        amplitudes,
        floor,
        float(coarse_t60[0]),
        float(initialized.fit.r_squared[0]),
    )


def nearest_frequency_indices(
    frequencies_hz: np.ndarray, targets_hz: np.ndarray
) -> np.ndarray:
    """Return unique FFT-bin indices nearest to requested frequencies."""

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    targets = np.asarray(targets_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must have non-empty shape (F,).")
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets_hz must have non-empty shape (Q,).")
    indices = [int(np.argmin(np.abs(frequencies - target))) for target in targets]
    return np.asarray(list(dict.fromkeys(indices)), dtype=np.int64)


def stratified_receiver_indices(
    condition_index: np.ndarray, count_per_condition: int
) -> np.ndarray:
    """Select evenly spaced RIR indices within every source condition."""

    condition_index = np.asarray(condition_index)
    selected: list[np.ndarray] = []
    for condition in np.unique(condition_index):
        available = np.flatnonzero(condition_index == condition)
        count = min(count_per_condition, available.size)
        local = np.linspace(0, available.size - 1, count).astype(np.int64)
        selected.append(available[np.unique(local)])
    return np.concatenate(selected).astype(np.int64, copy=False)


def _fit_one_frequency(
    cache_path: str,
    metadata_path: str,
    frequency_index: int,
    receiver_indices: np.ndarray | None,
    n_components: int,
    initialization: str,
    max_iter: int,
    tol: float,
    result_path: str,
) -> str:
    power_cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if receiver_indices is None:
        observed_power = np.asarray(
            power_cache[:, frequency_index : frequency_index + 1, :],
            dtype=np.float64,
        )
    else:
        observed_power = np.asarray(
            power_cache[receiver_indices, frequency_index, :], dtype=np.float64
        )[:, np.newaxis, :]
    with np.load(metadata_path, allow_pickle=False) as metadata:
        times_s = np.asarray(metadata["times_s"], dtype=np.float64)
        frequency_hz = float(metadata["frequencies_hz"][frequency_index])

    rates, amplitudes, floor, coarse_t60, coarse_r_squared = initial_parameters(
        observed_power, times_s, n_components, initialization
    )
    initial_t60 = np.asarray(rate_to_t60(rates[0]))
    result = cw_decay_sage(
        observed_power,
        times_s,
        rates,
        rate_bounds_per_s=(
            t60_to_rate(T60_BOUNDS_S[1]),
            t60_to_rate(T60_BOUNDS_S[0]),
        ),
        component_weight_power=COMPONENT_WEIGHT_POWER,
        initial_amplitudes=amplitudes,
        initial_noise_floor=floor,
        estimate_noise_floor=True,
        max_iter=max_iter,
        tol=tol,
        rate_method="newton",
    )
    fitted_t60 = np.asarray(rate_to_t60(result.rates_per_s[0]))
    order = np.argsort(fitted_t60)
    fitted_t60 = fitted_t60[order]
    fitted_amplitudes = result.amplitudes[:, 0, order]
    history_t60 = np.sort(
        np.asarray(rate_to_t60(result.rate_history_per_s[:, 0, :])), axis=1
    )
    np.savez_compressed(
        result_path,
        frequency_index=np.asarray(frequency_index),
        frequency_hz=np.asarray(frequency_hz),
        n_components=np.asarray(n_components),
        initialization=np.asarray(initialization),
        coarse_t60_s=np.asarray(coarse_t60),
        coarse_r_squared=np.asarray(coarse_r_squared),
        initial_t60_s=initial_t60,
        estimated_t60_s=fitted_t60,
        estimated_amplitudes=fitted_amplitudes,
        estimated_noise_floor=result.noise_floor[:, 0],
        objective_history=result.objective_history,
        t60_history_s=history_t60,
        n_iter=np.asarray(result.n_iter),
        converged=np.asarray(result.converged),
    )
    return result_path


def _read_fit(path: str | Path) -> FrequencyFit:
    with np.load(path, allow_pickle=False) as result:
        return FrequencyFit(
            frequency_index=int(result["frequency_index"]),
            frequency_hz=float(result["frequency_hz"]),
            n_components=int(result["n_components"]),
            initialization=str(result["initialization"]),
            coarse_t60_s=float(result["coarse_t60_s"]),
            coarse_r_squared=float(result["coarse_r_squared"]),
            initial_t60_s=result["initial_t60_s"].copy(),
            estimated_t60_s=result["estimated_t60_s"].copy(),
            estimated_amplitudes=result["estimated_amplitudes"].copy(),
            estimated_noise_floor=result["estimated_noise_floor"].copy(),
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
    n_components: int,
    initialization: str,
    max_iter: int,
    tol: float,
    jobs: int,
    result_dir: Path,
) -> list[FrequencyFit]:
    """Run independently stopped frequency fits with per-bin checkpoints."""

    result_dir.mkdir(parents=True, exist_ok=False)
    fits: list[FrequencyFit] = []
    with ProcessPoolExecutor(max_workers=min(jobs, frequency_indices.size)) as pool:
        futures = {}
        for frequency_index in frequency_indices:
            result_path = result_dir / f"frequency_{int(frequency_index):03d}.npz"
            future = pool.submit(
                _fit_one_frequency,
                str(cache_path),
                str(metadata_path),
                int(frequency_index),
                receiver_indices,
                n_components,
                initialization,
                max_iter,
                tol,
                str(result_path),
            )
            futures[future] = int(frequency_index)
        for count, future in enumerate(as_completed(futures), start=1):
            fit = _read_fit(future.result())
            fits.append(fit)
            ratio = fit.objective_history[-1] / fit.objective_history[0]
            print(
                f"fit {count:03d}/{frequency_indices.size:03d}: "
                f"K={n_components}, {initialization}, {fit.frequency_hz:7.2f} Hz, "
                f"sweeps={fit.n_iter:4d}, IS ratio={ratio:.5f}, "
                f"T60={np.round(fit.estimated_t60_s, 3)} s",
                flush=True,
            )
    return sorted(fits, key=lambda fit: fit.frequency_index)


def pilot_is_healthy(fits: list[FrequencyFit]) -> bool:
    """Check finite parameters and cap weighted objective growth at 10 percent."""

    if not fits:
        return False
    return all(
        np.all(np.isfinite(fit.estimated_t60_s))
        and np.all(np.isfinite(fit.estimated_amplitudes))
        and np.all(np.isfinite(fit.estimated_noise_floor))
        and np.all(np.isfinite(fit.objective_history))
        and fit.objective_history[-1] <= 1.10 * fit.objective_history[0]
        for fit in fits
    )


def save_summary(
    fits: list[FrequencyFit],
    receiver_indices: np.ndarray,
    max_iter: int,
    output_dir: Path,
    prefix: str,
) -> None:
    """Save numerical, CSV, and compact graphical summaries for one fit set."""

    frequencies = np.asarray([fit.frequency_hz for fit in fits])
    coarse_t60 = np.asarray([fit.coarse_t60_s for fit in fits])
    coarse_r_squared = np.asarray([fit.coarse_r_squared for fit in fits])
    initial_t60 = np.stack([fit.initial_t60_s for fit in fits])
    estimated_t60 = np.stack([fit.estimated_t60_s for fit in fits])
    amplitudes = np.stack([fit.estimated_amplitudes for fit in fits], axis=1)
    floors = np.stack([fit.estimated_noise_floor for fit in fits], axis=1)
    objective = np.full((len(fits), max_iter + 1), np.nan)
    t60_history = np.full((len(fits), max_iter + 1, fits[0].n_components), np.nan)
    for index, fit in enumerate(fits):
        length = fit.n_iter + 1
        objective[index, :length] = fit.objective_history
        t60_history[index, :length] = fit.t60_history_s
    n_iter = np.asarray([fit.n_iter for fit in fits], dtype=np.int64)
    converged = np.asarray([fit.converged for fit in fits])
    initial_objective = objective[:, 0]
    final_objective = np.asarray([fit.objective_history[-1] for fit in fits])
    np.savez_compressed(
        output_dir / f"{prefix}_results.npz",
        frequencies_hz=frequencies,
        receiver_indices=receiver_indices,
        coarse_t60_s=coarse_t60,
        coarse_r_squared=coarse_r_squared,
        initial_t60_s=initial_t60,
        estimated_t60_s=estimated_t60,
        estimated_amplitudes=amplitudes,
        estimated_noise_floor=floors,
        objective_history=objective,
        t60_history_s=t60_history,
        n_iter=n_iter,
        converged=converged,
        n_components=np.asarray(fits[0].n_components),
        initialization=np.asarray(fits[0].initialization),
        t60_bounds_s=np.asarray(T60_BOUNDS_S),
        component_weight_power=np.asarray(COMPONENT_WEIGHT_POWER),
        estimate_noise_floor=np.asarray(True),
    )

    with (output_dir / f"{prefix}_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frequency_hz",
                "coarse_t60_s",
                "coarse_r_squared",
                *[f"estimated_t60_{k + 1}_s" for k in range(fits[0].n_components)],
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
                    fit.coarse_t60_s,
                    fit.coarse_r_squared,
                    *fit.estimated_t60_s,
                    fit.n_iter,
                    fit.converged,
                    initial_objective[index],
                    final_objective[index],
                    final_objective[index] / initial_objective[index],
                ]
            )

    if fits[0].n_components == 2:
        plot_decay_times(
            output_dir / f"{prefix}_results.npz",
            output_dir / f"{prefix}_summary.png",
        )


def _save_png_and_pdf(fig, output_path: str | Path) -> Path:
    """Save a Matplotlib figure as both PNG and PDF and return the PNG path."""

    output = Path(output_path)
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    return output


def plot_decay_times(
    result_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot two fitted energy decay times and the pooled linear estimate.

    ``result_path`` must contain frequency centers ``(F,)``, fitted energy
    ``T60`` values ``(F,2)``, and pooled linear energy ``T60`` values ``(F,)``,
    all in hertz and seconds. The figure is saved in PNG and PDF formats.
    """

    plt = _pyplot()
    with np.load(result_path, allow_pickle=False) as result:
        frequencies_hz = np.asarray(result["frequencies_hz"], dtype=np.float64)
        estimated_t60_s = np.asarray(result["estimated_t60_s"], dtype=np.float64)
        coarse_t60_s = np.asarray(result["coarse_t60_s"], dtype=np.float64)
    if estimated_t60_s.shape != (frequencies_hz.size, 2):
        raise ValueError("result must contain two-slope T60 values with shape (F,2).")
    if coarse_t60_s.shape != frequencies_hz.shape:
        raise ValueError("pooled linear T60 values must have shape (F,).")

    with plt.rc_context({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.25, 'lines.markersize': 4, 'grid.linewidth': 0.5, 'pdf.fonttype': 42}):
        fig, axis = plt.subplots(figsize=(3.45, 2.3), layout="constrained")
        axis.plot(
            frequencies_hz,
            estimated_t60_s[:, 1],
            color=LONG_SLOPE_RGB,
            linewidth=1.25,
            label="longer decay",
        )
        axis.plot(
            frequencies_hz,
            estimated_t60_s[:, 0],
            color=SHORT_SLOPE_RGB,
            linewidth=1.25,
            label="shorter decay",
        )
        axis.plot(
            frequencies_hz,
            coarse_t60_s,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="linear fit",
        )
        axis.set_xscale("log", base=10.0)
        axis.set_xlim(100.0, 10_000.0)
        axis.set_xticks(
            [100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0],
            labels=["100", "200", "500", "1k", "2k", "5k", "10k"],
        )
        axis.set_ylim(0.0, 2.0)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("T60 (s)")
        axis.legend()
        axis.grid(alpha=0.2)
        output = _save_png_and_pdf(fig, output_path)
        plt.close(fig)
        return output


def amplitude_mixture_rgb(
    amplitudes: np.ndarray,
    *,
    reference_db: float,
    dynamic_range_db: float = 40.0,
    dark_background: bool = False,
) -> np.ndarray:
    """Mix short- and long-slope amplitudes into an order-independent RGB map.

    Parameters
    ----------
    amplitudes
        Positive variance amplitudes with final axis ``(short,long)`` and any
        leading shape.
    reference_db
        Zero-level reference in power decibels. Values at or above this level
        receive maximum color strength.
    dynamic_range_db
        Positive displayed range below ``reference_db`` in power decibels.
    dark_background
        If true, interpolate the specified dark, blue, coral, and warm-light
        bivariate corners. If false, weak values are white and stronger
        mixtures darken.

    Returns
    -------
    ndarray
        RGB values with shape ``amplitudes.shape[:-1] + (3,)``. Blue and coral
        identify short and long slopes, and the mixture is symmetric.
    """

    values = np.asarray(amplitudes, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("amplitudes must have a final axis of length two.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("amplitudes must contain finite positive values.")
    if not np.isfinite(reference_db):
        raise ValueError("reference_db must be finite.")
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0.0:
        raise ValueError("dynamic_range_db must be finite and positive.")

    amplitude_db = 10.0 * np.log10(values)
    strength = np.clip(
        (amplitude_db - (reference_db - dynamic_range_db)) / dynamic_range_db,
        0.0,
        1.0,
    )
    short_strength = strength[..., 0]
    long_strength = strength[..., 1]
    if dark_background:
        return (
            ((1.0 - short_strength) * (1.0 - long_strength))[..., np.newaxis]
            * WEAK_MIXTURE_RGB
            + (short_strength * (1.0 - long_strength))[..., np.newaxis]
            * SHORT_SLOPE_RGB
            + ((1.0 - short_strength) * long_strength)[..., np.newaxis] * LONG_SLOPE_RGB
            + (short_strength * long_strength)[..., np.newaxis] * STRONG_MIXTURE_RGB
        )

    total_strength = short_strength + long_strength
    hue = (
        short_strength[..., np.newaxis] * SHORT_SLOPE_RGB
        + long_strength[..., np.newaxis] * LONG_SLOPE_RGB
    ) / np.maximum(total_strength[..., np.newaxis], np.finfo(np.float64).tiny)
    opacity = 1.0 - (1.0 - short_strength) * (1.0 - long_strength)
    return 1.0 - opacity[..., np.newaxis] * (1.0 - hue)


def rho_mixture_rgb(rho: np.ndarray) -> np.ndarray:
    """Return RGB colors for final fast, slow, and floor contributions.

    ``rho`` has final axis ``(rho_1, rho_2, rho_0)`` for fast decay, slow
    decay, and constant floor. It must be finite, non-negative, and sum to
    one. The output has shape ``rho.shape[:-1] + (3,)``; pure components use
    ``SHORT_SLOPE_RGB``, ``LONG_SLOPE_RGB``, and ``FLOOR_RHO_RGB``.
    """

    values = np.asarray(rho, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("rho must have a final axis of length three.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("rho must be finite and non-negative.")
    if not np.allclose(np.sum(values, axis=-1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("rho must sum to one along its final axis.")
    return (
        values[..., 0, np.newaxis] * SHORT_SLOPE_RGB
        + values[..., 1, np.newaxis] * LONG_SLOPE_RGB
        + values[..., 2, np.newaxis] * FLOOR_RHO_RGB
    )


def plot_final_rho_space_time(
    result_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    target_frequency_hz: float,
) -> Path:
    """Plot final rho over receiver position and elapsed time at one bin.

    The fitted archive supplies final two-slope amplitudes/floors and RTs;
    metadata supplies frame times and room-to-hallway positions. The selected
    bin is nearest ``target_frequency_hz``. Room/LOS and Hallway/LOS panels use blue for
    ``rho_1``, orange for ``rho_2``, and dark gray for ``rho_0``.
    """

    if not np.isfinite(target_frequency_hz) or target_frequency_hz <= 0.0:
        raise ValueError("target_frequency_hz must be finite and positive.")
    plt = _pyplot()
    with np.load(result_path, allow_pickle=False) as result:
        t60_s = np.asarray(result["estimated_t60_s"], dtype=np.float64)
        amplitudes = np.asarray(result["estimated_amplitudes"], dtype=np.float64)
        floors = np.asarray(result["estimated_noise_floor"], dtype=np.float64)
        frequencies_hz = np.asarray(result["frequencies_hz"], dtype=np.float64)
        receiver_indices = np.asarray(result["receiver_indices"], dtype=np.int64)
    if (
        t60_s.ndim != 2 or t60_s.shape[1] != 2
        or amplitudes.shape != (receiver_indices.size, frequencies_hz.size, 2)
        or floors.shape != amplitudes.shape[:2]
        or not all(np.all(np.isfinite(x)) for x in (t60_s, amplitudes, floors))
        or np.any(t60_s <= 0.0) or np.any(amplitudes <= 0.0) or np.any(floors <= 0.0)
    ):
        raise ValueError("result must contain finite positive two-slope fit parameters.")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        times_s = np.asarray(metadata["times_s"], dtype=np.float64)
        condition_names = np.asarray(metadata["condition_names"])
        condition_index = np.asarray(metadata["rir_condition_index"])[receiver_indices]
        positions_m = np.asarray(metadata["listener_positions_m"])[receiver_indices, 0]
    if times_s.ndim != 1 or times_s.size < 2 or np.any(np.diff(times_s) <= 0.0):
        raise ValueError("metadata times_s must be a strictly increasing 1-D array.")

    frequency_index = int(np.argmin(np.abs(frequencies_hz - target_frequency_hz)))
    atoms = np.exp(
        -t60_to_rate(t60_s[frequency_index])[:, np.newaxis] * times_s[np.newaxis, :]
    )
    components = amplitudes[:, frequency_index, :, np.newaxis] * atoms[np.newaxis]
    total = np.sum(components, axis=1) + floors[:, frequency_index, np.newaxis]
    rho = np.concatenate(
        (
            components / total[:, np.newaxis, :],
            (floors[:, frequency_index, np.newaxis] / total)[:, np.newaxis, :],
        ), axis=1,
    ).transpose(0, 2, 1)
    colors = rho_mixture_rgb(rho)

    display_names = {
        "room_los": "Source in Room, LOS",
        "hallway_los": "Source in Hallway, LOS",
    }
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8}):
        figure = plt.figure(figsize=(3.45, 3.1))
        grid = figure.add_gridspec(2, 1, left=0.14, right=0.98, bottom=0.19, top=0.95, hspace=0.26)
        axes = [figure.add_subplot(grid[index, 0]) for index in range(2)]
        for panel, (name, axis) in enumerate(zip(display_names, axes, strict=True)):
            condition = list(condition_names).index(name)
            mask = condition_index == condition
            order = np.argsort(positions_m[mask])
            x = positions_m[mask][order]
            axis.imshow(
                np.transpose(colors[mask][order], (1, 0, 2)), origin="lower",
                aspect="auto", interpolation="nearest",
                extent=(x[0], x[-1], times_s[0], times_s[-1]),
            )
            axis.axvline(2.5, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
            axis.set(title=display_names[name], xlim=(0.0, 5.0), ylim=(times_s[0], times_s[-1]))
            axis.title.set_fontsize(7)
            axis.set_xticks([0.0, 2.5, 5.0], labels=["0", "2.5", "5"])
            if panel == 0:
                axis.set_ylabel("Elapsed time (s)")
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_ylabel("Elapsed time (s)")
        figure.text(0.58, 0.1, r"Room $\leftarrow\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\rightarrow$ Hallway", ha="center", fontsize=7)
        figure.text(0.58, 0.1, "Receiver position (m)", ha="center", fontsize=8)
        figure.legend(
            handles=[
                plt.Line2D([], [], color=SHORT_SLOPE_RGB, linewidth=3, label=r"$\rho_1$ fast decay"),
                plt.Line2D([], [], color=LONG_SLOPE_RGB, linewidth=3, label=r"$\rho_2$ slow decay"),
                plt.Line2D([], [], color=FLOOR_RHO_RGB, linewidth=3, label=r"$\rho_0$ floor"),
            ],
            loc="lower center", bbox_to_anchor=(0.58, 0), ncols=3, frameon=False, fontsize=7,
        )
        output = Path(output_path).with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300)
        figure.savefig(output.with_suffix(".pdf"))
        plt.close(figure)
    return output


def plot_amplitude_mixture(
    result_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    dynamic_range_db: float = 40.0,
    dark_background: bool = True,
) -> Path:
    """Plot both fitted slope amplitudes over position and frequency.

    ``result_path`` must contain two-slope amplitudes ``(R,F,2)`` and the
    corresponding receiver indices. The SOFA listener x coordinate is shown
    from 0 to 5 m, with the doorway marked at 2.5 m. A global power-dB
    reference across all four conditions is retained. Three panels show room
    without LOS, hallway with LOS, and hallway without LOS, followed by a
    half-width bivariate key. Returns the PNG path; a fixed-size PDF is also saved.
    """

    plt = _pyplot()

    with np.load(result_path, allow_pickle=False) as result:
        amplitudes = np.asarray(result["estimated_amplitudes"], dtype=np.float64)
        frequencies_hz = np.asarray(result["frequencies_hz"], dtype=np.float64)
        receiver_indices = np.asarray(result["receiver_indices"], dtype=np.int64)
    if amplitudes.ndim != 3 or amplitudes.shape[2] != 2:
        raise ValueError("result must contain two-slope amplitudes with shape (R,F,2).")
    if amplitudes.shape[:2] != (receiver_indices.size, frequencies_hz.size):
        raise ValueError(
            "result amplitude axes do not match receivers and frequencies."
        )
    with np.load(metadata_path, allow_pickle=False) as metadata:
        condition_names = np.asarray(metadata["condition_names"])
        condition_index = np.asarray(metadata["rir_condition_index"])[receiver_indices]
        positions_m = np.asarray(metadata["listener_positions_m"])[receiver_indices, 0]

    reference_db = float(np.max(10.0 * np.log10(amplitudes)))
    mixed_rgb = amplitude_mixture_rgb(
        amplitudes,
        reference_db=reference_db,
        dynamic_range_db=dynamic_range_db,
        dark_background=dark_background,
    )
    display_names = {
        "room_los": "Source in Room, LOS",
        "room_no_los": "Source in Room, no LOS",
        "hallway_los": "Source in Hallway, LOS",
        "hallway_no_los": "Source in Hallway, no LOS",
    }

    fig = plt.figure(figsize=(7.10, 1.7))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1, 1, 1, 0.5),
        left=0.07,
        right=0.93,
        bottom=0.25,
        top=0.86,
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    selected_names = ("room_no_los", "hallway_los", "hallway_no_los")
    for panel, (name, axis) in enumerate(zip(selected_names, axes, strict=True)):
        condition = list(condition_names).index(name)
        mask = condition_index == condition
        order = np.argsort(positions_m[mask])
        x = positions_m[mask][order]
        image = np.transpose(mixed_rgb[mask][order], (1, 0, 2))
        axis.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(x[0], x[-1], frequencies_hz[0], frequencies_hz[-1]),
        )
        axis.axvline(2.5, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
        axis.set_title(display_names[name], fontsize=8)
        axis.set_xlim(0.0, 5.0)
        axis.set_yscale("log")
        axis.set_ylim(frequencies_hz[0], frequencies_hz[-1])
        axis.set_yticks(
            [250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0],
            labels=["250", "500", "1k", "2k", "4k", "8k"],
            minor=False,
        )
        axis.set_yticks([], minor=True)
        axis.tick_params(axis="both", labelsize=8)
        axis.set_xticks([0, 2.5, 5], labels=["0", "2.5", "5"])
        if panel == 0:
            axis.set_ylabel("Frequency (Hz)", fontsize=9)
        else:
            axis.tick_params(axis="y", labelleft=False)

    fig.text(
        (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2,
        0.025,
        "Receiver position (m)",
        ha="center",
        fontsize=9,
    )
    key_axis = fig.add_subplot(grid[0, 3])
    levels_db = np.linspace(-dynamic_range_db, 0.0, 151)
    short_db, long_db = np.meshgrid(levels_db, levels_db)
    key_amplitudes = 10.0 ** (
        (np.stack((short_db, long_db), axis=-1) + reference_db) / 10.0
    )
    key_axis.imshow(
        amplitude_mixture_rgb(
            key_amplitudes,
            reference_db=reference_db,
            dynamic_range_db=dynamic_range_db,
            dark_background=dark_background,
        ),
        origin="lower",
        extent=(-dynamic_range_db, 0, -dynamic_range_db, 0),
        aspect="auto",
    )
    key_axis.set_title("Amplitude (dB)", fontsize=8)
    key_axis.set_xticks([-dynamic_range_db, 0])
    key_axis.set_yticks([-dynamic_range_db, 0])
    key_axis.tick_params(labelsize=8)
    key_axis.tick_params(axis="y", pad=1, length=2)
    key_axis.yaxis.tick_right()
    key_axis.yaxis.set_label_position("right")
    key_axis.set_xlabel("Fast decay", fontsize=8)
    key_axis.set_ylabel("Slow decay", fontsize=8, labelpad=1)
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return output


def _configuration_name(n_components: int, initialization: str) -> str:
    return f"K{n_components}_{initialization.replace('-', '_')}"


def _unique_configurations(
    components: list[int], initializations: list[str]
) -> list[tuple[int, str]]:
    configurations: list[tuple[int, str]] = []
    for n_components in dict.fromkeys(components):
        for initialization in dict.fromkeys(initializations):
            if n_components == 1 and initialization == "log-spaced":
                continue
            configurations.append((n_components, initialization))
    return configurations


def main() -> None:
    """Prepare the cache or run the explicitly selected experiment stage."""

    args = _arguments()
    if (
        args.rebuild_cache
        or args.stage == "prepare"
        or not (args.cache.is_file() and args.cache_metadata.is_file())
    ):
        prepare_power_cache(args.dataset_dir, args.cache, args.cache_metadata)
    validate_power_cache(args.cache, args.cache_metadata)
    if args.stage == "prepare":
        return

    output_dir = create_run_output_dir(args.output_root)
    power = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    with np.load(args.cache_metadata, allow_pickle=False) as metadata:
        frequencies_hz = np.asarray(metadata["frequencies_hz"])
        condition_index = np.asarray(metadata["rir_condition_index"])
        run_metadata = {name: metadata[name].copy() for name in metadata.files}
    np.savez_compressed(
        output_dir / "analysis_metadata.npz",
        **run_metadata,
        experiment_stage=np.asarray(args.stage),
        dataset_dir=np.asarray(str(args.dataset_dir.resolve())),
        power_cache=np.asarray(str(args.cache.resolve())),
    )

    if args.stage == "pilot":
        frequency_indices = nearest_frequency_indices(
            frequencies_hz, PILOT_TARGET_FREQUENCIES_HZ
        )
        receiver_indices = stratified_receiver_indices(
            condition_index, args.pilot_rirs_per_condition
        )
        configurations = _unique_configurations(
            args.pilot_components, args.initializations
        )
        max_iter = args.pilot_max_iter
        print(
            "pilot FFT bins: "
            + ", ".join(f"{frequencies_hz[index]:g}" for index in frequency_indices)
            + " Hz",
            flush=True,
        )
    else:
        if args.highest_bins > frequencies_hz.size:
            raise ValueError("--highest-bins exceeds the cached frequency count.")
        first = frequencies_hz.size - args.highest_bins if args.highest_bins else 0
        frequency_indices = np.arange(first, frequencies_hz.size, dtype=np.int64)
        receiver_indices = np.arange(power.shape[0], dtype=np.int64)
        configurations = _unique_configurations(
            args.full_components, args.initializations
        )
        max_iter = args.max_iter

    health: list[tuple[str, bool]] = []
    for n_components, initialization in configurations:
        name = _configuration_name(n_components, initialization)
        fits = run_frequency_fits(
            args.cache,
            args.cache_metadata,
            frequency_indices,
            receiver_indices,
            n_components,
            initialization,
            max_iter,
            args.tol,
            args.jobs,
            output_dir / f"{name}_frequency_results",
        )
        save_summary(fits, receiver_indices, max_iter, output_dir, name)
        if n_components == 2:
            plot_final_rho_space_time(
                output_dir / f"{name}_results.npz",
                output_dir / "analysis_metadata.npz",
                output_dir / f"{name}_rho_space_time",
                target_frequency_hz=args.rho_frequency_hz,
            )
        health.append((name, pilot_is_healthy(fits)))

    with (output_dir / "configuration_health.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["configuration", "numerically_healthy"])
        writer.writerows(health)
    print(
        "configuration health: "
        + ", ".join(f"{name}={healthy}" for name, healthy in health),
        flush=True,
    )
    print(f"{args.stage} outputs: {output_dir}", flush=True)
    if args.show:
        plt = _pyplot()
        plt.show()


if __name__ == "__main__":
    main()
