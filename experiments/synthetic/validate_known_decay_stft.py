"""Validate the supplied-atom SAGE stage on exact-model observations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from common_slope_nmf import (
    AmplitudeSAGEResult,
    amplitude_sage,
    exponential_atoms,
    sample_complex_gaussian,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir

SAMPLE_RATE_HZ = 24_000
DURATION_S = 2.0
FRAME_SIZE_SAMPLES = 256
HOP_SIZE_SAMPLES = 128
N_BINS = 512
T60_S = 1.0
DECAY_AMPLITUDE = 10.0 ** (0.0 / 10.0)
FLOOR_LEVELS_DB = (None, -30.0, -60.0)
INITIAL_DECAY_AMPLITUDE = 0.25
INITIAL_NOISE_FLOOR = 0.01
SEED = 20260724


@dataclass(frozen=True)
class ExperimentRun:
    """Data and estimates for one known-floor condition.

    Attributes
    ----------
    label
        Human-readable floor condition.
    slug
        Filesystem-safe condition name.
    floor
        Generating time-invariant variance floor in power units.
    stft
        Generated circular complex-Gaussian coefficients, shape ``(F, N)``.
    result
        Amplitude-step SAGE estimates for the condition.
    """

    label: str
    slug: str
    floor: float
    stft: np.ndarray
    result: AmplitudeSAGEResult


def frame_times(
    duration_s: float,
    sample_rate_hz: int,
    frame_size_samples: int,
    hop_size_samples: int,
) -> np.ndarray:
    """Return elapsed times for complete, unpadded frames.

    Parameters
    ----------
    duration_s
        Signal duration in seconds.
    sample_rate_hz
        Sampling rate in samples per second.
    frame_size_samples
        Frame length in samples.
    hop_size_samples
        Frame advance in samples.

    Returns
    -------
    ndarray
        Frame-start elapsed times in seconds, shape ``(N,)``. The first frame
        is the decay origin, so its elapsed time is zero.
    """

    if duration_s <= 0.0 or not np.isfinite(duration_s):
        raise ValueError("duration_s must be finite and positive.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    n_samples = int(round(duration_s * sample_rate_hz))
    if frame_size_samples <= 0 or frame_size_samples > n_samples:
        raise ValueError(
            "frame_size_samples must be positive and no greater than the signal."
        )
    if hop_size_samples <= 0:
        raise ValueError("hop_size_samples must be positive.")

    n_frames = 1 + (n_samples - frame_size_samples) // hop_size_samples
    return (
        np.arange(n_frames, dtype=np.float64)
        * hop_size_samples
        / sample_rate_hz
    )


def exact_amplitude_intervals(
    observed_power: np.ndarray,
    decay_atom: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact confidence intervals for one decay and no floor.

    Parameters
    ----------
    observed_power
        Positive instantaneous powers, shape ``(F, N)``.
    decay_atom
        Positive unit-amplitude decay kernel, shape ``(N,)``.
    confidence
        Confidence level strictly between zero and one.

    Returns
    -------
    lower, upper
        Exact confidence limits for the variance amplitude in each bin, each
        with shape ``(F,)`` and the same power units as ``observed_power``.
    """

    powers = np.asarray(observed_power, dtype=np.float64)
    atom = np.asarray(decay_atom, dtype=np.float64)
    if powers.ndim != 2 or atom.ndim != 1:
        raise ValueError(
            "observed_power must be (F, N) and decay_atom must be (N,)."
        )
    if powers.shape[1] != atom.size:
        raise ValueError(
            "observed_power and decay_atom must have the same frame count."
        )
    if np.any(powers <= 0.0) or np.any(atom <= 0.0):
        raise ValueError("observed_power and decay_atom must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")

    normalized_sum = np.sum(powers / atom, axis=1)
    degrees_of_freedom = 2 * atom.size
    tail = 0.5 * (1.0 - confidence)
    lower = 2.0 * normalized_sum / chi2.ppf(
        1.0 - tail, degrees_of_freedom
    )
    upper = 2.0 * normalized_sum / chi2.ppf(
        tail, degrees_of_freedom
    )
    return lower, upper


def fisher_standard_errors(
    dictionary: np.ndarray, amplitudes: np.ndarray
) -> np.ndarray:
    """Return complex-Gaussian variance-model Fisher standard errors.

    Parameters
    ----------
    dictionary
        Fixed positive variance atoms, shape ``(Q, N)``.
    amplitudes
        Positive variance amplitudes, shape ``(Q,)``.

    Returns
    -------
    ndarray
        Asymptotic standard errors for the amplitudes, shape ``(Q,)``, in
        the same power units as ``amplitudes``.
    """

    atoms = np.asarray(dictionary, dtype=np.float64)
    weights = np.asarray(amplitudes, dtype=np.float64)
    if atoms.ndim != 2 or weights.shape != (atoms.shape[0],):
        raise ValueError(
            "dictionary must be (Q, N) and amplitudes must be (Q,)."
        )
    if np.any(atoms <= 0.0) or np.any(weights <= 0.0):
        raise ValueError("dictionary and amplitudes must be positive.")

    variance = weights @ atoms
    weighted_atoms = atoms / variance
    information = weighted_atoms @ weighted_atoms.T
    return np.sqrt(np.diag(np.linalg.inv(information)))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5_000,
        help="Maximum SAGE sweeps for each decay-plus-floor fit.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for the timestamped run directory.",
    )
    args = parser.parse_args()
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    return args


def _condition(floor_db: float | None) -> tuple[str, str, float]:
    if floor_db is None:
        return "No floor", "no_floor", 0.0
    floor = 10.0 ** (floor_db / 10.0)
    magnitude = int(abs(floor_db))
    return f"{floor_db:.0f} dB floor", f"floor_m{magnitude}_db", floor


def _generate_and_fit(
    decay_atom: np.ndarray,
    floor_db: float | None,
    rng: np.random.Generator,
    max_iter: int,
) -> ExperimentRun:
    label, slug, floor = _condition(floor_db)
    variance = np.broadcast_to(
        DECAY_AMPLITUDE * decay_atom + floor,
        (N_BINS, decay_atom.size),
    )
    stft = sample_complex_gaussian(variance, rng=rng)
    power = np.abs(stft) ** 2

    if floor == 0.0:
        dictionary = decay_atom[np.newaxis, :]
        initial_amplitudes = np.full(
            (N_BINS, 1), INITIAL_DECAY_AMPLITUDE
        )
        fit_max_iter = 3
    else:
        dictionary = np.vstack([decay_atom, np.ones_like(decay_atom)])
        initial_amplitudes = np.tile(
            [INITIAL_DECAY_AMPLITUDE, INITIAL_NOISE_FLOOR],
            (N_BINS, 1),
        )
        fit_max_iter = max_iter

    result = amplitude_sage(
        power,
        dictionary,
        initial_amplitudes=initial_amplitudes,
        max_iter=fit_max_iter,
        tol=1e-12,
    )
    return ExperimentRun(label, slug, floor, stft, result)



def main() -> None:
    """Generate three exact-model STFTs, fit them, and save diagnostics."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    times_s = frame_times(
        DURATION_S,
        SAMPLE_RATE_HZ,
        FRAME_SIZE_SAMPLES,
        HOP_SIZE_SAMPLES,
    )
    decay_atom = exponential_atoms(times_s, t60_to_rate(T60_S))[0]
    rng = np.random.default_rng(SEED)
    runs = [
        _generate_and_fit(decay_atom, floor_db, rng, args.max_iter)
        for floor_db in FLOOR_LEVELS_DB
    ]

    n_runs = len(runs)
    n_bins = np.abs(runs[0].stft).shape[0]
    observed_power = np.stack([np.abs(run.stft) ** 2 for run in runs])
    fitted_variance = np.stack([run.result.variance for run in runs])
    decay_amplitudes = np.stack([run.result.amplitudes[:, 0] for run in runs])
    floor_amplitudes = np.full((n_runs, n_bins), np.nan)
    for index, run in enumerate(runs):
        if run.floor > 0.0:
            floor_amplitudes[index] = run.result.amplitudes[:, 1]
    max_hist = max(run.result.objective_history.size for run in runs)
    objective_history = np.full((n_runs, max_hist), np.nan)
    for index, run in enumerate(runs):
        history = run.result.objective_history
        objective_history[index, : history.size] = history
    archive_path = output_dir / "known_decay_stft_results.npz"
    np.savez_compressed(
        archive_path,
        times_s=times_s,
        labels=np.array([run.label for run in runs]),
        slugs=np.array([run.slug for run in runs]),
        floors=np.array([run.floor for run in runs]),
        observed_power=observed_power,
        fitted_variance=fitted_variance,
        decay_amplitudes=decay_amplitudes,
        floor_amplitudes=floor_amplitudes,
        objective_history=objective_history,
        decay_amplitude=np.asarray(DECAY_AMPLITUDE),
    )

    no_floor_run = runs[0]
    no_floor_power = np.abs(no_floor_run.stft) ** 2
    exact_mle = np.mean(no_floor_power / decay_atom, axis=1)
    lower_ci, upper_ci = exact_amplitude_intervals(
        no_floor_power, decay_atom
    )
    theoretical_no_floor_sd = DECAY_AMPLITUDE / np.sqrt(times_s.size)

    print(f"seed={SEED}")
    print(
        f"sample_rate_hz={SAMPLE_RATE_HZ}, duration_s={DURATION_S}, "
        f"frame_size={FRAME_SIZE_SAMPLES}, hop_size={HOP_SIZE_SAMPLES}"
    )
    print(
        f"bins={N_BINS}, frames={times_s.size}, "
        f"last_frame_time_s={times_s[-1]:.9f}"
    )
    print(
        f"energy_t60_s={T60_S}, decay_amplitude={DECAY_AMPLITUDE}, "
        f"initial_decay={INITIAL_DECAY_AMPLITUDE}, "
        f"initial_floor={INITIAL_NOISE_FLOOR}"
    )
    print(
        "no_floor_sage_vs_closed_form_max_abs_error="
        f"{np.max(np.abs(no_floor_run.result.amplitudes[:, 0] - exact_mle)):.3e}"
    )
    print(
        "no_floor_theory_mean_sd="
        f"{DECAY_AMPLITUDE:.6f}, {theoretical_no_floor_sd:.6f}"
    )
    print(
        "no_floor_exact_95pct_ci_coverage="
        f"{np.mean((lower_ci <= DECAY_AMPLITUDE) & (DECAY_AMPLITUDE <= upper_ci)):.6f}"
    )

    for run in runs:
        decay_estimates = run.result.amplitudes[:, 0]
        print(
            f"condition={run.label!r}, decay_mean_sd="
            f"{np.mean(decay_estimates):.6f}, "
            f"{np.std(decay_estimates, ddof=1):.6f}"
        )
        if run.floor > 0.0:
            floor_estimates = run.result.amplitudes[:, 1]
            dictionary = np.vstack([decay_atom, np.ones_like(decay_atom)])
            standard_errors = fisher_standard_errors(
                dictionary,
                np.array([DECAY_AMPLITUDE, run.floor]),
            )
            print(
                f"condition={run.label!r}, floor_mean_sd="
                f"{np.mean(floor_estimates):.9g}, "
                f"{np.std(floor_estimates, ddof=1):.9g}, "
                f"fisher_decay_floor_sd={standard_errors[0]:.6f}, "
                f"{standard_errors[1]:.9g}"
            )
        print(
            f"condition={run.label!r}, sweeps={run.result.n_iter}, "
            f"converged={run.result.converged}, objective_monotone="
            f"{bool(np.all(np.diff(run.result.objective_history) <= 1e-10))}"
        )

    print(f"saved={archive_path.resolve()}")


if __name__ == "__main__":
    main()
