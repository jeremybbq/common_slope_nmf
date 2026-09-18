"""Quantify fixed-rate two-decay amplitude inference under component masking.

This controlled experiment isolates the supplied-atom amplitude stage used by
the room-to-hallway analysis. The two energy-decay times are known exactly;
only their unit-origin variance amplitudes and a constant variance floor are
estimated. Five fast-to-slow amplitude ratios are tested with matched circular
complex-Gaussian STFT innovations so changes across cases are caused by the
amplitude ratio rather than a different Monte Carlo draw.
"""


from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    fit_amplitudes,
    exponential_features,
    sample_complex_gaussian,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir

SEED = 20260908

N_REALIZATIONS = 100

N_FRAMES = 250

HOP_S = 256.0 / 48_000.0

TRUE_T60_S = np.array([0.50, 1.50], dtype=np.float64)

FAST_TO_SLOW_DB = np.array([10.0, 0.0, -10.0, -15.0, -20.0])

TOTAL_DECAY_AMPLITUDE = 1.0

NOISE_FLOOR_DB = -50.0

N_HEAD_FRAMES = 8

N_TAIL_FRAMES = 20

COMPONENT_LABELS = ("fast decay", "slow decay", "noise floor")

@dataclass(frozen=True)
class AmplitudeInferenceRun:
    """Ground truth, observations, and fixed-rate SAGE estimates.

    Attributes
    ----------
    seed
        Random seed used for the shared complex-Gaussian innovations.
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    dictionary
        Two exponential atoms followed by a constant atom, shape ``(3,N)``.
    ratio_db
        Fast-to-slow unit-origin power-amplitude ratios in dB, shape ``(C,)``.
    true_amplitudes
        Fast, slow, and floor variance amplitudes, shape ``(C,3)``.
    coefficients
        Circular complex-Gaussian STFT coefficients, shape ``(C,R,N)``.
    observed_power
        Instantaneous STFT powers, shape ``(C,R,N)``.
    initial_amplitudes
        Equal decay-amplitude and tail-floor starts, shape ``(C,R,3)``.
    estimated_amplitudes
        Fitted variance amplitudes, shape ``(C,R,3)``.
    n_sweep_evaluations, converged
        Complete ordinary SAGE sweeps and combined loss/stability flags,
        each shape ``(C,)``. The diagnostic probe sweep is excluded.
    fixed_point_residual
        Final maximum amplitude change under one extra SAGE sweep, divided
        by each realization's mean observed power, shape ``(C,)``.
        The probe does not modify the saved estimates.
    objective_history
        Total IS loss before fitting and after each sweep, shape
        ``(C, max_sweep_evaluations + 1)``, with trailing NaN padding.
    max_sweep_evaluations, objective_tol, fixed_point_tol
        Saved sweep budget and dimensionless stopping/diagnostic tolerances.
    initial_objective, final_objective
        Summed IS objectives for each ratio case, each shape ``(C,)``.
    """

    seed: int
    times_s: np.ndarray
    dictionary: np.ndarray
    ratio_db: np.ndarray
    true_amplitudes: np.ndarray
    coefficients: np.ndarray
    observed_power: np.ndarray
    initial_amplitudes: np.ndarray
    estimated_amplitudes: np.ndarray
    n_sweep_evaluations: np.ndarray
    converged: np.ndarray
    fixed_point_residual: np.ndarray
    objective_history: np.ndarray
    max_sweep_evaluations: int
    objective_tol: float
    fixed_point_tol: float
    initial_objective: np.ndarray
    final_objective: np.ndarray

def amplitude_pair_from_ratio_db(
    fast_to_slow_db: np.ndarray | float,
    total_amplitude: float = TOTAL_DECAY_AMPLITUDE,
) -> np.ndarray:
    """Return fast/slow variance amplitudes with a fixed sum.

    Parameters
    ----------
    fast_to_slow_db
        Power-decibel ratio ``10 log10(A_fast / A_slow)``, scalar or shape
        ``(C,)``.
    total_amplitude
        Positive unit-origin sum ``A_fast + A_slow`` in variance units.

    Returns
    -------
    ndarray
        Fast and slow amplitudes on the final axis, shape ``(...,2)``.
    """

    ratios_db = np.asarray(fast_to_slow_db, dtype=np.float64)
    if not np.all(np.isfinite(ratios_db)):
        raise ValueError("fast_to_slow_db must contain finite values.")
    if not np.isfinite(total_amplitude) or total_amplitude <= 0.0:
        raise ValueError("total_amplitude must be finite and positive.")
    linear_ratio = np.power(10.0, ratios_db / 10.0)
    slow = total_amplitude / (1.0 + linear_ratio)
    fast = total_amplitude - slow
    return np.stack((fast, slow), axis=-1)


def fixed_rate_dictionary(times_s: np.ndarray) -> np.ndarray:
    """Return known fast, slow, and floor atoms with shape ``(3,N)``.

    ``times_s`` has shape ``(N,)`` and is measured in seconds. The two decay
    rows are unitless energy atoms for :data:`TRUE_T60_S`; the last row is one.
    """

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_s must have non-empty shape (N,).")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times_s must contain finite non-negative values.")
    decays = exponential_features(times, t60_to_rate(TRUE_T60_S))
    return np.vstack((decays, np.ones(times.size, dtype=np.float64)))


def equal_decay_initialization(observed_power: np.ndarray) -> np.ndarray:
    """Initialize two equal decay amplitudes and a tail-derived floor.

    Parameters
    ----------
    observed_power
        Positive instantaneous power observations, shape ``(R,N)``.

    Returns
    -------
    ndarray
        Positive initial fast, slow, and floor amplitudes, shape ``(R,3)``.
        The decay signal is the first-eight-frame mean minus the final-20-frame
        mean, divided equally between the two supplied decay atoms.
    """

    power = np.asarray(observed_power, dtype=np.float64)
    if power.ndim != 2 or power.shape[1] < N_HEAD_FRAMES + N_TAIL_FRAMES:
        raise ValueError(
            "observed_power must have shape (R,N) with at least 28 frames."
        )
    if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError("observed_power must contain finite positive values.")
    floor = np.mean(power[:, -N_TAIL_FRAMES:], axis=1)
    head = np.mean(power[:, :N_HEAD_FRAMES], axis=1)
    signal = np.maximum(head - floor, np.finfo(np.float64).tiny)
    return np.column_stack((0.5 * signal, 0.5 * signal, floor))


def run_amplitude_inference(
    *,
    seed: int = SEED,
    n_realizations: int = N_REALIZATIONS,
    n_frames: int = N_FRAMES,
    max_sweep_evaluations: int = 20_000,
    objective_tol: float = 1e-10,
    fixed_point_tol: float = 1e-8,
) -> AmplitudeInferenceRun:
    """Generate matched exact-model STFTs and estimate their amplitudes.

    Parameters
    ----------
    seed
        Random seed used for one shared circular-Gaussian innovation array.
    n_realizations
        Number ``R`` of independent STFT realizations per amplitude ratio.
    n_frames
        Number ``N`` of frames in every realization.
    max_sweep_evaluations
        Maximum complete ordinary SAGE sweeps in each ratio case.
    objective_tol
        Non-negative relative total IS-objective stopping tolerance per case.
    fixed_point_tol
        Non-negative tolerance for the final stability probe. A case is marked
        converged only if both the loss stop and this probe pass; this
        diagnostic does not extend an early loss-based stop.

    Returns
    -------
    AmplitudeInferenceRun
        Complete ground truth, synthetic STFT coefficients, starts, estimates,
        and convergence diagnostics. Rates are fixed and never estimated.
    """

    if n_realizations <= 0 or n_frames < N_HEAD_FRAMES + N_TAIL_FRAMES:
        raise ValueError("use positive realizations and at least 28 frames.")
    if max_sweep_evaluations <= 0:
        raise ValueError("max_sweep_evaluations must be positive.")
    for name, value in (
        ("objective_tol", objective_tol),
        ("fixed_point_tol", fixed_point_tol),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    times_s = np.arange(n_frames, dtype=np.float64) * HOP_S
    dictionary = fixed_rate_dictionary(times_s)
    decay_amplitudes = amplitude_pair_from_ratio_db(FAST_TO_SLOW_DB)
    noise_floor = 10.0 ** (NOISE_FLOOR_DB / 10.0)
    true_amplitudes = np.column_stack(
        (decay_amplitudes, np.full(FAST_TO_SLOW_DB.size, noise_floor))
    )
    exact_variance = true_amplitudes @ dictionary

    # Common random numbers make each ratio use the same standardized STFT
    # innovations while retaining independent realizations within a case.
    innovations = sample_complex_gaussian(
        np.ones((n_realizations, n_frames), dtype=np.float64),
        rng=np.random.default_rng(seed),
    )
    coefficients = np.sqrt(exact_variance[:, np.newaxis, :]) * innovations
    observed_power = np.abs(coefficients) ** 2

    initial_amplitudes = np.empty(
        (FAST_TO_SLOW_DB.size, n_realizations, 3), dtype=np.float64
    )
    estimated_amplitudes = np.empty_like(initial_amplitudes)
    n_sweep_evaluations = np.empty(FAST_TO_SLOW_DB.size, dtype=np.int64)
    converged = np.empty(FAST_TO_SLOW_DB.size, dtype=bool)
    fixed_point_residual = np.empty(FAST_TO_SLOW_DB.size, dtype=np.float64)
    initial_objective = np.empty(FAST_TO_SLOW_DB.size, dtype=np.float64)
    final_objective = np.empty(FAST_TO_SLOW_DB.size, dtype=np.float64)

    objective_history = np.full((FAST_TO_SLOW_DB.size, max_sweep_evaluations + 1), np.nan)

    for case_index, ratio_db in enumerate(FAST_TO_SLOW_DB):
        initial_amplitudes[case_index] = equal_decay_initialization(
            observed_power[case_index]
        )
        result = fit_amplitudes(
            observed_power[case_index],
            dictionary,
            initial_amplitudes=initial_amplitudes[case_index],
            max_iter=max_sweep_evaluations,
            tol=objective_tol,
        )
        estimated_amplitudes[case_index] = result.amplitudes
        n_sweeps = result.loss_history.size - 1
        n_sweep_evaluations[case_index] = n_sweeps
        probe = fit_amplitudes(
            observed_power[case_index], dictionary, result.amplitudes,
            max_iter=1, tol=0.0,
        )
        scale = np.mean(observed_power[case_index], axis=1, keepdims=True)
        fixed_point_residual[case_index] = np.max(
            np.abs(probe.amplitudes - result.amplitudes) / scale
        )
        converged[case_index] = (
            result.converged and fixed_point_residual[case_index] <= fixed_point_tol
        )
        objective_history[case_index, : n_sweeps + 1] = result.loss_history
        initial_objective[case_index] = result.loss_history[0]
        final_objective[case_index] = result.loss_history[-1]
        print(
            f"ratio={ratio_db:+g} dB: sweeps={n_sweeps}, "
            f"converged={converged[case_index]}",
            flush=True,
        )

    return AmplitudeInferenceRun(
        seed=seed,
        times_s=times_s,
        dictionary=dictionary,
        ratio_db=FAST_TO_SLOW_DB.copy(),
        true_amplitudes=true_amplitudes,
        coefficients=coefficients,
        observed_power=observed_power,
        initial_amplitudes=initial_amplitudes,
        estimated_amplitudes=estimated_amplitudes,
        n_sweep_evaluations=n_sweep_evaluations,
        converged=converged,
        fixed_point_residual=fixed_point_residual,
        objective_history=objective_history,
        max_sweep_evaluations=max_sweep_evaluations,
        objective_tol=objective_tol,
        fixed_point_tol=fixed_point_tol,
        initial_objective=initial_objective,
        final_objective=final_objective,
    )


def summarize_run(run: AmplitudeInferenceRun) -> list[dict[str, float | str]]:
    """Return empirical estimator summaries for all ratio/component pairs.

    Linear statistics are in variance units. Relative errors and quantiles are
    power decibels of ``estimate / truth``. The returned list has ``3*C`` rows.
    """

    rows: list[dict[str, float | str]] = []
    for case_index, ratio_db in enumerate(run.ratio_db):
        for component_index, label in enumerate(COMPONENT_LABELS):
            truth = run.true_amplitudes[case_index, component_index]
            estimates = run.estimated_amplitudes[case_index, :, component_index]
            relative_error_db = 10.0 * np.log10(estimates / truth)
            rows.append(
                {
                    "fast_to_slow_db": float(ratio_db),
                    "component": label,
                    "true_amplitude": float(truth),
                    "mean_estimate": float(np.mean(estimates)),
                    "bias": float(np.mean(estimates) - truth),
                    "standard_deviation": float(np.std(estimates, ddof=1)),
                    "coefficient_of_variation": float(
                        np.std(estimates, ddof=1) / truth
                    ),
                    "relative_error_db_q05": float(
                        np.quantile(relative_error_db, 0.05)
                    ),
                    "relative_error_db_median": float(
                        np.median(relative_error_db)
                    ),
                    "relative_error_db_q95": float(
                        np.quantile(relative_error_db, 0.95)
                    ),
                }
            )
    return rows


def case_diagnostics(run: AmplitudeInferenceRun) -> list[dict[str, float]]:
    """Return component-tradeoff and total-amplitude diagnostics per case."""

    rows: list[dict[str, float]] = []
    errors = run.estimated_amplitudes[:, :, :2] - run.true_amplitudes[:, None, :2]
    for case_index, ratio_db in enumerate(run.ratio_db):
        total_error = np.sum(errors[case_index], axis=1)
        rows.append(
            {
                "fast_to_slow_db": float(ratio_db),
                "fast_slow_error_correlation": float(
                    np.corrcoef(errors[case_index].T)[0, 1]
                ),
                "total_decay_amplitude_bias": float(np.mean(total_error)),
                "total_decay_amplitude_standard_deviation": float(
                    np.std(total_error, ddof=1)
                ),
                "sweep_evaluations": float(
                    run.n_sweep_evaluations[case_index]
                ),
                "converged": float(run.converged[case_index]),
                "fixed_point_residual": float(
                    run.fixed_point_residual[case_index]
                ),
                "final_over_initial_is": float(
                    run.final_objective[case_index] / run.initial_objective[case_index]
                ),
            }
        )
    return rows


def save_run(run: AmplitudeInferenceRun, output_dir: Path) -> None:
    """Save reproducible arrays and empirical CSV summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "amplitude_inference_results.npz",
        times_s=run.times_s,
        dictionary=run.dictionary,
        fast_to_slow_db=run.ratio_db,
        true_t60_s=TRUE_T60_S,
        true_amplitudes=run.true_amplitudes,
        stft_coefficients=run.coefficients,
        observed_power=run.observed_power,
        initial_amplitudes=run.initial_amplitudes,
        estimated_amplitudes=run.estimated_amplitudes,
        n_sweep_evaluations=run.n_sweep_evaluations,
        converged=run.converged,
        fixed_point_residual=run.fixed_point_residual,
        objective_history=run.objective_history,
        max_sweep_evaluations=run.max_sweep_evaluations,
        objective_tol=run.objective_tol,
        fixed_point_tol=run.fixed_point_tol,
        solver=np.asarray("amplitude_sage"),
        initial_objective=run.initial_objective,
        final_objective=run.final_objective,
        hop_s=np.asarray(HOP_S),
        noise_floor_db=np.asarray(NOISE_FLOOR_DB),
        seed=np.asarray(run.seed),
    )

    summary = summarize_run(run)
    with (output_dir / "amplitude_inference_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    diagnostics = case_diagnostics(run)
    with (output_dir / "amplitude_inference_case_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)


def load_run(path: str | Path) -> AmplitudeInferenceRun:
    """Load an ordinary-SAGE NPZ with the shapes and units of AmplitudeInferenceRun.

    ``path`` is a saved ``amplitude_inference_results.npz``. Returns the
    complete run for analysis and plotting without invoking an estimator.
    """

    aliases = {"ratio_db": "fast_to_slow_db", "coefficients": "stft_coefficients"}
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["solver"]) != "amplitude_sage":
            raise ValueError("expected an ordinary amplitude-SAGE archive")
        values = {name: archive[aliases.get(name, name)]
                  for name in AmplitudeInferenceRun.__dataclass_fields__}
    for name in ("seed", "max_sweep_evaluations", "objective_tol", "fixed_point_tol"):
        values[name] = values[name].item()
    return AmplitudeInferenceRun(**values)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--max-sweep-evaluations", type=int, default=20_000)
    parser.add_argument("--objective-tol", type=float, default=1e-10)
    parser.add_argument("--fixed-point-tol", type=float, default=1e-8)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.realizations <= 0:
        parser.error("--realizations must be positive")
    if args.frames < N_HEAD_FRAMES + N_TAIL_FRAMES:
        parser.error("--frames must be at least 28")
    if args.max_sweep_evaluations <= 0:
        parser.error("--max-sweep-evaluations must be positive")
    for name in ("objective_tol", "fixed_point_tol"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def main() -> None:
    """Run the controlled amplitude-inference Monte Carlo experiment."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    run = run_amplitude_inference(
        seed=args.seed,
        n_realizations=args.realizations,
        n_frames=args.frames,
        max_sweep_evaluations=args.max_sweep_evaluations,
        objective_tol=args.objective_tol,
        fixed_point_tol=args.fixed_point_tol,
    )
    save_run(run, output_dir)
    print(f"results={output_dir / 'amplitude_inference_results.npz'}")
    print(f"summary={output_dir / 'amplitude_inference_summary.csv'}")


if __name__ == "__main__":
    main()
