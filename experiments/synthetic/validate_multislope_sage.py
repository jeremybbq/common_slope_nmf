"""Run a reproducible stochastic two-slope common-rate SAGE experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    DecaySAGEResult,
    decay_sage,
    exponential_variance,
    rate_to_t60,
    pseudo_decay_sage,
    sample_complex_gaussian,
    t60_to_rate,
)
from experiments.synthetic.validate_known_decay_stft import frame_times
from experiments._run_output import create_run_output_dir

SEED = 20260724
N_RIRS = 512
N_FREQUENCIES = 1
N_COMPONENTS = 2
SAMPLE_RATE_HZ = 24_000
DURATION_S = 2.0
FRAME_SIZE_SAMPLES = 256
HOP_SIZE_SAMPLES = 128
T60_RANGE_S = (0.5, 3.0)
AMPLITUDE_MEAN_DB = np.array([-10.0, -10.0])
AMPLITUDE_STD_DB = 10.0 / 3.0
AMPLITUDE_CORRELATION = -0.8
NOISE_MEAN_DB = -40.0
NOISE_STD_DB = 2.0 / 3.0
INITIAL_T60_S = np.array([[2.0, 2.0]])
INITIAL_NOISE_DB = -35.0


@dataclass(frozen=True)
class MultislopeData:
    """One exact-model two-slope dataset.

    Attributes
    ----------
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    t60_s
        Shared energy-decay times in seconds, shape ``(1, 2)``.
    amplitudes_db
        Unit-origin component variance amplitudes in dB, shape ``(R, 2)``.
    noise_floor_db
        Time-invariant variance floors in dB, shape ``(R,)``.
    variance
        Generating coefficient variances, shape ``(R, 1, N)``.
    observed_power
        Circular complex-Gaussian instantaneous powers, shape ``(R, 1, N)``.
    """

    times_s: np.ndarray
    t60_s: np.ndarray
    amplitudes_db: np.ndarray
    noise_floor_db: np.ndarray
    variance: np.ndarray
    observed_power: np.ndarray


def generate_multislope_data(seed: int = SEED) -> MultislopeData:
    """Generate the seeded two-slope complex-Gaussian experiment.

    Parameters
    ----------
    seed
        NumPy random seed controlling rates, amplitudes, floors, and samples.

    Returns
    -------
    MultislopeData
        Shared rates, per-RIR parameters, exact variances, and observed powers.

    Notes
    -----
    Component and floor levels are Gaussian in power dB and are converted with
    ``10**(level_db / 10)``. This preserves positivity in linear variance units.
    """

    rng = np.random.default_rng(seed)
    times_s = frame_times(
        DURATION_S,
        SAMPLE_RATE_HZ,
        FRAME_SIZE_SAMPLES,
        HOP_SIZE_SAMPLES,
    )
    t60_s = np.sort(
        rng.uniform(*T60_RANGE_S, size=N_COMPONENTS)
    )[np.newaxis, :]
    covariance_db2 = AMPLITUDE_STD_DB**2 * np.array(
        [
            [1.0, AMPLITUDE_CORRELATION],
            [AMPLITUDE_CORRELATION, 1.0],
        ]
    )
    amplitudes_db = rng.multivariate_normal(
        AMPLITUDE_MEAN_DB, covariance_db2, size=N_RIRS
    )
    noise_floor_db = rng.normal(
        NOISE_MEAN_DB, NOISE_STD_DB, size=N_RIRS
    )
    amplitudes = 10.0 ** (amplitudes_db[:, np.newaxis, :] / 10.0)
    noise_floor = 10.0 ** (noise_floor_db[:, np.newaxis] / 10.0)
    variance = exponential_variance(
        times_s,
        t60_to_rate(t60_s),
        amplitudes,
        noise_floor=noise_floor,
    )
    stft = sample_complex_gaussian(variance, rng=rng)
    return MultislopeData(
        times_s=times_s,
        t60_s=t60_s,
        amplitudes_db=amplitudes_db,
        noise_floor_db=noise_floor_db,
        variance=variance,
        observed_power=np.abs(stft) ** 2,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2_000,
        help="Maximum number of complete SAGE component sweeps.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Relative outer objective-decrease tolerance.",
    )
    parser.add_argument(
        "--rate-method",
        choices=("newton", "bisection"),
        default="newton",
        help="Profile-rate solver (default: newton).",
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
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    return args


def _fit(
    data: MultislopeData,
    args: argparse.Namespace,
    *,
    component_weight_power: float | None = None,
) -> DecaySAGEResult:
    initial_power = np.mean(data.observed_power[:, :, :8], axis=2)
    initial_amplitudes = np.repeat(
        (initial_power / N_COMPONENTS)[:, :, np.newaxis],
        N_COMPONENTS,
        axis=2,
    )
    initial_noise_floor = np.full(
        (N_RIRS, N_FREQUENCIES), 10.0 ** (INITIAL_NOISE_DB / 10.0)
    )
    estimator = (
        decay_sage
        if component_weight_power is None
        else pseudo_decay_sage
    )
    estimator_arguments: dict[str, float] = {}
    if component_weight_power is not None:
        estimator_arguments["component_weight_power"] = component_weight_power
    return estimator(
        data.observed_power,
        data.times_s,
        t60_to_rate(INITIAL_T60_S),
        rate_bounds_per_s=(
            t60_to_rate(T60_RANGE_S[1]),
            t60_to_rate(T60_RANGE_S[0]),
        ),
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=True,
        max_iter=args.max_iter,
        tol=args.tol,
        rate_method=args.rate_method,
        diagnostic_interval=100,
        **estimator_arguments,
    )


def _ordered_estimates(
    result: DecaySAGEResult,
) -> tuple[np.ndarray, np.ndarray]:
    estimated_t60_s = rate_to_t60(result.rates_per_s)[0]
    order = np.argsort(estimated_t60_s)
    return estimated_t60_s[order], result.amplitudes[:, 0, order]



def _correlation_slug() -> str:
    return f"rho_{AMPLITUDE_CORRELATION:+.2f}".replace("-", "m").replace(
        "+", "p"
    ).replace(".", "p")


def _experiment_slug() -> str:
    initial_slug = "_".join(
        f"{value:.2f}".replace(".", "p") for value in INITIAL_T60_S[0]
    )
    return f"{_correlation_slug()}_init_{initial_slug}"



def run_experiment(
    args: argparse.Namespace, *, component_weight_power: float | None = None
) -> Path:
    """Run the shared exact or weighted multislope experiment.

    Parameters
    ----------
    args
        Parsed controls containing ``max_iter``, ``tol``, ``rate_method``,
        and ``output_root``.
    component_weight_power
        ``None`` selects ordinary SAGE. A non-negative value selects
        pseudo-SAGE with fixed M-step weights ``rho_k ** power``.

    Returns
    -------
    Path
        Timestamped directory containing the NPZ diagnostics.
    """

    output_dir = create_run_output_dir(args.output_root)
    data = generate_multislope_data()
    result = _fit(
        data, args, component_weight_power=component_weight_power
    )
    estimator_label = (
        "SAGE"
        if component_weight_power is None
        else rf"pseudo-SAGE ($\rho^{{{component_weight_power:g}}}$ weighting)"
    )
    estimated_t60_s, estimated_amplitudes = _ordered_estimates(result)
    estimated_amplitudes_db = 10.0 * np.log10(estimated_amplitudes)
    estimated_floor_db = 10.0 * np.log10(result.noise_floor[:, 0])

    slug = f"multislope_{_experiment_slug()}"
    t60_history_s = np.sort(
        rate_to_t60(result.rate_history_per_s[:, 0, :]), axis=1
    )

    diagnostics_path = (
        output_dir
        / f"multislope_{_experiment_slug()}_diagnostics.npz"
    )
    saved_diagnostics: dict[str, np.ndarray] = {
        "objective_history": result.objective_history,
        "rate_history_per_s": result.rate_history_per_s,
        "t60_history_s": t60_history_s,
        "true_t60_s": data.t60_s,
        "times_s": data.times_s,
        "observed_power": data.observed_power,
        "true_variance": data.variance,
        "fitted_variance": result.variance,
        "true_amplitudes_db": data.amplitudes_db,
        "true_noise_floor_db": data.noise_floor_db,
        "estimated_amplitudes": estimated_amplitudes,
        "estimated_t60_s": estimated_t60_s,
        "estimated_noise_floor": result.noise_floor,
        "estimator_label": np.asarray(estimator_label),
        "slug": np.asarray(slug),
        "component_weight_power": np.asarray(
            np.nan
            if component_weight_power is None
            else component_weight_power
        ),
    }
    if result.update_diagnostics is not None:
        saved_diagnostics.update(
            update_sweep_indices=result.update_diagnostics.sweep_indices,
            scaled_total_error=(
                result.update_diagnostics.scaled_total_error
            ),
            component_weight=result.update_diagnostics.component_weight,
            profile_moment=result.update_diagnostics.profile_moment,
            profile_weight_sum=(
                result.update_diagnostics.profile_weight_sum
            ),
        )
    np.savez_compressed(diagnostics_path, **saved_diagnostics)

    print(f"seed={SEED}")
    print(
        f"rirs={N_RIRS}, frequencies={N_FREQUENCIES}, "
        f"frames={data.times_s.size}, last_frame_time_s={data.times_s[-1]:.9f}"
    )
    print(
        "amplitude_distribution_db="
        f"mean={AMPLITUDE_MEAN_DB.tolist()}, std={AMPLITUDE_STD_DB:.6f}, "
        f"rho={AMPLITUDE_CORRELATION:.3f}"
    )
    print(
        "realized_amplitude_mean_std_correlation_db="
        f"{np.mean(data.amplitudes_db, axis=0).tolist()}, "
        f"{np.std(data.amplitudes_db, axis=0, ddof=1).tolist()}, "
        f"{np.corrcoef(data.amplitudes_db.T)[0, 1]:.6f}"
    )
    print(
        "noise_distribution_db="
        f"mean={NOISE_MEAN_DB:.6f}, std={NOISE_STD_DB:.6f}; "
        "realized_mean_std="
        f"{np.mean(data.noise_floor_db):.6f}, "
        f"{np.std(data.noise_floor_db, ddof=1):.6f}"
    )
    print(f"true_t60_s={data.t60_s[0].tolist()}")
    print(f"initial_t60_s={INITIAL_T60_S[0].tolist()}")
    print(f"estimator={estimator_label}")
    print(f"estimated_t60_s={estimated_t60_s.tolist()}")
    print(
        "t60_absolute_error_s="
        f"{np.abs(estimated_t60_s - data.t60_s[0]).tolist()}"
    )
    print(
        "amplitude_rmse_db="
        f"{np.sqrt(np.mean((estimated_amplitudes_db - data.amplitudes_db) ** 2, axis=0)).tolist()}"
    )
    print(
        "noise_floor_rmse_db="
        f"{np.sqrt(np.mean((estimated_floor_db - data.noise_floor_db) ** 2)):.6f}"
    )
    print(
        "variance_log_rmse_db="
        f"{np.sqrt(np.mean((10.0 * np.log10(result.variance / data.variance)) ** 2)):.6f}"
    )
    print(
        f"rate_method={args.rate_method}, sweeps={result.n_iter}, "
        f"converged={result.converged}, final_is={result.objective_history[-1]:.9g}, "
        "objective_monotone="
        f"{bool(np.all(np.diff(result.objective_history) <= 1e-8))}"
    )
    print(f"saved={diagnostics_path.resolve()}")
    return output_dir.resolve()


def main() -> None:
    """Generate, fit, and summarize the common two-slope experiment."""

    run_experiment(_arguments())


if __name__ == "__main__":
    main()
