"""Sweep stochastic two-slope detection over independent frequency bins.

Each frequency bin receives its own pair of uniformly sampled energy-decay
times.  Frequency index is only an arbitrary experiment label: this script
does not impose frequency smoothness or share rates between bins.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_slope_nmf import (
    exponential_variance,
    init_decay_sage,
    pseudo_decay_sage,
    rate_to_t60,
    sample_power,
    sample_simplex_amplitudes,
    sample_t60,
    t60_to_rate,
)
from common_slope_nmf import plotting as decay_plots
from examples._run_output import create_run_output_dir

SEED = 20260817
N_RIRS = 512
N_PAIRS = 100
N_COMPONENTS = 2
N_FRAMES = 374
HOP_S = 128.0 / 24_000.0
T60_RANGE_S = (0.5, 3.0)
DIRICHLET_ALPHA = 0.5
NOISE_MEAN_DB = -40.0
NOISE_STD_DB = 2.0 / 3.0
COMPONENT_WEIGHT_POWER = 2.0
DEFAULT_TOL = 1e-6


def frequency_batches(
    n_frequencies: int, batch_size: int
) -> list[slice]:
    """Return contiguous frequency slices covering ``range(n_frequencies)``.

    Parameters
    ----------
    n_frequencies
        Positive number of independent frequency bins ``F``.
    batch_size
        Positive maximum number of bins fitted in one SAGE call.

    Returns
    -------
    list of slice
        Non-overlapping slices that cover every frequency exactly once.
    """

    if n_frequencies <= 0 or batch_size <= 0:
        raise ValueError("n_frequencies and batch_size must be positive.")
    return [
        slice(start, min(start + batch_size, n_frequencies))
        for start in range(0, n_frequencies, batch_size)
    ]


def order_decay_components(
    estimated_t60_s: np.ndarray,
    estimated_amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order fitted components by increasing energy-``T60`` in every bin.

    Parameters
    ----------
    estimated_t60_s
        Fitted decay times in seconds, shape ``(F,K)``.
    estimated_amplitudes
        Fitted unit-origin variance amplitudes, shape ``(R,F,K)``.

    Returns
    -------
    ordered_t60_s, ordered_amplitudes, order
        Decay times ``(F,K)``, consistently permuted amplitudes ``(R,F,K)``,
        and the integer component permutation ``(F,K)``.
    """

    t60_s = np.asarray(estimated_t60_s, dtype=np.float64)
    amplitudes = np.asarray(estimated_amplitudes, dtype=np.float64)
    if t60_s.ndim != 2:
        raise ValueError("estimated_t60_s must have shape (F,K).")
    if amplitudes.ndim != 3 or amplitudes.shape[1:] != t60_s.shape:
        raise ValueError("estimated_amplitudes must have shape (R,F,K).")
    order = np.argsort(t60_s, axis=1)
    ordered_t60_s = np.take_along_axis(t60_s, order, axis=1)
    amplitude_order = np.broadcast_to(order[np.newaxis, :, :], amplitudes.shape)
    ordered_amplitudes = np.take_along_axis(
        amplitudes, amplitude_order, axis=2
    )
    return ordered_t60_s, ordered_amplitudes, order


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n-rirs", type=int, default=N_RIRS)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help=(
            "Frequency bins per SAGE call. Smaller values reduce memory and "
            "localize the stopping gate; use 1 for strictly per-bin stopping."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help="Relative observed-objective decrease tolerance (default: 1e-6).",
    )
    parser.add_argument(
        "--component-weight-power",
        type=float,
        default=COMPONENT_WEIGHT_POWER,
    )
    parser.add_argument(
        "--rate-method", choices=("newton", "bisection"), default="newton"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for name in ("n_pairs", "n_rirs", "n_frames", "batch_size", "max_iter"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_frames < 17:
        parser.error("--n-frames must be at least 17 for 8-frame head/tail summaries")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    if (
        not np.isfinite(args.component_weight_power)
        or args.component_weight_power < 0.0
    ):
        parser.error("--component-weight-power must be finite and non-negative")
    args.batch_size = min(args.batch_size, args.n_pairs)
    return args



def _is_per_frequency(
    observed_power: np.ndarray, fitted_variance: np.ndarray
) -> np.ndarray:
    """Return summed IS divergence for each frequency, shape ``(F,)``."""

    ratio = observed_power / fitted_variance
    return np.sum(ratio - np.log(ratio) - 1.0, axis=(0, 2))


def _write_csv(
    path: Path,
    true_t60_s: np.ndarray,
    coarse_t60_s: np.ndarray,
    estimated_t60_s: np.ndarray,
    absolute_error_s: np.ndarray,
    amplitude_rmse_db: np.ndarray,
    floor_rmse_db: np.ndarray,
    variance_rmse_db: np.ndarray,
    final_is: np.ndarray,
    dominant_counts: np.ndarray,
    batch_index: np.ndarray,
    n_iter: np.ndarray,
    converged: np.ndarray,
) -> None:
    fieldnames = [
        "frequency_index",
        "batch_index",
        "true_short_t60_s",
        "true_long_t60_s",
        "true_separation_s",
        "coarse_init_t60_s",
        "estimated_short_t60_s",
        "estimated_long_t60_s",
        "short_abs_error_s",
        "long_abs_error_s",
        "maximum_abs_error_s",
        "short_amplitude_rmse_db",
        "long_amplitude_rmse_db",
        "noise_floor_rmse_db",
        "variance_rmse_db",
        "final_is_divergence",
        "short_dominant_rirs",
        "long_dominant_rirs",
        "sweeps",
        "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frequency_index in range(true_t60_s.shape[0]):
            writer.writerow(
                {
                    "frequency_index": frequency_index,
                    "batch_index": int(batch_index[frequency_index]),
                    "true_short_t60_s": true_t60_s[frequency_index, 0],
                    "true_long_t60_s": true_t60_s[frequency_index, 1],
                    "true_separation_s": np.diff(
                        true_t60_s[frequency_index]
                    )[0],
                    "coarse_init_t60_s": coarse_t60_s[frequency_index],
                    "estimated_short_t60_s": estimated_t60_s[
                        frequency_index, 0
                    ],
                    "estimated_long_t60_s": estimated_t60_s[
                        frequency_index, 1
                    ],
                    "short_abs_error_s": absolute_error_s[frequency_index, 0],
                    "long_abs_error_s": absolute_error_s[frequency_index, 1],
                    "maximum_abs_error_s": np.max(
                        absolute_error_s[frequency_index]
                    ),
                    "short_amplitude_rmse_db": amplitude_rmse_db[
                        frequency_index, 0
                    ],
                    "long_amplitude_rmse_db": amplitude_rmse_db[
                        frequency_index, 1
                    ],
                    "noise_floor_rmse_db": floor_rmse_db[frequency_index],
                    "variance_rmse_db": variance_rmse_db[frequency_index],
                    "final_is_divergence": final_is[frequency_index],
                    "short_dominant_rirs": dominant_counts[frequency_index, 0],
                    "long_dominant_rirs": dominant_counts[frequency_index, 1],
                    "sweeps": int(n_iter[frequency_index]),
                    "converged": bool(converged[frequency_index]),
                }
            )



def main() -> None:
    """Generate, fit, summarize, and plot the decay-detection sweep."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    rng = np.random.default_rng(args.seed)
    times_s = np.arange(args.n_frames, dtype=np.float64) * HOP_S
    true_t60_s = sample_t60(
        args.n_pairs,
        N_COMPONENTS,
        T60_RANGE_S,
        min_separation_s=0.0,
        rng=rng,
    )
    true_rates_per_s = np.asarray(t60_to_rate(true_t60_s))
    true_amplitudes = sample_simplex_amplitudes(
        args.n_rirs,
        args.n_pairs,
        N_COMPONENTS,
        concentration=DIRICHLET_ALPHA,
        total_amplitude=1.0,
        rng=rng,
    )
    noise_levels_db = rng.normal(
        NOISE_MEAN_DB, NOISE_STD_DB, size=(args.n_rirs, args.n_pairs)
    )
    true_noise_floor = 10.0 ** (noise_levels_db / 10.0)
    dominant_counts = np.stack(
        [
            np.sum(np.argmax(true_amplitudes, axis=2) == component_index, axis=0)
            for component_index in range(N_COMPONENTS)
        ],
        axis=1,
    )

    estimated_t60_s = np.empty_like(true_t60_s)
    estimated_amplitudes = np.empty_like(true_amplitudes)
    estimated_noise_floor = np.empty_like(true_noise_floor)
    coarse_t60_s = np.empty(args.n_pairs)
    amplitude_rmse_db = np.empty_like(true_t60_s)
    floor_rmse_db = np.empty(args.n_pairs)
    variance_rmse_db = np.empty(args.n_pairs)
    final_is = np.empty(args.n_pairs)
    n_iter = np.empty(args.n_pairs, dtype=np.int64)
    converged = np.empty(args.n_pairs, dtype=bool)
    batch_index = np.empty(args.n_pairs, dtype=np.int64)
    t60_history_s = np.full(
        (args.max_iter + 1, args.n_pairs, N_COMPONENTS), np.nan
    )
    batches = frequency_batches(args.n_pairs, args.batch_size)
    objective_history = np.full((len(batches), args.max_iter + 1), np.nan)
    example_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    rate_bounds = (
        t60_to_rate(T60_RANGE_S[1]),
        t60_to_rate(T60_RANGE_S[0]),
    )

    for current_batch_index, frequency_slice in enumerate(batches):
        variance = exponential_variance(
            times_s,
            true_rates_per_s[frequency_slice],
            true_amplitudes[:, frequency_slice, :],
            noise_floor=true_noise_floor[:, frequency_slice],
        )
        observed_power = sample_power(variance, rng=rng)
        init = init_decay_sage(
            observed_power,
            times_s,
            N_COMPONENTS,
            rate_bounds_per_s=rate_bounds,
            n_head_frames=8,
            n_tail_frames=8,
            floor_margin_db=6.0,
        )
        result = pseudo_decay_sage(
            observed_power,
            times_s,
            init.rates_per_s,
            rate_bounds_per_s=rate_bounds,
            component_weight_power=args.component_weight_power,
            initial_amplitudes=init.amplitudes,
            initial_noise_floor=init.noise_floor,
            estimate_noise_floor=True,
            max_iter=args.max_iter,
            tol=args.tol,
            rate_method=args.rate_method,
        )
        labeled_t60_s = np.asarray(rate_to_t60(result.rates_per_s))
        ordered_t60_s, ordered_amplitudes, _ = order_decay_components(
            labeled_t60_s, result.amplitudes
        )
        history_s = np.sort(
            rate_to_t60(result.rate_history_per_s), axis=2
        )
        history_length = result.n_iter + 1
        estimated_t60_s[frequency_slice] = ordered_t60_s
        estimated_amplitudes[:, frequency_slice, :] = ordered_amplitudes
        estimated_noise_floor[:, frequency_slice] = result.noise_floor
        coarse_t60_s[frequency_slice] = rate_to_t60(init.fit.rate_per_s)
        t60_history_s[:history_length, frequency_slice] = history_s
        objective_history[current_batch_index, :history_length] = (
            result.objective_history
        )
        n_iter[frequency_slice] = result.n_iter
        converged[frequency_slice] = result.converged
        batch_index[frequency_slice] = current_batch_index

        true_amplitudes_db = 10.0 * np.log10(
            true_amplitudes[:, frequency_slice, :]
        )
        estimated_amplitudes_db = 10.0 * np.log10(ordered_amplitudes)
        amplitude_rmse_db[frequency_slice] = np.sqrt(
            np.mean(
                (estimated_amplitudes_db - true_amplitudes_db) ** 2, axis=0
            )
        )
        floor_rmse_db[frequency_slice] = np.sqrt(
            np.mean(
                (
                    10.0 * np.log10(result.noise_floor)
                    - noise_levels_db[:, frequency_slice]
                )
                ** 2,
                axis=0,
            )
        )
        variance_rmse_db[frequency_slice] = np.sqrt(
            np.mean(
                (
                    10.0 * np.log10(result.variance)
                    - 10.0 * np.log10(variance)
                )
                ** 2,
                axis=(0, 2),
            )
        )
        final_is[frequency_slice] = _is_per_frequency(
            observed_power, result.variance
        )
        if example_maps is None:
            example_maps = (
                observed_power[:, 0, :].copy(),
                variance[:, 0, :].copy(),
                result.variance[:, 0, :].copy(),
            )
        print(
            f"batch={current_batch_index + 1:02d}/{len(batches):02d}, "
            f"frequencies={frequency_slice.start}:{frequency_slice.stop}, "
            f"sweeps={result.n_iter}, converged={result.converged}"
        )

    absolute_error_s = np.abs(estimated_t60_s - true_t60_s)
    csv_path = output_dir / "decay_detection_summary.csv"
    _write_csv(
        csv_path,
        true_t60_s,
        coarse_t60_s,
        estimated_t60_s,
        absolute_error_s,
        amplitude_rmse_db,
        floor_rmse_db,
        variance_rmse_db,
        final_is,
        dominant_counts,
        batch_index,
        n_iter,
        converged,
    )
    archive_path = output_dir / "decay_detection_results.npz"
    np.savez_compressed(
        archive_path,
        seed=np.asarray(args.seed),
        times_s=times_s,
        true_t60_s=true_t60_s,
        true_rates_per_s=true_rates_per_s,
        true_amplitudes=true_amplitudes,
        true_noise_floor=true_noise_floor,
        dominant_counts=dominant_counts,
        coarse_t60_s=coarse_t60_s,
        estimated_t60_s=estimated_t60_s,
        estimated_amplitudes=estimated_amplitudes,
        estimated_noise_floor=estimated_noise_floor,
        absolute_error_s=absolute_error_s,
        amplitude_rmse_db=amplitude_rmse_db,
        floor_rmse_db=floor_rmse_db,
        variance_rmse_db=variance_rmse_db,
        final_is=final_is,
        batch_index=batch_index,
        n_iter=n_iter,
        converged=converged,
        t60_history_s=t60_history_s,
        objective_history=objective_history,
        tolerance=np.asarray(args.tol),
        max_iter=np.asarray(args.max_iter),
        batch_size=np.asarray(args.batch_size),
        dirichlet_alpha=np.asarray(DIRICHLET_ALPHA),
        component_weight_power=np.asarray(args.component_weight_power),
    )

    figures = [
        (decay_plots.plot_true_vs_estimated_t60(true_t60_s, estimated_t60_s), "true_vs_estimated_t60.png"),
        (decay_plots.plot_t60_pair_plane(true_t60_s, estimated_t60_s), "t60_pair_plane.png"),
        (decay_plots.plot_t60_error_vs_separation(true_t60_s, absolute_error_s), "t60_error_vs_separation.png"),
        (decay_plots.plot_amplitude_rmse_vs_separation(true_t60_s, amplitude_rmse_db), "amplitude_error_vs_separation.png"),
        (decay_plots.plot_signed_amplitude_bias_vs_separation(true_t60_s, true_amplitudes, estimated_amplitudes), "signed_amplitude_error_vs_separation.png"),
        (decay_plots.plot_t60_error_by_frequency(absolute_error_s), "t60_error_by_frequency.png"),
        (decay_plots.plot_maximum_t60_error_cdf(absolute_error_s), "maximum_t60_error_cdf.png"),
        (decay_plots.plot_objective_history(objective_history, normalize=True, log_y=True, title="Observed objective convergence by frequency batch"), "objective_convergence.png"),
        (decay_plots.plot_t60_error_evolution(t60_history_s, true_t60_s, 0), "short_t60_error_evolution.png"),
        (decay_plots.plot_t60_error_evolution(t60_history_s, true_t60_s, 1), "long_t60_error_evolution.png"),
    ]
    saved_paths = [
        decay_plots.save_figure(
            figure, output_dir, filename, show=args.show
        )
        for figure, filename in figures
    ]
    if example_maps is None:
        raise RuntimeError("internal error: no frequency batch was evaluated")
    saved_paths.append(
        decay_plots.save_figure(
            decay_plots.plot_variance_maps(
                times_s,
                example_maps,
                ("Observed power", "Exact variance", "Fitted variance"),
            ),
            output_dir,
            "example_frequency_variance_maps.png",
            show=args.show,
        )
    )

    maximum_error_s = np.max(absolute_error_s, axis=1)
    print(f"median_maximum_t60_error_s={np.median(maximum_error_s):.6f}")
    print(f"p90_maximum_t60_error_s={np.quantile(maximum_error_s, 0.9):.6f}")
    print(f"maximum_t60_error_s={np.max(maximum_error_s):.6f}")
    print(f"converged_frequency_fraction={np.mean(converged):.3f}")
    print(f"saved={csv_path.resolve()}")
    print(f"saved={archive_path.resolve()}")
    for path in saved_paths:
        print(f"saved={path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
