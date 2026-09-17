"""Map sensitivity to the two initial T60 values on one fixed dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    DecaySAGEResult,
    pseudo_decay_sage,
    rate_to_t60,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir
from experiments.synthetic.validate_multislope_sage import (
    INITIAL_NOISE_DB,
    N_COMPONENTS,
    N_FREQUENCIES,
    N_RIRS,
    T60_RANGE_S,
    MultislopeData,
    generate_multislope_data,
)

INITIAL_AMPLITUDE_FRAMES = 8


def _weight_slug(power: float) -> str:
    return f"rho_p{power:g}".replace(".", "p")


def initial_t60_pairs(t60_grid_s: np.ndarray) -> np.ndarray:
    """Return every ordered pair from a one-dimensional T60 grid.

    Parameters
    ----------
    t60_grid_s
        Positive decay times in seconds, shape ``(G,)``.

    Returns
    -------
    np.ndarray
        Ordered initial decay-time pairs, shape ``(G**2, 2)``. Component 1
        varies fastest, so reshaping a result to ``(G,G)`` places component 1
        on the horizontal axis and component 2 on the vertical axis.
    """

    grid = np.asarray(t60_grid_s, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("t60_grid_s must be one-dimensional with at least 2 values.")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("t60_grid_s must contain finite positive values.")
    component_1, component_2 = np.meshgrid(grid, grid, indexing="xy")
    return np.column_stack((component_1.ravel(), component_2.ravel()))


def equal_amplitude_initialization(
    observed_power: np.ndarray,
    *,
    n_components: int = N_COMPONENTS,
    n_initial_frames: int = INITIAL_AMPLITUDE_FRAMES,
) -> np.ndarray:
    """Split initial observed power equally across decay components.

    Parameters
    ----------
    observed_power
        Positive observed power, shape ``(R,F,N)``.
    n_components
        Number of decay components receiving equal shares.
    n_initial_frames
        Number of leading frames used for the per-``(R,F)`` power estimate.

    Returns
    -------
    np.ndarray
        Positive unit-origin variance amplitudes, shape ``(R,F,K)``.
    """

    power = np.asarray(observed_power, dtype=np.float64)
    if power.ndim != 3 or power.shape[2] < n_initial_frames:
        raise ValueError(
            "observed_power must have shape (R,F,N) with enough initial frames."
        )
    if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError("observed_power must contain finite positive values.")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    if n_initial_frames <= 0:
        raise ValueError("n_initial_frames must be positive.")
    initial_power = np.mean(power[:, :, :n_initial_frames], axis=2)
    return np.repeat(
        (initial_power / n_components)[:, :, np.newaxis],
        n_components,
        axis=2,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=6,
        help="Number of initial T60 values per component (default: 6).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum sweeps for every initialization (default: 500).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Relative outer observed-objective tolerance.",
    )
    parser.add_argument(
        "--component-weight-power",
        type=float,
        default=2.0,
        help="Pseudo-SAGE component-strength exponent (default: 2).",
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
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    if (
        not np.isfinite(args.component_weight_power)
        or args.component_weight_power < 0.0
    ):
        parser.error("--component-weight-power must be finite and non-negative")
    return args


def _fit_one_start(
    data: MultislopeData,
    initial_t60_s: np.ndarray,
    initial_amplitudes: np.ndarray,
    args: argparse.Namespace,
) -> DecaySAGEResult:
    initial_noise_floor = np.full(
        (N_RIRS, N_FREQUENCIES), 10.0 ** (INITIAL_NOISE_DB / 10.0)
    )
    return pseudo_decay_sage(
        data.observed_power,
        data.times_s,
        t60_to_rate(initial_t60_s[np.newaxis, :]),
        rate_bounds_per_s=(
            t60_to_rate(T60_RANGE_S[1]),
            t60_to_rate(T60_RANGE_S[0]),
        ),
        component_weight_power=args.component_weight_power,
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=True,
        max_iter=args.max_iter,
        tol=args.tol,
        rate_method=args.rate_method,
    )



def _write_csv(
    path: Path,
    initial_pairs_s: np.ndarray,
    final_labeled_t60_s: np.ndarray,
    final_t60_s: np.ndarray,
    absolute_error_s: np.ndarray,
    n_iter: np.ndarray,
    converged: np.ndarray,
    final_objective: np.ndarray,
    objective_monotone: np.ndarray,
    variance_log_rmse_db: np.ndarray,
    amplitude_rmse_db: np.ndarray,
    noise_floor_rmse_db: np.ndarray,
) -> None:
    fieldnames = [
        "initial_t60_component_1_s",
        "initial_t60_component_2_s",
        "final_t60_component_1_s",
        "final_t60_component_2_s",
        "final_t60_short_s",
        "final_t60_long_s",
        "absolute_error_short_s",
        "absolute_error_long_s",
        "max_absolute_error_s",
        "n_iter",
        "converged",
        "final_is_objective",
        "objective_monotone",
        "variance_log_rmse_db",
        "amplitude_rmse_short_db",
        "amplitude_rmse_long_db",
        "noise_floor_rmse_db",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, initial_pair_s in enumerate(initial_pairs_s):
            writer.writerow(
                {
                    "initial_t60_component_1_s": initial_pair_s[0],
                    "initial_t60_component_2_s": initial_pair_s[1],
                    "final_t60_component_1_s": final_labeled_t60_s[index, 0],
                    "final_t60_component_2_s": final_labeled_t60_s[index, 1],
                    "final_t60_short_s": final_t60_s[index, 0],
                    "final_t60_long_s": final_t60_s[index, 1],
                    "absolute_error_short_s": absolute_error_s[index, 0],
                    "absolute_error_long_s": absolute_error_s[index, 1],
                    "max_absolute_error_s": np.max(absolute_error_s[index]),
                    "n_iter": n_iter[index],
                    "converged": bool(converged[index]),
                    "final_is_objective": final_objective[index],
                    "objective_monotone": bool(objective_monotone[index]),
                    "variance_log_rmse_db": variance_log_rmse_db[index],
                    "amplitude_rmse_short_db": amplitude_rmse_db[index, 0],
                    "amplitude_rmse_long_db": amplitude_rmse_db[index, 1],
                    "noise_floor_rmse_db": noise_floor_rmse_db[index],
                }
            )


def run_sweep(args: argparse.Namespace) -> Path:
    """Run and save the fixed-data decay-initialization basin experiment.

    Parameters
    ----------
    args
        Parsed grid, estimator, output, and display controls.

    Returns
    -------
    Path
        Timestamped directory containing numerical results and plots.
    """

    output_dir = create_run_output_dir(args.output_root)
    data = generate_multislope_data()
    t60_grid_s = np.linspace(*T60_RANGE_S, args.grid_size)
    initial_pairs_s = initial_t60_pairs(t60_grid_s)
    initial_amplitudes = equal_amplitude_initialization(data.observed_power)
    true_t60_s = data.t60_s[0]
    n_runs = initial_pairs_s.shape[0]

    final_labeled_t60_s = np.empty((n_runs, N_COMPONENTS))
    final_t60_s = np.empty_like(final_labeled_t60_s)
    absolute_error_s = np.empty_like(final_t60_s)
    n_iter = np.empty(n_runs, dtype=np.int64)
    converged = np.empty(n_runs, dtype=np.bool_)
    initial_objective = np.empty(n_runs)
    final_objective = np.empty(n_runs)
    objective_monotone = np.empty(n_runs, dtype=np.bool_)
    maximum_objective_increase = np.empty(n_runs)
    variance_log_rmse_db = np.empty(n_runs)
    amplitude_rmse_db = np.empty((n_runs, N_COMPONENTS))
    noise_floor_rmse_db = np.empty(n_runs)
    objective_history = np.full((n_runs, args.max_iter + 1), np.nan)
    t60_history_s = np.full(
        (n_runs, args.max_iter + 1, N_COMPONENTS), np.nan
    )
    labeled_t60_history_s = np.full_like(t60_history_s, np.nan)

    for run_index, initial_pair_s in enumerate(initial_pairs_s):
        result = _fit_one_start(
            data, initial_pair_s, initial_amplitudes, args
        )
        estimated_t60_s = rate_to_t60(result.rates_per_s)[0]
        order = np.argsort(estimated_t60_s)
        ordered_t60_s = estimated_t60_s[order]
        ordered_amplitudes_db = 10.0 * np.log10(
            result.amplitudes[:, 0, order]
        )
        objective_steps = np.diff(result.objective_history)
        history_length = result.objective_history.size
        labeled_history = rate_to_t60(result.rate_history_per_s[:, 0, :])
        ordered_history = np.sort(labeled_history, axis=1)

        final_labeled_t60_s[run_index] = estimated_t60_s
        final_t60_s[run_index] = ordered_t60_s
        absolute_error_s[run_index] = np.abs(ordered_t60_s - true_t60_s)
        n_iter[run_index] = result.n_iter
        converged[run_index] = result.converged
        initial_objective[run_index] = result.objective_history[0]
        final_objective[run_index] = result.objective_history[-1]
        objective_monotone[run_index] = np.all(objective_steps <= 1e-8)
        maximum_objective_increase[run_index] = max(
            0.0, float(np.max(objective_steps))
        )
        variance_log_rmse_db[run_index] = np.sqrt(
            np.mean(
                (10.0 * np.log10(result.variance / data.variance)) ** 2
            )
        )
        amplitude_rmse_db[run_index] = np.sqrt(
            np.mean(
                (ordered_amplitudes_db - data.amplitudes_db) ** 2,
                axis=0,
            )
        )
        estimated_floor_db = 10.0 * np.log10(result.noise_floor[:, 0])
        noise_floor_rmse_db[run_index] = np.sqrt(
            np.mean((estimated_floor_db - data.noise_floor_db) ** 2)
        )
        objective_history[run_index, :history_length] = (
            result.objective_history
        )
        t60_history_s[run_index, :history_length] = ordered_history
        labeled_t60_history_s[run_index, :history_length] = labeled_history
        print(
            f"run={run_index + 1:02d}/{n_runs}, "
            f"initial_t60_s={initial_pair_s.tolist()}, "
            f"final_t60_s={ordered_t60_s.tolist()}, "
            f"max_error_s={np.max(absolute_error_s[run_index]):.6f}, "
            f"sweeps={result.n_iter}, converged={result.converged}"
        )

    grid_shape = (args.grid_size, args.grid_size)
    maximum_error_grid_s = np.max(absolute_error_s, axis=1).reshape(grid_shape)
    n_iter_grid = n_iter.reshape(grid_shape)
    objective_excess_grid = (
        final_objective - np.min(final_objective)
    ).reshape(grid_shape)
    converged_grid = converged.reshape(grid_shape)
    slug = f"multislope_initial_t60_{_weight_slug(args.component_weight_power)}"

    csv_path = output_dir / f"{slug}_summary.csv"
    _write_csv(
        csv_path,
        initial_pairs_s,
        final_labeled_t60_s,
        final_t60_s,
        absolute_error_s,
        n_iter,
        converged,
        final_objective,
        objective_monotone,
        variance_log_rmse_db,
        amplitude_rmse_db,
        noise_floor_rmse_db,
    )
    archive_path = output_dir / f"{slug}_results.npz"
    np.savez_compressed(
        archive_path,
        t60_grid_s=t60_grid_s,
        initial_t60_s=initial_pairs_s,
        true_t60_s=true_t60_s,
        final_labeled_t60_s=final_labeled_t60_s,
        final_t60_s=final_t60_s,
        absolute_error_s=absolute_error_s,
        n_iter=n_iter,
        converged=converged,
        initial_objective=initial_objective,
        final_objective=final_objective,
        objective_monotone=objective_monotone,
        maximum_objective_increase=maximum_objective_increase,
        variance_log_rmse_db=variance_log_rmse_db,
        amplitude_rmse_db=amplitude_rmse_db,
        noise_floor_rmse_db=noise_floor_rmse_db,
        objective_history=objective_history,
        t60_history_s=t60_history_s,
        labeled_t60_history_s=labeled_t60_history_s,
        component_weight_power=np.asarray(args.component_weight_power),
        max_iter=np.asarray(args.max_iter),
        tolerance=np.asarray(args.tol),
    )

    print(f"true_t60_s={true_t60_s.tolist()}")
    print(f"grid_t60_s={t60_grid_s.tolist()}")
    print(
        f"converged_runs={int(np.sum(converged))}/{n_runs}, "
        f"monotone_runs={int(np.sum(objective_monotone))}/{n_runs}"
    )
    print(
        "max_t60_error_s="
        f"min={np.min(maximum_error_grid_s):.6f}, "
        f"median={np.median(maximum_error_grid_s):.6f}, "
        f"max={np.max(maximum_error_grid_s):.6f}"
    )
    print(f"saved={csv_path.resolve()}")
    print(f"saved={archive_path.resolve()}")
    return output_dir.resolve()


def main() -> None:
    """Run the decay-initialization basin sweep."""

    run_sweep(_arguments())


if __name__ == "__main__":
    main()
