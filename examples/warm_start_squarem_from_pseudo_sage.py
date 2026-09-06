"""Warm-start SQUAREM from the seeded two-slope pseudo-SAGE fit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from common_slope_nmf import (
    decay_squarem,
    pseudo_decay_sage,
    rate_to_t60,
    t60_to_rate,
)
from examples._run_output import create_run_output_dir
from examples.compare_multislope_convergence import (
    _initial_parameters,
    _rate_bounds,
)
from examples.validate_multislope_sage import (
    INITIAL_T60_S,
    generate_multislope_data,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-weight-power",
        type=float,
        default=1.0,
        help="Pseudo-SAGE component-strength exponent (default: 1).",
    )
    parser.add_argument(
        "--pseudo-sweeps",
        type=int,
        default=2_000,
        help="Maximum pseudo-SAGE component sweeps (default: 2000).",
    )
    parser.add_argument(
        "--squarem-sweep-evaluations",
        type=int,
        default=2_000,
        help="Fresh SQUAREM base-sweep evaluation budget (default: 2000).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Relative observed-objective tolerance shared by both stages.",
    )
    parser.add_argument(
        "--fixed-point-tol",
        type=float,
        default=1e-6,
        help="Additional dimensionless SQUAREM fixed-point tolerance.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for the timestamped output.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the comparison figure."
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.component_weight_power)
        or args.component_weight_power < 0.0
    ):
        parser.error("--component-weight-power must be finite and non-negative")
    if args.pseudo_sweeps <= 0:
        parser.error("--pseudo-sweeps must be positive")
    if args.squarem_sweep_evaluations <= 0:
        parser.error("--squarem-sweep-evaluations must be positive")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    if not np.isfinite(args.fixed_point_tol) or args.fixed_point_tol < 0.0:
        parser.error("--fixed-point-tol must be finite and non-negative")
    return args


def _plot_histories(
    pseudo_objective: np.ndarray,
    squarem_sweeps: np.ndarray,
    squarem_objective: np.ndarray,
    pseudo_label: str,
) -> plt.Figure:
    """Plot the two-stage loss histories against complete sweep evaluations."""

    pseudo_sweeps = np.arange(pseudo_objective.size)
    offset_squarem_sweeps = pseudo_sweeps[-1] + squarem_sweeps
    best = float(min(np.min(pseudo_objective), np.min(squarem_objective)))
    gap_floor = max(np.finfo(np.float64).eps * abs(best), 1e-12)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(pseudo_sweeps, pseudo_objective, label=pseudo_label)
    axes[0].plot(
        offset_squarem_sweeps,
        squarem_objective,
        label="warm-started SQUAREM",
    )
    axes[0].axvline(
        pseudo_sweeps[-1],
        color="black",
        linestyle="--",
        linewidth=1,
        label="optimizer switch",
    )
    axes[0].set(
        title="Continuous two-stage trajectory",
        xlabel="Total complete sweep evaluations",
        ylabel="Summed observed IS divergence",
    )

    axes[1].semilogy(
        squarem_sweeps,
        np.maximum(squarem_objective - best, gap_floor),
        color="tab:red",
    )
    axes[1].set(
        title="SQUAREM warm-start phase",
        xlabel="Additional ordinary-SAGE sweep evaluations",
        ylabel="IS divergence above final retained value",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize="small")
    figure.suptitle("Pseudo-SAGE followed by safeguarded SQUAREM")
    figure.tight_layout()
    return figure


def main() -> None:
    """Fit pseudo-SAGE, continue with SQUAREM, and save complete states."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    data = generate_multislope_data()
    initial_amplitudes, initial_noise_floor = _initial_parameters(data)

    print("running pseudo-SAGE warm-start stage...", flush=True)
    pseudo_start = perf_counter()
    pseudo = pseudo_decay_sage(
        data.observed_power,
        data.times_s,
        t60_to_rate(INITIAL_T60_S),
        rate_bounds_per_s=_rate_bounds(),
        component_weight_power=args.component_weight_power,
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=True,
        max_iter=args.pseudo_sweeps,
        tol=args.tol,
        rate_method="newton",
    )
    pseudo_elapsed_s = perf_counter() - pseudo_start

    print("running SQUAREM from the complete pseudo-SAGE state...", flush=True)
    squarem_start = perf_counter()
    squarem = decay_squarem(
        data.observed_power,
        data.times_s,
        pseudo.rates_per_s,
        rate_bounds_per_s=_rate_bounds(),
        initial_amplitudes=pseudo.amplitudes,
        initial_noise_floor=pseudo.noise_floor,
        estimate_noise_floor=True,
        max_sweep_evaluations=args.squarem_sweep_evaluations,
        objective_tol=args.tol,
        fixed_point_tol=args.fixed_point_tol,
        rate_method="newton",
    )
    squarem_elapsed_s = perf_counter() - squarem_start

    pseudo_t60_s = np.sort(rate_to_t60(pseudo.rates_per_s)[0])
    squarem_t60_s = np.sort(rate_to_t60(squarem.rates_per_s)[0])
    pseudo_label = f"pseudo-SAGE (rho^{args.component_weight_power:g})"
    if not np.isclose(
        pseudo.objective_history[-1],
        squarem.objective_history[0],
        rtol=1e-13,
        atol=0.0,
    ):
        raise RuntimeError("SQUAREM did not start at the pseudo-SAGE endpoint.")

    figure = _plot_histories(
        pseudo.objective_history,
        squarem.sweep_evaluation_history,
        squarem.objective_history,
        pseudo_label,
    )
    figure_path = output_dir / "pseudo_sage_to_squarem_convergence.png"
    figure.savefig(figure_path, dpi=180)
    if not args.show:
        plt.close(figure)

    results_path = output_dir / "pseudo_sage_to_squarem_results.npz"
    np.savez_compressed(
        results_path,
        true_t60_s=data.t60_s,
        pseudo_objective_history=pseudo.objective_history,
        pseudo_rate_history_per_s=pseudo.rate_history_per_s,
        pseudo_final_rates_per_s=pseudo.rates_per_s,
        pseudo_final_amplitudes=pseudo.amplitudes,
        pseudo_final_noise_floor=pseudo.noise_floor,
        pseudo_final_variance=pseudo.variance,
        pseudo_converged=np.asarray(pseudo.converged),
        squarem_sweep_evaluation_history=squarem.sweep_evaluation_history,
        squarem_objective_history=squarem.objective_history,
        squarem_rate_history_per_s=squarem.rate_history_per_s,
        squarem_fixed_point_residual_history=(
            squarem.fixed_point_residual_history
        ),
        squarem_final_rates_per_s=squarem.rates_per_s,
        squarem_final_amplitudes=squarem.amplitudes,
        squarem_final_noise_floor=squarem.noise_floor,
        squarem_final_variance=squarem.variance,
        squarem_converged=np.asarray(squarem.converged),
        squarem_accepted_extrapolations=np.asarray(
            squarem.n_accepted_extrapolations
        ),
        squarem_rejected_extrapolations=np.asarray(
            squarem.n_rejected_extrapolations
        ),
    )

    summary_path = output_dir / "pseudo_sage_to_squarem_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "stage",
                "sweep_evaluations",
                "initial_is",
                "final_is",
                "final_t60_short_s",
                "final_t60_long_s",
                "absolute_t60_error_short_s",
                "absolute_t60_error_long_s",
                "elapsed_s",
                "converged",
            ),
        )
        writer.writeheader()
        for stage, sweeps, objective, t60_s, elapsed_s, converged in (
            (
                pseudo_label,
                pseudo.n_iter,
                pseudo.objective_history,
                pseudo_t60_s,
                pseudo_elapsed_s,
                pseudo.converged,
            ),
            (
                "warm-started SQUAREM",
                squarem.n_sweep_evaluations,
                squarem.objective_history,
                squarem_t60_s,
                squarem_elapsed_s,
                squarem.converged,
            ),
        ):
            writer.writerow(
                {
                    "stage": stage,
                    "sweep_evaluations": sweeps,
                    "initial_is": f"{objective[0]:.12g}",
                    "final_is": f"{objective[-1]:.12g}",
                    "final_t60_short_s": f"{t60_s[0]:.12g}",
                    "final_t60_long_s": f"{t60_s[1]:.12g}",
                    "absolute_t60_error_short_s": (
                        f"{abs(t60_s[0] - data.t60_s[0, 0]):.12g}"
                    ),
                    "absolute_t60_error_long_s": (
                        f"{abs(t60_s[1] - data.t60_s[0, 1]):.12g}"
                    ),
                    "elapsed_s": f"{elapsed_s:.9g}",
                    "converged": converged,
                }
            )

    print(f"true_t60_s={data.t60_s[0].tolist()}")
    print(
        f"pseudo_sage: sweeps={pseudo.n_iter}, converged={pseudo.converged}, "
        f"final_is={pseudo.objective_history[-1]:.9f}, "
        f"final_t60_s={pseudo_t60_s.tolist()}, elapsed_s={pseudo_elapsed_s:.6g}"
    )
    print(
        "warm_squarem: "
        f"evaluations={squarem.n_sweep_evaluations}, "
        f"converged={squarem.converged}, "
        f"final_is={squarem.objective_history[-1]:.9f}, "
        f"final_t60_s={squarem_t60_s.tolist()}, "
        f"final_fixed_point_residual="
        f"{squarem.fixed_point_residual_history[-1]:.9g}, "
        f"accepted={squarem.n_accepted_extrapolations}, "
        f"rejected={squarem.n_rejected_extrapolations}, "
        f"elapsed_s={squarem_elapsed_s:.6g}"
    )
    print(f"saved={figure_path.resolve()}")
    print(f"saved={results_path.resolve()}")
    print(f"saved={summary_path.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
