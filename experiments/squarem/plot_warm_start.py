"""Plots for pseudo-SAGE to SQUAREM warm-start continuation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments._run_output import save_figure


def plot_results(results_path: Path, *, show: bool = False) -> Path:
    """Load a warm-start NPZ and write the two-stage loss figure."""

    with np.load(results_path, allow_pickle=False) as payload:
        pseudo_objective = payload["pseudo_objective_history"]
        squarem_sweeps = payload["squarem_sweep_evaluation_history"]
        squarem_objective = payload["squarem_objective_history"]
        power = float(payload["component_weight_power"])
    pseudo_label = f"pseudo-SAGE (rho^{power:g})"
    pseudo_sweeps = np.arange(pseudo_objective.size)
    offset_squarem_sweeps = pseudo_sweeps[-1] + squarem_sweeps
    best = float(min(np.min(pseudo_objective), np.min(squarem_objective)))
    gap_floor = max(np.finfo(np.float64).eps * abs(best), 1e-12)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(pseudo_sweeps, pseudo_objective, label=pseudo_label)
    axes[0].plot(offset_squarem_sweeps, squarem_objective, label="warm-started SQUAREM")
    axes[0].axvline(pseudo_sweeps[-1], color="black", linestyle="--", linewidth=1, label="optimizer switch")
    axes[0].set(title="Continuous two-stage trajectory", xlabel="Total complete sweep evaluations", ylabel="Summed observed IS divergence")
    axes[1].semilogy(squarem_sweeps, np.maximum(squarem_objective - best, gap_floor), color="tab:red")
    axes[1].set(title="SQUAREM warm-start phase", xlabel="Additional ordinary-SAGE sweep evaluations", ylabel="IS divergence above final retained value")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize="small")
    figure.suptitle("Pseudo-SAGE followed by safeguarded SQUAREM")
    figure.tight_layout()
    path = save_figure(
        figure, results_path.parent, "pseudo_sage_to_squarem_convergence.png", show=show
    )
    print(f"saved={path}")
    if show:
        plt.show()
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_results(args.results, show=args.show)


if __name__ == "__main__":
    main()
