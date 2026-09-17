"""Plots for SQUAREM versus SAGE convergence comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments._run_output import save_figure


def plot_results(results_path: Path, *, show: bool = False) -> Path:
    """Load saved comparison histories and write the two-panel figure."""

    with np.load(results_path, allow_pickle=False) as payload:
        runs = []
        index = 0
        while f"method_{index}_label" in payload.files:
            runs.append(
                {
                    "label": str(payload[f"method_{index}_label"]),
                    "sweep_evaluations": payload[f"method_{index}_sweep_evaluations"],
                    "objective_history": payload[f"method_{index}_objective_history"],
                }
            )
            index += 1
    best = min(float(np.min(run["objective_history"])) for run in runs)
    gap_floor = max(np.finfo(np.float64).eps * abs(best), 1e-12)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for run in runs:
        axes[0].plot(run["sweep_evaluations"], run["objective_history"], label=run["label"])
        axes[1].semilogy(
            run["sweep_evaluations"],
            np.maximum(run["objective_history"] - best, gap_floor),
            label=run["label"],
        )
    axes[0].set_ylabel("Summed observed IS divergence")
    axes[1].set_ylabel("IS divergence above best retained value")
    for axis in axes:
        axis.set_xlabel("Complete ordinary-SAGE sweep evaluations")
        axis.grid(alpha=0.25)
    axes[0].set_title("Raw loss")
    axes[1].set_title("Convergence detail (log scale)")
    axes[0].legend(fontsize="small")
    figure.suptitle("Fixed seeded two-T60 convergence comparison")
    figure.tight_layout()
    path = save_figure(
        figure, results_path.parent, "multislope_convergence_comparison.png", show=show
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
