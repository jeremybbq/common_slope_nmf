"""Experiment plots loaded from saved NPZ results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from experiments._run_output import save_figure

def plot_generator_validation(
    times_s: ArrayLike,
    normalized_decay_db: ArrayLike,
    t60_s: float,
    normalized_power: ArrayLike,
    target_variances: ArrayLike,
    empirical_mean_power: ArrayLike,
) -> Figure:
    """Plot decay convention, normalized-power law, and power calibration."""

    times = np.asarray(times_s); decay_db = np.asarray(normalized_decay_db)
    power = np.asarray(normalized_power); target = np.asarray(target_variances)
    empirical = np.asarray(empirical_mean_power)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].plot(times, decay_db, linewidth=2); axes[0].axvline(t60_s, color="tab:red", linestyle="--"); axes[0].axhline(-60.0, color="tab:red", linestyle="--"); axes[0].scatter([t60_s], [-60.0], color="tab:red", zorder=3)
    axes[0].set(title="Energy-decay convention", xlabel="Elapsed time (s)", ylabel="Relative variance (dB)", ylim=(-125.0, 5.0)); axes[0].grid(alpha=0.25)
    axes[1].hist(power[:, 0], bins=100, range=(0.0, 7.0), density=True, alpha=0.65, label="Empirical")
    grid = np.linspace(0.0, 7.0, 400); axes[1].plot(grid, np.exp(-grid), linewidth=2, label="Exp(1)")
    axes[1].set(title=r"Normalized power $Y/V$", xlabel=r"$Y/V$", ylabel="Density"); axes[1].legend(); axes[1].grid(alpha=0.25)
    axes[2].loglog(target, empirical, "o", markersize=7, label="Empirical mean"); axes[2].loglog(target, target, linestyle="--", label="Identity")
    axes[2].set(title="Power calibration", xlabel="Requested variance", ylabel="Mean sampled power"); axes[2].legend(); axes[2].grid(alpha=0.25, which="both")
    figure.tight_layout(); return figure


def plot_results(results_path: Path, *, show: bool = False) -> Path:
    """Load a generator-validation NPZ and write its diagnostic figure."""

    with np.load(results_path, allow_pickle=False) as payload:
        figure = plot_generator_validation(
            payload["times_s"],
            payload["normalized_decay_db"],
            float(payload["t60_s"]),
            payload["normalized_power"],
            payload["target_variances"],
            payload["empirical_mean_power"],
        )
        output_dir = results_path.parent
    path = save_figure(
        figure, output_dir, "experiment_01_02_validation.png", show=show
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

