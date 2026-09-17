"""Experiment plots loaded from saved NPZ results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from experiments._run_output import save_figure

def plot_fixed_rate_validation(
    deterministic_histories: Sequence[ArrayLike],
    deterministic_labels: Sequence[str],
    true_amplitudes: ArrayLike,
    fitted_amplitudes: ArrayLike,
    log_parameter_ratio: ArrayLike,
    component_labels: Sequence[str],
    heldout_excess_nll: ArrayLike,
) -> Figure:
    """Plot the four-panel fixed-rate SAGE validation summary."""

    truth = np.asarray(true_amplitudes); fitted = np.asarray(fitted_amplitudes)
    ratios = np.asarray(log_parameter_ratio); heldout = np.asarray(heldout_excess_nll)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for raw, label in zip(deterministic_histories, deterministic_labels, strict=True):
        objective = np.maximum(np.asarray(raw), np.finfo(float).tiny)
        axes[0, 0].semilogy(np.arange(objective.size), objective, label=label)
    axes[0, 0].set(title="Exact-data SAGE convergence", xlabel="Complete component sweeps", ylabel="Summed IS divergence"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.25)
    axes[0, 1].loglog(truth.ravel(), fitted.ravel(), "o", label="SAGE estimates"); limits=(1e-6,2.0); axes[0, 1].loglog(limits, limits, "--", label="Identity")
    axes[0, 1].set(title="Exact amplitudes and floors", xlabel="True variance amplitude", ylabel="Estimated variance amplitude", xlim=limits, ylim=limits); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.25, which="both")
    boxplot_values = [ratios[:, :, index].ravel() for index in range(ratios.shape[2])]
    axes[1, 0].boxplot(boxplot_values, labels=component_labels, showfliers=False); axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(title=f"Sampling uncertainty over {ratios.shape[0]} seeds", ylabel=r"$\log_{10}(\widehat{a}/a)$"); axes[1, 0].grid(alpha=0.25, axis="y")
    axes[1, 1].hist(heldout, bins=30, alpha=0.75, color="tab:purple"); axes[1, 1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    return figure


def plot_results(results_path: Path, *, show: bool = False) -> Path:
    """Load a fixed-rate validation NPZ and write its diagnostic figure."""

    with np.load(results_path, allow_pickle=False) as payload:
        histories = [
            row[np.isfinite(row)] for row in payload["deterministic_histories"]
        ]
        figure = plot_fixed_rate_validation(
            histories,
            [str(label) for label in payload["deterministic_labels"]],
            payload["true_amplitudes"],
            payload["fitted_amplitudes"],
            payload["log_parameter_ratio"],
            [str(label) for label in payload["component_labels"]],
            payload["heldout_excess_nll"],
        )
    path = save_figure(
        figure, results_path.parent, "experiment_03_fixed_rate_sage.png", show=show
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
