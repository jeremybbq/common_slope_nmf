"""Plots for unweighted coupled-room SAGE refinements."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments._run_output import save_figure


def plot_results(results_path: Path, *, show: bool = False) -> Path:
    """Load a refinement NPZ and write the comparison figure."""

    with np.load(results_path, allow_pickle=False) as payload:
        labels = [str(label) for label in payload["labels"]]
        frequencies = payload["frequencies_hz"]
        final_t60 = payload["estimated_t60_s"]
        final_objective = payload["final_objective"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    line_styles = ("-", "--", ":", "-.")
    for branch_index, label in enumerate(labels):
        for component_index in range(final_t60.shape[2]):
            axes[0].plot(
                frequencies,
                final_t60[branch_index, :, component_index],
                marker="o",
                linestyle=line_styles[branch_index % len(line_styles)],
                color=f"C{component_index}",
                label=(
                    f"{label}, component {component_index + 1}"
                    if component_index == 0
                    else None
                ),
            )
    axes[0].set_ylabel("Final energy T60 (s)")
    axes[0].set_title("Ordinary SAGE from weighted-fit warm starts")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for branch_index, label in enumerate(labels):
        axes[1].plot(
            frequencies,
            final_objective[branch_index],
            marker="o",
            linestyle=line_styles[branch_index % len(line_styles)],
            label=label,
        )
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Final observed IS objective")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = save_figure(
        fig, results_path.parent, "unweighted_warm_start_comparison.png", show=show
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
