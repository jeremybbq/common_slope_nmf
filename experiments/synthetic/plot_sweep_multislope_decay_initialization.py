"""Experiment plots loaded from saved NPZ results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from experiments._run_output import save_figure

def plot_initialization_heatmap(
    values: ArrayLike,
    t60_grid_s: ArrayLike,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    value_format: str = ".3f",
) -> Figure:
    """Plot a square metric grid over ordered two-component initial ``T60`` values."""

    grid = np.asarray(t60_grid_s, dtype=np.float64)
    matrix = np.asarray(values, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or matrix.shape != (grid.size, grid.size):
        raise ValueError("values must have shape (G,G) for a grid of length G>=2.")
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    step = grid[1] - grid[0]
    image = axis.imshow(matrix, origin="lower", extent=(grid[0]-step/2, grid[-1]+step/2, grid[0]-step/2, grid[-1]+step/2), aspect="equal", cmap=cmap)
    if grid.size <= 9:
        for row, y in enumerate(grid):
            for column, x in enumerate(grid):
                axis.text(x, y, format(matrix[row, column], value_format), ha="center", va="center", fontsize=7, color="white")
    axis.set(title=title, xlabel=r"Initial component 1 $T_{60}$ (s)", ylabel=r"Initial component 2 $T_{60}$ (s)", xticks=grid, yticks=grid)
    figure.colorbar(image, ax=axis, label=colorbar_label); figure.tight_layout()
    return figure

def plot_initialization_basin(
    initial_t60_s: ArrayLike,
    final_labeled_t60_s: ArrayLike,
    maximum_error_s: ArrayLike,
    true_t60_s: ArrayLike,
    *,
    t60_range_s: tuple[float, float],
) -> Figure:
    """Plot arrows from ordered two-component starts to fitted endpoints.

    Parameters are decay times in seconds. Initial and final arrays have shape
    ``(P,2)``, maximum errors have shape ``(P,)``, and truth has shape ``(2,)``.
    Both truth permutations are marked because component labels are ordered
    during the sequential update but the physical model is permutation-free.
    """

    initial = np.asarray(initial_t60_s, dtype=np.float64)
    final = np.asarray(final_labeled_t60_s, dtype=np.float64)
    errors = np.asarray(maximum_error_s, dtype=np.float64)
    truth = np.asarray(true_t60_s, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 2 or final.shape != initial.shape:
        raise ValueError("initial and final decay times must have shape (P,2).")
    if errors.shape != (initial.shape[0],) or truth.shape != (2,):
        raise ValueError("errors must have shape (P,) and truth shape (2,).")
    figure, axis = plt.subplots(figsize=(7.2, 6.3))
    arrows = axis.quiver(
        initial[:, 0], initial[:, 1],
        final[:, 0] - initial[:, 0], final[:, 1] - initial[:, 1],
        errors, angles="xy", scale_units="xy", scale=1.0,
        cmap="viridis_r", width=0.006,
    )
    axis.scatter(final[:, 0], final[:, 1], c=errors, cmap="viridis_r", s=20, edgecolors="black", linewidths=0.3, zorder=3)
    permutations = np.stack((truth, truth[::-1]))
    axis.scatter(permutations[:, 0], permutations[:, 1], marker="*", s=220, color="red", edgecolors="black", label="Generating decay pair", zorder=4)
    lower, upper = t60_range_s
    margin = 0.08 * (upper - lower)
    axis.set(title="Initial-to-final decay-time basin map", xlabel=r"Labeled component 1 $T_{60}$ (s)", ylabel=r"Labeled component 2 $T_{60}$ (s)", xlim=(lower-margin, upper+margin), ylim=(lower-margin, upper+margin), aspect="equal")
    axis.grid(alpha=0.2); axis.legend(loc="upper left")
    figure.colorbar(arrows, ax=axis, label=r"Maximum absolute $T_{60}$ error (s)")
    figure.tight_layout()
    return figure


def plot_results(results_path: Path, *, show: bool = False) -> list[Path]:
    """Load an initialization-sweep NPZ and write its diagnostic figures."""

    from experiments.synthetic.sweep_multislope_decay_initialization import T60_RANGE_S

    output_dir = results_path.parent
    slug = results_path.name.removesuffix("_results.npz")
    with np.load(results_path, allow_pickle=False) as payload:
        t60_grid_s = payload["t60_grid_s"]
        grid_shape = (t60_grid_s.size, t60_grid_s.size)
        maximum_error_grid_s = np.max(payload["absolute_error_s"], axis=1).reshape(grid_shape)
        n_iter_grid = payload["n_iter"].reshape(grid_shape)
        objective_excess_grid = (
            payload["final_objective"] - np.min(payload["final_objective"])
        ).reshape(grid_shape)
        converged_grid = payload["converged"].reshape(grid_shape)
        figures = [
            (
                plot_initialization_basin(
                    payload["initial_t60_s"],
                    payload["final_labeled_t60_s"],
                    np.max(payload["absolute_error_s"], axis=1),
                    payload["true_t60_s"],
                    t60_range_s=T60_RANGE_S,
                ),
                f"{slug}_basin_arrows.png",
            ),
            (
                plot_initialization_heatmap(
                    maximum_error_grid_s, t60_grid_s,
                    title=r"Final maximum absolute $T_{60}$ error",
                    colorbar_label="Maximum absolute error (s)",
                    cmap="magma",
                ),
                f"{slug}_maximum_t60_error.png",
            ),
            (
                plot_initialization_heatmap(
                    n_iter_grid, t60_grid_s,
                    title="Sweeps used from each decay initialization",
                    colorbar_label="Complete component sweeps",
                    value_format=".0f",
                ),
                f"{slug}_convergence_sweeps.png",
            ),
            (
                plot_initialization_heatmap(
                    objective_excess_grid, t60_grid_s,
                    title="Final observed objective above the best initialization",
                    colorbar_label="Excess summed IS divergence",
                    cmap="magma",
                    value_format=".1f",
                ),
                f"{slug}_objective_excess.png",
            ),
            (
                plot_initialization_heatmap(
                    converged_grid.astype(np.float64), t60_grid_s,
                    title="Convergence within the sweep budget",
                    colorbar_label="Converged (1=yes, 0=no)",
                    cmap="cividis",
                    value_format=".0f",
                ),
                f"{slug}_converged.png",
            ),
        ]
    saved = [save_figure(figure, output_dir, filename, show=show) for figure, filename in figures]
    for path in saved:
        print(f"saved={path}")
    if show:
        plt.show()
    return saved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_results(args.results, show=args.show)


if __name__ == "__main__":
    main()
