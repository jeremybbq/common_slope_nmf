
"""Plots for the synthetic convergence experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import ArrayLike

from experiments.synthetic.convergence import METHOD_COLORS, METHOD_LABELS, TRUE_T60_S

def save_figure(
    figure: Figure,
    output_dir: Path,
    filename: str,
    *,
    show: bool = False,
    dpi: int = 180,
) -> Path:
    """Save a figure and return its absolute path.

    Parameters
    ----------
    figure
        Matplotlib figure to save.
    output_dir
        Existing output directory.
    filename
        Output filename including its extension.
    show
        Keep the figure open when true; otherwise close it after saving.
    dpi
        Positive raster resolution.

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """

    path = Path(output_dir) / filename
    figure.savefig(path, dpi=dpi)
    if not show:
        plt.close(figure)
    return path.resolve()


@plt.rc_context({"font.family": "Liberation Serif", "mathtext.fontset": "stix"})
def plot_weight_order_objectives(
    histories_by_case: Sequence[Sequence[ArrayLike]],
    method_labels: Sequence[str],
    case_labels: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    trajectories_by_case: Sequence[Sequence[ArrayLike]] | None = None,
    true_t60_s: ArrayLike | None = None,
) -> Figure:
    """Compare observed IS-loss excess histories on logarithmic axes.

    Parameters
    ----------
    histories_by_case
        Nested sequence indexed as ``[case][method]``. Each entry is a
        one-dimensional observed total IS-divergence history whose index is
        the number of completed component sweeps.
    method_labels
        Labels for the ``M`` optimization methods.
    case_labels
        Labels for the ``C`` decay-pair cases.
    colors
        Optional ``M`` Matplotlib colors, shared between panels.
    trajectories_by_case, true_t60_s
        Optional per-sweep T60 trajectories, indexed as ``[case][method]``,
        and the true two-element T60 pair in seconds. When supplied, dashed
        Euclidean T60-pair errors are overlaid on a linear right-hand axis.

    Returns
    -------
    Figure
        Figure with one loss panel per case and common method styling. Within
        each panel, all methods share
        ``Delta L = L - min_{method,sweep} L``. Exact-zero minima are omitted
        because zero is undefined on a logarithmic axis.
    """

    n_cases = len(histories_by_case)
    n_methods = len(method_labels)
    if n_cases == 0 or n_methods == 0 or len(case_labels) != n_cases:
        raise ValueError("cases and methods must both be non-empty.")
    if any(len(case) != n_methods for case in histories_by_case):
        raise ValueError("each case must contain one history per method.")
    if colors is not None and len(colors) != n_methods:
        raise ValueError("colors must match the number of methods.")
    if (trajectories_by_case is None) != (true_t60_s is None):
        raise ValueError("trajectories_by_case and true_t60_s must be supplied together.")
    if trajectories_by_case is not None:
        if len(trajectories_by_case) != n_cases or any(
            len(case) != n_methods for case in trajectories_by_case
        ):
            raise ValueError("each case must contain one T60 trajectory per method.")
        truth = np.sort(np.asarray(true_t60_s, dtype=np.float64))
        if truth.shape != (2,) or not np.all(np.isfinite(truth)):
            raise ValueError("true_t60_s must be finite with shape (2,).")

    figure, axes = plt.subplots(
        1,
        n_cases,
        figsize=((3.45, 2.1) if n_cases == 1 else (7.10, 3.4)),
        squeeze=False,
        sharey=False,
        layout="constrained",
    )
    for case_index, (case, case_label) in enumerate(
        zip(histories_by_case, case_labels, strict=True)
    ):
        axis = axes[0, case_index]
        case_values: list[np.ndarray] = []
        for raw_history in case:
            history = np.asarray(raw_history, dtype=np.float64)
            if (
                history.ndim != 1
                or history.size == 0
                or not np.all(np.isfinite(history))
                or np.any(history <= 0.0)
            ):
                raise ValueError(
                    "every objective history must be positive, finite, and 1-D."
                )
            case_values.append(history)
        best_loss = min(float(np.min(history)) for history in case_values)
        excess_values = [
            np.maximum(history - best_loss, 0.0) for history in case_values
        ]
        plotted_values = [
            np.where(excess > 0.0, excess, np.nan) for excess in excess_values
        ]
        for method_index, (values, method_label) in enumerate(
            zip(plotted_values, method_labels, strict=True)
        ):
            axis.plot(
                np.arange(values.size),
                values,
                label=method_label,
                color=None if colors is None else colors[method_index],
                linewidth=1.25,
            )
        positive = np.concatenate(
            [excess[excess > 0.0] for excess in excess_values]
        )
        if positive.size == 0:
            raise ValueError(
                "each case needs at least one loss above its shared minimum."
            )
        lower_log = float(np.log10(np.min(positive)))
        upper_log = float(np.log10(np.max(positive)))
        margin = max(0.025 * (upper_log - lower_log), 0.02)
        axis.set(
            xlabel="Iterations",
            ylabel=(
                "Excess loss "
                r"$\Delta\mathcal{L}_{\mathrm{IS}}(\mathbf{Y}\mid\mathbf{V})$"
                if case_index == 0
                else ""
            ),
            yscale="log",
            ylim=(10.0 ** (lower_log - margin), 10.0 ** (upper_log + margin)),
        )
        if case_label:
            axis.set_title(case_label, fontsize=(8 if n_cases == 1 else 9))
        axis.xaxis.label.set_size(8 if n_cases == 1 else 9)
        axis.yaxis.label.set_size(8 if n_cases == 1 else 9)
        axis.tick_params(labelsize=7 if n_cases == 1 else 8)
        axis.grid(alpha=0.22, linewidth=0.5)
        if trajectories_by_case is not None:
            error_axis = axis.twinx()
            for method_index, trajectory in enumerate(trajectories_by_case[case_index]):
                path = np.asarray(trajectory, dtype=np.float64)
                if (
                    path.ndim != 2
                    or path.shape[1] != 2
                    or path.shape[0] == 0
                    or not np.all(np.isfinite(path))
                ):
                    raise ValueError("each T60 trajectory must be finite with shape (S,2).")
                distance_s = np.linalg.norm(np.sort(path, axis=1) - truth, axis=1)
                error_axis.plot(
                    np.arange(distance_s.size),
                    distance_s,
                    color=None if colors is None else colors[method_index],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.9,
                )
            error_axis.set_ylabel("Estimated-to-true RT distance (s)")
            error_axis.yaxis.label.set_size(8 if n_cases == 1 else 9)
            error_axis.tick_params(labelsize=7 if n_cases == 1 else 8)
    axes[0, 0].legend(loc="upper right", fontsize=7 if n_cases == 1 else 8)
    return figure


@plt.rc_context({"font.family": "Liberation Serif", "mathtext.fontset": "stix"})
def plot_profiled_t60_loss_surface(
    short_t60_s: ArrayLike,
    long_t60_s: ArrayLike,
    profiled_loss: ArrayLike,
    true_t60_s: ArrayLike,
    trajectories_t60_s: Sequence[ArrayLike],
    method_labels: Sequence[str],
    *,
    method_colors: Sequence[str] | None = None,
) -> Figure:
    """Plot a profiled two-decay IS-loss surface and fitted trajectories.

    Parameters
    ----------
    short_t60_s, long_t60_s
        Strictly increasing grid coordinates in seconds, shapes ``(X,)`` and
        ``(Y,)``.
    profiled_loss
        Total IS divergence after optimizing amplitudes and the noise floor,
        shape ``(Y,X)``.
    true_t60_s
        True short/long decay pair in seconds, shape ``(2,)``.
    trajectories_t60_s
        Sequence of ``M`` per-sweep decay trajectories in seconds, each shape
        ``(S_m,2)``. Pairs are sorted internally at every sweep.
    method_labels
        Labels for the ``M`` trajectories.
    method_colors
        Optional ``M`` Matplotlib colors.

    Returns
    -------
    Figure
        Interpolated color rendering of profiled excess loss with raw-grid
        contours and optimization paths. Bicubic interpolation affects only
        the displayed raster, not the saved or contoured numerical surface.
    """

    short = np.asarray(short_t60_s, dtype=np.float64)
    long = np.asarray(long_t60_s, dtype=np.float64)
    loss = np.asarray(profiled_loss, dtype=np.float64)
    truth = np.sort(np.asarray(true_t60_s, dtype=np.float64))
    n_methods = len(trajectories_t60_s)
    if (
        short.ndim != 1
        or long.ndim != 1
        or short.size < 2
        or long.size < 2
        or np.any(np.diff(short) <= 0.0)
        or np.any(np.diff(long) <= 0.0)
    ):
        raise ValueError("T60 grids must be strictly increasing one-dimensional arrays.")
    if loss.shape != (long.size, short.size) or not np.all(np.isfinite(loss)):
        raise ValueError("profiled_loss must be finite with shape (Y,X).")
    if np.any(loss < 0.0):
        raise ValueError("profiled_loss must be non-negative.")
    if truth.shape != (2,) or not np.all(np.isfinite(truth)):
        raise ValueError("true_t60_s must be finite with shape (2,).")
    if n_methods == 0 or len(method_labels) != n_methods:
        raise ValueError("trajectories and labels must have equal non-zero length.")
    if method_colors is not None and len(method_colors) != n_methods:
        raise ValueError("method_colors must match the number of trajectories.")

    trajectories: list[np.ndarray] = []
    for raw in trajectories_t60_s:
        trajectory = np.asarray(raw, dtype=np.float64)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 2
            or trajectory.shape[0] == 0
            or not np.all(np.isfinite(trajectory))
        ):
            raise ValueError("each trajectory must be finite with shape (S,2).")
        trajectories.append(np.sort(trajectory, axis=1))

    excess = np.maximum(loss - np.min(loss), 0.0)
    positive = excess[excess > 0.0]
    if positive.size == 0:
        raise ValueError("profiled_loss must contain at least two distinct values.")
    display_floor = max(float(np.min(positive)) * 0.5, np.finfo(float).tiny)
    display = np.maximum(excess, display_floor)
    norm = LogNorm(vmin=display_floor, vmax=float(np.max(display)))

    figure, axis = plt.subplots(figsize=(3.45, 3.15))
    # Reserve the right side explicitly: axes appended by ``axes_grid1`` keep
    # the colorbar exactly as tall as the square data axes, but are invisible
    # to Matplotlib's constrained-layout calculation.
    figure.subplots_adjust(left=0.10, right=0.96, bottom=0.30, top=0.976)
    image = axis.imshow(
        display,
        origin="lower",
        extent=(short[0], short[-1], long[0], long[-1]),
        aspect="auto",
        interpolation="bicubic",
        cmap="viridis",
        norm=norm,
        rasterized=True,
    )
    contour_levels = np.geomspace(display_floor, float(np.max(display)), 9)
    axis.contour(
        short,
        long,
        display,
        levels=np.unique(contour_levels),
        colors="white",
        linewidths=0.45,
        alpha=0.42,
    )
    invalid_start = max(float(short[0]), float(long[0]))
    invalid_stop = float(short[-1])
    if invalid_stop > invalid_start:
        invalid_x = np.unique(
            [
                invalid_start,
                min(max(invalid_start, float(long[-1])), invalid_stop),
                invalid_stop,
            ]
        )
        axis.fill_between(
            invalid_x,
            long[0],
            np.minimum(invalid_x, long[-1]),
            color="#d9d9d9",
            edgecolor="none",
            linewidth=0.0,
            alpha=1.0,
            zorder=3,
        )
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for method_index, (trajectory, label) in enumerate(
        zip(trajectories, method_labels, strict=True)
    ):
        color = (
            default_colors[method_index % len(default_colors)]
            if method_colors is None
            else method_colors[method_index]
        )
        line = axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linewidth=1.25,
            label=label,
            zorder=4 + n_methods - method_index,
        )[0]
        line.get_path().should_simplify = False
        final = trajectory[-1]
        axis.scatter(final[0], final[1], marker="x", s=24, color=color, zorder=9)

    starts = np.stack([trajectory[0] for trajectory in trajectories])
    if np.allclose(starts, starts[0], rtol=0.0, atol=1e-12):
        axis.scatter(
            starts[0, 0],
            starts[0, 1],
            marker="s",
            s=26,
            facecolors="white",
            edgecolors="#111827",
            linewidths=0.9,
            label="Linear fit",
            zorder=10,
        )
    else:
        axis.scatter(
            starts[:, 0],
            starts[:, 1],
            marker="s",
            s=24,
            facecolors="white",
            edgecolors="#111827",
            linewidths=0.9,
            label="Linear fits",
            zorder=10,
        )

    minimum_index = np.unravel_index(np.argmin(loss), loss.shape)
    axis.scatter(
        short[minimum_index[1]],
        long[minimum_index[0]],
        marker="D",
        s=28,
        facecolors="white",
        edgecolors="#111827",
        linewidths=0.9,
        label="Profiled grid minimum",
        zorder=10,
    )
    axis.scatter(
        truth[0],
        truth[1],
        marker="*",
        s=52,
        color="#ef4444",
        edgecolors="none",
        linewidths=0.0,
        label="True RTs",
        zorder=11,
    )
    axis.set(
        xlabel="Fast decay RT (s)",
        ylabel="Slow decay RT (s)",
        xlim=(short[0], short[-1]),
        ylim=(long[0], long[-1]),
    )
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_zorder(20)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    axis.tick_params(labelsize=7)
    handles, labels = axis.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles, strict=True))
    legend_order = [
        *method_labels,
        "True RTs",
        "Linear fit" if "Linear fit" in label_to_handle else "Linear fits",
        "Profiled grid minimum",
    ]
    figure.legend(
        [label_to_handle[label] for label in legend_order],
        legend_order,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncols=2,
        frameon=False,
    )
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="4.5%", pad=0.10)
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.set_label("Profiled excess loss", fontsize=8, labelpad=1)
    colorbar.ax.tick_params(labelsize=7)
    return figure


def save_loss_figure(
    output_dir: Path,
    histories: list[list[np.ndarray]],
    *,
    show: bool,
    trajectories: list[np.ndarray] | None = None,
) -> tuple[Path, Path]:
    """Save the single-column log-scale excess-loss figure."""

    figure = plot_weight_order_objectives(
        histories,
        METHOD_LABELS,
        ("",),
        colors=METHOD_COLORS,
        trajectories_by_case=None if trajectories is None else [trajectories],
        true_t60_s=None if trajectories is None else TRUE_T60_S,
    )
    pdf_path = output_dir / "is_loss_convergence.pdf"
    figure.savefig(pdf_path)
    png_path = save_figure(
        figure,
        output_dir,
        "is_loss_convergence.png",
        show=show,
        dpi=300,
    )
    return pdf_path.resolve(), png_path


def save_surface_figure(
    output_dir: Path,
    short_t60_s: np.ndarray,
    long_t60_s: np.ndarray,
    loss: np.ndarray,
    trajectories_t60_s: list[np.ndarray],
    *,
    show: bool,
) -> tuple[Path, Path]:
    """Save single-column PDF/PNG surface figures and return their paths."""

    figure = plot_profiled_t60_loss_surface(
        short_t60_s,
        long_t60_s,
        loss,
        TRUE_T60_S,
        trajectories_t60_s,
        METHOD_LABELS,
        method_colors=METHOD_COLORS,
    )
    pdf_path = output_dir / "profiled_t60_loss_surface.pdf"
    figure.savefig(pdf_path)
    png_path = save_figure(
        figure,
        output_dir,
        "profiled_t60_loss_surface.png",
        show=show,
        dpi=300,
    )
    return pdf_path.resolve(), png_path


def _plot_saved_results(
    results_path: Path, output_dir: Path, *, show: bool
) -> list[Path]:
    """Regenerate available figures from a current or legacy archive."""

    saved: list[Path] = []
    with np.load(results_path, allow_pickle=False) as archive:
        history_key = (
            "objective_history"
            if "objective_history" in archive
            else "loss_objective_history"
        )
        padded = np.asarray(archive[history_key])
        histories = [[row[np.isfinite(row)] for row in case] for case in padded]
        if len(histories) > 1:
            histories = [histories[0]]
        trajectories = None
        if "t60_trajectory_s" in archive:
            padded_paths = np.asarray(archive["t60_trajectory_s"])
            trajectories = [
                path[np.all(np.isfinite(path), axis=1)] for path in padded_paths
            ]
        saved.extend(
            save_loss_figure(output_dir, histories, show=show, trajectories=trajectories)
        )
        surface_keys = {
            "surface_short_t60_s",
            "surface_long_t60_s",
            "surface_loss",
            "t60_trajectory_s",
        }
        if surface_keys.issubset(archive.files):
            if trajectories is None:
                padded_paths = np.asarray(archive["t60_trajectory_s"])
                trajectories = [
                    path[np.all(np.isfinite(path), axis=1)] for path in padded_paths
                ]
            saved.extend(
                save_surface_figure(
                    output_dir,
                    np.asarray(archive["surface_short_t60_s"]),
                    np.asarray(archive["surface_long_t60_s"]),
                    np.asarray(archive["surface_loss"]),
                    trajectories,
                    show=show,
                )
            )
    return saved


def main() -> None:
    """Load a saved NPZ and write figures beside it."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.results.is_file():
        raise FileNotFoundError(args.results)
    output_dir = args.results.parent
    paths = _plot_saved_results(args.results, output_dir, show=args.show)
    print(f"source={args.results.resolve()}")
    if isinstance(paths, (list, tuple)):
        for path in paths:
            print(f"saved={path}")
    else:
        print(f"saved={paths}")


if __name__ == "__main__":
    main()
