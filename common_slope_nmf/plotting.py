"""Reusable plots for common-slope decay-estimation results.

This module intentionally remains outside :mod:`common_slope_nmf.__init__` so
the numerical package can be imported without the optional Matplotlib extra.
All amplitude dB values use the power convention ``10 * log10(value)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


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


def plot_variance_maps(
    times_s: ArrayLike,
    maps: Sequence[ArrayLike],
    titles: Sequence[str],
    *,
    vmin_db: float = -65.0,
    vmax_db: float = 5.0,
    rir_label: str = "RIR index",
) -> Figure:
    """Plot side-by-side power/variance maps in dB.

    ``maps`` must contain positive arrays of common shape ``(R,N)`` and
    ``times_s`` has shape ``(N,)`` in seconds. Linear power inputs are
    converted with ``10 log10``. The returned figure is not saved.
    """

    times = np.asarray(times_s, dtype=np.float64)
    values = [np.asarray(item, dtype=np.float64) for item in maps]
    if len(values) == 0 or len(values) != len(titles):
        raise ValueError("maps and titles must have the same non-zero length.")
    shape = values[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in values):
        raise ValueError("every map must have the same shape (R,N).")
    if times.shape != (shape[1],) or any(np.any(item <= 0.0) for item in values):
        raise ValueError("times_s must match N and every map must be positive.")

    figure, axes = plt.subplots(
        1, len(values), figsize=(5.2 * len(values), 5.0), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, item, title in zip(axes, values, titles, strict=True):
        image = axis.imshow(
            10.0 * np.log10(item),
            origin="lower",
            aspect="auto",
            extent=(times[0], times[-1], 0, shape[0] - 1),
            vmin=vmin_db,
            vmax=vmax_db,
            cmap="magma",
        )
        axis.set(title=title, xlabel="Elapsed time (s)")
    axes[0].set_ylabel(rir_label)
    figure.colorbar(image, ax=axes, label="Power/variance (dB re 1)")
    figure.subplots_adjust(left=0.06, right=0.92, bottom=0.12, wspace=0.1)
    return figure


def plot_objective_history(
    histories: ArrayLike | Sequence[ArrayLike],
    *,
    labels: Sequence[str] | None = None,
    normalize: bool = False,
    log_y: bool = False,
    title: str = "Observed objective convergence",
) -> Figure:
    """Plot one or more complete-sweep objective histories.

    A two-dimensional array is interpreted row-wise; NaN padding is ignored.
    With ``normalize=True``, each curve is divided by its initial value.
    """

    raw = np.asarray(histories, dtype=np.float64)
    rows = raw[np.newaxis, :] if raw.ndim == 1 else raw
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError("histories must have shape (S,) or (B,S).")
    if labels is not None and len(labels) != rows.shape[0]:
        raise ValueError("labels must match the number of histories.")
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for index, row in enumerate(rows):
        finite = np.isfinite(row)
        values = row[finite]
        if values.size == 0:
            continue
        if normalize:
            values = values / values[0]
        axis.plot(
            np.flatnonzero(finite),
            values,
            alpha=0.7,
            label=None if labels is None else labels[index],
        )
    axis.set(
        title=title,
        xlabel="Complete component sweeps",
        ylabel="Objective / initial objective" if normalize else "Summed IS divergence",
        yscale="log" if log_y else "linear",
    )
    if labels is not None:
        axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_t60_trajectories(
    t60_history_s: ArrayLike,
    *,
    true_t60_s: ArrayLike | None = None,
    title: str = "Decay-time trajectories",
) -> Figure:
    """Plot sorted energy-``T60`` histories, shape ``(S,K)`` in seconds."""

    history = np.asarray(t60_history_s, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] == 0:
        raise ValueError("t60_history_s must have shape (S,K).")
    truth = None if true_t60_s is None else np.asarray(true_t60_s, dtype=np.float64)
    if truth is not None and truth.shape != (history.shape[1],):
        raise ValueError("true_t60_s must have shape (K,).")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for component_index in range(history.shape[1]):
        line = axis.plot(
            history[:, component_index],
            label=f"Estimated component {component_index + 1}",
        )[0]
        if truth is not None:
            axis.axhline(
                truth[component_index],
                color=line.get_color(),
                linestyle="--",
                label=f"True component {component_index + 1}",
            )
    axis.set(
        title=title,
        xlabel="Complete component sweeps",
        ylabel=r"Energy $T_{60}$ (s)",
    )
    axis.legend(ncols=2, fontsize="small")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_simplex_amplitude_shares(amplitudes: ArrayLike) -> Figure:
    """Plot a histogram of component-1 simplex shares from ``(R,F,K)`` data."""

    values = np.asarray(amplitudes, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 2 or np.any(values < 0.0):
        raise ValueError("amplitudes must be non-negative with shape (R,F,K>=2).")
    shares = values[..., 0] / np.sum(values, axis=2)
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.hist(shares.ravel(), bins=50, histtype="step", linewidth=2)
    axis.set(
        title="Simplex share for component 1",
        xlabel=r"$a_1 / \sum_k a_k$",
        ylabel="Count",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _two_component_arrays(
    true_t60_s: ArrayLike, values: ArrayLike, name: str
) -> tuple[np.ndarray, np.ndarray]:
    t60_s = np.asarray(true_t60_s, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if t60_s.ndim != 2 or t60_s.shape[1] != 2:
        raise ValueError("true_t60_s must have shape (F,2).")
    if array.shape != t60_s.shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape (F,2).")
    return t60_s, array


def plot_true_vs_estimated_t60(
    true_t60_s: ArrayLike, estimated_t60_s: ArrayLike
) -> Figure:
    """Scatter sorted true versus fitted two-component ``T60`` values."""

    truth, estimate = _two_component_arrays(
        true_t60_s, estimated_t60_s, "estimated_t60_s"
    )
    bounds = (min(np.min(truth), np.min(estimate)), max(np.max(truth), np.max(estimate)))
    figure, axis = plt.subplots(figsize=(6.3, 5.7))
    for index, label in enumerate(("Short component", "Long component")):
        axis.scatter(truth[:, index], estimate[:, index], s=24, alpha=0.72, label=label)
    axis.plot(bounds, bounds, "k--", linewidth=1.2, label="Identity")
    axis.set(
        title="Decay-time recovery over independent frequency bins",
        xlabel=r"True energy $T_{60}$ (s)",
        ylabel=r"Estimated energy $T_{60}$ (s)",
        xlim=bounds,
        ylim=bounds,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(); axis.grid(alpha=0.2); figure.tight_layout()
    return figure


def plot_t60_pair_plane(
    true_t60_s: ArrayLike, estimated_t60_s: ArrayLike
) -> Figure:
    """Plot true and fitted short/long ``T60`` pairs with connecting lines."""

    truth, estimate = _two_component_arrays(
        true_t60_s, estimated_t60_s, "estimated_t60_s"
    )
    bounds = (min(np.min(truth), np.min(estimate)), max(np.max(truth), np.max(estimate)))
    figure, axis = plt.subplots(figsize=(6.5, 5.8))
    axis.scatter(truth[:, 0], truth[:, 1], s=25, label="True pairs", alpha=0.8)
    axis.scatter(estimate[:, 0], estimate[:, 1], s=25, label="Estimated pairs", alpha=0.8)
    for true_pair, fitted_pair in zip(truth, estimate, strict=True):
        axis.plot([true_pair[0], fitted_pair[0]], [true_pair[1], fitted_pair[1]], color="0.65", linewidth=0.45, alpha=0.5)
    axis.plot(bounds, bounds, "k--", linewidth=1.0)
    axis.set(title="True and fitted two-slope pairs", xlabel=r"Short $T_{60}$ (s)", ylabel=r"Long $T_{60}$ (s)", xlim=bounds, ylim=bounds)
    axis.legend(); axis.grid(alpha=0.2); figure.tight_layout()
    return figure


def plot_t60_error_vs_separation(
    true_t60_s: ArrayLike, absolute_error_s: ArrayLike
) -> Figure:
    """Plot maximum absolute ``T60`` error against true pair separation."""

    truth, errors = _two_component_arrays(
        true_t60_s, absolute_error_s, "absolute_error_s"
    )
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    scatter = axis.scatter(np.diff(truth, axis=1)[:, 0], np.max(errors, axis=1), c=np.mean(truth, axis=1), cmap="viridis", s=34, alpha=0.82)
    figure.colorbar(scatter, ax=axis, label=r"Pair mean $T_{60}$ (s)")
    axis.set(title="Detection error versus true slope separation", xlabel=r"True pair separation $\Delta T_{60}$ (s)", ylabel=r"Maximum absolute $T_{60}$ error (s)")
    axis.grid(alpha=0.2); figure.tight_layout()
    return figure


def _plot_component_error_vs_separation(
    true_t60_s: np.ndarray,
    errors_db: np.ndarray,
    *,
    title: str,
    ylabel: str,
    zero_line: bool,
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharex=True, sharey=True, layout="constrained")
    separation_s = np.diff(true_t60_s, axis=1)[:, 0]
    colors = np.mean(true_t60_s, axis=1)
    for index, (axis, label) in enumerate(zip(axes, ("Short component", "Long component"), strict=True)):
        scatter = axis.scatter(separation_s, errors_db[:, index], c=colors, cmap="viridis", s=34, alpha=0.82)
        if zero_line:
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set(title=label, xlabel=r"True pair separation $\Delta T_{60}$ (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(ylabel)
    figure.colorbar(scatter, ax=axes, label=r"Pair mean $T_{60}$ (s)")
    figure.suptitle(title)
    return figure


def plot_amplitude_rmse_vs_separation(
    true_t60_s: ArrayLike, amplitude_rmse_db: ArrayLike
) -> Figure:
    """Plot per-component power-dB amplitude RMSE versus ``T60`` separation."""

    truth, errors = _two_component_arrays(
        true_t60_s, amplitude_rmse_db, "amplitude_rmse_db"
    )
    return _plot_component_error_vs_separation(
        truth, errors,
        title="Amplitude recovery error versus true slope separation",
        ylabel="Amplitude RMSE (power dB)", zero_line=False,
    )


def plot_signed_amplitude_bias_vs_separation(
    true_t60_s: ArrayLike,
    true_amplitudes: ArrayLike,
    estimated_amplitudes: ArrayLike,
) -> Figure:
    """Plot RIR-mean signed power-dB amplitude bias versus separation.

    Amplitudes have shape ``(R,F,2)``. Positive bias means overestimation.
    """

    truth = np.asarray(true_t60_s, dtype=np.float64)
    true_values = np.asarray(true_amplitudes, dtype=np.float64)
    estimates = np.asarray(estimated_amplitudes, dtype=np.float64)
    expected = (true_values.shape[0],) + truth.shape
    if truth.ndim != 2 or truth.shape[1] != 2 or true_values.shape != expected or estimates.shape != expected or np.any(true_values <= 0.0) or np.any(estimates <= 0.0):
        raise ValueError("inputs must have shapes (F,2) and matching positive (R,F,2).")
    bias_db = np.mean(10.0 * np.log10(estimates) - 10.0 * np.log10(true_values), axis=0)
    return _plot_component_error_vs_separation(
        truth, bias_db,
        title="Signed amplitude error versus true slope separation",
        ylabel="Mean signed amplitude error (power dB)", zero_line=True,
    )


def plot_t60_error_by_frequency(absolute_error_s: ArrayLike) -> Figure:
    """Plot short/long absolute ``T60`` errors over arbitrary frequency labels."""

    errors = np.asarray(absolute_error_s, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 2:
        raise ValueError("absolute_error_s must have shape (F,2).")
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.plot(errors[:, 0], ".-", linewidth=0.7, label="Short")
    axis.plot(errors[:, 1], ".-", linewidth=0.7, label="Long")
    axis.set(title="Final decay-time error by arbitrary frequency index", xlabel="Frequency-bin index (no physical ordering)", ylabel=r"Absolute $T_{60}$ error (s)")
    axis.legend(); axis.grid(alpha=0.2); figure.tight_layout()
    return figure


def plot_maximum_t60_error_cdf(absolute_error_s: ArrayLike) -> Figure:
    """Plot the empirical CDF of maximum pairwise absolute ``T60`` error."""

    errors = np.asarray(absolute_error_s, dtype=np.float64)
    maximum = np.sort(np.max(errors, axis=1))
    cumulative = np.arange(1, maximum.size + 1) / maximum.size
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.step(maximum, cumulative, where="post")
    axis.set(title="Empirical distribution of pairwise detection error", xlabel=r"Maximum absolute $T_{60}$ error (s)", ylabel="Fraction of frequency bins", ylim=(0.0, 1.02))
    axis.grid(alpha=0.2); figure.tight_layout()
    return figure


def plot_t60_error_evolution(
    t60_history_s: ArrayLike, true_t60_s: ArrayLike, component_index: int
) -> Figure:
    """Plot a frequency-by-sweep absolute ``T60`` error heatmap."""

    history = np.asarray(t60_history_s, dtype=np.float64)
    truth = np.asarray(true_t60_s, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != truth.shape:
        raise ValueError("history must have shape (S,F,K) matching truth (F,K).")
    if not 0 <= component_index < truth.shape[1]:
        raise ValueError("component_index is out of range.")
    error_s = np.abs(history[:, :, component_index] - truth[np.newaxis, :, component_index]).T
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    image = axis.imshow(error_s, origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
    figure.colorbar(image, ax=axis, label=r"Absolute $T_{60}$ error (s)")
    label = "Short" if component_index == 0 else "Long"
    axis.set(title=f"{label}-component error evolution", xlabel="Complete component sweeps", ylabel="Frequency-bin index")
    figure.tight_layout()
    return figure


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


def plot_spatial_map(
    values: ArrayLike,
    times_s: ArrayLike,
    *,
    title: str,
    values_are_db: bool = False,
    vmin_db: float = -65.0,
    vmax_db: float = 0.0,
) -> Figure:
    """Plot one ``(R,N)`` spatial power or variance map over elapsed time."""

    data = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if data.ndim != 2 or times.shape != (data.shape[1],):
        raise ValueError("values and times_s must have shapes (R,N) and (N,).")
    if not values_are_db:
        if np.any(data <= 0.0):
            raise ValueError("linear power/variance values must be positive.")
        data = 10.0 * np.log10(data)
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    image = axis.imshow(data, origin="lower", aspect="auto", extent=(times[0], times[-1], 0, data.shape[0]-1), vmin=vmin_db, vmax=vmax_db, cmap="magma")
    axis.set(title=title, xlabel="Elapsed time (s)", ylabel=r"RIR index $r$")
    figure.colorbar(image, ax=axis, label="Power/variance (dB re 1)")
    figure.tight_layout()
    return figure


def plot_amplitude_distributions(
    true_amplitudes_db: ArrayLike,
    estimated_amplitudes: ArrayLike,
    true_t60_s: ArrayLike,
    estimated_t60_s: ArrayLike,
    *,
    estimator_label: str,
    generating_mean_db: float | None = None,
    generating_std_db: float | None = None,
) -> Figure:
    """Compare per-component true and fitted amplitude distributions."""

    truth_db = np.asarray(true_amplitudes_db, dtype=np.float64)
    estimates = np.asarray(estimated_amplitudes, dtype=np.float64)
    true_t60 = np.asarray(true_t60_s, dtype=np.float64)
    fitted_t60 = np.asarray(estimated_t60_s, dtype=np.float64)
    if truth_db.ndim != 2 or estimates.shape != truth_db.shape or np.any(estimates <= 0.0):
        raise ValueError("amplitudes must have matching shapes (R,K) and estimates be positive.")
    if true_t60.shape != (truth_db.shape[1],) or fitted_t60.shape != true_t60.shape:
        raise ValueError("decay-time arrays must have shape (K,).")
    estimated_db = 10.0 * np.log10(estimates)
    lower = np.floor(min(np.min(truth_db), np.min(estimated_db))) - 1.0
    upper = np.ceil(max(np.max(truth_db), np.max(estimated_db))) + 1.0
    bins = np.linspace(lower, upper, 41)
    figure, axes = plt.subplots(1, truth_db.shape[1], figsize=(5.5*truth_db.shape[1], 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.hist(truth_db[:, index], bins=bins, density=True, histtype="step", linewidth=2.0, label="Generating samples")
        axis.hist(estimated_db[:, index], bins=bins, density=True, histtype="step", linewidth=2.0, label=f"{estimator_label} estimates")
        if generating_mean_db is not None and generating_std_db is not None:
            grid = np.linspace(lower, upper, 500)
            density = np.exp(-0.5*((grid-generating_mean_db)/generating_std_db)**2)/(generating_std_db*np.sqrt(2*np.pi))
            axis.plot(grid, density, "k--", linewidth=1.6, label="Generating Gaussian law")
        axis.set(title=f"Component {index+1}: true/fit $T_{{60}}$={true_t60[index]:.3f}/{fitted_t60[index]:.3f} s", xlabel="Unit-origin variance amplitude (dB re 1)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Density across RIRs"); axes[0].legend(fontsize="small")
    figure.suptitle("Generating and estimated amplitude distributions"); figure.tight_layout()
    return figure


def plot_joint_amplitudes(
    true_amplitudes_db: ArrayLike,
    estimated_amplitudes: ArrayLike,
    *,
    estimator_label: str,
) -> Figure:
    """Compare true and fitted two-component amplitude scatter plots."""

    truth = np.asarray(true_amplitudes_db, dtype=np.float64)
    estimates = np.asarray(estimated_amplitudes, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != 2 or estimates.shape != truth.shape or np.any(estimates <= 0.0):
        raise ValueError("amplitudes must have matching shapes (R,2).")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    for axis, values, title in zip(axes, (truth, 10*np.log10(estimates)), ("Generating amplitudes", f"{estimator_label} amplitude estimates"), strict=True):
        axis.scatter(values[:, 0], values[:, 1], s=12, alpha=0.55)
        axis.set(title=title, xlabel="Short-slope amplitude (dB re 1)", ylabel="Long-slope amplitude (dB re 1)")
        axis.grid(alpha=0.25)
    figure.tight_layout(); return figure


def plot_parameter_estimates(
    true_amplitudes_db: ArrayLike,
    estimated_amplitudes: ArrayLike,
    true_floor_db: ArrayLike,
    estimated_floor: ArrayLike,
) -> Figure:
    """Plot true and fitted amplitudes and floors over RIR index."""

    truth = np.asarray(true_amplitudes_db, dtype=np.float64)
    estimates = np.asarray(estimated_amplitudes, dtype=np.float64)
    floor_truth = np.asarray(true_floor_db, dtype=np.float64)
    floor_estimate = np.asarray(estimated_floor, dtype=np.float64)
    if truth.ndim != 2 or estimates.shape != truth.shape or floor_truth.shape != (truth.shape[0],) or floor_estimate.shape != floor_truth.shape or np.any(estimates <= 0.0) or np.any(floor_estimate <= 0.0):
        raise ValueError("parameter arrays have inconsistent shapes or non-positive estimates.")
    figure, axes = plt.subplots(truth.shape[1]+1, 1, figsize=(11.0, 2.7*(truth.shape[1]+1)), sharex=True)
    index = np.arange(truth.shape[0])
    estimated_db = 10*np.log10(estimates)
    for component in range(truth.shape[1]):
        axes[component].plot(index, truth[:, component], color="black", linewidth=1.0, label="Truth")
        axes[component].plot(index, estimated_db[:, component], linewidth=0.8, alpha=0.85, label="Estimate")
        axes[component].set_ylabel(f"Component {component+1}\n(dB re 1)"); axes[component].grid(alpha=0.2)
    axes[-1].plot(index, floor_truth, color="black", linewidth=1.0, label="Truth")
    axes[-1].plot(index, 10*np.log10(floor_estimate), linewidth=0.8, alpha=0.85, label="Estimate")
    axes[-1].set(xlabel=r"RIR index $r$", ylabel="Noise floor\n(dB re 1)"); axes[-1].grid(alpha=0.2)
    axes[0].legend(ncols=2); figure.suptitle("Per-RIR generating and estimated variance amplitudes"); figure.tight_layout()
    return figure


def make_scaled_error_animation(
    scaled_total_error: ArrayLike,
    sweep_indices: ArrayLike,
    times_s: ArrayLike,
) -> tuple[Figure, FuncAnimation]:
    """Create an animation of ``log(epsilon + 1) = log(Y/V)`` over ``(R,N)``."""

    error = np.asarray(scaled_total_error, dtype=np.float64)
    sweeps = np.asarray(sweep_indices)
    times = np.asarray(times_s, dtype=np.float64)
    if error.ndim != 3 or sweeps.shape != (error.shape[0],) or times.shape != (error.shape[2],):
        raise ValueError("errors must have shape (D,R,N), sweeps (D,), times (N,).")
    log_ratio = np.log1p(error)
    lower = min(float(np.min(log_ratio)), -np.finfo(float).eps); upper = max(float(np.max(log_ratio)), np.finfo(float).eps)
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    image = axis.imshow(log_ratio[0], origin="lower", aspect="auto", extent=(times[0], times[-1], 0, error.shape[1]-1), cmap="coolwarm", norm=TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper))
    title = axis.set_title(""); axis.set(xlabel="Elapsed time (s)", ylabel=r"RIR index $r$")
    figure.colorbar(image, ax=axis, label=r"$\log(\epsilon+1)=\log(Y/V)$"); figure.tight_layout()
    def update(frame: int):
        image.set_data(log_ratio[frame]); title.set_text(f"Scaled fitting-error evolution: sweep {sweeps[frame]}"); return image, title
    animation = FuncAnimation(figure, update, frames=error.shape[0], interval=500, blit=False, repeat=True); update(0)
    return figure, animation


def make_component_weight_animation(
    component_weight: ArrayLike,
    sweep_indices: ArrayLike,
    times_s: ArrayLike,
    *,
    colors: ArrayLike | None = None,
) -> tuple[Figure, FuncAnimation]:
    """Create an RGB animation for three component weights over ``(R,N)``."""

    weights = np.asarray(component_weight, dtype=np.float64)
    sweeps = np.asarray(sweep_indices); times = np.asarray(times_s, dtype=np.float64)
    if weights.ndim != 4 or weights.shape[2] != 3 or sweeps.shape != (weights.shape[0],) or times.shape != (weights.shape[3],):
        raise ValueError("weights must have shape (D,R,3,N).")
    vertices = np.asarray(colors if colors is not None else [[.90,.22,.12],[.14,.40,.90],[.12,.72,.32]], dtype=np.float64)
    rgb = np.clip(np.einsum("drkn,kc->drnc", weights, vertices), 0.0, 1.0)
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    image = axis.imshow(rgb[0], origin="lower", aspect="auto", extent=(times[0], times[-1], 0, weights.shape[1]-1))
    title = axis.set_title(""); axis.set(xlabel="Elapsed time (s)", ylabel=r"RIR index $r$")
    key_axis = axis.inset_axes((0.77, 0.68, 0.20, 0.25))
    height = np.sqrt(3.0) / 2.0
    x = np.linspace(0.0, 1.0, 180); y = np.linspace(0.0, height, 156)
    xx, yy = np.meshgrid(x, y)
    noise = yy / height; long = xx - 0.5 * noise; short = 1.0 - long - noise
    barycentric = np.stack((short, long, noise), axis=-1)
    inside = np.all(barycentric >= 0.0, axis=-1)
    rgba = np.zeros((*inside.shape, 4)); rgba[..., :3] = np.clip(barycentric @ vertices, 0.0, 1.0); rgba[..., 3] = inside
    key_axis.imshow(rgba, origin="lower", extent=(0.0, 1.0, 0.0, height), aspect="equal")
    key_axis.text(0.0, -0.04, "Short", ha="center", va="top", fontsize=8); key_axis.text(1.0, -0.04, "Long", ha="center", va="top", fontsize=8); key_axis.text(0.5, height+0.03, "Noise", ha="center", va="bottom", fontsize=8)
    key_axis.set(xlim=(-0.1, 1.1), ylim=(-0.12, height+0.12)); key_axis.axis("off")
    figure.tight_layout()
    def update(frame: int):
        image.set_data(rgb[frame]); title.set_text(f"Wiener-style component-strength evolution: sweep {sweeps[frame]}"); return image, title
    animation = FuncAnimation(figure, update, frames=weights.shape[0], interval=500, blit=False, repeat=True); update(0)
    return figure, animation


def plot_known_decay_spectrogram(
    observed_power: ArrayLike,
    estimated_variance: ArrayLike,
    times_s: ArrayLike,
    *,
    title: str,
    reference_power: float = 1.0,
    lower_db: float = -120.0,
) -> Figure:
    """Compare generated instantaneous power and fitted variance maps."""

    observed = np.asarray(observed_power, dtype=np.float64)
    fitted = np.asarray(estimated_variance, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if observed.ndim != 2 or fitted.shape != observed.shape or times.shape != (observed.shape[1],) or np.any(observed <= 0.0) or np.any(fitted <= 0.0) or reference_power <= 0.0:
        raise ValueError("maps must be positive (F,N), times (N,), and reference positive.")
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for axis, values, label in zip(axes, (observed, fitted), ("Generated instantaneous power", "SAGE estimated variance"), strict=True):
        image = axis.imshow(np.maximum(10*np.log10(values/reference_power), lower_db), origin="lower", aspect="auto", extent=(times[0], times[-1], 0, observed.shape[0]-1), vmin=lower_db, vmax=10.0, cmap="magma")
        axis.set(title=label, xlabel="Elapsed time (s)", ylabel="Abstract bin")
    colorbar_axis = figure.add_axes((0.91, 0.14, 0.018, 0.68))
    figure.colorbar(image, cax=colorbar_axis, label="Power or variance (dB re 1)")
    figure.suptitle(title, y=0.97); figure.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.82, wspace=0.18)
    return figure


def plot_objective_gaps(
    histories: Sequence[ArrayLike], labels: Sequence[str]
) -> Figure:
    """Plot positive objective gaps to each history's final sweep."""

    if len(histories) != len(labels):
        raise ValueError("histories and labels must have equal length.")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for raw, label in zip(histories, labels, strict=True):
        history = np.asarray(raw, dtype=np.float64)
        gap = history - history[-1]; positive = gap > 0.0
        axis.semilogy(np.flatnonzero(positive), gap[positive], marker="o", markersize=3, label=label)
    axis.set(title="SAGE likelihood convergence", xlabel="Complete component sweeps", ylabel="IS objective gap to final sweep")
    axis.legend(); axis.grid(alpha=0.25); figure.tight_layout(); return figure


def plot_binwise_amplitudes(
    amplitudes: Sequence[ArrayLike],
    labels: Sequence[str],
    *,
    truth: float,
    decibels: bool,
) -> Figure:
    """Plot binwise fitted variance amplitudes on linear or relative-dB scale."""

    if len(amplitudes) != len(labels) or truth <= 0.0:
        raise ValueError("amplitudes and labels must match and truth be positive.")
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    for raw, label in zip(amplitudes, labels, strict=True):
        values = np.asarray(raw, dtype=np.float64)
        if decibels:
            values = 10*np.log10(values/truth)
        axis.plot(np.arange(values.size), values, linewidth=0.8, alpha=0.85, label=label)
    axis.axhline(0.0 if decibels else truth, color="black", linestyle="--")
    scale = "dB" if decibels else "linear"
    axis.set(title=f"Decay-amplitude estimates ({scale} scale)", xlabel="Abstract bin", ylabel="Estimated amplitude (dB re true amplitude)" if decibels else "Estimated variance amplitude")
    axis.legend(); axis.grid(alpha=0.25); figure.tight_layout(); return figure


def plot_amplitude_estimator_distributions(
    amplitudes: Sequence[ArrayLike],
    labels: Sequence[str],
    *,
    truth: float,
    exact_grid: ArrayLike | None = None,
    exact_density: ArrayLike | None = None,
    exact_label: str = "Exact sampling law",
) -> Figure:
    """Plot fitted-amplitude histograms with an optional exact density."""

    if len(amplitudes) != len(labels):
        raise ValueError("amplitudes and labels must have equal length.")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for raw, label in zip(amplitudes, labels, strict=True):
        axis.hist(np.asarray(raw, dtype=np.float64), bins=30, density=True, histtype="step", linewidth=1.8, label=label)
    if exact_grid is not None and exact_density is not None:
        axis.plot(np.asarray(exact_grid), np.asarray(exact_density), color="black", linestyle="--", linewidth=2, label=exact_label)
    axis.axvline(truth, color="black", linewidth=1)
    axis.set(title="Decay-amplitude estimator distributions", xlabel="Estimated variance amplitude", ylabel="Density across bins")
    axis.legend(); axis.grid(alpha=0.25); figure.tight_layout(); return figure


def plot_binwise_noise_floors(
    floors: Sequence[ArrayLike], truths: Sequence[float], labels: Sequence[str]
) -> Figure:
    """Plot fitted and true positive variance floors across bins in power dB."""

    if not (len(floors) == len(truths) == len(labels)):
        raise ValueError("floors, truths, and labels must have equal length.")
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    for raw, truth, label in zip(floors, truths, labels, strict=True):
        values = np.asarray(raw, dtype=np.float64)
        line = axis.plot(np.arange(values.size), 10*np.log10(values), linewidth=0.8, alpha=0.85, label=f"{label}: estimates")[0]
        axis.axhline(10*np.log10(truth), color=line.get_color(), linestyle="--", label=f"{label}: truth")
    axis.set(title="Estimated noise floors", xlabel="Abstract bin", ylabel="Estimated variance floor (dB re 1)")
    axis.legend(ncols=2, fontsize="small"); axis.grid(alpha=0.25); figure.tight_layout(); return figure


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
    axes[1, 1].set(title="Held-out score relative to true variance", xlabel="Mean held-out NLL difference", ylabel="Seed count"); axes[1, 1].grid(alpha=0.25)
    return figure
