"""Experiment plots loaded from saved NPZ results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from experiments._run_output import save_figure

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


def plot_results(results_path: Path, *, show: bool = False) -> list[Path]:
    """Load a decay-detection NPZ and write its diagnostic figures."""

    output_dir = results_path.parent
    with np.load(results_path, allow_pickle=False) as payload:
        true_t60_s = payload["true_t60_s"]
        estimated_t60_s = payload["estimated_t60_s"]
        absolute_error_s = payload["absolute_error_s"]
        figures = [
            (plot_true_vs_estimated_t60(true_t60_s, estimated_t60_s), "true_vs_estimated_t60.png"),
            (plot_t60_pair_plane(true_t60_s, estimated_t60_s), "t60_pair_plane.png"),
            (plot_t60_error_vs_separation(true_t60_s, absolute_error_s), "t60_error_vs_separation.png"),
            (plot_amplitude_rmse_vs_separation(true_t60_s, payload["amplitude_rmse_db"]), "amplitude_error_vs_separation.png"),
            (plot_signed_amplitude_bias_vs_separation(true_t60_s, payload["true_amplitudes"], payload["estimated_amplitudes"]), "signed_amplitude_error_vs_separation.png"),
            (plot_t60_error_by_frequency(absolute_error_s), "t60_error_by_frequency.png"),
            (plot_maximum_t60_error_cdf(absolute_error_s), "maximum_t60_error_cdf.png"),
            (plot_objective_history(payload["objective_history"], normalize=True, log_y=True, title="Observed objective convergence by frequency batch"), "objective_convergence.png"),
            (plot_t60_error_evolution(payload["t60_history_s"], true_t60_s, 0), "short_t60_error_evolution.png"),
            (plot_t60_error_evolution(payload["t60_history_s"], true_t60_s, 1), "long_t60_error_evolution.png"),
            (plot_variance_maps(payload["times_s"], (payload["example_observed_power"], payload["example_true_variance"], payload["example_fitted_variance"]), ("Observed power", "Exact variance", "Fitted variance")), "example_frequency_variance_maps.png"),
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
