"""Experiment plots loaded from saved NPZ results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from experiments._run_output import save_figure

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


def plot_results(results_path: Path, *, show: bool = False) -> list[Path]:
    """Load a synth-init-fit NPZ and write its diagnostic figures."""

    output_dir = results_path.parent
    with np.load(results_path, allow_pickle=False) as payload:
        figures = [
            (
                plot_simplex_amplitude_shares(payload["true_amplitudes"]),
                "simplex_amplitude_shares.png",
            ),
            (
                plot_variance_maps(
                    payload["times_s"],
                    (
                        payload["observed_power"][:, 0],
                        payload["true_variance"][:, 0],
                        payload["fitted_variance"][:, 0],
                    ),
                    ("Observed power", "Exact variance", "Fitted variance"),
                ),
                "observed_exact_fitted_variance.png",
            ),
            (
                plot_t60_trajectories(
                    payload["t60_history_s"],
                    true_t60_s=payload["true_t60_s"],
                    title="Decay-time trajectories from pooled-log initialization",
                ),
                "t60_trajectories.png",
            ),
            (
                plot_objective_history(
                    payload["objective_history"],
                    title=r"Observed objective for pseudo-SAGE with $\rho^2$ weighting",
                ),
                "objective.png",
            ),
        ]
    saved = [
        save_figure(figure, output_dir, filename, show=show)
        for figure, filename in figures
    ]
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
