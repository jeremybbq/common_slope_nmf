
"""Plots for the synthetic decay-robustness experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from experiments.synthetic.decay_robustness import METHOD_KEYS, METHOD_LABELS

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


def plot_t60_amplitude_error_histograms(
    true_t60_s: ArrayLike,
    estimated_t60_s: ArrayLike,
    true_amplitudes: ArrayLike,
    estimated_amplitudes: ArrayLike,
    method_labels: Sequence[str],
) -> Figure:
    """Return four density histograms of signed T60 and amplitude errors.

    T60 inputs have shapes (F,2) and (M,F,2), in seconds. Corresponding
    amplitude inputs have shapes (R,F,2) and (M,R,F,2), in linear variance
    units. Components are sorted by T60 with their amplitudes permuted
    together. T60 histograms pool F pairs; amplitude histograms pool R*F
    entries, which are not independent across R within a fitted pair.
    T60 errors are estimate minus truth in seconds. Amplitude errors are
    10*log10(estimate/truth) in dB; amplitudes must be strictly positive.
    No outliers are discarded.
    """
    truth = np.asarray(true_t60_s, dtype=float)
    estimates = np.asarray(estimated_t60_s, dtype=float)
    amplitudes = np.asarray(true_amplitudes, dtype=float)
    fitted = np.asarray(estimated_amplitudes, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("true_t60_s must have shape (F,2).")
    if estimates.shape != (len(method_labels), *truth.shape):
        raise ValueError("estimated_t60_s must have shape (M,F,2).")
    if amplitudes.ndim != 3 or amplitudes.shape[1:] != truth.shape:
        raise ValueError("true_amplitudes must have shape (R,F,2).")
    if fitted.shape != (len(method_labels), *amplitudes.shape):
        raise ValueError("estimated_amplitudes must have shape (M,R,F,2).")
    if not method_labels or any(not x.size or not np.all(np.isfinite(x))
                               for x in (truth, estimates, amplitudes, fitted)):
        raise ValueError("inputs must be nonempty and finite.")
    order = np.argsort(truth, axis=-1)
    estimate_order = np.argsort(estimates, axis=-1)
    t_error = np.take_along_axis(estimates, estimate_order, axis=-1) - np.take_along_axis(truth, order, axis=-1)
    if np.any(amplitudes <= 0) or np.any(fitted <= 0):
        raise ValueError("amplitudes must be positive for dB errors.")
    a_error = 10.0 * (
        np.log10(np.take_along_axis(fitted, estimate_order[:, None, :, :], axis=-1))
        - np.log10(np.take_along_axis(amplitudes, order[None, :, :], axis=-1))
    )
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8}):
        figure, axes = plt.subplots(2, 2, figsize=(7.10, 4.5), layout="constrained")
        for row, errors in enumerate((t_error, a_error)):
            for component in range(2):
                axis = axes[row, component]
                values = errors[..., component].reshape(len(method_labels), -1)
                edges = np.histogram_bin_edges(values.ravel(), bins=25 if row == 0 else 60)
                for method, label in enumerate(method_labels):
                    axis.hist(values[method], bins=edges, density=True,
                              histtype="step", linewidth=1.25,
                              color=f"C{method}", label=label)
                axis.axvline(0, color="0.4", linestyle="--", linewidth=0.7)
                axis.set_xlabel(r"$\widehat T_{60}-T_{60}$ (s)" if row == 0
                                else r"$10\log_{10}(\widehat a/a)$ (dB)")
                if component == 0:
                    axis.set_ylabel("Density")
                if row == 0:
                    axis.set_title("Short T60" if component == 0 else "Long T60", fontsize=9)
        axes[0, 0].legend()
    return figure


def plot_rate_error_pairs(
    true_t60_s: ArrayLike,
    estimated_t60_by_method: Sequence[ArrayLike],
    method_labels: Sequence[str],
) -> Figure:
    """Scatter signed fast/slow energy-rate errors for paired decay estimates.

    Inputs are positive T60 values in seconds: truth ``(F,2)`` and one
    estimate ``(F,2)`` per method. Each pair is sorted by T60 before conversion
    using ``lambda = 6*log(10)/T60``. The returned figure plots fast-rate error
    on x and slow-rate error on y, both in inverse seconds. All pairs are kept;
    the scatter describes errors across cases, not fixed-parameter uncertainty.
    Marker area scales linearly with true rate separation, with a minimum
    area of 9 points squared and a maximum of 49. The mapping is shared by
    all methods and documented by a separate size legend.
    """
    truth = np.asarray(true_t60_s, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 2 or truth.shape[0] == 0:
        raise ValueError("truth must have nonempty shape (F,2).")
    if not np.all(np.isfinite(truth)) or np.any(truth <= 0):
        raise ValueError("T60 values must be finite and positive.")
    if len(estimated_t60_by_method) != len(method_labels) or not method_labels:
        raise ValueError("provide one label per method.")
    true_rates = 6 * np.log(10) / np.sort(truth, axis=1)
    separation = true_rates[:, 0] - true_rates[:, 1]
    size_scale = max(float(np.max(separation)), np.finfo(float).eps)
    marker_areas = 9.0 + 40.0 * separation / size_scale
    errors = []
    for raw in estimated_t60_by_method:
        estimate = np.asarray(raw, dtype=float)
        if (estimate.shape != truth.shape or not np.all(np.isfinite(estimate))
                or np.any(estimate <= 0)):
            raise ValueError("estimates must be positive, finite, and match truth.")
        errors.append(6 * np.log(10) / np.sort(estimate, axis=1) - true_rates)
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7}):
        figure, axis = plt.subplots(figsize=(3.45, 3.05), layout="constrained")
        axis.axhline(0, color="0.55", linewidth=0.6, zorder=0)
        axis.axvline(0, color="0.55", linewidth=0.6, zorder=0)
        for index in reversed(range(len(errors))):
            error = errors[index]
            axis.scatter(error[:, 0], error[:, 1], s=marker_areas, alpha=0.6,
                         color=f"C{index}", edgecolors="none",
                         label=method_labels[index], zorder=2+len(errors)-index)
        handles, labels = axis.get_legend_handles_labels()
        method_legend = axis.legend(handles[::-1], labels[::-1], loc="upper right")
        axis.add_artist(method_legend)
        size_values = np.unique(np.quantile(separation, [0.1, 0.5, 0.9]))
        size_handles = [
            axis.scatter([], [], s=9.0 + 40.0 * value / size_scale,
                         color="0.45", alpha=0.6, edgecolors="none")
            for value in size_values
        ]
        axis.legend(size_handles, [f"{value:.2f}" for value in size_values],
                    title=r"True $|\lambda_1-\lambda_2|$ (s$^{-1}$)",
                    title_fontsize=7, loc="lower left", labelspacing=0.7)
        axis.set_xlabel(r"$\widehat\lambda_{\mathrm{fast}}-\lambda_{\mathrm{fast}}$ (s$^{-1}$)")
        axis.set_ylabel(r"$\widehat\lambda_{\mathrm{slow}}-\lambda_{\mathrm{slow}}$ (s$^{-1}$)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.08)
    return figure


@plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
def plot_t60_error_vs_separation(
    true_t60_s: ArrayLike, absolute_error_s: ArrayLike
) -> Figure:
    """Plot maximum absolute ``T60`` error against true pair separation."""

    truth, errors = _two_component_arrays(
        true_t60_s, absolute_error_s, "absolute_error_s"
    )
    figure, axis = plt.subplots(figsize=(3.45, 2.8))
    scatter = axis.scatter(np.diff(truth, axis=1)[:, 0], np.max(errors, axis=1), c=np.mean(truth, axis=1), cmap="viridis", s=34, alpha=0.82)
    figure.colorbar(scatter, ax=axis, label=r"Pair mean $T_{60}$ (s)")
    axis.set( xlabel=r"True pair separation $\Delta T_{60}$ (s)", ylabel=r"Maximum absolute $T_{60}$ error (s)")
    axis.grid(alpha=0.2); figure.tight_layout()
    return figure


@plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
def _plot_component_error_vs_separation(
    true_t60_s: np.ndarray,
    errors_db: np.ndarray,
    *,
    title: str,
    ylabel: str,
    zero_line: bool,
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.10, 3.4), sharex=True, sharey=True, layout="constrained")
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


def save_identifiability_figure(
    output_dir: Path,
    true_t60_s: np.ndarray,
    estimated_t60_s: np.ndarray,
    converged: np.ndarray | None,
    *,
    show: bool,
) -> Path:
    """Save signed rate-error scatter as PDF/PNG; return the PNG path.

    Truth has shape (F,2), estimates (M,F,2), in seconds. All cases are
    included irrespective of optional convergence flags (M,F).
    """

    for method_index, key in enumerate(METHOD_KEYS):
        errors = np.abs(estimated_t60_s[method_index] - true_t60_s)
        separation_figure = plot_t60_error_vs_separation(true_t60_s, errors)
        separation_figure.axes[0].set_title(METHOD_LABELS[method_index], fontsize=8)
        # Keep unfinished fits visible instead of silently filtering them.
        unfinished = None if converged is None else ~np.asarray(
            converged[method_index], dtype=bool
        )
        if unfinished is not None and np.any(unfinished):
            separation_figure.axes[0].scatter(
                np.diff(true_t60_s, axis=1)[unfinished, 0],
                np.max(errors[unfinished], axis=1),
                marker="x", color="black", s=20, label="Stopping criteria not met",
            )
            separation_figure.axes[0].legend(fontsize=7)
        separation_figure.tight_layout()
        separation_figure.savefig(output_dir / f"{key}_t60_error_vs_separation.pdf")
        save_figure(separation_figure, output_dir,
                    f"{key}_t60_error_vs_separation.png", show=show)

    figure = plot_rate_error_pairs(
        true_t60_s, list(estimated_t60_s), METHOD_LABELS
    )
    figure.savefig(output_dir / "rate_error_pairs.pdf")
    return save_figure(
        figure,
        output_dir,
        "rate_error_pairs.png",
        show=show,
    )


def save_error_histograms(
    output_dir: Path, true_t60_s: np.ndarray, estimated_t60_s: np.ndarray,
    true_amplitudes: np.ndarray, estimated_amplitudes: np.ndarray,
    *, show: bool = False,
) -> Path:
    """Save error histograms as PDF/PNG from paired T60 and amplitude arrays.

    Shapes are (F,2), (M,F,2), (R,F,2), and (M,R,F,2), respectively.
    T60 is in seconds; amplitudes are in linear variance units.
    Returns the absolute PNG path.
    """
    for method_index, key in enumerate(METHOD_KEYS):
        error_db = 10.0 * (np.log10(estimated_amplitudes[method_index])
                           - np.log10(true_amplitudes))
        figures = (
            (plot_amplitude_rmse_vs_separation(
                true_t60_s, np.sqrt(np.mean(error_db ** 2, axis=0))), "amplitude_rmse"),
            (plot_signed_amplitude_bias_vs_separation(
                true_t60_s, true_amplitudes, estimated_amplitudes[method_index]), "amplitude_bias"),
        )
        for figure, label in figures:
            figure.savefig(output_dir / f"{key}_{label}_vs_separation.pdf")
            save_figure(figure, output_dir, f"{key}_{label}_vs_separation.png", show=show)

    figure = plot_t60_amplitude_error_histograms(
        true_t60_s, estimated_t60_s, true_amplitudes, estimated_amplitudes,
        METHOD_LABELS,
    )
    figure.savefig(output_dir / "t60_amplitude_error_histograms.pdf")
    path = save_figure(
        figure, output_dir, "t60_amplitude_error_histograms.png", show=show
    )
    print(f"saved={path}")
    return path


def _plot_saved_results(
    results_path: Path, output_dir: Path, *, show: bool
) -> Path:
    """Regenerate the pair-plane figure from a current or combined archive."""

    with np.load(results_path, allow_pickle=False) as archive:
        prefix = "" if "true_t60_s" in archive else "plane_"
        save_error_histograms(
            output_dir,
            archive[f"{prefix}true_t60_s"],
            archive[f"{prefix}estimated_t60_s"],
            archive[f"{prefix}true_amplitudes"],
            archive[f"{prefix}estimated_amplitudes"],
            show=show,
        )
        return save_identifiability_figure(
            output_dir,
            np.asarray(archive[f"{prefix}true_t60_s"]),
            np.asarray(archive[f"{prefix}estimated_t60_s"]),
            (
                np.asarray(archive[f"{prefix}frequency_converged"])
                if f"{prefix}frequency_converged" in archive
                else None
            ),
            show=show,
        )


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
