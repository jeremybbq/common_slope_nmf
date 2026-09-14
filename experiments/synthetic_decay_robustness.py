"""Measure decay-estimate robustness across independently sampled T60 pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    DecaySAGEResult,
    exponential_variance,
    init_decay_sage,
    pseudo_decay_sage,
    rate_to_t60,
    sample_power,
    sample_simplex_amplitudes,
    sample_t60,
    t60_to_rate,
)
from typing import Sequence
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from experiments._run_output import create_run_output_dir

SEED = 20260907
N_RIRS = 512
N_PAIRS = 100
N_COMPONENTS = 2
N_FRAMES = 374
HOP_S = 128.0 / 24_000.0
T60_RANGE_S = (0.5, 3.0)
DIRICHLET_ALPHA = 0.5
NOISE_MEAN_DB = -40.0
NOISE_STD_DB = 2.0 / 3.0
METHOD_KEYS = ("p1", "p2")
METHOD_LABELS = (r"Pseudo-SAGE ($p=1$)", r"Pseudo-SAGE ($p=2$)")
METHOD_POWERS = (1.0, 2.0)


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


def frequency_batches(
    n_frequencies: int, batch_size: int
) -> list[slice]:
    """Return contiguous frequency slices covering ``range(n_frequencies)``.

    Parameters
    ----------
    n_frequencies
        Positive number of independent frequency bins ``F``.
    batch_size
        Positive maximum number of bins fitted in one SAGE call.

    Returns
    -------
    list of slice
        Non-overlapping slices that cover every frequency exactly once.
    """

    if n_frequencies <= 0 or batch_size <= 0:
        raise ValueError("n_frequencies and batch_size must be positive.")
    return [
        slice(start, min(start + batch_size, n_frequencies))
        for start in range(0, n_frequencies, batch_size)
    ]

def order_decay_components(
    estimated_t60_s: np.ndarray,
    estimated_amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order fitted components by increasing energy-``T60`` in every bin.

    Parameters
    ----------
    estimated_t60_s
        Fitted decay times in seconds, shape ``(F,K)``.
    estimated_amplitudes
        Fitted unit-origin variance amplitudes, shape ``(R,F,K)``.

    Returns
    -------
    ordered_t60_s, ordered_amplitudes, order
        Decay times ``(F,K)``, consistently permuted amplitudes ``(R,F,K)``,
        and the integer component permutation ``(F,K)``.
    """

    t60_s = np.asarray(estimated_t60_s, dtype=np.float64)
    amplitudes = np.asarray(estimated_amplitudes, dtype=np.float64)
    if t60_s.ndim != 2:
        raise ValueError("estimated_t60_s must have shape (F,K).")
    if amplitudes.ndim != 3 or amplitudes.shape[1:] != t60_s.shape:
        raise ValueError("estimated_amplitudes must have shape (R,F,K).")
    order = np.argsort(t60_s, axis=1)
    ordered_t60_s = np.take_along_axis(t60_s, order, axis=1)
    amplitude_order = np.broadcast_to(order[np.newaxis, :, :], amplitudes.shape)
    ordered_amplitudes = np.take_along_axis(
        amplitudes, amplitude_order, axis=2
    )
    return ordered_t60_s, ordered_amplitudes, order


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n-rirs", type=int, default=N_RIRS)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Bins per fit; one gives independent per-bin stopping.",
    )
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Relative observed-loss stopping tolerance.",
    )
    parser.add_argument(
        "--decay-tol",
        type=float,
        default=1e-6,
        help=(
            "Maximum absolute log change in decay estimates. Stopping occurs "
            "when this or --tol is met."
        ),
    )
    parser.add_argument(
        "--rate-method", choices=("newton", "bisection"), default="newton"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--plot-results",
        type=Path,
        help="Regenerate only the figure from an existing result NPZ.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for name in ("n_pairs", "n_rirs", "n_frames", "batch_size", "max_iter"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_frames < 17:
        parser.error("--n-frames must be at least 17")
    for name in ("tol", "decay_tol"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    args.batch_size = min(args.batch_size, args.n_pairs)
    return args


def _rate_bounds() -> tuple[float, float]:
    """Return energy-decay-rate bounds in inverse seconds."""

    return (
        float(t60_to_rate(T60_RANGE_S[1])),
        float(t60_to_rate(T60_RANGE_S[0])),
    )


def _fit_method(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    initial_rates_per_s: np.ndarray,
    initial_amplitudes: np.ndarray,
    initial_noise_floor: np.ndarray,
    component_weight_power: float,
    *,
    max_iter: int,
    tol: float,
    decay_tol: float,
    rate_method: str,
) -> DecaySAGEResult:
    """Fit one pseudo-SAGE order to power ``(R,F,N)``."""

    return pseudo_decay_sage(
        observed_power,
        times_s,
        initial_rates_per_s,
        rate_bounds_per_s=_rate_bounds(),
        component_weight_power=component_weight_power,
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=True,
        max_iter=max_iter,
        tol=tol,
        decay_tol=decay_tol,
        rate_method=rate_method,
    )


def _ordered_fit(
    result: DecaySAGEResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted T60 ``(F,2)`` and correspondingly ordered amplitudes."""

    t60_s, amplitudes, _ = order_decay_components(
        rate_to_t60(result.rates_per_s), result.amplitudes
    )
    return t60_s, amplitudes


def _per_frequency_is(
    observed_power: np.ndarray, fitted_variance: np.ndarray
) -> np.ndarray:
    """Return observed IS divergence per frequency, shape ``(F,)``."""

    ratio = observed_power / fitted_variance
    return np.sum(ratio - np.log(ratio) - 1.0, axis=(0, 2))


def _stopping_flags(
    result: DecaySAGEResult, tol: float, decay_tol: float
) -> tuple[bool, bool]:
    """Return whether the final sweep meets the loss and decay gates."""

    previous_objective, final_objective = result.objective_history[-2:]
    decrease = previous_objective - final_objective
    roundoff = (
        64.0 * np.finfo(np.float64).eps * max(1.0, previous_objective)
    )
    loss_converged = (
        decrease >= -roundoff
        and decrease <= tol * max(1.0, previous_objective)
    )
    decay_change = np.max(
        np.abs(
            np.log(
                result.rate_history_per_s[-1]
                / result.rate_history_per_s[-2]
            )
        )
    )
    return loss_converged, bool(decay_change <= decay_tol)


def run_identifiability_comparison(
    rng: np.random.Generator,
    times_s: np.ndarray,
    *,
    n_pairs: int,
    n_rirs: int,
    batch_size: int,
    max_iter: int,
    tol: float,
    decay_tol: float,
    rate_method: str,
) -> dict[str, np.ndarray]:
    """Fit pseudo-SAGE p=1 and p=2 to independent random T60 pairs.

    Parameters
    ----------
    rng
        Seeded random generator controlling all synthetic quantities.
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    n_pairs
        Number of independently parameterized frequency-bin labels ``F``.
    n_rirs
        Number of independent RIR realizations ``R`` per pair.
    batch_size
        Number of frequency bins per estimator call. Use one for independent
        per-bin stopping decisions.
    max_iter
        Maximum complete component sweeps per fit.
    tol, decay_tol
        Dimensionless loss and outer decay-stability tolerances. Either gate
        may stop a fit.
    rate_method
        Profile-rate solver, ``"newton"`` or ``"bisection"``.

    Returns
    -------
    dict
        True/fitted T60 values in seconds, amplitudes and floors in variance
        units, per-frequency losses, sweep counts, and stopping diagnostics.
    """

    true_t60_s = sample_t60(
        n_pairs, N_COMPONENTS, T60_RANGE_S, rng=rng
    )
    true_rates_per_s = np.asarray(t60_to_rate(true_t60_s), dtype=np.float64)
    true_amplitudes = sample_simplex_amplitudes(
        n_rirs,
        n_pairs,
        N_COMPONENTS,
        concentration=DIRICHLET_ALPHA,
        total_amplitude=1.0,
        rng=rng,
    )
    noise_level_db = rng.normal(
        NOISE_MEAN_DB, NOISE_STD_DB, size=(n_rirs, n_pairs)
    )
    true_noise_floor = 10.0 ** (noise_level_db / 10.0)

    batches = frequency_batches(n_pairs, batch_size)
    n_methods = len(METHOD_KEYS)
    estimated_t60_s = np.empty((n_methods, n_pairs, N_COMPONENTS))
    estimated_amplitudes = np.empty(
        (n_methods, n_rirs, n_pairs, N_COMPONENTS)
    )
    estimated_noise_floor = np.empty((n_methods, n_rirs, n_pairs))
    final_is = np.empty((n_methods, n_pairs))
    batch_n_iter = np.empty((n_methods, len(batches)), dtype=np.int64)
    batch_converged = np.empty_like(batch_n_iter, dtype=bool)
    frequency_n_iter = np.empty((n_methods, n_pairs), dtype=np.int64)
    frequency_converged = np.empty((n_methods, n_pairs), dtype=bool)
    frequency_loss_converged = np.empty((n_methods, n_pairs), dtype=bool)
    frequency_decay_converged = np.empty((n_methods, n_pairs), dtype=bool)
    objective_history = np.full(
        (n_methods, len(batches), max_iter + 1), np.nan
    )

    rate_history_per_s = np.full((n_methods, max_iter + 1, n_pairs, N_COMPONENTS), np.nan)

    for batch_index, frequency_slice in enumerate(batches):
        exact_variance = exponential_variance(
            times_s,
            true_rates_per_s[frequency_slice],
            true_amplitudes[:, frequency_slice, :],
            noise_floor=true_noise_floor[:, frequency_slice],
        )
        observed_power = sample_power(exact_variance, rng=rng)
        init = init_decay_sage(
            observed_power,
            times_s,
            N_COMPONENTS,
            rate_bounds_per_s=_rate_bounds(),
            n_head_frames=8,
            n_tail_frames=8,
            floor_margin_db=6.0,
        )
        for method_index, power in enumerate(METHOD_POWERS):
            print(
                f"pair {batch_index + 1}/{len(batches)}: "
                f"{METHOD_LABELS[method_index]}",
                flush=True,
            )
            result = _fit_method(
                observed_power,
                times_s,
                init.rates_per_s,
                init.amplitudes,
                init.noise_floor,
                power,
                max_iter=max_iter,
                tol=tol,
                decay_tol=decay_tol,
                rate_method=rate_method,
            )
            rate_history_per_s[method_index, :result.n_iter + 1, frequency_slice] = result.rate_history_per_s
            ordered_t60_s, ordered_amplitudes = _ordered_fit(result)
            estimated_t60_s[method_index, frequency_slice] = ordered_t60_s
            estimated_amplitudes[
                method_index, :, frequency_slice, :
            ] = ordered_amplitudes
            estimated_noise_floor[
                method_index, :, frequency_slice
            ] = result.noise_floor
            final_is[method_index, frequency_slice] = _per_frequency_is(
                observed_power, result.variance
            )
            batch_n_iter[method_index, batch_index] = result.n_iter
            batch_converged[method_index, batch_index] = result.converged
            frequency_n_iter[method_index, frequency_slice] = result.n_iter
            frequency_converged[method_index, frequency_slice] = result.converged
            loss_gate, decay_gate = _stopping_flags(result, tol, decay_tol)
            frequency_loss_converged[
                method_index, frequency_slice
            ] = loss_gate
            frequency_decay_converged[
                method_index, frequency_slice
            ] = decay_gate
            objective_history[
                method_index, batch_index, : result.objective_history.size
            ] = result.objective_history

    return {
        "true_t60_s": true_t60_s,
        "true_amplitudes": true_amplitudes,
        "true_noise_floor": true_noise_floor,
        "estimated_t60_s": estimated_t60_s,
        "estimated_amplitudes": estimated_amplitudes,
        "estimated_noise_floor": estimated_noise_floor,
        "final_is": final_is,
        "batch_n_iter": batch_n_iter,
        "batch_converged": batch_converged,
        "frequency_n_iter": frequency_n_iter,
        "frequency_converged": frequency_converged,
        "frequency_loss_converged": frequency_loss_converged,
        "frequency_decay_converged": frequency_decay_converged,
        "objective_history": objective_history,
        "rate_history_per_s": rate_history_per_s,
    }


def save_identifiability_figure(
    output_dir: Path,
    true_t60_s: np.ndarray,
    estimated_t60_s: np.ndarray,
    converged: np.ndarray,
    *,
    show: bool,
) -> Path:
    """Save signed rate-error scatter as PDF/PNG; return the PNG path.

    Truth has shape (F,2), estimates (M,F,2), in seconds. All cases are
    included irrespective of the supplied convergence flags (M,F).
    """

    for method_index, key in enumerate(METHOD_KEYS):
        errors = np.abs(estimated_t60_s[method_index] - true_t60_s)
        separation_figure = plot_t60_error_vs_separation(true_t60_s, errors)
        separation_figure.axes[0].set_title(METHOD_LABELS[method_index], fontsize=8)
        # Keep unfinished fits visible instead of silently filtering them.
        unfinished = ~np.asarray(converged[method_index], dtype=bool)
        if np.any(unfinished):
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
            np.asarray(archive[f"{prefix}frequency_converged"]),
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


def main() -> None:
    """Run or replot the identifiability comparison and save outputs."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    if args.plot_results is not None:
        if not args.plot_results.is_file():
            raise FileNotFoundError(args.plot_results)
        figure_path = _plot_saved_results(
            args.plot_results, output_dir, show=args.show
        )
        print(f"source={args.plot_results.resolve()}")
        print(f"saved={figure_path}")
        return

    times_s = np.arange(args.n_frames, dtype=np.float64) * HOP_S
    plane_seed = np.random.SeedSequence(args.seed).spawn(2)[1]
    result = run_identifiability_comparison(
        np.random.default_rng(plane_seed),
        times_s,
        n_pairs=args.n_pairs,
        n_rirs=args.n_rirs,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        tol=args.tol,
        decay_tol=args.decay_tol,
        rate_method=args.rate_method,
    )
    figure_path = save_identifiability_figure(
        output_dir,
        result["true_t60_s"],
        result["estimated_t60_s"],
        result["frequency_converged"],
        show=args.show,
    )
    save_error_histograms(
        output_dir, result["true_t60_s"], result["estimated_t60_s"],
        result["true_amplitudes"], result["estimated_amplitudes"], show=args.show,
    )
    archive_path = output_dir / "t60_identifiability_results.npz"
    np.savez_compressed(
        archive_path,
        seed=np.asarray(args.seed),
        times_s=times_s,
        method_keys=np.asarray(METHOD_KEYS),
        method_labels=np.asarray(METHOD_LABELS),
        method_weight_powers=np.asarray(METHOD_POWERS),
        t60_range_s=np.asarray(T60_RANGE_S),
        dirichlet_alpha=np.asarray(DIRICHLET_ALPHA),
        noise_mean_db=np.asarray(NOISE_MEAN_DB),
        noise_std_db=np.asarray(NOISE_STD_DB),
        true_t60_s=result["true_t60_s"],
        true_amplitudes=result["true_amplitudes"],
        true_noise_floor=result["true_noise_floor"],
        estimated_t60_s=result["estimated_t60_s"],
        estimated_amplitudes=result["estimated_amplitudes"],
        estimated_noise_floor=result["estimated_noise_floor"],
        final_is=result["final_is"],
        batch_n_iter=result["batch_n_iter"],
        batch_converged=result["batch_converged"],
        frequency_n_iter=result["frequency_n_iter"],
        frequency_converged=result["frequency_converged"],
        frequency_loss_converged=result["frequency_loss_converged"],
        frequency_decay_converged=result["frequency_decay_converged"],
        objective_history=result["objective_history"],
        rate_history_per_s=result["rate_history_per_s"],
        rate_method=np.asarray(args.rate_method),
        max_iter=np.asarray(args.max_iter),
        tolerance=np.asarray(args.tol),
        decay_tolerance=np.asarray(args.decay_tol),
        batch_size=np.asarray(args.batch_size),
    )
    for method_index, key in enumerate(METHOD_KEYS):
        print(
            f"{key}: converged="
            f"{int(np.sum(result['frequency_converged'][method_index]))}/"
            f"{args.n_pairs}, median_sweeps="
            f"{np.median(result['frequency_n_iter'][method_index]):.1f}"
        )
    print(f"saved={figure_path}")
    print(f"saved={archive_path.resolve()}")


if __name__ == "__main__":
    main()
