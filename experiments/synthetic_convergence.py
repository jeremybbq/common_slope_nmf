"""Compare SAGE loss paths on a profiled two-decay loss surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    AmplitudeSAGEResult,
    DecaySAGEInit,
    DecaySAGEResult,
    amplitude_sage,
    decay_sage,
    exponential_variance,
    init_decay_sage,
    pseudo_decay_sage,
    rate_to_t60,
    sample_power,
    sample_simplex_amplitudes,
    t60_to_rate,
)
from typing import Sequence
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from experiments._run_output import create_run_output_dir

SEED = 20260907
N_RIRS = 256
N_COMPONENTS = 2
N_FRAMES = 256
HOP_S = 128.0 / 24_000.0
T60_RANGE_S = (0.5, 3.0)
TRUE_T60_S = np.array([0.80, 1.40], dtype=np.float64)
SHORT_T60_RANGE_S = (0.6, 1.4)
LONG_T60_RANGE_S = (1.0, 1.8)
SURFACE_GRID_SIZE = 21
DIRICHLET_ALPHA = 0.5
NOISE_MEAN_DB = -40.0
NOISE_STD_DB = 2.0 / 3.0
DECAY_LOSS_TOL = 0.0
SURFACE_LOSS_TOL = 1e-6
METHOD_KEYS = ("ordinary", "p1", "p2")
METHOD_LABELS = (
    r"SAGE ($p=0$)",
    r"RW-SAGE ($p=1$)",
    r"RW-SAGE ($p=2$)",
)
METHOD_POWERS: tuple[float | None, ...] = (None, 1.0, 2.0)
METHOD_COLORS = ("C2", "C0", "C1")


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

    figure, axes = plt.subplots(
        1,
        n_cases,
        figsize=((3.45, 2.3) if n_cases == 1 else (7.10, 3.4)),
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
            xlabel="Complete component sweeps",
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
    axes[0, 0].legend(fontsize=7 if n_cases == 1 else 8)
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
    figure.subplots_adjust(left=0.14, right=0.86, bottom=0.27, top=0.976)
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
        label="True RT60s",
        zorder=11,
    )
    axis.set(
        xlabel="Fast decay RT60 (s)",
        ylabel="Slow decay RT60 (s)",
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
        "True RT60s",
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
    colorbar_axis = divider.append_axes("right", size="4.5%", pad=0.16)
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.set_label("Profiled excess loss", fontsize=8, labelpad=1)
    colorbar.ax.tick_params(labelsize=7)
    return figure


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rirs", type=int, default=N_RIRS)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument(
        "--tol",
        type=float,
        default=DECAY_LOSS_TOL,
        help="Relative total observed-loss tolerance; zero uses all sweeps.",
    )
    parser.add_argument("--surface-grid-size", type=int, default=SURFACE_GRID_SIZE)
    parser.add_argument("--surface-max-iter", type=int, default=2_000)
    parser.add_argument(
        "--surface-tol",
        type=float,
        default=SURFACE_LOSS_TOL,
        help="Relative IS-loss tolerance for fixed-rate amplitude profiling.",
    )
    parser.add_argument(
        "--rate-method", choices=("newton", "bisection"), default="newton"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--plot-results",
        type=Path,
        help="Regenerate figures only from an existing result NPZ.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    for name in (
        "n_rirs",
        "n_frames",
        "max_iter",
        "surface_grid_size",
        "surface_max_iter",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_frames < 17:
        parser.error("--n-frames must be at least 17")
    if args.surface_grid_size < 2:
        parser.error("--surface-grid-size must be at least two")
    for name in ("tol", "surface_tol"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def _rate_bounds() -> tuple[float, float]:
    """Return energy-decay-rate bounds in inverse seconds."""

    return (
        float(t60_to_rate(T60_RANGE_S[1])),
        float(t60_to_rate(T60_RANGE_S[0])),
    )


def _init_from_data(
    observed_power: np.ndarray, times_s: np.ndarray
) -> DecaySAGEInit:
    """Return pooled-regression decay and equal-amplitude initialization."""

    return init_decay_sage(
        observed_power,
        times_s,
        N_COMPONENTS,
        rate_bounds_per_s=_rate_bounds(),
        n_head_frames=8,
        n_tail_frames=8,
        floor_margin_db=6.0,
    )


def _fit_method(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    init: DecaySAGEInit,
    component_weight_power: float | None,
    *,
    max_iter: int,
    tol: float,
    rate_method: str,
) -> DecaySAGEResult:
    """Fit one method to power ``(R,1,N)`` from the shared warm start."""

    common = dict(
        rate_bounds_per_s=_rate_bounds(),
        initial_amplitudes=init.amplitudes,
        initial_noise_floor=init.noise_floor,
        estimate_noise_floor=True,
        max_iter=max_iter,
        tol=tol,
        rate_method=rate_method,
    )
    if component_weight_power is None:
        return decay_sage(observed_power, times_s, init.rates_per_s, **common)
    return pseudo_decay_sage(
        observed_power,
        times_s,
        init.rates_per_s,
        component_weight_power=component_weight_power,
        **common,
    )


def run_loss_comparison(
    rng: np.random.Generator,
    times_s: np.ndarray,
    *,
    n_rirs: int,
    max_iter: int,
    tol: float,
    rate_method: str,
) -> dict[str, object]:
    """Fit the three SAGE orders to one controlled two-decay dataset.

    Parameters
    ----------
    rng
        Seeded random generator controlling amplitudes, floors, and powers.
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    n_rirs
        Number of independent RIR realizations ``R``.
    max_iter
        Maximum number of complete component sweeps.
    tol
        Relative observed IS-loss stopping tolerance.
    rate_method
        Profile-rate solver, ``"newton"`` or ``"bisection"``.

    Returns
    -------
    dict
        Ground truth, common warm initialization, observations, estimates,
        objective histories, and T60 trajectories. Component amplitudes have
        shape ``(R,1,2)`` and sum to one at time zero.
    """

    true_amplitudes = sample_simplex_amplitudes(
        n_rirs,
        1,
        N_COMPONENTS,
        concentration=DIRICHLET_ALPHA,
        total_amplitude=1.0,
        rng=rng,
    )
    noise_level_db = rng.normal(NOISE_MEAN_DB, NOISE_STD_DB, size=(n_rirs, 1))
    true_noise_floor = 10.0 ** (noise_level_db / 10.0)
    exact_variance = exponential_variance(
        times_s,
        t60_to_rate(TRUE_T60_S[np.newaxis, :]),
        true_amplitudes,
        noise_floor=true_noise_floor,
    )
    observed_power = sample_power(exact_variance, rng=rng)
    init = _init_from_data(observed_power, times_s)

    histories: list[np.ndarray] = []
    rate_histories: list[np.ndarray] = []
    estimated_t60_s = np.empty((len(METHOD_KEYS), N_COMPONENTS))
    estimated_rates_per_s = np.empty((len(METHOD_KEYS), 1, N_COMPONENTS))
    estimated_amplitudes = np.empty((len(METHOD_KEYS), n_rirs, 1, N_COMPONENTS))
    estimated_noise_floor = np.empty((len(METHOD_KEYS), n_rirs, 1))
    n_iter = np.empty(len(METHOD_KEYS), dtype=np.int64)
    converged = np.empty(len(METHOD_KEYS), dtype=bool)
    for method_index, power in enumerate(METHOD_POWERS):
        print(f"loss path: {METHOD_LABELS[method_index]}", flush=True)
        result = _fit_method(
            observed_power,
            times_s,
            init,
            power,
            max_iter=max_iter,
            tol=tol,
            rate_method=rate_method,
        )
        estimated_t60_s[method_index] = np.sort(
            rate_to_t60(result.rates_per_s[0])
        )
        estimated_rates_per_s[method_index] = result.rates_per_s
        estimated_amplitudes[method_index] = result.amplitudes
        estimated_noise_floor[method_index] = result.noise_floor
        n_iter[method_index] = result.n_iter
        converged[method_index] = result.converged
        histories.append(result.objective_history.copy())
        rate_histories.append(
            np.sort(rate_to_t60(result.rate_history_per_s[:, 0, :]), axis=1)
        )

    return {
        "histories": [histories],
        "t60_trajectories_s": rate_histories,
        "true_amplitudes": true_amplitudes,
        "true_noise_floor": true_noise_floor,
        "observed_power": observed_power,
        "exact_variance": exact_variance,
        "init_rates_per_s": init.rates_per_s,
        "init_amplitudes": init.amplitudes,
        "init_noise_floor": init.noise_floor,
        "initial_t60_s": np.sort(rate_to_t60(init.rates_per_s[0])),
        "estimated_t60_s": estimated_t60_s,
        "estimated_rates_per_s": estimated_rates_per_s,
        "estimated_amplitudes": estimated_amplitudes,
        "estimated_noise_floor": estimated_noise_floor,
        "n_iter": n_iter,
        "converged": converged,
    }


def _fixed_rate_atoms(times_s: np.ndarray, t60_s: np.ndarray) -> np.ndarray:
    """Return two exponential atoms and one constant-floor atom, shape ``(3,N)``."""

    decay_atoms = np.exp(
        -t60_to_rate(t60_s)[:, np.newaxis] * times_s[np.newaxis, :]
    )
    return np.concatenate((decay_atoms, np.ones((1, times_s.size))), axis=0)


def _profile_one_point(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    t60_s: np.ndarray,
    initial_joint_amplitudes: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> AmplitudeSAGEResult:
    """Profile amplitudes and the floor for one supplied T60 pair."""

    return amplitude_sage(
        observed_power,
        _fixed_rate_atoms(times_s, t60_s),
        initial_amplitudes=initial_joint_amplitudes,
        max_iter=max_iter,
        tol=tol,
    )


def profile_fixed_rate_loss_surface(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    short_t60_s: np.ndarray,
    long_t60_s: np.ndarray,
    initial_amplitudes: np.ndarray,
    initial_noise_floor: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    cold_check_t60_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Profile nuisance amplitudes over a two-dimensional T60 grid.

    Parameters
    ----------
    observed_power
        Strictly positive observed power, shape ``(R,N)``.
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    short_t60_s, long_t60_s
        Strictly increasing decay grids in seconds, shapes ``(X,)`` and
        ``(Y,)``.
    initial_amplitudes
        Common data-derived decay-amplitude start, shape ``(R,2)`` in
        variance units.
    initial_noise_floor
        Common data-derived floor start, shape ``(R,)`` in variance units.
    max_iter
        Maximum fixed-rate amplitude SAGE sweeps per grid point.
    tol
        Relative total IS-loss stopping tolerance.
    cold_check_t60_s
        Optional additional T60 pair in seconds, shape ``(2,)``. Its nearest
        grid point is included in the independent-start diagnostics.

    Returns
    -------
    dict
        Profiled total IS loss ``(Y,X)``, fitted decay amplitudes
        ``(Y,X,R,2)``, fitted floors ``(Y,X,R)``, iteration/convergence maps,
        and continuation-versus-independent diagnostics. Every point is fit
        once from a serpentine continuation and once from the common pooled
        initialization; the lower-loss converged state defines the reported
        profile. Sparse corner/center/check-location gaps are returned as a
        compact traversal-dependence summary.
    """

    power = np.asarray(observed_power, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    short = np.asarray(short_t60_s, dtype=np.float64)
    long = np.asarray(long_t60_s, dtype=np.float64)
    amplitudes = np.asarray(initial_amplitudes, dtype=np.float64)
    floor = np.asarray(initial_noise_floor, dtype=np.float64)
    if power.ndim != 2 or times.shape != (power.shape[1],):
        raise ValueError("observed_power and times_s must have shapes (R,N) and (N,).")
    if amplitudes.shape != (power.shape[0], 2) or floor.shape != (power.shape[0],):
        raise ValueError("initial amplitudes/floor must have shapes (R,2) and (R,).")
    if (
        short.ndim != 1
        or long.ndim != 1
        or short.size < 2
        or long.size < 2
        or np.any(np.diff(short) <= 0.0)
        or np.any(np.diff(long) <= 0.0)
    ):
        raise ValueError("T60 grids must be strictly increasing and non-trivial.")

    n_long, n_short, n_rirs = long.size, short.size, power.shape[0]
    loss = np.empty((n_long, n_short))
    fitted_amplitudes = np.empty((n_long, n_short, n_rirs, 2))
    fitted_floor = np.empty((n_long, n_short, n_rirs))
    n_iter = np.empty((n_long, n_short), dtype=np.int64)
    converged = np.empty((n_long, n_short), dtype=bool)
    continuation_loss = np.empty((n_long, n_short))
    independent_loss = np.empty((n_long, n_short))
    selected_independent = np.empty((n_long, n_short), dtype=bool)
    common_start = np.concatenate((amplitudes, floor[:, np.newaxis]), axis=1)
    continuation = common_start.copy()

    for long_index, long_value in enumerate(long):
        short_indices = (
            range(n_short) if long_index % 2 == 0 else range(n_short - 1, -1, -1)
        )
        for short_index in short_indices:
            continuation_result = _profile_one_point(
                power,
                times,
                np.array([short[short_index], long_value]),
                continuation,
                max_iter=max_iter,
                tol=tol,
            )
            independent_result = _profile_one_point(
                power,
                times,
                np.array([short[short_index], long_value]),
                common_start,
                max_iter=max_iter,
                tol=tol,
            )
            continuation_value = continuation_result.objective_history[-1]
            independent_value = independent_result.objective_history[-1]
            use_independent = (
                independent_result.converged and not continuation_result.converged
            ) or (
                independent_result.converged == continuation_result.converged
                and independent_value < continuation_value
            )
            result = independent_result if use_independent else continuation_result
            continuation = result.amplitudes
            continuation_loss[long_index, short_index] = continuation_value
            independent_loss[long_index, short_index] = independent_value
            selected_independent[long_index, short_index] = use_independent
            loss[long_index, short_index] = result.objective_history[-1]
            fitted_amplitudes[long_index, short_index] = result.amplitudes[:, :2]
            fitted_floor[long_index, short_index] = result.amplitudes[:, 2]
            n_iter[long_index, short_index] = result.n_iter
            converged[long_index, short_index] = result.converged

    cold_index_list = [
        [0, 0],
        [0, n_short - 1],
        [n_long // 2, n_short // 2],
        [n_long - 1, 0],
        [n_long - 1, n_short - 1],
    ]
    if cold_check_t60_s is not None:
        check_pair = np.asarray(cold_check_t60_s, dtype=np.float64)
        if check_pair.shape != (2,) or not np.all(np.isfinite(check_pair)):
            raise ValueError("cold_check_t60_s must be finite with shape (2,).")
        cold_index_list.append(
            [
                int(np.argmin(np.abs(long - check_pair[1]))),
                int(np.argmin(np.abs(short - check_pair[0]))),
            ]
        )
    cold_indices = np.unique(np.asarray(cold_index_list, dtype=np.int64), axis=0)
    check_long = cold_indices[:, 0]
    check_short = cold_indices[:, 1]
    cold_loss = independent_loss[check_long, check_short]
    warm_check_loss = continuation_loss[check_long, check_short]

    return {
        "loss": loss,
        "amplitudes": fitted_amplitudes,
        "noise_floor": fitted_floor,
        "n_iter": n_iter,
        "converged": converged,
        "continuation_loss": continuation_loss,
        "independent_loss": independent_loss,
        "selected_independent": selected_independent,
        "cold_check_indices": cold_indices,
        "cold_check_loss": cold_loss,
        "cold_check_relative_gap": np.abs(cold_loss - warm_check_loss)
        / np.maximum(1.0, np.abs(cold_loss)),
    }


def _padded_histories(
    histories: list[list[np.ndarray]], max_iter: int
) -> np.ndarray:
    """Pad nested histories to shape ``(case,method,max_iter+1)``."""

    padded = np.full((len(histories), len(METHOD_KEYS), max_iter + 1), np.nan)
    for case_index, case in enumerate(histories):
        for method_index, history in enumerate(case):
            padded[case_index, method_index, : history.size] = history
    return padded


def _padded_trajectories(
    trajectories: list[np.ndarray], max_iter: int
) -> np.ndarray:
    """Pad T60 paths to shape ``(method,max_iter+1,2)`` in seconds."""

    padded = np.full((len(trajectories), max_iter + 1, 2), np.nan)
    for method_index, trajectory in enumerate(trajectories):
        padded[method_index, : trajectory.shape[0]] = trajectory
    return padded


def save_loss_figure(
    output_dir: Path,
    histories: list[list[np.ndarray]],
    *,
    show: bool,
) -> tuple[Path, Path]:
    """Save single-column PDF/PNG loss figures and return their paths."""

    with plt.rc_context({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7}):
        raw_figure, axis = plt.subplots(figsize=(3.45, 2.3), layout="constrained")
        for history, label, color in zip(histories[0], METHOD_LABELS, METHOD_COLORS, strict=True):
            axis.plot(history, label=label, color=color, linewidth=1.25)
        axis.set(xlabel="Complete component sweeps", ylabel="Total IS loss")
        axis.legend()
        raw_figure.savefig(output_dir / "total_is_loss.pdf")
        save_figure(raw_figure, output_dir, "total_is_loss.png", show=show, dpi=300)

    figure = plot_weight_order_objectives(
        histories,
        METHOD_LABELS,
        ("",),
        colors=METHOD_COLORS,
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
        saved.extend(save_loss_figure(output_dir, histories, show=show))
        surface_keys = {
            "surface_short_t60_s",
            "surface_long_t60_s",
            "surface_loss",
            "t60_trajectory_s",
        }
        if surface_keys.issubset(archive.files):
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
    """Run or replot the convergence/surface experiment and save outputs."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    if args.plot_results is not None:
        if not args.plot_results.is_file():
            raise FileNotFoundError(args.plot_results)
        figure_paths = _plot_saved_results(
            args.plot_results, output_dir, show=args.show
        )
        print(f"source={args.plot_results.resolve()}")
        for path in figure_paths:
            print(f"saved={path}")
        return

    times_s = np.arange(args.n_frames, dtype=np.float64) * HOP_S
    result = run_loss_comparison(
        np.random.default_rng(args.seed),
        times_s,
        n_rirs=args.n_rirs,
        max_iter=args.max_iter,
        tol=args.tol,
        rate_method=args.rate_method,
    )
    histories = result["histories"]
    trajectories = result["t60_trajectories_s"]
    if not isinstance(histories, list) or not isinstance(trajectories, list):
        raise RuntimeError("internal error: histories have the wrong type")

    short_t60_s = np.linspace(
        *SHORT_T60_RANGE_S, args.surface_grid_size, dtype=np.float64
    )
    long_t60_s = np.linspace(
        *LONG_T60_RANGE_S, args.surface_grid_size, dtype=np.float64
    )
    print(
        f"profiling {args.surface_grid_size}x{args.surface_grid_size} T60 grid",
        flush=True,
    )
    surface = profile_fixed_rate_loss_surface(
        np.asarray(result["observed_power"])[:, 0, :],
        times_s,
        short_t60_s,
        long_t60_s,
        np.asarray(result["init_amplitudes"])[:, 0, :],
        np.asarray(result["init_noise_floor"])[:, 0],
        max_iter=args.surface_max_iter,
        tol=args.surface_tol,
        cold_check_t60_s=TRUE_T60_S,
    )

    figure_paths = [
        *save_loss_figure(output_dir, histories, show=args.show),
        *save_surface_figure(
            output_dir,
            short_t60_s,
            long_t60_s,
            surface["loss"],
            trajectories,
            show=args.show,
        ),
    ]
    archive_path = output_dir / "loss_convergence_results.npz"
    np.savez_compressed(
        archive_path,
        seed=np.asarray(args.seed),
        times_s=times_s,
        method_keys=np.asarray(METHOD_KEYS),
        method_labels=np.asarray(METHOD_LABELS),
        method_weight_powers=np.asarray([0.0, 1.0, 2.0]),
        true_t60_s=TRUE_T60_S,
        objective_history=_padded_histories(histories, args.max_iter),
        t60_trajectory_s=_padded_trajectories(trajectories, args.max_iter),
        true_amplitudes=result["true_amplitudes"],
        true_noise_floor=result["true_noise_floor"],
        observed_power=result["observed_power"],
        exact_variance=result["exact_variance"],
        init_rates_per_s=result["init_rates_per_s"],
        init_amplitudes=result["init_amplitudes"],
        init_noise_floor=result["init_noise_floor"],
        initial_t60_s=result["initial_t60_s"],
        estimated_t60_s=result["estimated_t60_s"],
        estimated_rates_per_s=result["estimated_rates_per_s"],
        estimated_amplitudes=result["estimated_amplitudes"],
        estimated_noise_floor=result["estimated_noise_floor"],
        rate_method=np.asarray(args.rate_method),
        n_iter=result["n_iter"],
        converged=result["converged"],
        surface_short_t60_s=short_t60_s,
        surface_long_t60_s=long_t60_s,
        surface_loss=surface["loss"],
        surface_amplitudes=surface["amplitudes"],
        surface_noise_floor=surface["noise_floor"],
        surface_n_iter=surface["n_iter"],
        surface_converged=surface["converged"],
        surface_continuation_loss=surface["continuation_loss"],
        surface_independent_loss=surface["independent_loss"],
        surface_selected_independent=surface["selected_independent"],
        surface_cold_check_indices=surface["cold_check_indices"],
        surface_cold_check_loss=surface["cold_check_loss"],
        surface_cold_check_relative_gap=surface["cold_check_relative_gap"],
        max_iter=np.asarray(args.max_iter),
        tolerance=np.asarray(args.tol),
        surface_max_iter=np.asarray(args.surface_max_iter),
        surface_tolerance=np.asarray(args.surface_tol),
        dirichlet_alpha=np.asarray(DIRICHLET_ALPHA),
    )
    print(f"initial_t60_s={np.asarray(result['initial_t60_s']).tolist()}")
    print(
        "surface_converged="
        f"{int(np.sum(surface['converged']))}/{surface['converged'].size}"
    )
    print(
        "max_cold_start_relative_gap="
        f"{float(np.max(surface['cold_check_relative_gap'])):.3e}"
    )
    for path in figure_paths:
        print(f"saved={path}")
    print(f"saved={archive_path.resolve()}")


if __name__ == "__main__":
    main()
