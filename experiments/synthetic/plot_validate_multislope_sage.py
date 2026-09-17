"""Experiment plots loaded from saved NPZ results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from experiments._run_output import save_figure

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


def _save_animation(animation, figure, output_dir: Path, filename: str, show: bool) -> Path:
    path = output_dir / filename
    animation.save(path, writer=PillowWriter(fps=2), dpi=120)
    if not show:
        plt.close(figure)
    return path.resolve()


def plot_results(results_path: Path, *, show: bool = False) -> list[Path]:
    """Load a two-slope SAGE NPZ and write its diagnostic figures."""

    from experiments.synthetic.validate_multislope_sage import (
        AMPLITUDE_MEAN_DB,
        AMPLITUDE_STD_DB,
        N_COMPONENTS,
    )

    output_dir = results_path.parent
    with np.load(results_path, allow_pickle=False) as payload:
        estimator_label = str(payload["estimator_label"])
        slug = str(payload["slug"])
        times_s = payload["times_s"]
        observed = payload["observed_power"]
        true_variance = payload["true_variance"]
        fitted = payload["fitted_variance"]
        estimated_amplitudes = payload["estimated_amplitudes"]
        estimated_t60_s = payload["estimated_t60_s"]
        figures = [
            (plot_spatial_map(observed[:, 0], times_s, title="Observed two-slope power over time and RIR index"), f"{slug}_spatial_observed_power_db.png"),
            (plot_spatial_map(fitted[:, 0], times_s, title=f"{estimator_label} fitted two-slope variance over time and RIR index"), f"{slug}_spatial_fitted_variance_db.png"),
            (plot_variance_maps(times_s, (observed[:, 0], true_variance[:, 0], fitted[:, 0]), ("Observed instantaneous power $Y$", "Generating variance $V$", r"Estimated variance $\widehat V$"), vmax_db=0.0, rir_label=r"RIR index $r$"), f"{slug}_observed_true_estimated_variance.png"),
            (plot_amplitude_distributions(payload["true_amplitudes_db"], estimated_amplitudes, payload["true_t60_s"][0], estimated_t60_s, estimator_label=estimator_label, generating_mean_db=AMPLITUDE_MEAN_DB[0], generating_std_db=AMPLITUDE_STD_DB), f"{slug}_amplitude_distributions_db.png"),
            (plot_joint_amplitudes(payload["true_amplitudes_db"], estimated_amplitudes, estimator_label=estimator_label), f"{slug}_joint_amplitudes_db.png"),
            (plot_parameter_estimates(payload["true_amplitudes_db"], estimated_amplitudes, payload["true_noise_floor_db"], payload["estimated_noise_floor"][:, 0]), f"{slug}_parameters_by_rir_db.png"),
            (plot_objective_history(payload["objective_history"], title=f"Two-slope {estimator_label} raw observed objective"), f"{slug}_raw_objective.png"),
            (plot_t60_trajectories(payload["t60_history_s"], true_t60_s=payload["true_t60_s"][0], title="Shared decay-time trajectories"), f"{slug}_t60_trajectories.png"),
        ]
        saved = [save_figure(figure, output_dir, filename, show=show) for figure, filename in figures]
        if "scaled_total_error" in payload.files:
            decay_order = np.argsort(estimated_t60_s)
            channel_order = np.concatenate((decay_order, [N_COMPONENTS]))
            saved.append(
                _save_animation(
                    *reversed(make_scaled_error_animation(
                        payload["scaled_total_error"][:, :, 0, :],
                        payload["update_sweep_indices"],
                        times_s,
                    )),
                    output_dir,
                    f"{slug}_scaled_error_evolution.gif",
                    show,
                )
            )
            saved.append(
                _save_animation(
                    *reversed(make_component_weight_animation(
                        payload["component_weight"][:, :, 0, channel_order, :],
                        payload["update_sweep_indices"],
                        times_s,
                    )),
                    output_dir,
                    f"{slug}_component_weight_evolution.gif",
                    show,
                )
            )
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
