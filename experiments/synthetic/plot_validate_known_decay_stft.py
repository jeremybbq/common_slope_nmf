"""Experiment plots loaded from saved NPZ results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from experiments._run_output import save_figure

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


def plot_results(results_path: Path, *, show: bool = False) -> list[Path]:
    """Load a known-decay STFT NPZ and write its diagnostic figures."""

    from scipy.stats import gamma

    from experiments.synthetic.validate_known_decay_stft import DECAY_AMPLITUDE

    output_dir = results_path.parent
    with np.load(results_path, allow_pickle=False) as payload:
        times_s = payload["times_s"]
        labels = [str(label) for label in payload["labels"]]
        slugs = [str(slug) for slug in payload["slugs"]]
        floors = payload["floors"]
        observed = payload["observed_power"]
        fitted = payload["fitted_variance"]
        decay_amplitudes = payload["decay_amplitudes"]
        floor_amplitudes = payload["floor_amplitudes"]
        histories = [
            row[np.isfinite(row)] for row in payload["objective_history"]
        ]
        truth = float(payload["decay_amplitude"])
        figures = []
        for index, (label, slug, floor) in enumerate(zip(labels, slugs, floors, strict=True)):
            lower_db = -120.0 if floor == 0.0 else 10.0 * np.log10(floor) - 30.0
            figures.append(
                (
                    plot_known_decay_spectrogram(
                        observed[index],
                        fitted[index],
                        times_s,
                        title=label,
                        reference_power=truth,
                        lower_db=lower_db,
                    ),
                    f"known_decay_spectrogram_{slug}.png",
                )
            )
        grid = np.linspace(0.65, 1.45, 500)
        figures.extend(
            [
                (
                    plot_objective_gaps(histories, labels),
                    "known_decay_sage_convergence.png",
                ),
                (
                    plot_amplitude_estimator_distributions(
                        list(decay_amplitudes),
                        labels,
                        truth=truth,
                        exact_grid=grid,
                        exact_density=gamma.pdf(
                            grid, a=times_s.size, scale=truth / times_s.size
                        ),
                        exact_label="Exact no-floor Gamma law",
                    ),
                    "known_decay_estimator_distributions.png",
                ),
                (
                    plot_binwise_amplitudes(
                        list(decay_amplitudes), labels, truth=truth, decibels=False
                    ),
                    "known_decay_decay_amplitudes_linear.png",
                ),
                (
                    plot_binwise_amplitudes(
                        list(decay_amplitudes), labels, truth=truth, decibels=True
                    ),
                    "known_decay_decay_amplitudes_db.png",
                ),
                (
                    plot_binwise_noise_floors(
                        [floor_amplitudes[index] for index, floor in enumerate(floors) if floor > 0.0],
                        [float(floor) for floor in floors if floor > 0.0],
                        [label for label, floor in zip(labels, floors, strict=True) if floor > 0.0],
                    ),
                    "known_decay_noise_floor_estimates_db.png",
                ),
            ]
        )
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
