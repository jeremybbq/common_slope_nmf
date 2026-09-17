"""Plots for coupled-room SAGE fits."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.coupled_rooms.fit import (
    N_COMPONENTS,
    PUBLISHED_OCTAVE_FREQUENCIES_HZ,
    PUBLISHED_OCTAVE_T60_S,
)
from experiments._run_output import save_figure


def plot_summary(results_path: Path, *, show: bool = False) -> Path:
    """Load a coupled-room NPZ and write the three-panel summary figure."""

    with np.load(results_path, allow_pickle=False) as payload:
        frequencies_hz = payload["frequencies_hz"]
        t60_s = payload["estimated_t60_s"]
        initial_t60_s = payload["fitted_initial_t60_s"]
        amplitudes = payload["estimated_amplitudes"]
        objective = payload["objective_history"]
    initial_objective = objective[:, 0]
    final_objective = np.array(
        [row[np.isfinite(row)][-1] for row in objective]
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for component_index in range(N_COMPONENTS):
        axes[0].semilogx(
            frequencies_hz,
            t60_s[:, component_index],
            ".-",
            label=f"estimated component {component_index + 1}",
        )
        axes[0].semilogx(
            PUBLISHED_OCTAVE_FREQUENCIES_HZ,
            PUBLISHED_OCTAVE_T60_S[:, component_index],
            "x",
            color=f"C{component_index}",
            markersize=7,
            markeredgewidth=1.5,
            label=(
                "published octave-band common slopes"
                if component_index == 0
                else None
            ),
        )
    common_initialization = np.allclose(
        initial_t60_s,
        initial_t60_s[:, :1],
        rtol=0.0,
        atol=0.0,
        equal_nan=False,
    )
    if common_initialization:
        axes[0].semilogx(
            frequencies_hz,
            initial_t60_s[:, 0],
            "k:",
            linewidth=1.5,
            label="common initial T60",
        )
    axes[0].set_ylabel("Energy T60 (s)")
    axes[0].set_title(
        "Weighted no-floor pseudo-SAGE and published octave-band results"
    )
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)

    median_amplitude_db = np.median(
        10.0 * np.log10(np.maximum(amplitudes, np.finfo(float).tiny)), axis=0
    )
    for component_index in range(N_COMPONENTS):
        axes[1].semilogx(
            frequencies_hz,
            median_amplitude_db[:, component_index],
            ".-",
            label=f"component {component_index + 1}",
        )
    axes[1].set_ylabel("Median unit-origin variance (dB)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].semilogx(
        frequencies_hz, final_objective / initial_objective, ".-", label="IS ratio"
    )
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[2].set(xlabel="Frequency (Hz)", ylabel="Final / initial IS objective")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    prefix = results_path.name.removesuffix("_results.npz")
    path = save_figure(fig, results_path.parent, f"{prefix}_summary.png", show=show)
    print(f"saved={path}")
    if show:
        plt.show()
    return path


def plot_results(results_path: Path, *, show: bool = False) -> Path:
    return plot_summary(results_path, show=show)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_results(args.results, show=args.show)


if __name__ == "__main__":
    main()
