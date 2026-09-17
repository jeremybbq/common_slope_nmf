
"""Plots for the synthetic amplitude-identifiability experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.synthetic.amplitude_identifiability import (
    AmplitudeInferenceRun,
    COMPONENT_LABELS,
    load_run,
)

def _pyplot():
    """Import pyplot with a writable cache directory."""

    import os
    import tempfile

    config_dir = Path(tempfile.gettempdir()) / "common_slope_nmf_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Serif",
            "mathtext.it": "Liberation Serif:italic",
            "mathtext.bf": "Liberation Serif:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def plot_run(run: AmplitudeInferenceRun, output_dir: Path, *, show: bool = False) -> Path:
    """Plot empirical relative-error distributions for both decay amplitudes."""

    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
    colors = ("#4385BE", "#E8705F")
    rng = np.random.default_rng(12345)
    raw_error_db = 10.0 * np.log10(
        run.estimated_amplitudes[:, :, :2] / run.true_amplitudes[:, None, :2]
    )
    finite_lower = max(-80.0, float(np.quantile(raw_error_db, 0.005)))
    finite_upper = min(80.0, float(np.quantile(raw_error_db, 0.995)))
    lower = min(-10.0, np.floor(finite_lower / 5.0) * 5.0)
    upper = max(10.0, np.ceil(finite_upper / 5.0) * 5.0)

    x = np.arange(run.ratio_db.size, dtype=np.float64)
    ratio_labels = [f"{value:+g}" + ("" if ok else "*")
                    for value, ok in zip(run.ratio_db, run.converged, strict=True)]
    for component_index, (axis, label, color) in enumerate(
        zip(axes, COMPONENT_LABELS[:2], colors, strict=True)
    ):
        errors = raw_error_db[:, :, component_index]
        displayed = np.clip(errors, lower, upper)
        for case_index in range(run.ratio_db.size):
            jitter = rng.uniform(-0.16, 0.16, size=errors.shape[1])
            axis.scatter(
                x[case_index] + jitter,
                displayed[case_index],
                s=11,
                color=color,
                alpha=0.30,
                linewidths=0.0,
            )
        q05, median, q95 = np.quantile(errors, [0.05, 0.5, 0.95], axis=1)
        median_plot = np.clip(median, lower, upper)
        axis.errorbar(
            x,
            median_plot,
            yerr=np.vstack(
                (
                    median_plot - np.clip(q05, lower, upper),
                    np.clip(q95, lower, upper) - median_plot,
                )
            ),
            fmt="o",
            color="black",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=5.5,
            capsize=3,
            linewidth=1.3,
            label="median and 5–95% interval",
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(label.capitalize())
        axis.set_xticks(x, labels=ratio_labels)
        axis.set_xlabel(r"True $10\log_{10}(A_\mathrm{fast}/A_\mathrm{slow})$ (dB)")
        axis.set_ylim(lower, upper)
        axis.grid(axis="y", alpha=0.2)
        clipped = np.count_nonzero((errors < lower) | (errors > upper))
        if clipped:
            axis.text(
                0.02,
                0.03,
                f"{clipped} estimates clipped at plot limits",
                transform=axis.transAxes,
                fontsize=9,
                va="bottom",
            )
    axes[0].set_ylabel(r"Amplitude error $10\log_{10}(\hat A/A)$ (dB)")
    axes[1].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "Known-rate amplitude inference from "
        f"{run.observed_power.shape[1]} matched complex-Gaussian STFTs"
    )
    if not np.all(run.converged):
        figure.text(0.5, 0.01, "* Loss/stability criteria not both met", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 1))

    output_path = output_dir / "amplitude_inference_relative_error.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def main() -> None:
    """Load a saved ordinary-SAGE NPZ and write its relative-error figure."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run = load_run(args.results)
    output_dir = args.results.parent
    print(f"figure={plot_run(run, output_dir, show=args.show)}")


if __name__ == "__main__":
    main()
