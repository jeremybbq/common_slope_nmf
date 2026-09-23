
"""Plots comparing Georg baselines with room-to-hallway SAGE fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.room_to_hallway.georg_benchmarks import N_SLOPES
from experiments.room_to_hallway.omni import LONG_SLOPE_RGB, SHORT_SLOPE_RGB
from experiments.room_to_hallway.plot_omni import _pyplot, _save_png_and_pdf

def plot_t60_comparison(
    proposed_result_path: str | Path,
    benchmark_result_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Overlay bandwise Georg baselines on the proposed-method T60 curves.

    Parameters
    ----------
    proposed_result_path
        NPZ containing ``frequencies_hz (F,)``, ``estimated_t60_s (F,2)``,
        and ``coarse_t60_s (F,)`` in hertz and seconds.
    benchmark_result_path
        NPZ containing ``band_centers_hz (B,)``, independent DecayFitNet
        estimates ``(R,B,2)``, and CommonSlopeAnalysis estimates ``(B,2)``.
    output_path
        Destination PNG path. A matching PDF is saved beside it.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG figure.

    Notes
    -----
    Each method's two slopes are ordered by increasing energy T60. The
    DecayFitNet estimates across receivers are shown as split violins: the
    short-slope density occupies the left half and the long-slope density the
    right half at each band center. CommonSlopeAnalysis has one shared pair per
    frequency band.
    """

    with np.load(proposed_result_path, allow_pickle=False) as proposed:
        frequencies_hz = np.asarray(
            proposed["frequencies_hz"], dtype=np.float64
        )
        proposed_t60_s = np.asarray(
            proposed["estimated_t60_s"], dtype=np.float64
        )
        coarse_t60_s = np.asarray(
            proposed["coarse_t60_s"], dtype=np.float64
        )
    with np.load(benchmark_result_path, allow_pickle=False) as benchmark:
        band_centers_hz = np.asarray(
            benchmark["band_centers_hz"], dtype=np.float64
        )
        decayfitnet_t60_s = np.asarray(
            benchmark["decayfitnet_t60_s"], dtype=np.float64
        )
        common_t60_s = np.asarray(
            benchmark["common_slope_t60_s"], dtype=np.float64
        )

    if proposed_t60_s.shape != (frequencies_hz.size, N_SLOPES):
        raise ValueError("proposed T60 values must have shape (F,2).")
    if coarse_t60_s.shape != frequencies_hz.shape:
        raise ValueError("coarse T60 values must have shape (F,).")
    if decayfitnet_t60_s.ndim != 3 or decayfitnet_t60_s.shape[1:] != (
        band_centers_hz.size,
        N_SLOPES,
    ):
        raise ValueError("DecayFitNet T60 values must have shape (R,B,2).")
    if common_t60_s.shape != (band_centers_hz.size, N_SLOPES):
        raise ValueError("common-slope T60 values must have shape (B,2).")
    if np.any(frequencies_hz <= 0.0) or np.any(band_centers_hz <= 0.0):
        raise ValueError("all plotted frequencies must be positive.")

    plt = _pyplot()
    with plt.rc_context({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.25, 'lines.markersize': 4, 'grid.linewidth': 0.5, 'pdf.fonttype': 42}):
        fig, axis = plt.subplots(figsize=(3.45, 2.3))
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.34)
        slope_colors = ("#109cff", "#ff9b10")
        slope_names = ("short", "long")

        for slope_index, (color, slope_name) in enumerate(
            zip(slope_colors, slope_names, strict=True)
        ):
            axis.plot(
                frequencies_hz,
                proposed_t60_s[:, slope_index],
                color=color,
                linewidth=1.25,
                zorder=2,
                label=f"proposed: {slope_name}",
            )
        axis.plot(
            frequencies_hz,
            coarse_t60_s,
            color="black",
            linestyle="--",
            linewidth=1.3,
            alpha=0.75,
            zorder=1,
            label="Log-linear fit",
        )

        violin_widths = 0.24 * band_centers_hz
        for slope_index, (color, slope_name) in enumerate(
            zip(slope_colors, slope_names, strict=True)
        ):
            violin = axis.violinplot(
                [decayfitnet_t60_s[:, band_index, slope_index]
                 for band_index in range(band_centers_hz.size)],
                positions=band_centers_hz,
                widths=violin_widths,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                points=100,
                bw_method="scott",
            )
            for band_index, body in enumerate(violin["bodies"]):
                center = band_centers_hz[band_index]
                vertices = body.get_paths()[0].vertices
                if slope_index == 0:
                    vertices[:, 0] = np.minimum(vertices[:, 0], center)
                else:
                    vertices[:, 0] = np.maximum(vertices[:, 0], center)
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_linewidth(1.1)
                body.set_alpha(0.32)
                body.set_zorder(2.5)
                if band_index == 0:
                    body.set_label(f"DecayFitNet: {slope_name}")
            axis.scatter(
                band_centers_hz,
                common_t60_s[:, slope_index],
                marker="x",
                color=color,
                s=20.0,
                linewidths=1.25,
                zorder=4,
                label=f"CommonSlopeAnalysis: {slope_name}",
            )

        upper_values = np.concatenate(
            (
                proposed_t60_s.ravel(),
                coarse_t60_s.ravel(),
                common_t60_s.ravel(),
                decayfitnet_t60_s.ravel(),
            )
        )
        finite_upper = upper_values[np.isfinite(upper_values)]
        if finite_upper.size == 0:
            raise ValueError("the T60 result files contain no finite values.")
        y_max = max(2.0, 1.08 * float(np.max(finite_upper)))

        axis.set_xscale("log", base=10.0)
        from matplotlib.ticker import NullFormatter

        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlim(200.0, 10_000.0)
        axis.set_xticks(
            [200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0],
            labels=["200", "500", "1k", "2k", "5k", "10k"],
        )
        axis.set_ylim(0.0, y_max)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("RT60 (s)")
        axis.grid(alpha=0.2)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_handles = [
            Line2D([], [], color="black", linestyle="--", linewidth=1.3,
                   label="Linear fit"),
            Line2D([], [], color="black", linewidth=1.25,
                   label="NMF/CW-SAGE"),
            Patch(facecolor="black", edgecolor="black", alpha=0.32,
                  label="DecayFitNet"),
            Line2D([], [], color="black", linestyle="none", marker="x",
                   markersize=4.5, markeredgewidth=1.25,
                   label="CommonSlopeAnalysis"),
        ]
        axis.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            frameon=False,
            fontsize=7,
            handlelength=1.5,
            columnspacing=0.8,
            labelspacing=0.25,
        )
        output = _save_png_and_pdf(fig, output_path)
        plt.close(fig)
        return output


def main() -> None:
    """Overlay Georg baselines on a proposed two-slope T60 result."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    path = plot_t60_comparison(args.proposed, args.benchmark, args.output)
    print(f"saved={path}")
    if args.show:
        plt = _pyplot()
        plt.show()


if __name__ == "__main__":
    main()
