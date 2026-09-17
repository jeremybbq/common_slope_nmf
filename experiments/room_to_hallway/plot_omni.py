"""Plots for room-to-hallway omnidirectional SAGE fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.room_to_hallway.omni import (
    LONG_SLOPE_RGB,
    SHORT_SLOPE_RGB,
    amplitude_mixture_rgb,
)

def _pyplot():
    """Import pyplot with a writable, process-safe cache directory."""

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


def _save_png_and_pdf(fig, output_path: str | Path) -> Path:
    """Save a Matplotlib figure as both PNG and PDF and return the PNG path."""

    output = Path(output_path)
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    return output


def plot_decay_times(
    result_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot two fitted energy decay times and the pooled linear estimate.

    ``result_path`` must contain frequency centers ``(F,)``, fitted energy
    ``T60`` values ``(F,2)``, and pooled linear energy ``T60`` values ``(F,)``,
    all in hertz and seconds. The figure is saved in PNG and PDF formats.
    """

    plt = _pyplot()
    with np.load(result_path, allow_pickle=False) as result:
        frequencies_hz = np.asarray(result["frequencies_hz"], dtype=np.float64)
        estimated_t60_s = np.asarray(result["estimated_t60_s"], dtype=np.float64)
        coarse_t60_s = np.asarray(result["coarse_t60_s"], dtype=np.float64)
    if estimated_t60_s.shape != (frequencies_hz.size, 2):
        raise ValueError("result must contain two-slope T60 values with shape (F,2).")
    if coarse_t60_s.shape != frequencies_hz.shape:
        raise ValueError("pooled linear T60 values must have shape (F,).")

    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.plot(
        frequencies_hz,
        estimated_t60_s[:, 1],
        color=LONG_SLOPE_RGB,
        linewidth=1.8,
        label="longer decay",
    )
    axis.plot(
        frequencies_hz,
        estimated_t60_s[:, 0],
        color=SHORT_SLOPE_RGB,
        linewidth=1.8,
        label="shorter decay",
    )
    axis.plot(
        frequencies_hz,
        coarse_t60_s,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="linear fit",
    )
    axis.set_xscale("log", base=10.0)
    axis.set_xlim(100.0, 10_000.0)
    axis.set_xticks(
        [100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0],
        labels=["100", "200", "500", "1k", "2k", "5k", "10k"],
    )
    axis.set_ylim(0.0, 2.0)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("T60 (s)")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    output = _save_png_and_pdf(fig, output_path)
    plt.close(fig)
    return output


def plot_amplitude_mixture(
    result_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    dynamic_range_db: float = 40.0,
    dark_background: bool = True,
) -> Path:
    """Plot both fitted slope amplitudes over position and frequency.

    ``result_path`` must contain two-slope amplitudes ``(R,F,2)`` and the
    corresponding receiver indices. The SOFA listener x coordinate is shown
    from 0 to 5 m, with the doorway marked at 2.5 m. A global power-dB
    reference makes all four source/visibility panels directly comparable.
    """

    plt = _pyplot()

    with np.load(result_path, allow_pickle=False) as result:
        amplitudes = np.asarray(result["estimated_amplitudes"], dtype=np.float64)
        frequencies_hz = np.asarray(result["frequencies_hz"], dtype=np.float64)
        receiver_indices = np.asarray(result["receiver_indices"], dtype=np.int64)
    if amplitudes.ndim != 3 or amplitudes.shape[2] != 2:
        raise ValueError("result must contain two-slope amplitudes with shape (R,F,2).")
    if amplitudes.shape[:2] != (receiver_indices.size, frequencies_hz.size):
        raise ValueError(
            "result amplitude axes do not match receivers and frequencies."
        )
    with np.load(metadata_path, allow_pickle=False) as metadata:
        condition_names = np.asarray(metadata["condition_names"])
        condition_index = np.asarray(metadata["rir_condition_index"])[receiver_indices]
        positions_m = np.asarray(metadata["listener_positions_m"])[receiver_indices, 0]

    reference_db = float(np.max(10.0 * np.log10(amplitudes)))
    mixed_rgb = amplitude_mixture_rgb(
        amplitudes,
        reference_db=reference_db,
        dynamic_range_db=dynamic_range_db,
        dark_background=dark_background,
    )
    display_names = {
        "room_los": "Source in Room, LOS",
        "room_no_los": "Source in Room, no LOS",
        "hallway_los": "Source in Hallway, LOS",
        "hallway_no_los": "Source in Hallway, no LOS",
    }

    fig = plt.figure(figsize=(11, 7.2))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        bottom=0.09,
        top=0.95,
        hspace=0.25,
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[index // 2, index % 2]) for index in range(4)]
    for condition, (name, axis) in enumerate(zip(condition_names, axes, strict=True)):
        mask = condition_index == condition
        order = np.argsort(positions_m[mask])
        x = positions_m[mask][order]
        image = np.transpose(mixed_rgb[mask][order], (1, 0, 2))
        axis.imshow(
            image,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(x[0], x[-1], frequencies_hz[0], frequencies_hz[-1]),
        )
        axis.axvline(2.5, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
        axis.set_title(display_names.get(str(name), str(name)))
        axis.set_xlim(0.0, 5.0)
        axis.set_yscale("log")
        axis.set_ylim(frequencies_hz[0], frequencies_hz[-1])
        axis.set_yticks(
            [250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0],
            labels=["250", "500", "1k", "2k", "4k", "8k"],
            minor=False,
        )
        axis.set_yticks([], minor=True)
        if condition >= 2:
            axis.set_xlabel("room ←          Positions (m)          → hallway")
        if condition % 2 == 0:
            axis.set_ylabel("Frequency (Hz)")

    output = _save_png_and_pdf(fig, output_path)
    plt.close(fig)
    return output



def plot_results(results_path: Path, *, show: bool = False, metadata: Path | None = None) -> list[Path]:
    """Plot decay times (and amplitude mixture when metadata is supplied)."""

    output_dir = results_path.parent
    prefix = results_path.name.removesuffix("_results.npz")
    saved = [plot_decay_times(results_path, output_dir / f"{prefix}_summary.png")]
    if metadata is not None:
        saved.append(
            plot_amplitude_mixture(
                results_path, metadata, output_dir / f"{prefix}_amplitude_mixture.png"
            )
        )
    if show:
        _pyplot().show()
    for path in saved:
        print(f"saved={path}")
    return saved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Saved NPZ archive or timestamped run directory.",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    results = args.results
    if results.is_dir():
        metadata = args.metadata
        if metadata is None:
            candidate = results / "analysis_metadata.npz"
            metadata = candidate if candidate.is_file() else None
        archives = sorted(
            path
            for path in results.glob("*_results.npz")
            if path.is_file()
        )
        if not archives:
            raise SystemExit(f"no *_results.npz files in {results}")
        for archive in archives:
            plot_results(archive, show=args.show, metadata=metadata)
        return
    plot_results(results, show=args.show, metadata=args.metadata)


if __name__ == "__main__":
    main()
