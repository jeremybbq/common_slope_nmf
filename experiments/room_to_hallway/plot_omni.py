
"""Plots for room-to-hallway omnidirectional SAGE fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common_slope_nmf import t60_to_rate
from experiments.room_to_hallway.omni import (
    FLOOR_RHO_RGB,
    LONG_SLOPE_RGB,
    RHO_TARGET_FREQUENCY_HZ,
    SHORT_SLOPE_RGB,
    amplitude_mixture_rgb,
    rho_mixture_rgb,
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
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
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

    with plt.rc_context({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.25, 'lines.markersize': 4, 'grid.linewidth': 0.5, 'pdf.fonttype': 42}):
        fig, axis = plt.subplots(figsize=(3.45, 2.3), layout="constrained")
        axis.plot(
            frequencies_hz,
            estimated_t60_s[:, 1],
            color=LONG_SLOPE_RGB,
            linewidth=1.25,
            label="longer decay",
        )
        axis.plot(
            frequencies_hz,
            estimated_t60_s[:, 0],
            color=SHORT_SLOPE_RGB,
            linewidth=1.25,
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
        output = _save_png_and_pdf(fig, output_path)
        plt.close(fig)
        return output


def plot_final_rho_space_time(
    result_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    target_frequency_hz: float,
) -> Path:
    """Plot final rho over receiver position and elapsed time at one bin.

    The fitted archive supplies final two-slope amplitudes/floors and RTs;
    metadata supplies frame times and room-to-hallway positions. The selected
    bin is nearest ``target_frequency_hz``. Room/LOS and Hallway/LOS panels use blue for
    ``rho_1``, orange for ``rho_2``, and dark gray for ``rho_0``.
    """

    if not np.isfinite(target_frequency_hz) or target_frequency_hz <= 0.0:
        raise ValueError("target_frequency_hz must be finite and positive.")
    plt = _pyplot()
    with np.load(result_path, allow_pickle=False) as result:
        t60_s = np.asarray(result["estimated_t60_s"], dtype=np.float64)
        amplitudes = np.asarray(result["estimated_amplitudes"], dtype=np.float64)
        floors = np.asarray(result["estimated_noise_floor"], dtype=np.float64)
        frequencies_hz = np.asarray(result["frequencies_hz"], dtype=np.float64)
        receiver_indices = np.asarray(result["receiver_indices"], dtype=np.int64)
    if (
        t60_s.ndim != 2 or t60_s.shape[1] != 2
        or amplitudes.shape != (receiver_indices.size, frequencies_hz.size, 2)
        or floors.shape != amplitudes.shape[:2]
        or not all(np.all(np.isfinite(x)) for x in (t60_s, amplitudes, floors))
        or np.any(t60_s <= 0.0) or np.any(amplitudes <= 0.0) or np.any(floors <= 0.0)
    ):
        raise ValueError("result must contain finite positive two-slope fit parameters.")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        times_s = np.asarray(metadata["times_s"], dtype=np.float64)
        condition_names = np.asarray(metadata["condition_names"])
        condition_index = np.asarray(metadata["rir_condition_index"])[receiver_indices]
        positions_m = np.asarray(metadata["listener_positions_m"])[receiver_indices, 0]
    if times_s.ndim != 1 or times_s.size < 2 or np.any(np.diff(times_s) <= 0.0):
        raise ValueError("metadata times_s must be a strictly increasing 1-D array.")

    frequency_index = int(np.argmin(np.abs(frequencies_hz - target_frequency_hz)))
    atoms = np.exp(
        -t60_to_rate(t60_s[frequency_index])[:, np.newaxis] * times_s[np.newaxis, :]
    )
    components = amplitudes[:, frequency_index, :, np.newaxis] * atoms[np.newaxis]
    total = np.sum(components, axis=1) + floors[:, frequency_index, np.newaxis]
    rho = np.concatenate(
        (
            components / total[:, np.newaxis, :],
            (floors[:, frequency_index, np.newaxis] / total)[:, np.newaxis, :],
        ), axis=1,
    ).transpose(0, 2, 1)
    colors = rho_mixture_rgb(rho)

    display_names = {
        "room_los": "Source in Room, LOS",
        "hallway_los": "Source in Hallway, LOS",
    }
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8}):
        figure = plt.figure(figsize=(3.45, 3.1))
        grid = figure.add_gridspec(2, 1, left=0.14, right=0.98, bottom=0.19, top=0.95, hspace=0.26)
        axes = [figure.add_subplot(grid[index, 0]) for index in range(2)]
        for panel, (name, axis) in enumerate(zip(display_names, axes, strict=True)):
            condition = list(condition_names).index(name)
            mask = condition_index == condition
            order = np.argsort(positions_m[mask])
            x = positions_m[mask][order]
            axis.imshow(
                np.transpose(colors[mask][order], (1, 0, 2)), origin="lower",
                aspect="auto", interpolation="nearest",
                extent=(x[0], x[-1], times_s[0], times_s[-1]),
            )
            axis.axvline(2.5, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
            axis.set(title=display_names[name], xlim=(0.0, 5.0), ylim=(times_s[0], times_s[-1]))
            axis.title.set_fontsize(7)
            axis.set_xticks([0.0, 2.5, 5.0], labels=["0", "2.5", "5"])
            if panel == 0:
                axis.set_ylabel("Elapsed time (s)")
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_ylabel("Elapsed time (s)")
        figure.text(0.58, 0.1, r"Room $\leftarrow\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\rightarrow$ Hallway", ha="center", fontsize=7)
        figure.text(0.58, 0.1, "Receiver position (m)", ha="center", fontsize=8)
        figure.legend(
            handles=[
                plt.Line2D([], [], color=SHORT_SLOPE_RGB, linewidth=3, label=r"$\rho_1$ fast decay"),
                plt.Line2D([], [], color=LONG_SLOPE_RGB, linewidth=3, label=r"$\rho_2$ slow decay"),
                plt.Line2D([], [], color=FLOOR_RHO_RGB, linewidth=3, label=r"$\rho_0$ floor"),
            ],
            loc="lower center", bbox_to_anchor=(0.58, 0), ncols=3, frameon=False, fontsize=7,
        )
        output = Path(output_path).with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300)
        figure.savefig(output.with_suffix(".pdf"))
        plt.close(figure)
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
    reference across all four conditions is retained. Three panels show room
    without LOS, hallway with LOS, and hallway without LOS, followed by a
    half-width bivariate key. Returns the PNG path; a fixed-size PDF is also saved.
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

    fig = plt.figure(figsize=(7.10, 1.7))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1, 1, 1, 0.5),
        left=0.07,
        right=0.93,
        bottom=0.25,
        top=0.86,
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    selected_names = ("room_no_los", "hallway_los", "hallway_no_los")
    for panel, (name, axis) in enumerate(zip(selected_names, axes, strict=True)):
        condition = list(condition_names).index(name)
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
        axis.set_title(display_names[name], fontsize=8)
        axis.set_xlim(0.0, 5.0)
        axis.set_yscale("log")
        axis.set_ylim(frequencies_hz[0], frequencies_hz[-1])
        axis.set_yticks(
            [250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0],
            labels=["250", "500", "1k", "2k", "4k", "8k"],
            minor=False,
        )
        axis.set_yticks([], minor=True)
        axis.tick_params(axis="both", labelsize=8)
        axis.set_xticks([0, 2.5, 5], labels=["0", "2.5", "5"])
        if panel == 0:
            axis.set_ylabel("Frequency (Hz)", fontsize=9)
        else:
            axis.tick_params(axis="y", labelleft=False)

    fig.text(
        (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2,
        0.025,
        "Receiver position (m)",
        ha="center",
        fontsize=9,
    )
    key_axis = fig.add_subplot(grid[0, 3])
    levels_db = np.linspace(-dynamic_range_db, 0.0, 151)
    short_db, long_db = np.meshgrid(levels_db, levels_db)
    key_amplitudes = 10.0 ** (
        (np.stack((short_db, long_db), axis=-1) + reference_db) / 10.0
    )
    key_axis.imshow(
        amplitude_mixture_rgb(
            key_amplitudes,
            reference_db=reference_db,
            dynamic_range_db=dynamic_range_db,
            dark_background=dark_background,
        ),
        origin="lower",
        extent=(-dynamic_range_db, 0, -dynamic_range_db, 0),
        aspect="auto",
    )
    key_axis.set_title("Amplitude (dB)", fontsize=8)
    key_axis.set_xticks([-dynamic_range_db, 0])
    key_axis.set_yticks([-dynamic_range_db, 0])
    key_axis.tick_params(labelsize=8)
    key_axis.tick_params(axis="y", pad=1, length=2)
    key_axis.yaxis.tick_right()
    key_axis.yaxis.set_label_position("right")
    key_axis.set_xlabel("Fast decay", fontsize=8)
    key_axis.set_ylabel("Slow decay", fontsize=8, labelpad=1)
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return output


def plot_run(
    run_dir: Path,
    *,
    rho_frequency_hz: float = RHO_TARGET_FREQUENCY_HZ,
    show: bool = False,
) -> list[Path]:
    """Write two-slope figures for every configuration NPZ in a run directory."""

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    metadata_path = run_dir / "analysis_metadata.npz"
    saved: list[Path] = []
    for result_path in sorted(run_dir.glob("*_results.npz")):
        with np.load(result_path, allow_pickle=False) as archive:
            n_components = int(np.asarray(archive["n_components"]))
        if n_components != 2:
            continue
        prefix = result_path.name.removesuffix("_results.npz")
        saved.append(
            plot_decay_times(result_path, run_dir / f"{prefix}_summary.png")
        )
        if metadata_path.is_file():
            saved.append(
                plot_final_rho_space_time(
                    result_path,
                    metadata_path,
                    run_dir / f"{prefix}_rho_space_time",
                    target_frequency_hz=rho_frequency_hz,
                )
            )
    if show:
        plt = _pyplot()
        plt.show()
    return saved


def main() -> None:
    """Load a saved room-to-hallway run directory and write its figures."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--rho-frequency-hz",
        type=float,
        default=RHO_TARGET_FREQUENCY_HZ,
        help="Nearest fitted frequency used for the final rho space-time map.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.rho_frequency_hz) or args.rho_frequency_hz <= 0.0:
        parser.error("--rho-frequency-hz must be finite and positive")
    paths = plot_run(
        args.results,
        rho_frequency_hz=args.rho_frequency_hz,
        show=args.show,
    )
    print(f"source={args.results.resolve()}")
    for path in paths:
        print(f"saved={path}")


if __name__ == "__main__":
    main()
