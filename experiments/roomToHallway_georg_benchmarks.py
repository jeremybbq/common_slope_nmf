"""Run fixed-K=2 DecayFitNet and CommonSlopeAnalysis room benchmarks.

The four raw v1.3 Meeting Room to Hallway SOFA files are analyzed in six
octave-like bands. DecayFitNet supplies independent two-slope parameters for
every RIR. Georg Götz's common-slope procedure then clusters those decay times
into two shared values per band and refits per-RIR amplitudes on Schroeder EDCs.
The external DecayFitNet ONNX model is loaded from its own checkout and is not
copied into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from common_slope_nmf import load_sofa_channel
from common_slope_nmf.georg_baselines import (
    DECAYFITNET_OUTPUT_SIZE,
    GEORG_BANDWIDTH_FACTOR,
    ExternalDecayFitNet,
    band_energy_decay_curves,
    determine_common_decay_times,
    edc_to_equivalent_rir_power_amplitudes,
    fit_common_slope_edcs,
)
from experiments._run_output import create_run_output_dir
from experiments.roomToHallway_omni import (
    CONDITION_FILES,
    DATASET_DIR,
    LONG_SLOPE_RGB,
    SHORT_SLOPE_RGB,
    _pyplot,
    _save_png_and_pdf,
)


SAMPLE_RATE_HZ = 48_000.0
BAND_CENTERS_HZ = np.array(
    [250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0]
)
N_SLOPES = 2
HISTOGRAM_RESOLUTION_S = 0.05
CLUSTER_SEED = 42
FILTER_ORDER = 5
FILTER_TAIL_DISCARD_FRACTION = 0.005
DECAYFITNET_EDC_TAIL_DISCARD_FRACTION = 0.05
DEFAULT_MODEL_DIR = Path.home() / "Documents" / "DecayFitNet" / "model"
PARTIAL_KEYS = (
    "decayfitnet_t60_s",
    "decayfitnet_amplitudes_normalized_edc",
    "decayfitnet_amplitudes_absolute_edc_energy",
    "decayfitnet_equivalent_rir_power_amplitudes",
    "decayfitnet_noise_normalized_per_sample",
    "decayfitnet_noise_absolute_per_sample",
    "decayfitnet_edc_normalization_energy",
    "decayfitnet_mse_db2",
    "common_slope_t60_s",
    "common_slope_amplitudes_normalized_edc",
    "common_slope_amplitudes_absolute_edc_energy",
    "common_slope_equivalent_rir_power_amplitudes",
    "common_slope_noise_normalized_edc_origin",
    "common_slope_noise_absolute_edc_energy",
    "common_slope_equivalent_noise_power_per_sample",
    "common_slope_mse_db2",
    "common_slope_fit_success",
    "common_slope_cluster_sizes",
)


def load_room_transition_omni(
    dataset_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Load the four raw SOFA conditions as one ordered omni RIR batch.

    Returns a dictionary containing RIR samples ``(R,N)``, listener/source
    positions ``(R,3)`` in metres, condition and within-file measurement
    indices ``(R,)``, condition names, filenames, and the scalar sample rate.
    """

    directory = Path(dataset_dir)
    rirs: list[np.ndarray] = []
    listener_positions: list[np.ndarray] = []
    source_positions: list[np.ndarray] = []
    condition_indices: list[np.ndarray] = []
    measurement_indices: list[np.ndarray] = []
    filenames: list[str] = []
    expected_samples = None
    for condition_index, (condition_name, filename) in enumerate(CONDITION_FILES):
        path = directory / filename
        rir, listener, source, info = load_sofa_channel(
            path, channel_index=0
        )
        if not np.isclose(info.sample_rate_hz, SAMPLE_RATE_HZ):
            raise ValueError(f"{filename} does not use the expected 48 kHz rate.")
        if expected_samples is None:
            expected_samples = info.n_samples
        elif info.n_samples != expected_samples:
            raise ValueError("all SOFA files must have equal RIR lengths.")
        rirs.append(rir)
        listener_positions.append(listener)
        source_positions.append(source)
        condition_indices.append(
            np.full(info.n_measurements, condition_index, dtype=np.int64)
        )
        measurement_indices.append(
            np.arange(info.n_measurements, dtype=np.int64)
        )
        filenames.append(filename)
        print(
            f"loaded {condition_name}: {info.n_measurements} omni RIRs",
            flush=True,
        )
    return {
        "rirs": np.concatenate(rirs),
        "listener_positions_m": np.concatenate(listener_positions),
        "source_positions_m": np.concatenate(source_positions),
        "condition_index": np.concatenate(condition_indices),
        "measurement_index": np.concatenate(measurement_indices),
        "condition_names": np.asarray(
            [name for name, _ in CONDITION_FILES], dtype="U32"
        ),
        "condition_filenames": np.asarray(filenames, dtype="U180"),
        "sample_rate_hz": np.asarray(SAMPLE_RATE_HZ),
    }


def decayfitnet_mse_db2(
    edcs: np.ndarray,
    t60_s: np.ndarray,
    amplitudes_normalized_edc: np.ndarray,
    noise_normalized_per_sample: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    """Return per-RIR DecayFitNet dB-domain EDC mean-squared errors.

    The input EDCs have shape ``(R,L)``. Times and amplitudes have shape
    ``(R,K)``; noise has shape ``(R,)``. The comparison uses 100 uniformly
    spaced samples and excludes the final five, matching the benchmark fit.
    """

    curves = np.asarray(edcs, dtype=np.float64)
    t60 = np.asarray(t60_s, dtype=np.float64)
    amplitudes = np.asarray(amplitudes_normalized_edc, dtype=np.float64)
    noise = np.asarray(noise_normalized_per_sample, dtype=np.float64)
    if curves.ndim != 2 or t60.shape != amplitudes.shape:
        raise ValueError("EDCs must be (R,L), with T60/amplitudes shaped (R,K).")
    if t60.shape[0] != curves.shape[0] or noise.shape != (curves.shape[0],):
        raise ValueError("DecayFitNet parameter RIR axes do not match EDCs.")
    sample_indices = np.linspace(
        0.0, curves.shape[1] - 1.0, DECAYFITNET_OUTPUT_SIZE
    )
    times_s = sample_indices / sample_rate_hz
    exponentials = np.exp(
        -np.log(1e6)
        * times_s[np.newaxis, np.newaxis, :]
        / t60[:, :, np.newaxis]
    )
    exponentials -= exponentials[:, :, -1:]
    fitted = np.sum(amplitudes[:, :, np.newaxis] * exponentials, axis=1)
    fitted += noise[:, np.newaxis] * (
        curves.shape[1] - sample_indices[np.newaxis, :]
    )
    normalized = curves / curves[:, :1]
    source_indices = np.arange(curves.shape[1], dtype=np.float64)
    observed = np.stack(
        [np.interp(sample_indices, source_indices, row) for row in normalized]
    )
    epsilon = np.finfo(np.float64).tiny
    residual_db = 10.0 * np.log10(np.maximum(fitted[:, :95], epsilon))
    residual_db -= 10.0 * np.log10(np.maximum(observed[:, :95], epsilon))
    return np.mean(residual_db**2, axis=1)


def run_one_band(
    rirs: np.ndarray,
    sample_rate_hz: float,
    band_hz: float,
    decayfitnet: ExternalDecayFitNet,
    *,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    """Run both Georg baselines for one band and return NPZ-ready arrays."""

    print(f"{band_hz:g} Hz: filtering and integrating EDCs", flush=True)
    edcs = band_energy_decay_curves(rirs, sample_rate_hz, band_hz)
    print(f"{band_hz:g} Hz: DecayFitNet K={N_SLOPES}", flush=True)
    independent = decayfitnet.estimate_edcs(edcs, sample_rate_hz)
    common_t60_s, clusters = determine_common_decay_times(
        independent.t60_s,
        N_SLOPES,
        histogram_resolution_s=HISTOGRAM_RESOLUTION_S,
        seed=CLUSTER_SEED,
    )
    print(
        f"{band_hz:g} Hz: common T60="
        + ", ".join(f"{value:.3f} s" for value in common_t60_s),
        flush=True,
    )
    common_fit = fit_common_slope_edcs(
        edcs,
        common_t60_s,
        sample_rate_hz,
        n_jobs=n_jobs,
    )

    normalization = independent.edc_normalization_energy
    independent_absolute = (
        independent.amplitudes_normalized_edc * normalization[:, np.newaxis]
    )
    common_t60_per_rir = np.broadcast_to(
        common_t60_s, common_fit.amplitudes_absolute_edc_energy.shape
    )
    result = {
        "band_center_hz": np.asarray(band_hz),
        "n_edc_samples": np.asarray(edcs.shape[1]),
        "decayfitnet_t60_s": independent.t60_s,
        "decayfitnet_amplitudes_normalized_edc": (
            independent.amplitudes_normalized_edc
        ),
        "decayfitnet_amplitudes_absolute_edc_energy": independent_absolute,
        "decayfitnet_equivalent_rir_power_amplitudes": (
            edc_to_equivalent_rir_power_amplitudes(
                independent_absolute, independent.t60_s, sample_rate_hz
            )
        ),
        "decayfitnet_noise_normalized_per_sample": (
            independent.noise_normalized_per_sample
        ),
        "decayfitnet_noise_absolute_per_sample": (
            independent.noise_normalized_per_sample * normalization
        ),
        "decayfitnet_edc_normalization_energy": normalization,
        "decayfitnet_mse_db2": decayfitnet_mse_db2(
            edcs,
            independent.t60_s,
            independent.amplitudes_normalized_edc,
            independent.noise_normalized_per_sample,
            sample_rate_hz,
        ),
        "common_slope_t60_s": common_t60_s,
        "common_slope_amplitudes_normalized_edc": (
            common_fit.amplitudes_normalized_edc
        ),
        "common_slope_amplitudes_absolute_edc_energy": (
            common_fit.amplitudes_absolute_edc_energy
        ),
        "common_slope_equivalent_rir_power_amplitudes": (
            edc_to_equivalent_rir_power_amplitudes(
                common_fit.amplitudes_absolute_edc_energy,
                common_t60_per_rir,
                sample_rate_hz,
            )
        ),
        "common_slope_noise_normalized_edc_origin": (
            common_fit.noise_normalized_edc_origin
        ),
        "common_slope_noise_absolute_edc_energy": (
            common_fit.noise_absolute_edc_energy
        ),
        "common_slope_equivalent_noise_power_per_sample": (
            common_fit.noise_absolute_edc_energy / edcs.shape[1]
        ),
        "common_slope_mse_db2": common_fit.mse_db2,
        "common_slope_fit_success": common_fit.success,
        "common_slope_cluster_sizes": np.asarray(
            [cluster.size for cluster in clusters], dtype=np.int64
        ),
    }
    del edcs
    return result


def combine_band_partials(
    partial_paths: list[Path],
) -> dict[str, np.ndarray]:
    """Stack ordered one-band NPZ results into receiver-by-band arrays."""

    if not partial_paths:
        raise ValueError("partial_paths must not be empty.")
    loaded = []
    try:
        loaded = [np.load(path, allow_pickle=False) for path in partial_paths]
        combined: dict[str, np.ndarray] = {
            "band_centers_hz": np.asarray(
                [float(values["band_center_hz"]) for values in loaded]
            ),
            "n_edc_samples": np.asarray(
                [int(values["n_edc_samples"]) for values in loaded]
            ),
        }
        receiver_first = {
            key for key in PARTIAL_KEYS if key not in {
                "common_slope_t60_s",
                "common_slope_cluster_sizes",
            }
        }
        for key in PARTIAL_KEYS:
            arrays = [np.asarray(values[key]) for values in loaded]
            axis = 1 if key in receiver_first else 0
            combined[key] = np.stack(arrays, axis=axis)
        return combined
    finally:
        for values in loaded:
            values.close()


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
                   label="NMR/CW-SAGE"),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_combined_results(
    output_dir: Path,
    band_results: dict[str, np.ndarray],
    dataset: dict[str, np.ndarray],
    model_dir: Path,
) -> Path:
    """Save combined estimates, amplitude conventions, and provenance."""

    model_path = model_dir / f"DecayFitNet_{N_SLOPES}slopes_v10.onnx"
    transform_path = model_dir / f"input_transform_{N_SLOPES}slopes.pkl"
    output_path = output_dir / "room_to_hallway_georg_benchmarks.npz"
    np.savez_compressed(
        output_path,
        **band_results,
        listener_positions_m=dataset["listener_positions_m"],
        source_positions_m=dataset["source_positions_m"],
        condition_index=dataset["condition_index"],
        measurement_index=dataset["measurement_index"],
        condition_names=dataset["condition_names"],
        condition_filenames=dataset["condition_filenames"],
        sample_rate_hz=dataset["sample_rate_hz"],
        receiver_channel_index=np.asarray(0),
        n_slopes=np.asarray(N_SLOPES),
        filter_order=np.asarray(FILTER_ORDER),
        bandwidth_factor=np.asarray(GEORG_BANDWIDTH_FACTOR),
        analyze_full_rir=np.asarray(True),
        filter_tail_discard_fraction=np.asarray(
            FILTER_TAIL_DISCARD_FRACTION
        ),
        decayfitnet_edc_tail_discard_fraction=np.asarray(
            DECAYFITNET_EDC_TAIL_DISCARD_FRACTION
        ),
        decayfitnet_output_size=np.asarray(DECAYFITNET_OUTPUT_SIZE),
        common_slope_histogram_resolution_s=np.asarray(
            HISTOGRAM_RESOLUTION_S
        ),
        common_slope_cluster_seed=np.asarray(CLUSTER_SEED),
        decayfitnet_model_path=np.asarray(str(model_path.resolve())),
        decayfitnet_model_sha256=np.asarray(_sha256(model_path)),
        decayfitnet_transform_path=np.asarray(str(transform_path.resolve())),
        decayfitnet_transform_sha256=np.asarray(_sha256(transform_path)),
        decayfitnet_amplitude_convention=np.asarray(
            "normalized Schroeder-EDC exponential coefficient"
        ),
        common_slope_amplitude_convention=np.asarray(
            "log-EDC-fit exponential coefficient"
        ),
        equivalent_rir_power_amplitude_note=np.asarray(
            "EDC coefficient times 1-exp(-6*ln(10)/(fs*T60)); "
            "band-filter dependent and not identical to STFT-bin amplitude"
        ),
    )
    return output_path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--bands",
        nargs="+",
        type=float,
        default=BAND_CENTERS_HZ.tolist(),
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    bands = np.asarray(args.bands, dtype=np.float64)
    if np.any(~np.isfinite(bands)) or np.any(bands <= 0.0):
        parser.error("--bands must contain finite positive frequencies")
    if np.any(np.diff(bands) <= 0.0):
        parser.error("--bands must be strictly increasing")
    if bands[-1] * GEORG_BANDWIDTH_FACTOR >= SAMPLE_RATE_HZ / 2.0:
        parser.error("the highest band edge must be below Nyquist")
    return args


def main() -> None:
    """Run all selected bands and save one combined benchmark NPZ."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    dataset = load_room_transition_omni(args.dataset_dir)
    rirs = dataset.pop("rirs")
    decayfitnet = ExternalDecayFitNet(args.model_dir, n_slopes=N_SLOPES)
    partial_paths: list[Path] = []
    for band_hz in args.bands:
        result = run_one_band(
            rirs,
            SAMPLE_RATE_HZ,
            float(band_hz),
            decayfitnet,
            n_jobs=args.jobs,
        )
        partial_path = output_dir / f"georg_benchmark_{int(band_hz):05d}_hz.npz"
        np.savez_compressed(partial_path, **result)
        partial_paths.append(partial_path)
        print(f"saved checkpoint {partial_path.name}", flush=True)
    combined = combine_band_partials(partial_paths)
    output_path = save_combined_results(
        output_dir, combined, dataset, args.model_dir.resolve()
    )
    print(f"combined benchmark={output_path}", flush=True)


if __name__ == "__main__":
    main()
