"""Run fixed-K=2 DecayFitNet and CommonSlopeAnalysis room benchmarks.

The four raw v1.3 Meeting Room to Hallway SOFA files are analyzed in six octave bands. DecayFitNet's Python toolbox, loaded from an external checkout, filters each RIR, forms the Schroeder EDC, and estimates two slopes. Georg Götz's common-slope procedure then clusters those decay times into two shared values per band. The checkout is not copied into this repository.
"""


from __future__ import annotations

import argparse
import hashlib
import sys
import types
from pathlib import Path

import numpy as np

from common_slope_nmf.baseline import (
    determine_common_decay_times,
    edc_to_equivalent_rir_power_amplitudes,
)
from experiments._run_output import create_run_output_dir
from experiments.datasets import load_sofa_channel
from experiments.room_to_hallway.omni import CONDITION_FILES, DATASET_DIR

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

DECAYFITNET_OUTPUT_SIZE = 100

# FilterByOctaves in the DecayFitNet Python toolbox.
BANDWIDTH_FACTOR = float(np.sqrt(2.0))

DEFAULT_DECAYFITNET_REPO = Path.home() / "Documents" / "DecayFitNet"

_TOOLBOX_PACKAGE = "decayfitnet_checkout"

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


def open_decayfitnet(
    repo_dir: str | Path,
    *,
    n_slopes: int,
    sample_rate_hz: float,
    filter_frequencies: list[float],
):
    """Load ``DecayFitNetToolbox`` from a DecayFitNet repository checkout.

    ``repo_dir`` is the repository root. Filtering and Schroeder integration come from ``python/toolbox``. The ONNX network and input transform come from ``model/``.
    """

    repo = Path(repo_dir).expanduser().resolve()
    toolbox_dir = repo / "python" / "toolbox"
    model_dir = repo / "model"
    if not (toolbox_dir / "DecayFitNetToolbox.py").is_file():
        raise FileNotFoundError(
            f"{repo} has no python/toolbox/DecayFitNetToolbox.py."
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{repo} has no model directory.")
    existing = sys.modules.get(_TOOLBOX_PACKAGE)
    if existing is None:
        package = types.ModuleType(_TOOLBOX_PACKAGE)
        package.__path__ = [str(toolbox_dir)]
        package.__package__ = _TOOLBOX_PACKAGE
        sys.modules[_TOOLBOX_PACKAGE] = package
    elif Path(existing.__path__[0]).resolve() != toolbox_dir:
        raise RuntimeError(
            f"{_TOOLBOX_PACKAGE} is already loaded from {existing.__path__[0]}."
        )
    try:
        from decayfitnet_checkout.DecayFitNetToolbox import DecayFitNetToolbox
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "onnx", "onnxruntime"}:
            raise ImportError(
                "DecayFitNet's Python toolbox requires torch, onnx, and onnxruntime."
            ) from exc
        raise

    toolbox = DecayFitNetToolbox(
        n_slopes=n_slopes,
        sample_rate=int(sample_rate_hz),
        filter_frequencies=[float(frequency) for frequency in filter_frequencies],
        model_dir=model_dir,
    )
    return toolbox, model_dir


def _decayfitnet_mse_db2(
    decayfitnet,
    rirs: np.ndarray,
    t60_s: np.ndarray,
    amplitudes_normalized_edc: np.ndarray,
    noise_normalized_per_sample: np.ndarray,
    sample_rate_hz: float,
) -> tuple[int, np.ndarray]:
    """Compare each Schroeder EDC with DecayFitNet's reconstructed EDC.

    The comparison uses Georg's ``decay_model`` and drops the final 5 percent, matching the DecayFitNet demo. Returns the Schroeder length and one mean-squared dB residual per RIR.
    """

    import torch
    from decayfitnet_checkout.core import decay_model, discard_last_n_percent

    true_edc, _ = decayfitnet._preprocess.schroeder(
        torch.as_tensor(rirs), analyse_full_rir=True
    )
    n_samples = int(true_edc.shape[-1])
    time_axis = torch.linspace(0, n_samples - 1, n_samples) / float(sample_rate_hz)
    fitted = decay_model(
        torch.as_tensor(t60_s).clone(),
        torch.as_tensor(amplitudes_normalized_edc).clone(),
        torch.as_tensor(noise_normalized_per_sample).reshape(-1, 1),
        time_axis,
        compensate_uli=True,
        backend="torch",
    )
    observed = discard_last_n_percent(true_edc, 5)
    predicted = discard_last_n_percent(fitted, 5)
    residual_db = 10.0 * torch.log10(observed) - 10.0 * torch.log10(predicted)
    mse = torch.mean(residual_db.square(), dim=-1).reshape(-1)
    return n_samples, mse.detach().cpu().numpy()


def run_one_band(
    rirs: np.ndarray,
    sample_rate_hz: float,
    band_hz: float,
    decayfitnet,
) -> dict[str, np.ndarray]:
    """Run both Georg baselines for one band and return NPZ-ready arrays."""

    if rirs.ndim != 2 or rirs.shape[0] > rirs.shape[1]:
        raise ValueError(
            "DecayFitNet expects RIRs shaped (receivers, samples) with more samples than receivers. "
            f"Got {getattr(rirs, 'shape', None)}."
        )
    print(f"{band_hz:g} Hz: DecayFitNet K={N_SLOPES}", flush=True)
    decayfitnet.set_filter_frequencies([float(band_hz)])
    parameters, norm_vals = decayfitnet.estimate_parameters(
        rirs, analyse_full_rir=True
    )
    t60_s = np.asarray(parameters[0], dtype=np.float64)
    amplitudes = np.asarray(parameters[1], dtype=np.float64)
    noise = np.asarray(parameters[2], dtype=np.float64).reshape(rirs.shape[0])
    normalization = np.asarray(norm_vals, dtype=np.float64).reshape(rirs.shape[0])
    if t60_s.shape != (rirs.shape[0], N_SLOPES) or amplitudes.shape != t60_s.shape:
        raise ValueError(
            "DecayFitNet returned unexpected parameter shapes "
            f"{t60_s.shape} and {amplitudes.shape}."
        )
    n_edc_samples, mse_db2 = _decayfitnet_mse_db2(
        decayfitnet,
        rirs,
        t60_s,
        amplitudes,
        noise,
        sample_rate_hz,
    )
    common_t60_s, cluster_sizes = determine_common_decay_times(
        t60_s,
        N_SLOPES,
        histogram_resolution_s=HISTOGRAM_RESOLUTION_S,
        seed=CLUSTER_SEED,
    )
    print(
        f"{band_hz:g} Hz: common T60="
        + ", ".join(f"{value:.3f} s" for value in common_t60_s),
        flush=True,
    )

    absolute = amplitudes * normalization[:, np.newaxis]
    return {
        "band_center_hz": np.asarray(band_hz),
        "n_edc_samples": np.asarray(n_edc_samples),
        "decayfitnet_t60_s": t60_s,
        "decayfitnet_amplitudes_normalized_edc": amplitudes,
        "decayfitnet_amplitudes_absolute_edc_energy": absolute,
        "decayfitnet_equivalent_rir_power_amplitudes": (
            edc_to_equivalent_rir_power_amplitudes(
                absolute, t60_s, sample_rate_hz
            )
        ),
        "decayfitnet_noise_normalized_per_sample": noise,
        "decayfitnet_noise_absolute_per_sample": noise * normalization,
        "decayfitnet_edc_normalization_energy": normalization,
        "decayfitnet_mse_db2": mse_db2,
        "common_slope_t60_s": common_t60_s,
        "common_slope_cluster_sizes": cluster_sizes,
    }


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
    repo_dir: Path,
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
        bandwidth_factor=np.asarray(BANDWIDTH_FACTOR),
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
        decayfitnet_repo_path=np.asarray(str(repo_dir.resolve())),
        decayfitnet_model_path=np.asarray(str(model_path.resolve())),
        decayfitnet_model_sha256=np.asarray(_sha256(model_path)),
        decayfitnet_transform_path=np.asarray(str(transform_path.resolve())),
        decayfitnet_transform_sha256=np.asarray(_sha256(transform_path)),
        decayfitnet_amplitude_convention=np.asarray(
            "normalized Schroeder-EDC exponential coefficient"
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
    parser.add_argument(
        "--decayfitnet-repo",
        type=Path,
        default=DEFAULT_DECAYFITNET_REPO,
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--bands",
        nargs="+",
        type=float,
        default=BAND_CENTERS_HZ.tolist(),
    )
    args = parser.parse_args()
    bands = np.asarray(args.bands, dtype=np.float64)
    if np.any(~np.isfinite(bands)) or np.any(bands <= 0.0):
        parser.error("--bands must contain finite positive frequencies")
    if np.any(np.diff(bands) <= 0.0):
        parser.error("--bands must be strictly increasing")
    if bands[-1] * BANDWIDTH_FACTOR >= SAMPLE_RATE_HZ / 2.0:
        parser.error("the highest band edge must be below Nyquist")
    return args


def main() -> None:
    """Run all selected bands and save one combined benchmark NPZ."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    dataset = load_room_transition_omni(args.dataset_dir)
    rirs = dataset.pop("rirs")
    decayfitnet, model_dir = open_decayfitnet(
        args.decayfitnet_repo,
        n_slopes=N_SLOPES,
        sample_rate_hz=SAMPLE_RATE_HZ,
        filter_frequencies=[float(band_hz) for band_hz in args.bands],
    )
    partial_paths: list[Path] = []
    for band_hz in args.bands:
        result = run_one_band(
            rirs,
            SAMPLE_RATE_HZ,
            float(band_hz),
            decayfitnet,
        )
        partial_path = output_dir / f"georg_benchmark_{int(band_hz):05d}_hz.npz"
        np.savez_compressed(partial_path, **result)
        partial_paths.append(partial_path)
        print(f"saved checkpoint {partial_path.name}", flush=True)
    combined = combine_band_partials(partial_paths)
    output_path = save_combined_results(
        output_dir,
        combined,
        dataset,
        Path(args.decayfitnet_repo).expanduser().resolve(),
        model_dir,
    )
    print(f"combined benchmark={output_path}", flush=True)


if __name__ == "__main__":
    main()
