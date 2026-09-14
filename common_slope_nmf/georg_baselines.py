"""Python ports needed to benchmark Georg Götz's decay-analysis methods.

The DecayFitNet ONNX weights and input transform remain external assets. This
module implements the NumPy/SciPy preprocessing and the common-slope clustering
and log-EDC amplitude fit described by the upstream MIT-licensed MATLAB code.
"""

from __future__ import annotations

import io
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.signal import butter, sosfilt


GEORG_BANDWIDTH_FACTOR = float(np.sqrt(1.5))
DECAYFITNET_OUTPUT_SIZE = 100
DECAYFITNET_EDC_NORM_EPS = 1e-10


@dataclass(frozen=True)
class DecayFitNetEstimate:
    """Fixed-order DecayFitNet estimates for one frequency band.

    Attributes
    ----------
    t60_s
        Sorted energy-decay times in seconds, shape ``(R,K)``.
    amplitudes_normalized_edc
        Dimensionless coefficients of the normalized Schroeder EDC model,
        shape ``(R,K)``.
    noise_normalized_per_sample
        Predicted stationary-noise power relative to the EDC normalization,
        shape ``(R,)``.
    edc_normalization_energy
        Initial backward-integrated in-band energy, shape ``(R,)``.
    """

    t60_s: NDArray[np.float64]
    amplitudes_normalized_edc: NDArray[np.float64]
    noise_normalized_per_sample: NDArray[np.float64]
    edc_normalization_energy: NDArray[np.float64]


@dataclass(frozen=True)
class CommonSlopeEDCFit:
    """Common-slope log-EDC amplitude estimates for one frequency band.

    Attributes
    ----------
    amplitudes_normalized_edc
        Dimensionless normalized-EDC coefficients, shape ``(R,K)``.
    amplitudes_absolute_edc_energy
        Coefficients multiplied by initial in-band EDC energy, shape ``(R,K)``.
    noise_normalized_edc_origin
        Dimensionless integrated-noise coefficient at EDC time zero,
        shape ``(R,)``; this is not per-sample stationary-noise power.
    noise_absolute_edc_energy
        Integrated-noise coefficient in absolute energy units, shape ``(R,)``.
    mse_db2
        Mean squared dB residual over the fitted EDC samples, shape ``(R,)``.
    success
        SciPy least-squares termination flags, shape ``(R,)``.
    """

    amplitudes_normalized_edc: NDArray[np.float64]
    amplitudes_absolute_edc_energy: NDArray[np.float64]
    noise_normalized_edc_origin: NDArray[np.float64]
    noise_absolute_edc_energy: NDArray[np.float64]
    mse_db2: NDArray[np.float64]
    success: NDArray[np.bool_]


def _positive_matrix(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must have non-empty shape (R,N).")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values.")
    return array


def octave_band_sos(
    center_frequency_hz: float,
    sample_rate_hz: float,
    *,
    order: int = 5,
    bandwidth_factor: float = GEORG_BANDWIDTH_FACTOR,
) -> NDArray[np.float64]:
    """Design Georg Götz's causal octave-like Butterworth band filter.

    Parameters
    ----------
    center_frequency_hz
        Positive band center frequency in hertz.
    sample_rate_hz
        Positive sampling frequency in samples per second.
    order
        Positive Butterworth order.
    bandwidth_factor
        Ratio from the center to either band edge. Georg's MATLAB analysis
        uses ``sqrt(1.5)``.

    Returns
    -------
    ndarray
        Second-order-section coefficients, shape ``(S,6)``.
    """

    center = float(center_frequency_hz)
    sample_rate = float(sample_rate_hz)
    if not np.isfinite(center) or center <= 0.0:
        raise ValueError("center_frequency_hz must be finite and positive.")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive.")
    if order <= 0:
        raise ValueError("order must be positive.")
    if not np.isfinite(bandwidth_factor) or bandwidth_factor <= 1.0:
        raise ValueError("bandwidth_factor must be finite and greater than one.")
    edges = center * np.array(
        [1.0 / bandwidth_factor, bandwidth_factor], dtype=np.float64
    )
    if edges[1] >= sample_rate / 2.0:
        raise ValueError("upper band edge must be below Nyquist.")
    return butter(
        order,
        edges,
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )


def band_energy_decay_curves(
    rirs: ArrayLike,
    sample_rate_hz: float,
    center_frequency_hz: float,
    *,
    discard_tail_fraction: float = 0.005,
) -> NDArray[np.float64]:
    """Filter RIRs and return backward-integrated in-band energy curves.

    Parameters
    ----------
    rirs
        Real RIR samples, shape ``(R,N)``.
    sample_rate_hz
        Sampling frequency in samples per second.
    center_frequency_hz
        Analysis-band center frequency in hertz.
    discard_tail_fraction
        Fraction removed after causal filtering to suppress end artefacts.

    Returns
    -------
    ndarray
        Positive Schroeder energy-decay curves, shape ``(R,L)`` and squared
        input-sample units. The full RIR is analyzed without onset cropping.
    """

    signals = np.asarray(rirs, dtype=np.float64)
    if signals.ndim != 2 or signals.size == 0:
        raise ValueError("rirs must have non-empty shape (R,N).")
    if not np.all(np.isfinite(signals)):
        raise ValueError("rirs must contain finite real samples.")
    if not np.isfinite(discard_tail_fraction) or not 0.0 <= discard_tail_fraction < 1.0:
        raise ValueError("discard_tail_fraction must lie in [0,1).")
    n_discard = int(np.rint(discard_tail_fraction * signals.shape[1]))
    if n_discard >= signals.shape[1] - 1:
        raise ValueError("discard_tail_fraction leaves too few samples.")
    filtered = sosfilt(
        octave_band_sos(center_frequency_hz, sample_rate_hz),
        signals,
        axis=1,
    )
    if n_discard:
        filtered = filtered[:, :-n_discard].copy()
    np.square(filtered, out=filtered)
    reversed_energy = filtered[:, ::-1]
    np.cumsum(reversed_energy, axis=1, out=reversed_energy)
    return np.maximum(filtered, np.finfo(np.float64).tiny)


def _resample_rows(values: NDArray[np.float64], output_size: int) -> NDArray[np.float64]:
    source_x = np.arange(values.shape[1], dtype=np.float64)
    target_x = np.linspace(0.0, values.shape[1] - 1.0, output_size)
    return np.stack([np.interp(target_x, source_x, row) for row in values])


class _TorchScalarUnpickler(pickle.Unpickler):
    """Read the scalar tensor used in legacy DecayFitNet transforms sans torch."""

    def find_class(self, module: str, name: str):  # noqa: ANN001
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda payload: np.frombuffer(payload[-4:], dtype="<f4").copy()
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, offset, size, stride, requires_grad, hooks: float(
                np.ravel(storage)[offset]
            )
        return super().find_class(module, name)


def load_decayfitnet_normfactor(path: str | Path) -> float:
    """Load the scalar EDC dB normalization from an external model pickle."""

    transform_path = Path(path)
    try:
        with transform_path.open("rb") as stream:
            transform = pickle.load(stream)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        with transform_path.open("rb") as stream:
            transform = _TorchScalarUnpickler(stream).load()
    value = float(np.asarray(transform["edcs_db_normfactor"]))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("DecayFitNet EDC normalization must be positive.")
    return value


class ExternalDecayFitNet:
    """Torch-free ONNX adapter for externally stored DecayFitNet assets.

    The adapter does not package or copy DecayFitNet. ``model_dir`` must point
    to the external checkout's model directory containing the fixed-order ONNX
    network and its input-transform pickle.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        n_slopes: int = 2,
        output_size: int = DECAYFITNET_OUTPUT_SIZE,
    ) -> None:
        if n_slopes not in (1, 2, 3):
            raise ValueError("n_slopes must be one, two, or three.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "DecayFitNet benchmarking requires onnxruntime; install the "
                "project's decayfitnet extra."
            ) from exc

        directory = Path(model_dir).expanduser().resolve()
        model_path = directory / f"DecayFitNet_{n_slopes}slopes_v10.onnx"
        transform_path = directory / f"input_transform_{n_slopes}slopes.pkl"
        if not model_path.is_file() or not transform_path.is_file():
            raise FileNotFoundError(
                "external DecayFitNet model or input transform is missing from "
                f"{directory}."
            )
        self.n_slopes = n_slopes
        self.output_size = output_size
        self.model_path = model_path
        self.normfactor = load_decayfitnet_normfactor(transform_path)
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def estimate_edcs(
        self,
        edcs: ArrayLike,
        sample_rate_hz: float,
    ) -> DecayFitNetEstimate:
        """Estimate fixed-order decay parameters from in-band EDCs.

        Parameters
        ----------
        edcs
            Positive full-RIR Schroeder curves, shape ``(R,L)``, in energy
            units before normalization.
        sample_rate_hz
            Original RIR sampling frequency in samples per second.

        Returns
        -------
        DecayFitNetEstimate
            Sorted energy ``T60`` values, normalized EDC amplitudes, normalized
            per-sample noise powers, and absolute EDC normalization energies.
        """

        curves = _positive_matrix("edcs", edcs)
        sample_rate = float(sample_rate_hz)
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive.")
        normalization = curves[:, 0].copy()
        normalized = curves / normalization[:, np.newaxis]
        edc_db = 10.0 * np.log10(normalized + DECAYFITNET_EDC_NORM_EPS)
        n_adjust = curves.shape[1] / self.output_size
        t_adjust = 10.0 / (curves.shape[1] / sample_rate)
        last = int(np.rint(0.95 * edc_db.shape[1]))
        network_input = _resample_rows(edc_db[:, :last], self.output_size)
        network_input = (2.0 * network_input / self.normfactor + 1.0).astype(
            np.float32
        )

        outputs = self._session.run(None, {self._input_name: network_input})
        t60_s = np.asarray(outputs[0], dtype=np.float64) / t_adjust
        amplitudes = np.asarray(outputs[1], dtype=np.float64)
        noise = np.power(
            10.0, np.clip(np.asarray(outputs[2], dtype=np.float64), -32.0, 32.0)
        ).ravel()
        noise /= n_adjust
        if t60_s.shape != (curves.shape[0], self.n_slopes):
            raise ValueError("DecayFitNet returned an unexpected T60 shape.")
        order = np.argsort(t60_s, axis=1)
        return DecayFitNetEstimate(
            t60_s=np.take_along_axis(t60_s, order, axis=1),
            amplitudes_normalized_edc=np.take_along_axis(
                amplitudes, order, axis=1
            ),
            noise_normalized_per_sample=noise,
            edc_normalization_energy=normalization,
        )


def _histogram_edges(
    minimum: float, maximum: float, resolution: float
) -> NDArray[np.float64]:
    lower = np.floor(minimum / resolution) * resolution
    upper = np.ceil(maximum / resolution) * resolution
    return np.arange(lower, upper + resolution / 2.0, resolution)


def _kmedians_1d(
    values: NDArray[np.float64],
    n_clusters: int,
    *,
    seed: int,
    n_init: int = 5,
    max_iter: int = 10_000,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    generator = np.random.default_rng(seed)
    unique = np.unique(values)
    if unique.size < n_clusters:
        raise ValueError("fewer distinct positive values than requested clusters.")
    best_cost = np.inf
    best_labels = np.empty(values.size, dtype=np.int64)
    best_centers = np.empty(n_clusters, dtype=np.float64)
    for _ in range(n_init):
        centers = generator.choice(unique, size=n_clusters, replace=False).astype(float)
        labels = np.full(values.size, -1, dtype=np.int64)
        for _ in range(max_iter):
            updated_labels = np.argmin(
                np.abs(values[:, np.newaxis] - centers[np.newaxis, :]), axis=1
            )
            if np.array_equal(updated_labels, labels):
                break
            labels = updated_labels
            for cluster in range(n_clusters):
                members = values[labels == cluster]
                if members.size:
                    centers[cluster] = np.median(members)
        cost = float(np.sum(np.abs(values - centers[labels])))
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers


def determine_common_decay_times(
    t60_s: ArrayLike,
    n_common_slopes: int,
    *,
    histogram_resolution_s: float = 0.05,
    seed: int = 42,
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
    """Cluster per-RIR energy decay times into Georg-style common slopes.

    Positive entries of ``t60_s`` are clustered by one-dimensional L1
    k-medians. Each common time is the center of the most populated histogram
    bin within its sorted cluster.

    Returns
    -------
    common_t60_s, clusters
        Sorted shared energy decay times in seconds, shape ``(K,)``, and the
        corresponding tuple of per-RIR DecayFitNet samples in seconds.
    """

    values = np.asarray(t60_s, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)].ravel()
    if n_common_slopes <= 0:
        raise ValueError("n_common_slopes must be positive.")
    if positive.size < n_common_slopes:
        raise ValueError("too few positive decay times for requested clusters.")
    if not np.isfinite(histogram_resolution_s) or histogram_resolution_s <= 0.0:
        raise ValueError("histogram_resolution_s must be finite and positive.")
    labels, centers = _kmedians_1d(
        positive, n_common_slopes, seed=seed
    )
    cluster_order = np.argsort(centers)
    edges = _histogram_edges(
        float(np.min(positive)),
        float(np.max(positive)),
        histogram_resolution_s,
    )
    common = np.empty(n_common_slopes, dtype=np.float64)
    clusters: list[NDArray[np.float64]] = []
    for output_index, cluster_index in enumerate(cluster_order):
        members = positive[labels == cluster_index]
        if members.size == 0:
            raise RuntimeError("k-medians produced an empty common-slope cluster.")
        counts, _ = np.histogram(members, bins=edges)
        peak = int(np.argmax(counts))
        common[output_index] = np.mean(edges[peak : peak + 2])
        clusters.append(members.copy())
    return common, tuple(clusters)


def common_slope_edc_kernel(
    t60_s: ArrayLike, times_s: ArrayLike
) -> NDArray[np.float64]:
    """Return Georg's exponential EDC atoms and linear integrated-noise atom.

    Energy ``T60`` values have shape ``(K,)`` in seconds and ``times_s`` has
    shape ``(N,)`` in seconds. The result has shape ``(N,K+1)``.
    """

    t60 = np.asarray(t60_s, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if t60.ndim != 1 or t60.size == 0 or np.any(t60 <= 0.0):
        raise ValueError("t60_s must have positive shape (K,).")
    if times.ndim != 1 or times.size < 2 or np.any(times < 0.0):
        raise ValueError("times_s must have non-negative shape (N,), N >= 2.")
    exponentials = np.exp(
        -np.log(1e6) * times[:, np.newaxis] / t60[np.newaxis, :]
    )
    noise = np.linspace(1.0, 1.0 / times.size, times.size)[:, np.newaxis]
    return np.hstack((exponentials, noise))


def fit_common_slope_edcs(
    edcs: ArrayLike,
    common_t60_s: ArrayLike,
    sample_rate_hz: float,
    *,
    output_size: int = DECAYFITNET_OUTPUT_SIZE,
    fit_fraction: float = 0.95,
    n_jobs: int = 1,
) -> CommonSlopeEDCFit:
    """Fit common-slope amplitudes to full-RIR Schroeder curves.

    The upstream procedure normalizes each EDC by its first value, resamples it
    to 100 points, and minimizes squared dB residuals over the first 95 points
    with non-negative bounded amplitudes and a linear integrated-noise atom.

    Parameters
    ----------
    edcs
        Positive in-band backward-integrated energy curves, shape ``(R,L)``.
    common_t60_s
        Supplied shared energy decay times in seconds, shape ``(K,)``.
    sample_rate_hz
        Original RIR sampling frequency in samples per second.
    output_size
        Number of uniformly spaced EDC points used for fitting.
    fit_fraction
        Leading fraction of resampled EDC points included in the dB loss.
    n_jobs
        Number of concurrent per-RIR least-squares fits.

    Returns
    -------
    CommonSlopeEDCFit
        Normalized and absolute EDC amplitudes, integrated-noise coefficients,
        dB-domain mean-squared residuals, and fit-success flags.
    """

    curves = _positive_matrix("edcs", edcs)
    common = np.asarray(common_t60_s, dtype=np.float64)
    if common.ndim != 1 or common.size == 0 or np.any(common <= 0.0):
        raise ValueError("common_t60_s must have positive shape (K,).")
    if sample_rate_hz <= 0.0 or not np.isfinite(sample_rate_hz):
        raise ValueError("sample_rate_hz must be finite and positive.")
    if output_size < 2:
        raise ValueError("output_size must be at least two.")
    if not 0.0 < fit_fraction <= 1.0:
        raise ValueError("fit_fraction must lie in (0,1].")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive.")

    normalization = curves[:, 0].copy()
    normalized = curves / normalization[:, np.newaxis]
    downsampled = _resample_rows(normalized, output_size)
    times_s = np.linspace(
        0.0, (curves.shape[1] - 1.0) / sample_rate_hz, output_size
    )
    kernel = common_slope_edc_kernel(common, times_s)
    n_fit = int(np.rint(fit_fraction * output_size))
    kernel_fit = kernel[:n_fit]
    factor = 10.0 / np.log(10.0)
    lower = np.zeros(common.size + 1, dtype=np.float64)
    upper = np.concatenate((np.full(common.size, 10.0), [1.0]))
    initial = np.concatenate((np.ones(common.size), [1e-10]))
    epsilon = np.finfo(np.float64).eps

    def fit_one(index: int):
        target_db = 10.0 * np.log10(np.maximum(downsampled[index, :n_fit], epsilon))

        def residual(weights: NDArray[np.float64]) -> NDArray[np.float64]:
            model = np.maximum(kernel_fit @ weights, epsilon)
            return 10.0 * np.log10(model) - target_db

        def jacobian(weights: NDArray[np.float64]) -> NDArray[np.float64]:
            model = np.maximum(kernel_fit @ weights, epsilon)
            return factor * kernel_fit / model[:, np.newaxis]

        fit = least_squares(
            residual,
            initial,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            ftol=1e-9,
            xtol=1e-12,
            max_nfev=5_000,
        )
        return fit.x, float(np.mean(residual(fit.x) ** 2)), bool(fit.success)

    if n_jobs > 1 and curves.shape[0] > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, curves.shape[0])) as executor:
            fits = list(executor.map(fit_one, range(curves.shape[0])))
    else:
        fits = [fit_one(index) for index in range(curves.shape[0])]
    weights = np.stack([fit[0] for fit in fits])
    amplitudes_normalized = weights[:, : common.size]
    noise_normalized = weights[:, -1]
    return CommonSlopeEDCFit(
        amplitudes_normalized_edc=amplitudes_normalized,
        amplitudes_absolute_edc_energy=(
            amplitudes_normalized * normalization[:, np.newaxis]
        ),
        noise_normalized_edc_origin=noise_normalized,
        noise_absolute_edc_energy=noise_normalized * normalization,
        mse_db2=np.asarray([fit[1] for fit in fits]),
        success=np.asarray([fit[2] for fit in fits]),
    )


def edc_to_equivalent_rir_power_amplitudes(
    amplitudes_edc: ArrayLike,
    t60_s: ArrayLike,
    sample_rate_hz: float,
) -> NDArray[np.float64]:
    """Convert EDC coefficients to discrete-time RIR power amplitudes.

    ``amplitudes_edc`` and ``t60_s`` must broadcast on their final slope axis.
    The returned power amplitudes use the same absolute energy units multiplied
    by ``1 - exp(-6 log(10)/(fs*T60))``. They remain band-filter dependent and
    are not numerically identical to STFT-bin variance amplitudes.

    Returns
    -------
    ndarray
        Equivalent unit-origin per-sample RIR power amplitudes, with the
        broadcast shape of ``amplitudes_edc`` and ``t60_s``.
    """

    amplitudes = np.asarray(amplitudes_edc, dtype=np.float64)
    t60 = np.asarray(t60_s, dtype=np.float64)
    if np.any(amplitudes < 0.0) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes_edc must contain finite non-negative values.")
    if np.any(t60 <= 0.0) or not np.all(np.isfinite(t60)):
        raise ValueError("t60_s must contain finite positive values.")
    factor = -np.expm1(-6.0 * np.log(10.0) / (sample_rate_hz * t60))
    return amplitudes * factor
