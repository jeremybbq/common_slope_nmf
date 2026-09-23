"""Common-slope clustering of DecayFitNet energy decay times.

DecayFitNet filtering, Schroeder integration, resampling, and network inference stay in an external DecayFitNet checkout. This module keeps the common-decay-time clustering from Georg Götz's MATLAB analysis that the Python toolbox does not provide.

References
----------
[1] G. Götz, S. J. Schlecht, and V. Pulkki, "Common-slope modeling of late reverberation," IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 31, pp. 3945-3957, 2023. https://doi.org/10.1109/TASLP.2023.3317572
[2] G. Götz, R. Falcón Pérez, S. J. Schlecht, and V. Pulkki, "Neural network for multi-exponential sound energy decay analysis," J. Acoust. Soc. Am., vol. 152, no. 2, pp. 942-953, 2022. https://doi.org/10.1121/10.0013416
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Cluster per-RIR energy decay times into Georg-style common slopes.

    Positive entries of ``t60_s`` are clustered by one-dimensional L1 k-medians. Each common time is the center of the most populated histogram bin within its sorted cluster.

    Returns
    -------
    common_t60_s, cluster_sizes
        Sorted shared energy decay times in seconds, shape ``(K,)``, and the number of positive DecayFitNet samples assigned to each cluster, shape ``(K,)``.
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
    cluster_sizes = np.empty(n_common_slopes, dtype=np.int64)
    for output_index, cluster_index in enumerate(cluster_order):
        members = positive[labels == cluster_index]
        if members.size == 0:
            raise RuntimeError("k-medians produced an empty common-slope cluster.")
        counts, _ = np.histogram(members, bins=edges)
        peak = int(np.argmax(counts))
        common[output_index] = np.mean(edges[peak : peak + 2])
        cluster_sizes[output_index] = members.size
    return common, cluster_sizes


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
