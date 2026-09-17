import pickle

import numpy as np

from common_slope_nmf.georg_baselines import (
    GEORG_BANDWIDTH_FACTOR,
    band_energy_decay_curves,
    common_slope_edc_kernel,
    determine_common_decay_times,
    edc_to_equivalent_rir_power_amplitudes,
    fit_common_slope_edcs,
    load_decayfitnet_normfactor,
)


def test_band_energy_decay_prefers_passband_tone():
    sample_rate_hz = 8_000.0
    times_s = np.arange(4_000) / sample_rate_hz
    rirs = np.stack(
        (
            np.sin(2.0 * np.pi * 1_000.0 * times_s),
            np.sin(2.0 * np.pi * 100.0 * times_s),
        )
    )

    edcs = band_energy_decay_curves(rirs, sample_rate_hz, 1_000.0)

    assert GEORG_BANDWIDTH_FACTOR == np.sqrt(1.5)
    assert edcs.shape == (2, 3_980)
    assert edcs[0, 0] > 1_000.0 * edcs[1, 0]
    assert np.all(np.diff(edcs, axis=1) <= 0.0)


def test_common_time_clustering_uses_histogram_modes():
    estimates = np.array(
        [
            [0.49, 1.48],
            [0.51, 1.51],
            [0.52, 1.52],
            [0.80, 1.49],
        ]
    )

    common, clusters = determine_common_decay_times(
        estimates, 2, histogram_resolution_s=0.05, seed=42
    )

    np.testing.assert_allclose(common, [0.525, 1.475], atol=1e-14)
    assert len(clusters) == 2
    assert sum(cluster.size for cluster in clusters) == estimates.size


def test_common_slope_edc_fit_recovers_exact_decay_coefficients():
    sample_rate_hz = 200.0
    length = 301
    times_s = np.arange(length) / sample_rate_hz
    t60_s = np.array([0.40, 1.20])
    decay_atoms = np.exp(
        -np.log(1e6) * times_s[:, np.newaxis] / t60_s[np.newaxis, :]
    )
    amplitudes = np.array([[0.7, 0.3], [0.2, 0.8]])
    edcs = amplitudes @ decay_atoms.T

    result = fit_common_slope_edcs(edcs, t60_s, sample_rate_hz)

    assert np.all(result.success)
    np.testing.assert_allclose(
        result.amplitudes_normalized_edc,
        amplitudes,
        atol=5e-4,
    )
    assert np.max(result.noise_normalized_edc_origin) < 3e-6
    assert np.max(result.mse_db2) < 1e-6


def test_edc_to_rir_power_conversion_matches_geometric_sum():
    sample_rate_hz = 1_000.0
    t60_s = np.array([0.5, 1.5])
    edc_amplitudes = np.array([[2.0, 4.0]])
    power = edc_to_equivalent_rir_power_amplitudes(
        edc_amplitudes, t60_s, sample_rate_hz
    )
    ratios = np.exp(-6.0 * np.log(10.0) / (sample_rate_hz * t60_s))

    np.testing.assert_allclose(power / (1.0 - ratios), edc_amplitudes)


def test_plain_python_decayfitnet_transform_is_supported(tmp_path):
    path = tmp_path / "input_transform.pkl"
    with path.open("wb") as stream:
        pickle.dump({"edcs_db_normfactor": 141.5}, stream)

    assert load_decayfitnet_normfactor(path) == 141.5
