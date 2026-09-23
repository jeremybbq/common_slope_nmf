import numpy as np

from common_slope_nmf.baseline import (
    determine_common_decay_times,
    edc_to_equivalent_rir_power_amplitudes,
)


def test_common_time_clustering_uses_histogram_modes():
    estimates = np.array(
        [
            [0.49, 1.48],
            [0.51, 1.51],
            [0.52, 1.52],
            [0.80, 1.49],
        ]
    )

    common, cluster_sizes = determine_common_decay_times(
        estimates, 2, histogram_resolution_s=0.05, seed=42
    )

    np.testing.assert_allclose(common, [0.525, 1.475], atol=1e-14)
    np.testing.assert_array_equal(cluster_sizes, [4, 4])


def test_edc_to_rir_power_conversion_matches_geometric_sum():
    sample_rate_hz = 1_000.0
    t60_s = np.array([0.5, 1.5])
    edc_amplitudes = np.array([[2.0, 4.0]])
    power = edc_to_equivalent_rir_power_amplitudes(
        edc_amplitudes, t60_s, sample_rate_hz
    )
    ratios = np.exp(-6.0 * np.log(10.0) / (sample_rate_hz * t60_s))

    np.testing.assert_allclose(power / (1.0 - ratios), edc_amplitudes)


def test_package_does_not_export_baseline_helpers():
    import common_slope_nmf

    for name in (
        "determine_common_decay_times",
        "edc_to_equivalent_rir_power_amplitudes",
    ):
        assert not hasattr(common_slope_nmf, name)
