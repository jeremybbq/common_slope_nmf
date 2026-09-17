import numpy as np
import pytest

from common_slope_nmf import exponential_variance
from common_slope_nmf.synth import (
    sample_complex_gaussian,
    sample_multislope_data,
    sample_power,
    sample_simplex_amplitudes,
    sample_t60,
)


def test_complex_gaussian_power_has_calibrated_exponential_statistics():
    variances = np.array([1.0, 1e-1, 1e-3, 1e-6])
    coefficients = sample_complex_gaussian(
        variances,
        rng=np.random.default_rng(20260724),
        n_realizations=50_000,
    )

    normalized_power = np.abs(coefficients) ** 2 / variances
    np.testing.assert_allclose(
        np.mean(normalized_power, axis=0), 1.0, atol=0.025, rtol=0.0
    )
    np.testing.assert_allclose(
        np.var(normalized_power, axis=0), 1.0, atol=0.07, rtol=0.0
    )
    np.testing.assert_allclose(
        np.median(normalized_power, axis=0),
        np.log(2.0),
        atol=0.025,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.quantile(normalized_power, 0.95, axis=0),
        -np.log(0.05),
        atol=0.12,
        rtol=0.0,
    )


def test_real_and_imaginary_parts_each_have_half_the_total_variance():
    variances = np.array([1.0, 1e-2, 1e-6])
    coefficients = sample_complex_gaussian(
        variances,
        rng=np.random.default_rng(101),
        n_realizations=50_000,
    )
    component_scale = np.sqrt(variances / 2.0)

    normalized_real = coefficients.real / component_scale
    normalized_imaginary = coefficients.imag / component_scale

    np.testing.assert_allclose(
        np.mean(normalized_real, axis=0), 0.0, atol=0.02, rtol=0.0
    )
    np.testing.assert_allclose(
        np.mean(normalized_imaginary, axis=0), 0.0, atol=0.02, rtol=0.0
    )
    np.testing.assert_allclose(
        np.var(normalized_real, axis=0), 1.0, atol=0.025, rtol=0.0
    )
    np.testing.assert_allclose(
        np.var(normalized_imaginary, axis=0), 1.0, atol=0.025, rtol=0.0
    )


def test_seeded_sampling_is_reproducible_and_power_preserves_shape():
    variance = np.array([[1.0, 0.5], [0.1, 0.01]])

    first = sample_power(
        variance, rng=np.random.default_rng(42), n_realizations=3
    )
    second = sample_power(
        variance, rng=np.random.default_rng(42), n_realizations=3
    )

    assert first.shape == (3, 2, 2)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0.0)


@pytest.mark.parametrize("variance", [0.0, -1.0, np.inf, np.nan, 1.0 + 1.0j])
def test_sampling_rejects_invalid_variance(variance):
    with pytest.raises(ValueError):
        sample_complex_gaussian(variance)


@pytest.mark.parametrize("n_realizations", [0, -1])
def test_sampling_rejects_nonpositive_realization_count(n_realizations):
    with pytest.raises(ValueError):
        sample_complex_gaussian(1.0, n_realizations=n_realizations)


def test_sampling_rejects_noninteger_realization_count_and_legacy_rng():
    with pytest.raises(TypeError):
        sample_complex_gaussian(1.0, n_realizations=1.5)
    with pytest.raises(TypeError, match="Generator"):
        sample_complex_gaussian(1.0, rng=np.random.RandomState(0))


def test_simplex_amplitudes_have_fixed_sum_and_corner_coverage():
    amplitudes = sample_simplex_amplitudes(
        900,
        1,
        3,
        concentration=0.2,
        rng=np.random.default_rng(4),
    )[:, 0]

    np.testing.assert_allclose(np.sum(amplitudes, axis=1), 1.0, atol=2e-16)
    assert np.mean(np.max(amplitudes, axis=1) > 0.8) > 0.55
    dominant_counts = np.bincount(np.argmax(amplitudes, axis=1), minlength=3)
    assert np.all(dominant_counts > 240)


def test_sample_t60_enforces_sorted_minimum_separation():
    t60_s = sample_t60(
        4,
        3,
        (0.4, 3.2),
        min_separation_s=0.35,
        rng=np.random.default_rng(18),
    )

    assert t60_s.shape == (4, 3)
    assert np.all((t60_s > 0.4) & (t60_s < 3.2))
    assert np.all(np.diff(t60_s, axis=1) >= 0.35)


def test_multislope_data_matches_exact_model_and_is_reproducible():
    times_s = np.arange(51, dtype=np.float64) * 0.01
    arguments = dict(
        times_s=times_s,
        n_rirs=30,
        n_frequencies=2,
        n_components=3,
        t60_range_s=(0.4, 2.5),
        amplitude_concentration=0.25,
        noise_mean_db=-35.0,
        noise_std_db=1.0,
        min_t60_separation_s=0.2,
    )

    first = sample_multislope_data(
        **arguments, rng=np.random.default_rng(82)
    )
    second = sample_multislope_data(
        **arguments, rng=np.random.default_rng(82)
    )

    assert first.t60_s.shape == (2, 3)
    assert first.amplitudes.shape == (30, 2, 3)
    assert first.variance.shape == (30, 2, times_s.size)
    assert first.dominant_component.shape == (30, 2)
    np.testing.assert_allclose(np.sum(first.amplitudes, axis=2), 1.0)
    expected_variance = exponential_variance(
        times_s,
        first.rates_per_s,
        first.amplitudes,
        noise_floor=first.noise_floor,
    )
    np.testing.assert_array_equal(first.variance, expected_variance)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.t60_s, second.t60_s)
