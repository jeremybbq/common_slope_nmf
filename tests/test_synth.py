import numpy as np
import pytest

from common_slope_nmf.synth import sample_complex_gaussian, sample_power


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
