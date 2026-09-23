import numpy as np
import pytest

from common_slope_nmf.loss import is_divergence


def test_is_divergence_is_zero_at_identity_and_scale_invariant():
    power = np.array([[1e-6, 0.2, 3.0], [0.01, 2.0, 10.0]])
    variance = np.array([[2e-6, 0.1, 4.0], [0.02, 1.5, 8.0]])

    np.testing.assert_array_equal(
        is_divergence(power, power, reduction="none"),
        np.zeros_like(power),
    )
    assert is_divergence(power, variance) == pytest.approx(
        is_divergence(1e7 * power, 1e7 * variance), rel=1e-14
    )


def test_is_divergence_matches_direct_formula():
    power = np.array([0.2, 1.0, 3.0])
    variance = np.array([0.4, 0.8, 2.0])
    ratio = power / variance

    assert is_divergence(power, variance) == pytest.approx(
        np.sum(ratio - np.log(ratio) - 1.0)
    )
    assert is_divergence(power, variance, reduction="mean") == pytest.approx(
        np.mean(ratio - np.log(ratio) - 1.0)
    )


def test_is_divergence_remains_finite_for_extreme_small_ratio():
    power = np.array([1e-300, 1e-200])
    variance = np.array([1.0, 1e100])

    divergence = is_divergence(power, variance, reduction="none")

    expected_log_ratio = np.log(power) - np.log(variance)
    expected = np.expm1(expected_log_ratio) - expected_log_ratio
    np.testing.assert_allclose(divergence, expected, rtol=1e-15)
    assert np.all(np.isfinite(divergence))


def test_none_reduction_preserves_shape():
    power = np.array([[1.0, 2.0], [0.5, 0.25]])
    variance = np.array([[0.8, 1.5], [0.4, 0.5]])

    values = is_divergence(power, variance, reduction="none")

    assert values.shape == power.shape
    assert np.all(values >= 0.0)


@pytest.mark.parametrize(
    ("power", "variance"),
    [
        ([0.0, 1.0], [1.0, 1.0]),
        ([1.0, -1.0], [1.0, 1.0]),
        ([1.0, np.nan], [1.0, 1.0]),
    ],
)
def test_objectives_reject_invalid_values(power, variance):
    with pytest.raises(ValueError):
        is_divergence(power, variance)


def test_objectives_reject_mismatched_shapes_and_unknown_reduction():
    with pytest.raises(ValueError, match="same shape"):
        is_divergence(np.ones((2, 3)), np.ones((4, 3)))
    with pytest.raises(ValueError, match="reduction"):
        is_divergence(1.0, 1.0, reduction="median")
