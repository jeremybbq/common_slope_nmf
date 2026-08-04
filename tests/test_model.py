import numpy as np
import pytest

from multislope_nmf.model import (
    exponential_atoms,
    exponential_variance,
    rate_to_t60,
    t60_to_rate,
)


def test_energy_t60_convention_and_db_slope():
    t60_s = 0.6
    amplitude = 2.5
    times_s = np.arange(121, dtype=np.float64) * 0.01
    rate_per_s = t60_to_rate(t60_s)

    variance = exponential_variance(times_s, rate_per_s, amplitude)
    normalized_db = 10.0 * np.log10(variance / variance[0])
    expected_db = -60.0 * times_s / t60_s

    np.testing.assert_allclose(variance[0], amplitude, rtol=1e-12)
    np.testing.assert_allclose(
        variance[60] / variance[0], 1e-6, rtol=1e-12
    )
    np.testing.assert_allclose(normalized_db, expected_db, atol=1e-12)


def test_t60_rate_round_trip_for_scalar_and_array():
    t60_s = np.array([0.25, 0.6, 1.2])

    np.testing.assert_allclose(rate_to_t60(t60_to_rate(t60_s)), t60_s)
    assert rate_to_t60(t60_to_rate(0.6)) == pytest.approx(0.6)


def test_atoms_have_explicit_component_axis_and_unit_origin():
    times_s = np.array([0.0, 0.1, 0.2])
    rates_per_s = np.array([2.0, 5.0])

    atoms = exponential_atoms(times_s, rates_per_s)

    assert atoms.shape == (2, 3)
    np.testing.assert_array_equal(atoms[:, 0], np.ones(2))
    assert np.all(np.diff(atoms, axis=-1) < 0.0)

    scalar_atom = exponential_atoms(times_s, 2.0)
    assert scalar_atom.shape == (1, 3)


def test_variance_broadcasts_shared_rates_over_rirs_and_adds_floor():
    times_s = np.array([0.0, 0.5])
    rates_per_s = np.array([[1.0, 3.0], [2.0, 4.0]])  # (F, K)
    amplitudes = np.ones((3, 2, 2))  # (R, F, K)
    floors = np.full((3, 2), 0.25)

    variance = exponential_variance(
        times_s, rates_per_s, amplitudes, noise_floor=floors
    )

    assert variance.shape == (3, 2, 2)
    np.testing.assert_allclose(variance[..., 0], 2.25)
    np.testing.assert_allclose(
        variance[0, 0, 1], np.exp(-0.5) + np.exp(-1.5) + 0.25
    )


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (t60_to_rate, 0.0),
        (t60_to_rate, -1.0),
        (rate_to_t60, np.inf),
    ],
)
def test_parameter_transforms_reject_invalid_values(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_forward_model_rejects_invalid_physical_parameters():
    with pytest.raises(ValueError, match="one-dimensional"):
        exponential_atoms([[0.0, 0.1]], [1.0])
    with pytest.raises(ValueError, match="positive"):
        exponential_variance([0.0, 0.1], [0.0], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        exponential_variance([0.0, 0.1], [1.0], [-1.0])
    with pytest.raises(ValueError, match="noise_floor"):
        exponential_variance([0.0, 0.1], [1.0], [1.0], noise_floor=-1.0)
