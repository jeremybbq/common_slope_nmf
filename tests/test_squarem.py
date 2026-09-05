import numpy as np
import pytest

from common_slope_nmf import (
    amplitude_sage,
    decay_sage,
    exponential_atoms,
    exponential_variance,
    t60_to_rate,
)
from common_slope_nmf.squarem import amplitude_squarem, decay_squarem


def _dictionary(times_s):
    return np.vstack(
        [
            exponential_atoms(times_s, t60_to_rate([0.25, 0.8])),
            np.ones((1, times_s.size)),
        ]
    )


def test_step_one_matches_two_complete_sage_sweeps():
    times_s = np.arange(61, dtype=np.float64) * 0.01
    atoms = _dictionary(times_s)
    true_amplitudes = np.array([[1.0, 0.08, 1e-4], [0.2, 0.7, 3e-4]])
    power = true_amplitudes @ atoms
    initial = np.ones_like(true_amplitudes)

    expected = amplitude_sage(
        power,
        atoms,
        initial_amplitudes=initial,
        max_iter=2,
        tol=0.0,
    )
    accelerated = amplitude_squarem(
        power,
        atoms,
        initial_amplitudes=initial,
        max_sweep_evaluations=2,
        objective_tol=0.0,
        fixed_point_tol=0.0,
        initial_step_max=1.0,
    )

    np.testing.assert_array_equal(accelerated.amplitudes, expected.amplitudes)
    np.testing.assert_array_equal(accelerated.variance, expected.variance)
    assert accelerated.n_sweep_evaluations == 2
    assert accelerated.n_accepted_extrapolations == 0
    assert accelerated.n_rejected_extrapolations == 0


def test_squarem_recovers_difficult_amplitudes_monotonically():
    times_s = np.arange(81, dtype=np.float64) * 0.01
    atoms = _dictionary(times_s)
    true_amplitudes = np.array([[1.0, 0.1, 1e-4], [0.2, 0.8, 3e-4]])
    power = true_amplitudes @ atoms

    result = amplitude_squarem(
        power,
        atoms,
        initial_amplitudes=np.ones_like(true_amplitudes),
        max_sweep_evaluations=400,
        objective_tol=1e-11,
        fixed_point_tol=1e-8,
    )

    assert result.converged
    assert np.max(np.diff(result.objective_history)) <= 1e-12
    assert result.n_accepted_extrapolations > 0
    np.testing.assert_allclose(
        result.amplitudes, true_amplitudes, rtol=2e-4, atol=1e-12
    )
    np.testing.assert_allclose(result.variance, power, rtol=1e-5)


def test_infeasible_or_nonmonotone_rows_fall_back_safely():
    times_s = np.arange(61, dtype=np.float64) * 0.01
    atoms = _dictionary(times_s)
    true_amplitudes = np.array(
        [
            [1.0, 0.08, 1e-5],
            [0.1, 1.0, 3e-5],
            [0.8, 0.3, 1e-4],
            [0.25, 0.7, 3e-6],
        ]
    )
    power = true_amplitudes @ atoms

    result = amplitude_squarem(
        power,
        atoms,
        initial_amplitudes=np.ones_like(true_amplitudes),
        max_sweep_evaluations=30,
        objective_tol=0.0,
        fixed_point_tol=0.0,
        initial_step_max=100.0,
        step_max_limit=100.0,
    )

    assert result.n_rejected_extrapolations > 0
    assert np.all(np.isfinite(result.amplitudes))
    assert np.all(result.amplitudes > 0.0)
    assert np.max(np.diff(result.objective_history)) <= 1e-12


def test_decay_step_one_matches_two_complete_sage_sweeps():
    times_s = np.arange(51, dtype=np.float64) * 0.01
    true_rates_per_s = np.array([[4.0], [9.0]])
    true_amplitudes = np.array([[[0.7], [1.2]], [[1.3], [0.4]]], dtype=np.float64)
    power = exponential_variance(times_s, true_rates_per_s, true_amplitudes)
    initial_rates_per_s = np.full_like(true_rates_per_s, 6.0)
    initial_amplitudes = np.ones_like(true_amplitudes)

    expected = decay_sage(
        power,
        times_s,
        initial_rates_per_s,
        rate_bounds_per_s=(1.0, 12.0),
        initial_amplitudes=initial_amplitudes,
        estimate_noise_floor=False,
        max_iter=2,
        tol=0.0,
    )
    accelerated = decay_squarem(
        power,
        times_s,
        initial_rates_per_s,
        rate_bounds_per_s=(1.0, 12.0),
        initial_amplitudes=initial_amplitudes,
        estimate_noise_floor=False,
        max_sweep_evaluations=2,
        objective_tol=0.0,
        fixed_point_tol=0.0,
        initial_step_max=1.0,
    )

    np.testing.assert_array_equal(accelerated.rates_per_s, expected.rates_per_s)
    np.testing.assert_array_equal(accelerated.amplitudes, expected.amplitudes)
    np.testing.assert_array_equal(accelerated.variance, expected.variance)
    np.testing.assert_array_equal(accelerated.noise_floor, 0.0)
    assert accelerated.n_sweep_evaluations == 2


def test_decay_squarem_preserves_constraints_and_original_objective():
    times_s = np.arange(41, dtype=np.float64) * 0.01
    true_rates_per_s = np.array([[5.0, 12.0], [4.0, 9.0]])
    true_amplitudes = np.array(
        [
            [[0.8, 0.2], [0.5, 0.4]],
            [[0.3, 0.7], [0.9, 0.1]],
        ]
    )
    true_floor = np.array([[1e-3, 2e-3], [3e-3, 1e-3]])
    power = exponential_variance(
        times_s,
        true_rates_per_s,
        true_amplitudes,
        noise_floor=true_floor,
    )

    result = decay_squarem(
        power,
        times_s,
        np.array([[7.0, 10.0], [6.0, 11.0]]),
        rate_bounds_per_s=(2.0, 15.0),
        initial_amplitudes=np.ones_like(true_amplitudes),
        initial_noise_floor=np.full_like(true_floor, 1e-2),
        max_sweep_evaluations=12,
        objective_tol=0.0,
        fixed_point_tol=0.0,
    )

    assert result.n_accepted_extrapolations > 0
    assert np.max(np.diff(result.objective_history)) <= 1e-10
    assert np.all((result.rates_per_s >= 2.0) & (result.rates_per_s <= 15.0))
    assert np.all(result.amplitudes > 0.0)
    assert np.all(result.noise_floor > 0.0)
    assert np.all(result.variance > 0.0)
    assert result.rate_history_per_s.shape[0] == result.objective_history.size


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_sweep_evaluations", 0, "positive"),
        ("objective_tol", -1.0, "non-negative"),
        ("fixed_point_tol", -1.0, "non-negative"),
        ("initial_step_max", 0.5, "at least one"),
        ("step_factor", 1.0, "greater than one"),
        ("step_max_limit", 0.5, "at least initial_step_max"),
    ],
)
def test_squarem_rejects_invalid_controls(name, value, message):
    arguments = {name: value}
    with pytest.raises((TypeError, ValueError), match=message):
        amplitude_squarem(np.ones((1, 3)), np.ones((1, 3)), **arguments)
