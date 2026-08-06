import numpy as np
import pytest
from scipy.optimize import minimize

from common_slope_nmf import (
    exponential_atoms,
    fixed_dictionary_sage,
    gaussian_variance_nll,
    is_divergence,
    sample_power,
    t60_to_rate,
)


def _two_decay_dictionary(times_s):
    decay_atoms = exponential_atoms(
        times_s, t60_to_rate(np.array([0.25, 0.8]))
    )
    return np.vstack([decay_atoms, np.ones((1, times_s.size))])


def test_single_component_amplitudes_are_exact_after_one_sweep():
    times_s = np.arange(121, dtype=np.float64) * 0.01
    dictionary = exponential_atoms(times_s, t60_to_rate(0.6))
    true_amplitudes = np.array([[0.2], [0.5], [1.0], [2.0]])
    power = true_amplitudes @ dictionary

    result = fixed_dictionary_sage(
        power,
        dictionary,
        initial_amplitudes=np.full_like(true_amplitudes, 7.0),
        max_iter=5,
        tol=0.0,
    )

    np.testing.assert_allclose(
        result.amplitudes, true_amplitudes, rtol=2e-14, atol=0.0
    )
    np.testing.assert_allclose(
        result.variance, power, rtol=2e-14, atol=0.0
    )
    assert result.objective_history[1] == pytest.approx(0.0, abs=1e-27)


def test_two_components_and_weak_floor_recover_exact_variance():
    times_s = np.arange(121, dtype=np.float64) * 0.01
    dictionary = _two_decay_dictionary(times_s)
    true_amplitudes = np.array(
        [
            [1.00, 0.08, 1e-5],
            [0.10, 1.00, 3e-5],
            [0.80, 0.30, 1e-4],
            [0.25, 0.70, 3e-6],
        ]
    )
    power = true_amplitudes @ dictionary

    result = fixed_dictionary_sage(
        power,
        dictionary,
        initial_amplitudes=np.ones_like(true_amplitudes),
        max_iter=70_000,
        tol=1e-13,
    )

    assert result.converged
    assert np.max(np.diff(result.objective_history)) <= 1e-12
    np.testing.assert_allclose(
        result.variance, result.amplitudes @ dictionary, rtol=1e-15
    )
    np.testing.assert_allclose(
        result.amplitudes, true_amplitudes, rtol=3e-4, atol=1e-12
    )
    np.testing.assert_allclose(
        result.variance, power, rtol=3e-5, atol=1e-12
    )


@pytest.mark.parametrize(
    ("t60_s", "amplitudes", "end_s"),
    [
        ([0.55, 0.60], [1.0, 0.7, 1e-3], 1.2),
        ([0.25, 0.80], [1.0, 0.1, 1e-5], 0.25),
        ([0.25, 0.80], [1.0, 1e-3, 1e-4], 1.2),
    ],
)
def test_difficult_cases_remain_finite_and_monotone(
    t60_s, amplitudes, end_s
):
    times_s = np.arange(0.0, end_s + 0.005, 0.01)
    dictionary = np.vstack(
        [
            exponential_atoms(times_s, t60_to_rate(t60_s)),
            np.ones((1, times_s.size)),
        ]
    )
    power = np.asarray(amplitudes)[np.newaxis, :] @ dictionary

    result = fixed_dictionary_sage(power, dictionary, max_iter=500, tol=0.0)

    assert np.all(np.isfinite(result.amplitudes))
    assert np.all(np.isfinite(result.variance))
    assert result.objective_history[-1] < result.objective_history[0]
    assert np.max(np.diff(result.objective_history)) <= 1e-12


def test_seeded_stochastic_fit_matches_independent_scipy_reference():
    times_s = np.arange(121, dtype=np.float64) * 0.01
    dictionary = _two_decay_dictionary(times_s)
    true_amplitudes = np.array([[0.8, 0.3, 1e-4]])
    true_variance = true_amplitudes @ dictionary
    power = sample_power(true_variance, rng=np.random.default_rng(7))

    result = fixed_dictionary_sage(
        power, dictionary, max_iter=5_000, tol=1e-12
    )

    def objective_and_gradient(log_amplitudes):
        amplitudes = np.exp(log_amplitudes)
        variance = amplitudes @ dictionary
        derivative_variance = 1.0 / variance - power[0] / variance**2
        gradient = amplitudes * (dictionary @ derivative_variance)
        return (
            gaussian_variance_nll(power[0], variance),
            gradient,
        )

    reference = minimize(
        objective_and_gradient,
        np.log(result.amplitudes[0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-40.0, 10.0)] * dictionary.shape[0],
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 10_000},
    )
    reference_variance = np.exp(reference.x) @ dictionary

    assert reference.success
    assert result.converged
    assert is_divergence(power[0], result.variance[0]) == pytest.approx(
        is_divergence(power[0], reference_variance), abs=1e-7
    )


def test_solver_rejects_invalid_shapes_and_controls():
    with pytest.raises(ValueError, match="same frame count"):
        fixed_dictionary_sage(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(ValueError, match="shape"):
        fixed_dictionary_sage(
            np.ones((2, 3)),
            np.ones((2, 3)),
            initial_amplitudes=np.ones((1, 2)),
        )
    with pytest.raises(ValueError, match="positive"):
        fixed_dictionary_sage(np.ones((1, 3)), np.array([[1.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        fixed_dictionary_sage(np.ones((1, 3)), np.ones((1, 3)), tol=-1.0)
