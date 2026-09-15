import numpy as np
import pytest

from common_slope_nmf import (
    decay_sage,
    exponential_variance,
    cw_decay_sage,
    t60_to_rate,
    update_rates_bisection,
    update_rates_newton,
)


@pytest.mark.parametrize(
    "rate_updater", [update_rates_newton, update_rates_bisection]
)
def test_weighted_rate_profile_recovers_exact_rate_and_amplitude(rate_updater):
    times_s = np.arange(81, dtype=np.float64) * 0.0125
    true_rates_per_s = np.array([2.5, 8.0])
    true_amplitudes = np.array([[0.3, 1.1], [1.4, 0.55]])
    posterior_power = true_amplitudes[:, :, np.newaxis] * np.exp(
        -true_rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )
    frame_weights = 0.1 + np.linspace(0.0, 1.0, times_s.size) ** 2
    surrogate_weights = np.empty_like(posterior_power)
    surrogate_weights[0, 0] = frame_weights
    surrogate_weights[0, 1] = frame_weights[::-1]
    surrogate_weights[1, 0] = 0.3 + np.sin(times_s) ** 2
    surrogate_weights[1, 1] = 0.2 + np.cos(times_s) ** 2

    rates_per_s, amplitudes = rate_updater(
        posterior_power,
        times_s,
        np.array([6.0, 4.0]),
        (1.0, 12.0),
        surrogate_weights=surrogate_weights,
    )

    np.testing.assert_allclose(rates_per_s, true_rates_per_s, rtol=3e-11)
    np.testing.assert_allclose(amplitudes, true_amplitudes, rtol=3e-11)


def test_rate_solvers_agree_for_interior_and_boundary_optima():
    times_s = np.arange(101, dtype=np.float64) * 0.01
    generating_rates_per_s = np.array([3.0, 0.5, 15.0])
    generating_amplitudes = np.array(
        [[0.4, 0.8, 1.2], [1.4, 0.7, 0.3]]
    )
    posterior_power = generating_amplitudes[:, :, np.newaxis] * np.exp(
        -generating_rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )
    initial_rates_per_s = np.array([9.0, 5.0, 2.0])
    lower_rates_per_s = np.ones(3)
    upper_rates_per_s = np.full(3, 10.0)
    expected_rates_per_s = np.array([3.0, 1.0, 10.0])
    amplitude_scale = np.mean(
        np.exp(
            (expected_rates_per_s - generating_rates_per_s)[:, np.newaxis]
            * times_s
        ),
        axis=1,
    )
    expected_amplitudes = generating_amplitudes * amplitude_scale

    newton_rates, newton_amplitudes = update_rates_newton(
        posterior_power,
        times_s,
        initial_rates_per_s,
        (lower_rates_per_s, upper_rates_per_s),
    )
    bisection_rates, bisection_amplitudes = update_rates_bisection(
        posterior_power,
        times_s,
        initial_rates_per_s,
        (lower_rates_per_s, upper_rates_per_s),
    )

    np.testing.assert_allclose(
        newton_rates, expected_rates_per_s, rtol=2e-12
    )
    np.testing.assert_allclose(
        bisection_rates, expected_rates_per_s, rtol=2e-12
    )
    np.testing.assert_allclose(
        newton_amplitudes, expected_amplitudes, rtol=2e-12
    )
    np.testing.assert_allclose(
        bisection_amplitudes, expected_amplitudes, rtol=2e-12
    )


@pytest.mark.parametrize("rate_method", ["newton", "bisection"])
def test_decay_sage_dispatches_both_rate_methods(rate_method):
    times_s = np.arange(81, dtype=np.float64) * 0.01
    true_rates_per_s = np.array([[4.0], [9.0]])
    true_amplitudes = np.array([[[0.7], [1.2]], [[1.3], [0.4]]])
    power = exponential_variance(
        times_s, true_rates_per_s, true_amplitudes
    )

    result = decay_sage(
        power,
        times_s,
        np.full_like(true_rates_per_s, 6.0),
        rate_bounds_per_s=(1.0, 12.0),
        initial_amplitudes=np.ones_like(true_amplitudes),
        estimate_noise_floor=False,
        max_iter=2,
        tol=0.0,
        rate_method=rate_method,
    )

    np.testing.assert_allclose(
        result.rates_per_s, true_rates_per_s, rtol=3e-11
    )
    np.testing.assert_allclose(
        result.amplitudes, true_amplitudes, rtol=3e-11
    )
    assert np.max(np.diff(result.objective_history)) <= 1e-10


def test_single_decay_rates_and_amplitudes_recover_exact_model():
    times_s = np.arange(151, dtype=np.float64) * 0.01
    true_t60_s = np.array([[0.35], [0.55], [0.80], [1.10]])
    true_rates_per_s = t60_to_rate(true_t60_s)
    true_amplitudes = np.array(
        [
            [[0.4], [0.7], [1.0], [1.3]],
            [[1.4], [1.1], [0.8], [0.5]],
            [[0.9], [0.6], [1.2], [0.75]],
        ]
    )
    power = exponential_variance(
        times_s, true_rates_per_s, true_amplitudes
    )
    initial_rates_per_s = np.full_like(
        true_rates_per_s, t60_to_rate(0.65)
    )

    result = decay_sage(
        power,
        times_s,
        initial_rates_per_s,
        initial_amplitudes=np.full_like(true_amplitudes, 2.0),
        rate_bounds_per_s=(t60_to_rate(1.5), t60_to_rate(0.2)),
        estimate_noise_floor=False,
        max_iter=5,
        tol=0.0,
        diagnostic_interval=1,
    )

    np.testing.assert_allclose(
        result.rates_per_s, true_rates_per_s, rtol=2e-10
    )
    np.testing.assert_allclose(
        result.amplitudes, true_amplitudes, rtol=2e-10
    )
    np.testing.assert_allclose(result.variance, power, rtol=2e-10)
    np.testing.assert_array_equal(result.noise_floor, 0.0)
    assert result.rate_history_per_s.shape == (
        result.n_iter + 1,
        *true_rates_per_s.shape,
    )
    np.testing.assert_array_equal(
        result.rate_history_per_s[0], initial_rates_per_s
    )
    np.testing.assert_array_equal(
        result.rate_history_per_s[-1], result.rates_per_s
    )
    diagnostics = result.update_diagnostics
    assert diagnostics is not None
    np.testing.assert_array_equal(
        diagnostics.sweep_indices, np.arange(1, result.n_iter + 1)
    )
    assert diagnostics.scaled_total_error.shape == (
        result.n_iter,
        *power.shape,
    )
    assert diagnostics.component_weight.shape == (
        result.n_iter,
        power.shape[0],
        power.shape[1],
        true_rates_per_s.shape[1] + 1,
        power.shape[2],
    )
    assert diagnostics.profile_moment.shape == (
        result.n_iter,
        power.shape[0],
        power.shape[1],
        true_rates_per_s.shape[1],
    )
    assert diagnostics.profile_weight_sum.shape == (
        result.n_iter,
        power.shape[0],
        power.shape[1],
        true_rates_per_s.shape[1],
    )
    np.testing.assert_allclose(
        diagnostics.scaled_total_error[0], 0.0, atol=1e-14
    )
    np.testing.assert_allclose(diagnostics.component_weight[0, :, :, 0], 1.0)
    np.testing.assert_array_equal(diagnostics.component_weight[0, :, :, 1], 0.0)
    np.testing.assert_allclose(
        diagnostics.profile_moment[0, :, :, 0],
        times_s.size * true_amplitudes[:, :, 0],
        rtol=2e-12,
    )
    np.testing.assert_array_equal(
        diagnostics.profile_weight_sum, times_s.size
    )
    np.testing.assert_allclose(
        np.sum(diagnostics.component_weight, axis=3), 1.0, atol=1e-15
    )
    assert np.max(np.diff(result.objective_history)) <= 1e-10


def test_cw_decay_sage_zero_power_matches_ordinary_sage():
    times_s = np.arange(61, dtype=np.float64) * 0.01
    true_rates_per_s = np.array([[3.0, 9.0]])
    true_amplitudes = np.array([[[0.8, 0.15]], [[0.2, 1.1]]])
    true_floor = np.array([[2e-3], [4e-3]])
    power = exponential_variance(
        times_s,
        true_rates_per_s,
        true_amplitudes,
        noise_floor=true_floor,
    )
    common_arguments = dict(
        rate_bounds_per_s=(1.0, 12.0),
        initial_amplitudes=np.full_like(true_amplitudes, 0.4),
        initial_noise_floor=np.full_like(true_floor, 1e-2),
        max_iter=4,
        tol=0.0,
        diagnostic_interval=1,
    )

    exact = decay_sage(
        power,
        times_s,
        np.array([[5.0, 7.0]]),
        **common_arguments,
    )
    weighted = cw_decay_sage(
        power,
        times_s,
        np.array([[5.0, 7.0]]),
        component_weight_power=0.0,
        **common_arguments,
    )

    np.testing.assert_allclose(weighted.rates_per_s, exact.rates_per_s)
    np.testing.assert_allclose(weighted.amplitudes, exact.amplitudes)
    np.testing.assert_allclose(weighted.noise_floor, exact.noise_floor)
    np.testing.assert_allclose(weighted.variance, exact.variance)
    np.testing.assert_allclose(
        weighted.objective_history, exact.objective_history
    )


@pytest.mark.parametrize("component_weight_power", [1.0, 2.0])
def test_cw_decay_sage_records_weighted_profile_quantities(
    component_weight_power,
):
    times_s = np.arange(51, dtype=np.float64) * 0.01
    rates_per_s = np.array([[4.0, 10.0]])
    amplitudes = np.array([[[1.0, 0.1]], [[0.15, 0.9]]])
    floor = np.array([[1e-3], [2e-3]])
    power = exponential_variance(
        times_s, rates_per_s, amplitudes, noise_floor=floor
    )

    result = cw_decay_sage(
        power,
        times_s,
        np.array([[5.0, 8.0]]),
        rate_bounds_per_s=(1.0, 14.0),
        component_weight_power=component_weight_power,
        initial_amplitudes=np.full_like(amplitudes, 0.5),
        initial_noise_floor=np.full_like(floor, 5e-3),
        max_iter=3,
        tol=0.0,
        diagnostic_interval=1,
    )

    diagnostics = result.update_diagnostics
    assert diagnostics is not None
    assert np.all(np.isfinite(result.objective_history))
    assert np.all(diagnostics.profile_weight_sum > 0.0)
    assert np.all(diagnostics.profile_weight_sum <= times_s.size)
    np.testing.assert_allclose(
        diagnostics.profile_moment[-1] / diagnostics.profile_weight_sum[-1],
        result.amplitudes,
        rtol=2e-12,
    )


def test_decay_and_floor_updates_recover_separated_components():
    times_s = np.arange(201, dtype=np.float64) * 0.01
    true_t60_s = np.array(
        [
            [0.30, 0.90],
            [0.36, 1.00],
            [0.42, 1.10],
        ]
    )
    true_rates_per_s = t60_to_rate(true_t60_s)
    true_amplitudes = np.array(
        [
            [[1.0, 0.20], [0.8, 0.35], [1.2, 0.15]],
            [[0.3, 1.00], [0.5, 0.80], [0.25, 1.10]],
            [[0.9, 0.45], [1.1, 0.25], [0.7, 0.55]],
            [[0.4, 0.75], [0.6, 0.65], [0.5, 0.90]],
        ]
    )
    true_floor = np.array(
        [
            [1e-4, 2e-4, 1.5e-4],
            [2e-4, 1e-4, 2.5e-4],
            [1.2e-4, 1.8e-4, 1e-4],
            [2.2e-4, 1.3e-4, 1.7e-4],
        ]
    )
    power = exponential_variance(
        times_s,
        true_rates_per_s,
        true_amplitudes,
        noise_floor=true_floor,
    )
    lower_rates = np.broadcast_to(
        t60_to_rate([0.60, 1.40]), true_t60_s.shape
    )
    upper_rates = np.broadcast_to(
        t60_to_rate([0.20, 0.65]), true_t60_s.shape
    )

    result = decay_sage(
        power,
        times_s,
        true_rates_per_s,
        initial_amplitudes=true_amplitudes,
        initial_noise_floor=true_floor,
        rate_bounds_per_s=(lower_rates, upper_rates),
        max_iter=3,
        tol=0.0,
    )

    assert np.max(np.diff(result.objective_history)) <= 2e-10
    np.testing.assert_allclose(
        result.rates_per_s, true_rates_per_s, rtol=2e-12
    )
    np.testing.assert_allclose(
        result.amplitudes, true_amplitudes, rtol=2e-12
    )
    np.testing.assert_allclose(result.noise_floor, true_floor, rtol=2e-12)
    np.testing.assert_allclose(result.variance, power, rtol=2e-12)


def test_decay_sage_rejects_invalid_rate_bounds():
    power = np.ones((2, 3, 10))
    times_s = np.arange(10, dtype=np.float64) * 0.01
    rates_per_s = np.ones((3, 1))

    try:
        decay_sage(
            power,
            times_s,
            rates_per_s,
            rate_bounds_per_s=(2.0, 1.0),
        )
    except ValueError as error:
        assert "lower" in str(error)
    else:
        raise AssertionError("invalid rate bounds were accepted")


def test_decay_sage_rejects_unknown_rate_method():
    with pytest.raises(ValueError, match="rate_method"):
        decay_sage(
            np.ones((1, 1, 10)),
            np.arange(10, dtype=np.float64) * 0.01,
            np.ones((1, 1)),
            rate_bounds_per_s=(0.5, 2.0),
            rate_method="secant",
        )


@pytest.mark.parametrize("diagnostic_interval", [0, -1, 1.5, True])
def test_decay_sage_rejects_invalid_diagnostic_interval(
    diagnostic_interval,
):
    error_type = TypeError if diagnostic_interval in {1.5, True} else ValueError
    with pytest.raises(error_type, match="diagnostic_interval"):
        decay_sage(
            np.ones((1, 1, 10)),
            np.arange(10, dtype=np.float64) * 0.01,
            np.ones((1, 1)),
            rate_bounds_per_s=(0.5, 2.0),
            diagnostic_interval=diagnostic_interval,
        )


@pytest.mark.parametrize(
    ("component_weight_power", "error_type"),
    [(-1.0, ValueError), (np.inf, ValueError), (True, TypeError)],
)
def test_cw_decay_sage_rejects_invalid_weight_power(
    component_weight_power, error_type
):
    with pytest.raises(error_type, match="component_weight_power"):
        cw_decay_sage(
            np.ones((1, 1, 10)),
            np.arange(10, dtype=np.float64) * 0.01,
            np.ones((1, 1)),
            rate_bounds_per_s=(0.5, 2.0),
            component_weight_power=component_weight_power,
        )


@pytest.mark.parametrize(
    ("decay_tol", "error_type"),
    [(-1.0, ValueError), (np.inf, ValueError), (True, TypeError)],
)
def test_decay_sage_rejects_invalid_outer_decay_tolerance(
    decay_tol, error_type
):
    with pytest.raises(error_type, match="decay_tol"):
        cw_decay_sage(
            np.ones((1, 1, 10)),
            np.arange(10, dtype=np.float64) * 0.01,
            np.ones((1, 1)),
            rate_bounds_per_s=(0.5, 2.0),
            decay_tol=decay_tol,
        )


def test_outer_decay_tolerance_stops_when_either_gate_is_met():
    times_s = np.arange(61, dtype=np.float64) * 0.01
    power = exponential_variance(
        times_s,
        np.array([[3.0, 9.0]]),
        np.array([[[0.8, 0.2]], [[0.2, 0.9]]]),
        noise_floor=np.array([[2e-3], [3e-3]]),
    )
    common = dict(
        rate_bounds_per_s=(1.0, 12.0),
        component_weight_power=1.0,
        initial_amplitudes=np.full((2, 1, 2), 0.4),
        initial_noise_floor=np.full((2, 1), 1e-2),
        max_iter=2,
        tol=0.0,
    )

    decay_stopped = cw_decay_sage(
        power,
        times_s,
        np.array([[5.0, 7.0]]),
        decay_tol=1e6,
        **common,
    )
    objective_only = cw_decay_sage(
        power,
        times_s,
        np.array([[5.0, 7.0]]),
        decay_tol=None,
        **common,
    )

    assert decay_stopped.converged
    assert decay_stopped.n_iter == 1
    assert objective_only.n_iter == 2


def test_rate_update_rejects_zero_total_surrogate_weight():
    with pytest.raises(ValueError, match="positive frame sums"):
        update_rates_newton(
            np.ones((1, 1, 10)),
            np.arange(10, dtype=np.float64) * 0.01,
            np.ones(1),
            (0.5, 2.0),
            surrogate_weights=np.zeros((1, 1, 10)),
        )
