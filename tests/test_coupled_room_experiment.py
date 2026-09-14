import numpy as np

from experiments.fit_coupled_rooms import (
    FrequencyFit,
    _stack_fits,
    initial_decay_rates,
    pilot_is_healthy,
)
from common_slope_nmf import rate_to_t60, t60_to_rate


def _fit(index, initial_objective, final_objective, *, finite=True):
    t60_s = np.array([0.7, 1.4, 3.5])
    if not finite:
        t60_s[0] = np.nan
    return FrequencyFit(
        frequency_index=index,
        frequency_hz=250.0 * (index + 1),
        initial_t60_s=np.array([0.73, 1.43, 3.48]),
        t60_s=t60_s,
        amplitudes=np.ones((5, 3)),
        objective_history=np.array([initial_objective, final_objective]),
        t60_history_s=np.stack((np.array([0.73, 1.43, 3.48]), t60_s)),
        n_iter=1,
        converged=True,
    )


def test_pilot_health_allows_small_pseudo_sage_increase_only():
    assert pilot_is_healthy([_fit(0, 100.0, 90.0), _fit(1, 100.0, 109.0)])
    assert not pilot_is_healthy([_fit(0, 100.0, 111.0)])
    assert not pilot_is_healthy([_fit(0, 100.0, 90.0, finite=False)])


def test_stack_fits_preserves_frequency_and_pads_histories():
    fits = [_fit(0, 100.0, 80.0), _fit(1, 120.0, 90.0)]

    (
        frequencies,
        initial_t60_s,
        t60_s,
        amplitudes,
        objective,
        t60_history,
    ) = _stack_fits(fits, max_iter=3)

    np.testing.assert_array_equal(frequencies, [250.0, 500.0])
    assert initial_t60_s.shape == (2, 3)
    assert t60_s.shape == (2, 3)
    assert amplitudes.shape == (5, 2, 3)
    assert objective.shape == (2, 4)
    assert t60_history.shape == (2, 4, 3)
    np.testing.assert_array_equal(objective[:, :2], [[100.0, 80.0], [120.0, 90.0]])
    assert np.all(np.isnan(objective[:, 2:]))


def test_equal_log_linear_initialization_repeats_pooled_decay_rate():
    times_s = np.arange(101, dtype=np.float64) * 0.01
    true_t60_s = np.array([0.8, 1.6])
    rates_per_s = t60_to_rate(true_t60_s)
    amplitudes = np.array([[0.2, 1.5], [2.0, 0.4], [0.7, 3.0]])
    power = amplitudes[:, :, np.newaxis] * np.exp(
        -rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )

    initial_rates = initial_decay_rates(
        power, times_s, "equal-log-linear"
    )

    assert initial_rates.shape == (2, 3)
    np.testing.assert_allclose(
        rate_to_t60(initial_rates),
        np.repeat(true_t60_s[:, np.newaxis], 3, axis=1),
        rtol=2e-15,
    )
