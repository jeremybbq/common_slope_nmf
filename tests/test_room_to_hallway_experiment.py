import numpy as np

from common_slope_nmf import rate_to_t60, t60_to_rate
from experiments.roomToHallway_omni import (
    LONG_SLOPE_RGB,
    SHORT_SLOPE_RGB,
    STRONG_MIXTURE_RGB,
    WEAK_MIXTURE_RGB,
    FrequencyFit,
    amplitude_mixture_rgb,
    initial_parameters,
    nearest_frequency_indices,
    pilot_is_healthy,
    stratified_receiver_indices,
)


def test_amplitude_mixture_is_white_when_weak_and_uses_both_hues():
    reference_db = 0.0
    amplitudes = np.array(
        [
            [1e-5, 1e-5],
            [1.0, 1e-5],
            [1e-5, 1.0],
            [1.0, 1.0],
        ]
    )

    rgb = amplitude_mixture_rgb(
        amplitudes, reference_db=reference_db, dynamic_range_db=40.0
    )

    np.testing.assert_allclose(rgb[0], np.ones(3))
    np.testing.assert_allclose(rgb[1], SHORT_SLOPE_RGB)
    np.testing.assert_allclose(rgb[2], LONG_SLOPE_RGB)
    np.testing.assert_allclose(rgb[3], (SHORT_SLOPE_RGB + LONG_SLOPE_RGB) / 2.0)


def test_dark_amplitude_mixture_reaches_all_four_configured_corners():
    amplitudes = np.array(
        [
            [1e-4, 1e-4],
            [1.0, 1e-4],
            [1e-4, 1.0],
            [1.0, 1.0],
        ]
    )

    rgb = amplitude_mixture_rgb(
        amplitudes,
        reference_db=0.0,
        dynamic_range_db=40.0,
        dark_background=True,
    )

    np.testing.assert_allclose(rgb[0], WEAK_MIXTURE_RGB)
    np.testing.assert_allclose(rgb[1], SHORT_SLOPE_RGB)
    np.testing.assert_allclose(rgb[2], LONG_SLOPE_RGB)
    np.testing.assert_allclose(rgb[3], STRONG_MIXTURE_RGB)


def test_initial_parameters_split_signal_power_equally_with_floor():
    times_s = np.arange(250, dtype=np.float64) * 256.0 / 48_000.0
    rates = t60_to_rate(np.array([0.9]))
    amplitudes = np.array([0.3, 0.5, 0.8, 1.2])[:, np.newaxis, np.newaxis]
    floors = np.array([1e-5, 2e-5, 3e-5, 4e-5])[:, np.newaxis, np.newaxis]
    power = amplitudes * np.exp(-rates[np.newaxis, :, np.newaxis] * times_s)
    power += floors

    initial_rates, initial_amplitudes, initial_floor, coarse_t60, r_squared = (
        initial_parameters(power, times_s, 3, "equal-log-linear")
    )

    assert initial_rates.shape == (1, 3)
    assert initial_amplitudes.shape == (4, 1, 3)
    assert initial_floor.shape == (4, 1)
    np.testing.assert_allclose(initial_amplitudes[:, :, 0], initial_amplitudes[:, :, 1])
    np.testing.assert_allclose(initial_amplitudes[:, :, 1], initial_amplitudes[:, :, 2])
    np.testing.assert_allclose(rate_to_t60(initial_rates), np.full((1, 3), coarse_t60))
    assert np.all(initial_floor > 0.0)
    assert np.isfinite(r_squared)


def test_log_spaced_initialization_is_ordered_and_bounded():
    times_s = np.arange(250, dtype=np.float64) * 256.0 / 48_000.0
    power = np.exp(-t60_to_rate(0.8) * times_s)[np.newaxis, np.newaxis, :]
    power += 1e-6

    rates, _, _, _, _ = initial_parameters(power, times_s, 3, "log-spaced")
    t60_s = rate_to_t60(rates[0])

    assert np.all(np.diff(t60_s) > 0.0)
    assert 0.15 <= t60_s[0] < t60_s[-1] <= 3.0
    np.testing.assert_allclose(t60_s[1] ** 2, t60_s[0] * t60_s[2])


def test_frequency_and_receiver_pilot_selection_are_deterministic():
    frequencies = np.arange(2, 86, dtype=np.float64) * 93.75
    indices = nearest_frequency_indices(
        frequencies, np.array([250.0, 500.0, 1000.0, 8000.0])
    )
    np.testing.assert_array_equal(
        frequencies[indices], [281.25, 468.75, 1031.25, 7968.75]
    )

    conditions = np.repeat(np.arange(4), 101)
    selected = stratified_receiver_indices(conditions, 25)
    assert selected.size == 100
    np.testing.assert_array_equal(np.bincount(conditions[selected]), np.full(4, 25))


def test_pilot_health_checks_weighted_objective_growth():
    def fit(final_objective):
        return FrequencyFit(
            frequency_index=0,
            frequency_hz=281.25,
            n_components=2,
            initialization="equal-log-linear",
            coarse_t60_s=0.8,
            coarse_r_squared=0.9,
            initial_t60_s=np.array([0.8, 0.8]),
            estimated_t60_s=np.array([0.6, 1.0]),
            estimated_amplitudes=np.ones((4, 2)),
            estimated_noise_floor=np.full(4, 1e-4),
            objective_history=np.array([100.0, final_objective]),
            t60_history_s=np.array([[0.8, 0.8], [0.6, 1.0]]),
            n_iter=1,
            converged=False,
        )

    assert pilot_is_healthy([fit(109.0)])
    assert not pilot_is_healthy([fit(111.0)])
