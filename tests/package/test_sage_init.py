import numpy as np

from common_slope_nmf.sage import init_decay_sage


def test_decay_sage_init_uses_pooled_rate_equal_amplitudes_and_tail_floor():
    times_s = np.arange(101, dtype=np.float64) * 0.01
    true_rates_per_s = np.array([3.0, 7.0])
    rir_amplitudes = np.array(
        [[0.1, 2.0], [1.0, 0.3], [4.0, 0.05], [0.4, 1.2]]
    )
    power = rir_amplitudes[:, :, np.newaxis] * np.exp(
        -true_rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )

    init = init_decay_sage(
        power,
        times_s,
        3,
        rate_bounds_per_s=(1.0, 10.0),
        n_head_frames=6,
        n_tail_frames=5,
    )

    expected_head_power = np.mean(power[:, :, :6], axis=2)
    expected_tail_power = np.mean(power[:, :, -5:], axis=2)
    assert init.rates_per_s.shape == (2, 3)
    assert init.amplitudes.shape == (4, 2, 3)
    np.testing.assert_allclose(
        init.rates_per_s,
        np.repeat(true_rates_per_s[:, np.newaxis], 3, axis=1),
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        np.sum(init.amplitudes, axis=2), expected_head_power
    )
    np.testing.assert_allclose(init.amplitudes[:, :, 0], init.amplitudes[:, :, 1])
    np.testing.assert_allclose(init.amplitudes[:, :, 1], init.amplitudes[:, :, 2])
    np.testing.assert_allclose(init.noise_floor, expected_tail_power)
    np.testing.assert_allclose(init.fit.rate_per_s, true_rates_per_s, rtol=2e-15)


def test_decay_sage_init_clips_coarse_rate_to_common_component_bounds():
    times_s = np.arange(101, dtype=np.float64) * 0.01
    true_rates_per_s = np.array([0.4, 14.0])
    power = np.exp(
        -true_rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )
    lower = np.array([[1.0, 2.0], [1.0, 2.0]])
    upper = np.array([[9.0, 10.0], [9.0, 10.0]])

    init = init_decay_sage(
        power,
        times_s,
        2,
        rate_bounds_per_s=(lower, upper),
        n_tail_frames=5,
        floor_margin_db=None,
    )

    np.testing.assert_array_equal(init.rates_per_s[0], 2.0)
    np.testing.assert_array_equal(init.rates_per_s[1], 9.0)
    np.testing.assert_allclose(init.fit.rate_per_s, true_rates_per_s, rtol=3e-15)
