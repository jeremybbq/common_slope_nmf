import numpy as np

from common_slope_nmf.sage import init_decay


def test_init_decay_uses_pooled_rate_head_amplitude_and_tail_floor():
    times_s = np.arange(101, dtype=np.float64) * 0.01
    true_rates_per_s = np.array([3.0, 7.0])
    rir_amplitudes = np.array(
        [[0.1, 2.0], [1.0, 0.3], [4.0, 0.05], [0.4, 1.2]]
    )
    power = rir_amplitudes[:, :, np.newaxis] * np.exp(
        -true_rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )

    rate_per_s, amplitudes, noise_floor = init_decay(
        power,
        times_s,
        n_head_frames=6,
        n_tail_frames=5,
    )

    expected_head_power = np.mean(power[:, :, :6], axis=2)
    expected_tail_power = np.mean(power[:, :, -5:], axis=2)
    assert rate_per_s.shape == (2,)
    assert amplitudes.shape == (4, 2)
    np.testing.assert_allclose(rate_per_s, true_rates_per_s, rtol=2e-15)
    np.testing.assert_allclose(amplitudes, expected_head_power)
    np.testing.assert_allclose(noise_floor, expected_tail_power)
