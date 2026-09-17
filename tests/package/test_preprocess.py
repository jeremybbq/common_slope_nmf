import numpy as np
import pytest

from common_slope_nmf.preprocess import (
    fit_coarse_decay,
    head_power,
    tail_power,
)


def test_head_and_tail_power_average_requested_frames():
    power = np.arange(1, 25, dtype=np.float64).reshape(2, 1, 12)

    np.testing.assert_allclose(
        head_power(power, 3), np.mean(power[:, :, :3], axis=2)
    )
    np.testing.assert_allclose(
        tail_power(power, 4), np.mean(power[:, :, -4:], axis=2)
    )


def test_pooled_log_fit_discards_varying_rir_intercepts():
    times_s = np.arange(101, dtype=np.float64) * 0.01
    rates_per_s = np.array([3.0, 8.0])
    amplitudes = np.array(
        [[0.02, 5.0], [0.3, 1.2], [4.0, 0.05], [1.1, 0.8]]
    )
    power = amplitudes[:, :, np.newaxis] * np.exp(
        -rates_per_s[np.newaxis, :, np.newaxis] * times_s
    )

    fit = fit_coarse_decay(power, times_s, n_tail_frames=7)

    np.testing.assert_allclose(fit.rate_per_s, rates_per_s, rtol=2e-15)
    np.testing.assert_allclose(fit.r_squared, 1.0, atol=2e-15)
    np.testing.assert_array_equal(fit.frame_mask[:, -7:], False)
    np.testing.assert_array_equal(fit.frame_mask[:, :-7], True)


def test_floor_margin_uses_one_common_mask_per_frequency():
    times_s = np.arange(81, dtype=np.float64) * 0.01
    rates_per_s = np.array([4.0, 7.0])
    amplitudes = np.array(
        [[1.0, 0.4], [0.5, 1.2], [1.5, 0.2]]
    )
    floor = np.array([[1e-2, 2e-2], [2e-2, 1e-2], [1.5e-2, 1.5e-2]])
    power = (
        amplitudes[:, :, np.newaxis]
        * np.exp(-rates_per_s[np.newaxis, :, np.newaxis] * times_s)
        + floor[:, :, np.newaxis]
    )

    fit = fit_coarse_decay(
        power,
        times_s,
        noise_floor=floor,
        n_tail_frames=5,
        floor_margin_db=6.0,
    )

    expected_mask = np.mean(power, axis=0) > (
        np.mean(floor, axis=0)[:, np.newaxis] * 10.0 ** 0.6
    )
    expected_mask[:, -5:] = False
    np.testing.assert_array_equal(fit.frame_mask, expected_mask)
    assert np.all(fit.rate_per_s > 0.0)


def test_coarse_fit_rejects_nondecaying_power():
    with pytest.raises(ValueError, match="positive decay rate"):
        fit_coarse_decay(
            np.ones((2, 1, 20)),
            np.arange(20, dtype=np.float64) * 0.01,
        )
