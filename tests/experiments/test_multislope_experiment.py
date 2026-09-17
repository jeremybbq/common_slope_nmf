import numpy as np

from common_slope_nmf import exponential_variance, t60_to_rate
from experiments.synthetic.validate_multislope_sage import (
    N_RIRS,
    generate_multislope_data,
)


def test_seeded_multislope_generator_matches_requested_model():
    data = generate_multislope_data()

    assert data.t60_s.shape == (1, 2)
    assert data.amplitudes_db.shape == (N_RIRS, 2)
    assert data.noise_floor_db.shape == (N_RIRS,)
    assert data.variance.shape == (N_RIRS, 1, data.times_s.size)
    assert data.observed_power.shape == data.variance.shape
    assert np.all(np.diff(data.t60_s[0]) > 0.0)
    assert np.all((data.t60_s > 0.5) & (data.t60_s < 3.0))

    realized_correlation = np.corrcoef(data.amplitudes_db.T)[0, 1]
    assert -0.83 < realized_correlation < -0.77
    assert np.all((data.amplitudes_db > -20.0) & (data.amplitudes_db < 0.0))
    assert np.mean(
        (data.noise_floor_db > -42.0)
        & (data.noise_floor_db < -38.0)
    ) > 0.99

    expected_variance = exponential_variance(
        data.times_s,
        t60_to_rate(data.t60_s),
        10.0 ** (data.amplitudes_db[:, np.newaxis, :] / 10.0),
        noise_floor=10.0 ** (data.noise_floor_db[:, np.newaxis] / 10.0),
    )
    np.testing.assert_array_equal(data.variance, expected_variance)
