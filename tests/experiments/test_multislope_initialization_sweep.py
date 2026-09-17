import numpy as np
import pytest

from experiments.synthetic.sweep_multislope_decay_initialization import (
    equal_amplitude_initialization,
    initial_t60_pairs,
)


def test_initial_t60_pairs_include_all_component_orders():
    grid_s = np.array([0.5, 1.5, 3.0])

    pairs_s = initial_t60_pairs(grid_s)

    expected_s = np.array(
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [3.0, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
            [3.0, 1.5],
            [0.5, 3.0],
            [1.5, 3.0],
            [3.0, 3.0],
        ]
    )
    np.testing.assert_array_equal(pairs_s, expected_s)


def test_equal_amplitude_initialization_preserves_total_initial_power():
    power = np.arange(1, 25, dtype=np.float64).reshape(2, 1, 12)

    amplitudes = equal_amplitude_initialization(
        power, n_components=2, n_initial_frames=8
    )

    expected_power = np.mean(power[:, :, :8], axis=2)
    assert amplitudes.shape == (2, 1, 2)
    np.testing.assert_allclose(np.sum(amplitudes, axis=2), expected_power)
    np.testing.assert_allclose(amplitudes[:, :, 0], amplitudes[:, :, 1])


@pytest.mark.parametrize(
    "grid_s", [np.array([1.0]), np.array([0.5, -1.0]), np.array([0.5, np.nan])]
)
def test_initial_t60_pairs_reject_invalid_grids(grid_s):
    with pytest.raises(ValueError, match="t60_grid_s"):
        initial_t60_pairs(grid_s)
