import inspect

import numpy as np
import pytest

from common_slope_nmf import cw_decay_sage, decay_sage
from experiments.synthetic_decay_robustness import (
    frequency_batches,
    order_decay_components,
)


def test_decay_estimators_use_pragmatic_outer_tolerance_default():
    assert inspect.signature(decay_sage).parameters["tol"].default == 1e-6
    assert inspect.signature(cw_decay_sage).parameters["tol"].default == 1e-6
    assert inspect.signature(decay_sage).parameters["decay_tol"].default is None
    assert (
        inspect.signature(cw_decay_sage).parameters["decay_tol"].default
        is None
    )


def test_frequency_batches_cover_all_bins_once():
    batches = frequency_batches(11, 4)

    assert batches == [slice(0, 4), slice(4, 8), slice(8, 11)]
    covered = np.concatenate(
        [np.arange(batch.start, batch.stop) for batch in batches]
    )
    np.testing.assert_array_equal(covered, np.arange(11))


@pytest.mark.parametrize("n_frequencies,batch_size", [(0, 1), (2, 0)])
def test_frequency_batches_reject_nonpositive_sizes(n_frequencies, batch_size):
    with pytest.raises(ValueError, match="positive"):
        frequency_batches(n_frequencies, batch_size)


def test_order_decay_components_applies_frequency_specific_permutations():
    t60_s = np.array([[2.0, 0.5], [0.8, 2.5]])
    amplitudes = np.array(
        [
            [[20.0, 5.0], [8.0, 25.0]],
            [[21.0, 6.0], [9.0, 26.0]],
        ]
    )

    ordered_t60_s, ordered_amplitudes, order = order_decay_components(
        t60_s, amplitudes
    )

    np.testing.assert_array_equal(order, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(ordered_t60_s, [[0.5, 2.0], [0.8, 2.5]])
    np.testing.assert_array_equal(
        ordered_amplitudes,
        [
            [[5.0, 20.0], [8.0, 25.0]],
            [[6.0, 21.0], [9.0, 26.0]],
        ],
    )


@pytest.mark.parametrize(
    "t60_s,amplitudes",
    [
        (np.ones(2), np.ones((2, 1, 2))),
        (np.ones((1, 2)), np.ones((2, 2, 1))),
    ],
)
def test_order_decay_components_rejects_shape_mismatch(t60_s, amplitudes):
    with pytest.raises(ValueError, match="shape"):
        order_decay_components(t60_s, amplitudes)
