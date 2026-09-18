import numpy as np
import pytest

from common_slope_nmf import (
    global_energy_onset,
    rir_stft_power,
    select_stft_frames,
)


def test_global_energy_onset_is_a_placeholder():
    with pytest.raises(NotImplementedError, match="placeholder"):
        global_energy_onset(np.zeros((2, 8)), 1_000.0)


def test_stft_power_preserves_explicit_units_and_shapes():
    rng = np.random.default_rng(5)
    rirs = rng.normal(size=(4, 1024))

    transformed = rir_stft_power(
        rirs,
        24_000.0,
        frame_size_samples=256,
        hop_size_samples=128,
        fft_size_samples=384,
        minimum_frequency_hz=62.5,
        maximum_frequency_hz=8_000.0,
    )

    assert transformed.observed_power.shape == (4, 128, 7)
    np.testing.assert_array_equal(transformed.frequencies_hz, np.arange(1, 129) * 62.5)
    np.testing.assert_allclose(np.diff(transformed.frame_time_s), 128.0 / 24_000.0)
    assert np.all(transformed.observed_power > 0.0)


def test_select_stft_frames_discards_and_resets_decay_origin():
    rng = np.random.default_rng(12)
    transformed = rir_stft_power(
        rng.normal(size=(2, 4096)),
        48_000.0,
        frame_size_samples=512,
        hop_size_samples=256,
        fft_size_samples=512,
        minimum_frequency_hz=187.5,
        maximum_frequency_hz=8_000.0,
    )

    selected = select_stft_frames(
        transformed, discard_initial_frames=3, discard_final_frames=4
    )

    np.testing.assert_array_equal(
        selected.observed_power, transformed.observed_power[:, :, 3:-4]
    )
    assert selected.frame_time_s[0] == 0.0
    np.testing.assert_allclose(np.diff(selected.frame_time_s), 256.0 / 48_000.0)


@pytest.mark.parametrize(
    "minimum_hz,maximum_hz",
    [(-1.0, 100.0), (100.0, 13_000.0), (1000.0, 500.0)],
)
def test_stft_power_rejects_invalid_frequency_intervals(minimum_hz, maximum_hz):
    with pytest.raises(ValueError, match="frequency limits"):
        rir_stft_power(
            np.ones((1, 512)),
            24_000.0,
            minimum_frequency_hz=minimum_hz,
            maximum_frequency_hz=maximum_hz,
        )


def test_package_does_not_export_dataset_loaders():
    import common_slope_nmf

    for name in (
        "inspect_sofa_dataset",
        "inspect_srir_dataset",
        "load_sofa_channel",
        "load_srir_channel",
    ):
        assert not hasattr(common_slope_nmf, name)
