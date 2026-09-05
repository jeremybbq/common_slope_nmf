import h5py
import numpy as np
import pytest

from common_slope_nmf import (
    global_energy_onset,
    inspect_srir_dataset,
    load_srir_channel,
    resample_rirs,
    rir_stft_power,
)


def _write_test_dataset(path):
    n_channels, n_receivers, n_samples = 3, 7, 64
    time = np.arange(n_samples, dtype=np.float64)
    srirs = np.empty((n_channels, n_receivers, n_samples))
    for channel in range(n_channels):
        for receiver in range(n_receivers):
            srirs[channel, receiver] = 1000 * channel + 100 * receiver + time
    positions = np.stack(
        (np.arange(n_receivers), np.arange(n_receivers) + 0.5, np.ones(n_receivers))
    )
    with h5py.File(path, "w") as stream:
        group = stream.create_group("srirDataset")
        group.create_dataset("srirs", data=srirs)
        group.create_dataset("rcvPos", data=positions)
        group.create_dataset("fs", data=np.array([[48_000.0]]))
    return srirs, positions


def test_inspect_and_load_srir_channel_detect_axes(tmp_path):
    path = tmp_path / "srirs.mat"
    srirs, positions = _write_test_dataset(path)

    info = inspect_srir_dataset(path)
    rirs, selected_positions, loaded_info = load_srir_channel(
        path,
        channel_index=1,
        receiver_indices=np.array([0, 3, 6]),
        time_chunk_samples=11,
    )

    assert info == loaded_info
    assert info.shape == (3, 7, 64)
    assert (info.channel_axis, info.receiver_axis, info.time_axis) == (0, 1, 2)
    assert info.sample_rate_hz == 48_000.0
    np.testing.assert_array_equal(rirs, srirs[1, [0, 3, 6]])
    np.testing.assert_array_equal(selected_positions, positions[:2, [0, 3, 6]].T)


def test_load_srir_channel_requires_increasing_indices(tmp_path):
    path = tmp_path / "srirs.mat"
    _write_test_dataset(path)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_srir_channel(path, receiver_indices=np.array([3, 1]))


def test_global_energy_onset_uses_one_pooled_threshold():
    rirs = np.zeros((3, 300))
    rirs[0, 100] = 1.0
    rirs[1, 120] = 0.5
    rirs[2, 140] = 0.25

    onset, pooled = global_energy_onset(
        rirs,
        1000.0,
        threshold_db=-20.0,
        smoothing_duration_s=0.001,
    )

    assert onset == 100
    assert pooled.shape == (300,)
    assert pooled[100] == np.max(pooled)


def test_resample_and_stft_power_preserve_explicit_units_and_shapes():
    rng = np.random.default_rng(5)
    source = rng.normal(size=(4, 2048))

    resampled = resample_rirs(source, 48_000.0, 24_000.0)
    transformed = rir_stft_power(
        resampled,
        24_000.0,
        frame_size_samples=256,
        hop_size_samples=128,
        fft_size_samples=384,
        minimum_frequency_hz=62.5,
        maximum_frequency_hz=8_000.0,
    )

    assert resampled.shape == (4, 1024)
    assert transformed.observed_power.shape == (4, 128, 7)
    np.testing.assert_array_equal(
        transformed.frequencies_hz, np.arange(1, 129) * 62.5
    )
    np.testing.assert_allclose(
        np.diff(transformed.times_s), 128.0 / 24_000.0
    )
    assert np.all(transformed.observed_power > 0.0)


@pytest.mark.parametrize(
    "minimum_hz,maximum_hz",
    [(-1.0, 100.0), (100.0, 13_000.0), (1000.0, 500.0)],
)
def test_stft_power_rejects_invalid_frequency_intervals(
    minimum_hz, maximum_hz
):
    with pytest.raises(ValueError, match="frequency limits"):
        rir_stft_power(
            np.ones((1, 512)),
            24_000.0,
            minimum_frequency_hz=minimum_hz,
            maximum_frequency_hz=maximum_hz,
        )
