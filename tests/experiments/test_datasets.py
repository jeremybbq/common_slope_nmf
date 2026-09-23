import h5py
import numpy as np
import pytest

from experiments.datasets import (
    inspect_sofa_dataset,
    inspect_srir_dataset,
    load_sofa_channel,
    load_srir_channel,
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


def _write_test_sofa(path):
    n_measurements, n_receivers, n_samples = 5, 4, 32
    ir = np.arange(n_measurements * n_receivers * n_samples, dtype=np.float64).reshape(
        n_measurements, n_receivers, n_samples
    )
    listener = np.column_stack(
        (np.arange(n_measurements), np.zeros(n_measurements), np.ones(n_measurements))
    )
    source = np.array([[2.0, 3.0, 1.5]])
    with h5py.File(path, "w") as stream:
        stream.attrs["SOFAConventions"] = np.bytes_(b"SingleRoomDRIR")
        stream.create_dataset("Data.IR", data=ir)
        stream.create_dataset("Data.SamplingRate", data=np.array([48_000.0]))
        stream.create_dataset("ListenerPosition", data=listener)
        stream.create_dataset("SourcePosition", data=source)
    return ir, listener, source


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


def test_inspect_and_load_sofa_channel(tmp_path):
    path = tmp_path / "room.sofa"
    ir, listener, source = _write_test_sofa(path)

    info = inspect_sofa_dataset(path)
    rirs, loaded_listener, loaded_source, loaded_info = load_sofa_channel(
        path,
        channel_index=2,
        measurement_indices=np.array([0, 2, 4]),
        time_chunk_samples=7,
    )

    assert info == loaded_info
    assert info.shape == (5, 4, 32)
    assert info.sample_rate_hz == 48_000.0
    assert info.convention == "SingleRoomDRIR"
    np.testing.assert_array_equal(rirs, ir[[0, 2, 4], 2])
    np.testing.assert_array_equal(loaded_listener, listener[[0, 2, 4]])
    np.testing.assert_array_equal(loaded_source, np.repeat(source, 3, axis=0))
