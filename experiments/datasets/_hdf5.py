"""Optional h5py access for experiment dataset readers."""

from __future__ import annotations


def h5py_module():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Reading MATLAB v7.3 / SOFA HDF5 data requires h5py; install the "
            "project's experiments extra."
        ) from exc
    return h5py


def decode_hdf5_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
