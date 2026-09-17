"""Optional HDF5 helpers for experiment dataset readers."""

from __future__ import annotations


def h5py_module():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Reading SOFA or MATLAB v7.3 SRIR data requires h5py; install the "
            "project's experiments extra."
        ) from exc
    return h5py


def decode_hdf5_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
