from pathlib import Path

import numpy as np
import pytest

from experiments.room_to_hallway.georg_benchmarks import (
    BAND_CENTERS_HZ,
    N_SLOPES,
    PARTIAL_KEYS,
    combine_band_partials,
    open_decayfitnet,
)
from experiments.room_to_hallway.plot_georg_benchmarks import plot_t60_comparison


def test_benchmark_uses_agreed_fixed_k2_six_bands():
    assert N_SLOPES == 2
    np.testing.assert_array_equal(
        BAND_CENTERS_HZ,
        [250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0],
    )


def test_open_decayfitnet_requires_a_checkout(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="DecayFitNetToolbox.py"):
        open_decayfitnet(
            tmp_path,
            n_slopes=2,
            sample_rate_hz=48_000.0,
            filter_frequencies=[1_000.0],
        )


def test_band_checkpoints_stack_receiver_and_shared_axes(tmp_path: Path):
    paths = []
    for band_index, band_hz in enumerate((250.0, 500.0)):
        values = {
            "band_center_hz": np.asarray(band_hz),
            "n_edc_samples": np.asarray(100),
        }
        for key in PARTIAL_KEYS:
            if key in {"common_slope_t60_s", "common_slope_cluster_sizes"}:
                values[key] = np.full(2, band_index, dtype=np.float64)
            elif "amplitudes" in key or key == "decayfitnet_t60_s":
                values[key] = np.full((3, 2), band_index, dtype=np.float64)
            else:
                values[key] = np.full(3, band_index, dtype=np.float64)
        path = tmp_path / f"band_{band_index}.npz"
        np.savez(path, **values)
        paths.append(path)

    combined = combine_band_partials(paths)

    assert combined["decayfitnet_t60_s"].shape == (3, 2, 2)
    assert combined["common_slope_t60_s"].shape == (2, 2)
    assert combined["common_slope_cluster_sizes"].shape == (2, 2)
    np.testing.assert_array_equal(combined["band_centers_hz"], [250.0, 500.0])


def test_t60_comparison_plot_saves_png_and_pdf(tmp_path: Path):
    proposed_path = tmp_path / "proposed.npz"
    benchmark_path = tmp_path / "benchmarks.npz"
    output_path = tmp_path / "comparison.png"
    np.savez(
        proposed_path,
        frequencies_hz=np.array([250.0, 500.0, 1_000.0]),
        estimated_t60_s=np.array(
            [[0.5, 1.4], [0.55, 1.5], [0.6, 1.6]]
        ),
        coarse_t60_s=np.array([1.0, 1.1, 1.2]),
    )
    np.savez(
        benchmark_path,
        band_centers_hz=np.array([250.0, 1_000.0]),
        decayfitnet_t60_s=np.array(
            [
                [[0.45, 1.3], [0.55, 1.5]],
                [[0.50, 1.4], [0.60, 1.7]],
                [[0.55, 1.5], [0.65, 1.9]],
            ]
        ),
        common_slope_t60_s=np.array([[0.5, 1.4], [0.6, 1.7]]),
    )

    saved_path = plot_t60_comparison(
        proposed_path, benchmark_path, output_path
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.with_suffix(".pdf").is_file()
