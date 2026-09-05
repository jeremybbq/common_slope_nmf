from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from common_slope_nmf import plotting


def test_detection_plot_builders_return_figures(tmp_path):
    true_t60_s = np.array([[0.6, 1.4], [1.0, 1.2], [1.5, 2.8]])
    estimated_t60_s = true_t60_s + np.array(
        [[0.01, -0.02], [0.05, 0.04], [-0.02, 0.01]]
    )
    absolute_error_s = np.abs(estimated_t60_s - true_t60_s)
    amplitude_rmse_db = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])
    true_amplitudes = np.full((4, 3, 2), 0.5)
    estimated_amplitudes = true_amplitudes * np.array([1.2, 0.8])

    figures = [
        plotting.plot_true_vs_estimated_t60(true_t60_s, estimated_t60_s),
        plotting.plot_t60_pair_plane(true_t60_s, estimated_t60_s),
        plotting.plot_t60_error_vs_separation(
            true_t60_s, absolute_error_s
        ),
        plotting.plot_amplitude_rmse_vs_separation(
            true_t60_s, amplitude_rmse_db
        ),
        plotting.plot_signed_amplitude_bias_vs_separation(
            true_t60_s, true_amplitudes, estimated_amplitudes
        ),
    ]

    for index, figure in enumerate(figures):
        path = plotting.save_figure(
            figure, tmp_path, f"figure_{index}.png"
        )
        assert path == (tmp_path / f"figure_{index}.png").resolve()
        assert path.is_file()


def test_variance_and_history_plots_accept_result_shaped_arrays():
    times_s = np.linspace(0.0, 0.2, 5)
    maps = (np.ones((3, 5)), np.full((3, 5), 0.5))
    objective = np.array([[10.0, 8.0, 7.0], [9.0, 7.5, np.nan]])
    history = np.array([[1.0, 2.0], [0.9, 2.1], [0.8, 2.2]])

    figures = [
        plotting.plot_variance_maps(times_s, maps, ("A", "B")),
        plotting.plot_objective_history(
            objective, normalize=True, log_y=True
        ),
        plotting.plot_t60_trajectories(
            history, true_t60_s=np.array([0.75, 2.25])
        ),
    ]

    assert all(len(figure.axes) >= 1 for figure in figures)
    for figure in figures:
        matplotlib.pyplot.close(figure)


def test_plotting_rejects_mismatched_component_shapes():
    with pytest.raises(ValueError, match="shape"):
        plotting.plot_amplitude_rmse_vs_separation(
            np.ones((3, 2)), np.ones((2, 2))
        )


def test_initialization_heatmap_requires_square_grid():
    with pytest.raises(ValueError, match="shape"):
        plotting.plot_initialization_heatmap(
            np.ones((2, 3)),
            np.array([0.5, 1.0]),
            title="test",
            colorbar_label="value",
        )
