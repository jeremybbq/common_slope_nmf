import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from experiments.synthetic import plot_convergence as convergence
from experiments.synthetic import plot_decay_robustness as robustness


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
        robustness.plot_t60_error_vs_separation(
            true_t60_s, absolute_error_s
        ),
        robustness.plot_amplitude_rmse_vs_separation(
            true_t60_s, amplitude_rmse_db
        ),
        robustness.plot_signed_amplitude_bias_vs_separation(
            true_t60_s, true_amplitudes, estimated_amplitudes
        ),
    ]

    for index, figure in enumerate(figures):
        path = robustness.save_figure(
            figure, tmp_path, f"figure_{index}.png"
        )
        assert path == (tmp_path / f"figure_{index}.png").resolve()
        assert path.is_file()


def test_weight_order_comparison_plots_accept_multiple_methods_and_cases():
    histories = [
        [np.array([10.0, 8.0, 7.0]), np.array([10.0, 7.8]), np.array([10.0, 7.5])],
        [np.array([12.0, 9.0]), np.array([12.0, 8.5]), np.array([12.0, 8.0])],
    ]
    labels = ("ordinary", "p=1", "p=2")

    figures = [
        convergence.plot_weight_order_objectives(
            histories,
            labels,
            ("separated", "close"),
            trajectories_by_case=[
                [
                    np.array([[0.7, 1.3], [0.61, 1.39], [0.6, 1.4]]),
                    np.array([[0.7, 1.3], [0.6, 1.4]]),
                    np.array([[0.7, 1.3], [0.62, 1.41]]),
                ],
                [
                    np.array([[0.7, 1.3], [0.61, 1.39]]),
                    np.array([[0.7, 1.3], [0.6, 1.4]]),
                    np.array([[0.7, 1.3], [0.62, 1.41]]),
                ],
            ],
            true_t60_s=np.array([0.6, 1.4]),
        ),
    ]

    assert [len(figure.axes) for figure in figures] == [4]
    loss_axes = figures[0].axes[:2]
    error_axes = figures[0].axes[2:]
    assert figures[0]._suptitle is None
    np.testing.assert_allclose(figures[0].get_size_inches(), (7.10, 3.4))
    assert [axis.get_yscale() for axis in loss_axes] == ["log", "log"]
    assert loss_axes[0].get_ylabel().startswith("Excess loss")
    assert loss_axes[1].get_ylabel() == ""
    assert loss_axes[0].xaxis.label.get_fontfamily() == ["Liberation Serif"]
    np.testing.assert_allclose(
        loss_axes[0].lines[0].get_ydata()[:2],
        [3.0, 1.0],
    )
    assert np.isnan(loss_axes[0].lines[0].get_ydata()[2])
    assert [axis.get_ylabel() for axis in error_axes] == [
        "Estimated-to-true RT distance (s)",
        "Estimated-to-true RT distance (s)",
    ]
    np.testing.assert_allclose(
        error_axes[0].lines[0].get_ydata(),
        [np.hypot(0.1, -0.1), np.hypot(0.01, -0.01), 0.0],
    )
    for figure in figures:
        matplotlib.pyplot.close(figure)


def test_profiled_t60_surface_uses_display_only_bicubic_interpolation():
    short_t60_s = np.linspace(0.5, 1.1, 3)
    long_t60_s = np.linspace(0.9, 1.5, 3)
    loss = np.array(
        [[12.0, 8.0, 9.0], [7.0, 5.0, 6.0], [10.0, 8.0, 11.0]]
    )
    trajectories = (
        np.array([[0.7, 1.0], [0.6, 1.4]]),
        np.array([[0.8, 1.1], [0.61, 1.39]]),
    )

    figure = convergence.plot_profiled_t60_loss_surface(
        short_t60_s,
        long_t60_s,
        loss,
        np.array([0.6, 1.4]),
        trajectories,
        ("p=1", "p=2"),
    )

    assert len(figure.axes) == 2
    np.testing.assert_allclose(figure.get_size_inches(), (3.45, 3.15))
    assert figure.axes[0].get_title() == ""
    assert figure.axes[0].images[0].get_interpolation() == "bicubic"
    np.testing.assert_allclose(figure.axes[0].get_xlim(), (0.5, 1.1))
    np.testing.assert_allclose(figure.axes[0].get_ylim(), (0.9, 1.5))
    assert figure.axes[0].get_xlabel() == "Fast decay RT (s)"
    assert figure.axes[0].get_ylabel() == "Slow decay RT (s)"
    assert figure.axes[1].get_ylabel() == "Profiled excess loss"
    assert all(line.get_marker() == "None" for line in figure.axes[0].lines)
    assert all(not line.get_path().should_simplify for line in figure.axes[0].lines)
    assert figure.axes[0].xaxis.label.get_fontfamily() == ["Liberation Serif"]
    assert not figure.legends[0].get_frame_on()

    collections = {
        collection.get_label(): collection
        for collection in figure.axes[0].collections
    }
    true_marker = collections["True RTs"]
    minimum_marker = collections["Profiled grid minimum"]
    linear_fit_marker = collections["Linear fits"]
    assert true_marker.get_sizes()[0] == 52
    assert true_marker.get_edgecolors().size == 0
    np.testing.assert_allclose(
        true_marker.get_facecolors()[0, :3],
        matplotlib.colors.to_rgb("#ef4444"),
    )
    assert minimum_marker.get_sizes()[0] == 28
    np.testing.assert_allclose(minimum_marker.get_facecolors()[0, :3], (1, 1, 1))
    np.testing.assert_allclose(
        minimum_marker.get_edgecolors()[0, :3],
        matplotlib.colors.to_rgb("#111827"),
    )
    assert linear_fit_marker.get_sizes()[0] == 24

    mask_color = matplotlib.colors.to_rgb("#d9d9d9")
    symmetry_mask = next(
        collection
        for collection in figure.axes[0].collections
        if collection.get_facecolors().size
        and np.allclose(collection.get_facecolors()[0, :3], mask_color)
    )
    mask_vertices = symmetry_mask.get_paths()[0].vertices
    for corner in ((0.9, 0.9), (1.1, 0.9), (1.1, 1.1)):
        assert np.any(np.all(np.isclose(mask_vertices, corner), axis=1))
    assert symmetry_mask.get_hatch() is None
    assert all(
        spine.get_zorder() > symmetry_mask.get_zorder()
        for spine in figure.axes[0].spines.values()
    )

    figure.canvas.draw()
    surface_position = figure.axes[0].get_position()
    colorbar_position = figure.axes[1].get_position()
    np.testing.assert_allclose(
        (colorbar_position.y0, colorbar_position.y1),
        (surface_position.y0, surface_position.y1),
        atol=1e-10,
    )
    matplotlib.pyplot.close(figure)


def test_plotting_rejects_mismatched_component_shapes():
    with pytest.raises(ValueError, match="shape"):
        robustness.plot_amplitude_rmse_vs_separation(
            np.ones((3, 2)), np.ones((2, 2))
        )
