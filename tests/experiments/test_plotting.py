import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from experiments._run_output import save_figure
from experiments.squarem.plot_compare_convergence import (
    plot_results as plot_compare_convergence,
)
from experiments.synthetic.plot_sweep_decay_detection import (
    plot_amplitude_rmse_vs_separation,
    plot_objective_history,
    plot_signed_amplitude_bias_vs_separation,
    plot_t60_error_vs_separation,
    plot_t60_pair_plane,
    plot_true_vs_estimated_t60,
    plot_variance_maps,
)
from experiments.synthetic.plot_sweep_multislope_decay_initialization import (
    plot_initialization_heatmap,
)
from experiments.synthetic.plot_validate_decay_and_generator import (
    plot_results as plot_generator_results,
)
from experiments.synthetic.plot_validate_multislope_sage import plot_t60_trajectories


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
        plot_true_vs_estimated_t60(true_t60_s, estimated_t60_s),
        plot_t60_pair_plane(true_t60_s, estimated_t60_s),
        plot_t60_error_vs_separation(true_t60_s, absolute_error_s),
        plot_amplitude_rmse_vs_separation(true_t60_s, amplitude_rmse_db),
        plot_signed_amplitude_bias_vs_separation(
            true_t60_s, true_amplitudes, estimated_amplitudes
        ),
    ]

    for index, figure in enumerate(figures):
        path = save_figure(figure, tmp_path, f"figure_{index}.png")
        assert path == (tmp_path / f"figure_{index}.png").resolve()
        assert path.is_file()
        assert path.with_suffix(".pdf").is_file()


def test_variance_and_history_plots_accept_result_shaped_arrays():
    times_s = np.linspace(0.0, 0.2, 5)
    maps = (np.ones((3, 5)), np.full((3, 5), 0.5))
    objective = np.array([[10.0, 8.0, 7.0], [9.0, 7.5, np.nan]])
    history = np.array([[1.0, 2.0], [0.9, 2.1], [0.8, 2.2]])

    figures = [
        plot_variance_maps(times_s, maps, ("A", "B")),
        plot_objective_history(objective, normalize=True, log_y=True),
        plot_t60_trajectories(history, true_t60_s=np.array([0.75, 2.25])),
    ]

    assert all(len(figure.axes) >= 1 for figure in figures)
    for figure in figures:
        matplotlib.pyplot.close(figure)


def test_plotting_rejects_mismatched_component_shapes():
    with pytest.raises(ValueError, match="shape"):
        plot_amplitude_rmse_vs_separation(np.ones((3, 2)), np.ones((2, 2)))


def test_initialization_heatmap_requires_square_grid():
    with pytest.raises(ValueError, match="shape"):
        plot_initialization_heatmap(
            np.ones((2, 3)),
            np.array([0.5, 1.0]),
            title="test",
            colorbar_label="value",
        )


def test_plot_builders_load_fake_npz_archives(tmp_path):
    rng = np.random.default_rng(0)
    generator_path = tmp_path / "generator_validation_results.npz"
    np.savez_compressed(
        generator_path,
        times_s=np.linspace(0.0, 1.2, 13),
        normalized_decay_db=np.linspace(0.0, -60.0, 13),
        t60_s=np.asarray(0.6),
        normalized_power=rng.exponential(size=(20, 4)),
        target_variances=np.array([1.0, 1e-1, 1e-3, 1e-6]),
        empirical_mean_power=np.array([1.0, 1e-1, 1e-3, 1e-6]),
    )
    generator_figure = plot_generator_results(generator_path)
    assert generator_figure.is_file()
    assert (tmp_path / "experiment_01_02_validation.png").is_file()

    comparison_path = tmp_path / "multislope_convergence_histories.npz"
    np.savez_compressed(
        comparison_path,
        method_0_label=np.asarray("ordinary SAGE"),
        method_0_sweep_evaluations=np.arange(4),
        method_0_objective_history=np.array([4.0, 3.0, 2.5, 2.4]),
        method_1_label=np.asarray("SQUAREM"),
        method_1_sweep_evaluations=np.array([0, 2, 4]),
        method_1_objective_history=np.array([4.0, 2.2, 2.0]),
    )
    comparison_figure = plot_compare_convergence(comparison_path)
    assert comparison_figure.is_file()
    assert (tmp_path / "multislope_convergence_comparison.png").is_file()
