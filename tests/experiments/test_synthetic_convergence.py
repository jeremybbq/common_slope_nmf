import numpy as np

from experiments.synthetic.convergence import (
    DECAY_LOSS_TOL,
    LONG_T60_RANGE_S,
    N_FRAMES,
    N_RIRS,
    SHORT_T60_RANGE_S,
    SURFACE_GRID_SIZE,
    SURFACE_LOSS_TOL,
    TRUE_T60_S,
    profile_fixed_rate_loss_surface,
    run_loss_comparison,
)


def test_loss_surface_experiment_uses_requested_lightweight_setup():
    np.testing.assert_allclose(
        TRUE_T60_S,
        [0.80, 1.40],
        rtol=0.0,
        atol=1e-15,
    )
    assert (N_RIRS, N_FRAMES) == (256, 256)
    assert SURFACE_GRID_SIZE == 21
    assert SHORT_T60_RANGE_S == (0.6, 1.4)
    assert LONG_T60_RANGE_S == (1.0, 1.8)
    assert DECAY_LOSS_TOL == 0.0
    assert SURFACE_LOSS_TOL == 1e-6


def test_small_loss_comparison_uses_uniform_complementary_amplitudes():
    times_s = np.arange(100, dtype=np.float64) * (128.0 / 24_000.0)
    result = run_loss_comparison(
        np.random.default_rng(7),
        times_s,
        n_rirs=32,
        max_iter=1,
        tol=1e-4,
    )

    amplitudes = np.asarray(result["true_amplitudes"])
    assert amplitudes.shape == (32, 1, 2)
    np.testing.assert_allclose(
        amplitudes[:, :, 0],
        np.random.default_rng(7).uniform(0.0, 1.0, size=(32, 1)),
    )
    np.testing.assert_allclose(amplitudes[:, :, 1], 1.0 - amplitudes[:, :, 0])
    np.testing.assert_allclose(np.sum(amplitudes, axis=2), 1.0)
    initial_t60_s = np.asarray(result["initial_t60_s"])
    assert initial_t60_s.shape == (2,)
    np.testing.assert_allclose(initial_t60_s[0], initial_t60_s[1])
    assert np.asarray(result["estimated_t60_s"]).shape == (3, 2)
    histories = result["histories"]
    assert isinstance(histories, list)
    assert len(histories) == 1
    assert len(histories[0]) == 3
    trajectories = result["t60_trajectories_s"]
    assert isinstance(trajectories, list)
    assert all(path.shape == (2, 2) for path in trajectories)


def test_small_profiled_surface_saves_warm_and_cold_fit_diagnostics():
    times_s = np.arange(40, dtype=np.float64) * 0.01
    observed_power = np.ones((8, times_s.size))
    surface = profile_fixed_rate_loss_surface(
        observed_power,
        times_s,
        np.linspace(0.5, 1.1, 3),
        np.linspace(0.9, 1.5, 3),
        np.full((8, 2), 0.4),
        np.full(8, 0.2),
        max_iter=2,
        tol=1e-4,
    )

    assert surface["loss"].shape == (3, 3)
    assert surface["amplitudes"].shape == (3, 3, 8, 2)
    assert surface["noise_floor"].shape == (3, 3, 8)
    assert surface["n_iter"].shape == (3, 3)
    assert surface["converged"].shape == (3, 3)
    assert surface["cold_check_indices"].shape == (5, 2)
    assert np.all(np.isfinite(surface["cold_check_relative_gap"]))
