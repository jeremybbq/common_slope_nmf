import numpy as np

from experiments.synthetic.amplitude_identifiability import (
    FAST_TO_SLOW_DB,
    NOISE_FLOOR_DB,
    TRUE_T60_S,
    amplitude_pair_from_ratio_db,
    case_diagnostics,
    equal_decay_initialization,
    run_amplitude_inference,
    summarize_run,
)


def test_amplitude_ratios_include_masked_fast_minus_15_db_case():
    np.testing.assert_array_equal(
        FAST_TO_SLOW_DB,
        [10.0, 0.0, -10.0, -15.0, -20.0],
    )
    pairs = amplitude_pair_from_ratio_db(FAST_TO_SLOW_DB)
    np.testing.assert_allclose(np.sum(pairs, axis=1), 1.0)
    np.testing.assert_allclose(
        10.0 * np.log10(pairs[:, 0] / pairs[:, 1]),
        FAST_TO_SLOW_DB,
        atol=2e-14,
    )
    np.testing.assert_allclose(TRUE_T60_S, [0.5, 1.5])
    assert NOISE_FLOOR_DB == -50.0


def test_equal_initialization_divides_only_decay_signal_equally():
    power = np.ones((2, 40), dtype=np.float64)
    power[:, :8] = np.array([[5.0], [9.0]])
    power[:, -20:] = np.array([[1.0], [3.0]])
    initialized = equal_decay_initialization(power)

    np.testing.assert_allclose(initialized[:, 0], initialized[:, 1])
    np.testing.assert_allclose(initialized, [[2.0, 2.0, 1.0], [3.0, 3.0, 3.0]])


def test_small_run_uses_matched_complex_gaussian_innovations():
    run = run_amplitude_inference(
        seed=13,
        n_realizations=7,
        n_frames=40,
        max_sweep_evaluations=3,
        objective_tol=0.0,
        fixed_point_tol=0.0,
    )

    assert run.dictionary.shape == (3, 40)
    assert run.true_amplitudes.shape == (5, 3)
    assert run.coefficients.shape == (5, 7, 40)
    assert run.estimated_amplitudes.shape == (5, 7, 3)
    np.testing.assert_allclose(run.observed_power, np.abs(run.coefficients) ** 2)
    exact_variance = run.true_amplitudes @ run.dictionary
    standardized_power = run.observed_power / exact_variance[:, None, :]
    np.testing.assert_allclose(
        standardized_power,
        np.broadcast_to(standardized_power[0], standardized_power.shape),
        rtol=2e-14,
        atol=0.0,
    )
    assert np.all(run.estimated_amplitudes > 0.0)
    assert np.all(run.final_objective <= run.initial_objective + 1e-10)
    assert len(summarize_run(run)) == 15
    assert len(case_diagnostics(run)) == 5


def test_ordinary_sage_history_and_archive_round_trip(tmp_path):
    from common_slope_nmf import amplitude_sage
    from experiments.synthetic.amplitude_identifiability import load_run, save_run

    run = run_amplitude_inference(
        seed=13, n_realizations=7, n_frames=40,
        max_sweep_evaluations=3, objective_tol=0.0, fixed_point_tol=0.0,
    )
    expected = amplitude_sage(
        run.observed_power[0], run.dictionary, run.initial_amplitudes[0],
        max_iter=3, tol=0.0,
    )
    np.testing.assert_allclose(run.estimated_amplitudes[0], expected.amplitudes, rtol=1e-14)
    np.testing.assert_allclose(run.objective_history[0], expected.objective_history, rtol=1e-14)
    assert not np.any(run.converged)
    assert np.all(np.diff(run.objective_history, axis=1) <= 1e-10)
    save_run(run, tmp_path)
    restored = load_run(tmp_path / 'amplitude_inference_results.npz')
    for name in run.__dataclass_fields__:
        np.testing.assert_equal(getattr(restored, name), getattr(run, name))
