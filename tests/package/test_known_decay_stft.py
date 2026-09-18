import numpy as np
from scipy.stats import chi2

from common_slope_nmf import (
    exponential_features,
    fit_amplitudes,
    sample_complex_gaussian,
    t60_to_rate,
)

SAMPLE_RATE_HZ = 24_000
DURATION_S = 2.0
FRAME_SIZE_SAMPLES = 256
HOP_SIZE_SAMPLES = 128
N_BINS = 512
T60_S = 1.0
DECAY_AMPLITUDE = 1.0
SEED = 20260724


def _frame_times():
    n_samples = round(DURATION_S * SAMPLE_RATE_HZ)
    n_frames = 1 + (
        n_samples - FRAME_SIZE_SAMPLES
    ) // HOP_SIZE_SAMPLES
    return (
        np.arange(n_frames, dtype=np.float64)
        * HOP_SIZE_SAMPLES
        / SAMPLE_RATE_HZ
    )


def test_known_decay_no_floor_matches_exact_inference():
    times_s = _frame_times()
    assert times_s.shape == (374,)
    assert times_s[0] == 0.0
    assert times_s[-1] == 373 * HOP_SIZE_SAMPLES / SAMPLE_RATE_HZ

    atom = exponential_features(times_s, t60_to_rate(T60_S))[0]
    variance = np.broadcast_to(
        DECAY_AMPLITUDE * atom, (N_BINS, times_s.size)
    )
    coefficients = sample_complex_gaussian(
        variance, rng=np.random.default_rng(SEED)
    )
    power = np.abs(coefficients) ** 2

    result = fit_amplitudes(
        power,
        atom[np.newaxis, :],
        initial_amplitudes=np.full((N_BINS, 1), 4.0),
        max_iter=1,
        tol=0.0,
    )
    exact_mle = np.mean(power / atom, axis=1)
    np.testing.assert_allclose(
        result.amplitudes[:, 0], exact_mle, rtol=4e-15, atol=0.0
    )

    theoretical_sd = DECAY_AMPLITUDE / np.sqrt(times_s.size)
    mean_standard_error = theoretical_sd / np.sqrt(N_BINS)
    assert abs(np.mean(exact_mle) - DECAY_AMPLITUDE) < 4 * mean_standard_error
    np.testing.assert_allclose(
        np.std(exact_mle, ddof=1), theoretical_sd, rtol=0.12
    )

    normalized_sum = np.sum(power / atom, axis=1)
    degrees_of_freedom = 2 * times_s.size
    lower = 2.0 * normalized_sum / chi2.ppf(
        0.975, degrees_of_freedom
    )
    upper = 2.0 * normalized_sum / chi2.ppf(
        0.025, degrees_of_freedom
    )
    coverage = np.mean(
        (lower <= DECAY_AMPLITUDE) & (DECAY_AMPLITUDE <= upper)
    )
    assert 0.91 <= coverage <= 0.99


def test_known_decay_with_floors_matches_fisher_sampling_scale():
    times_s = _frame_times()
    atom = exponential_features(times_s, t60_to_rate(T60_S))[0]
    dictionary = np.vstack([atom, np.ones_like(atom)])
    no_floor_variance = np.broadcast_to(
        DECAY_AMPLITUDE * atom, (N_BINS, times_s.size)
    )

    rng = np.random.default_rng(SEED)
    sample_complex_gaussian(no_floor_variance, rng=rng)
    for floor in (1e-3, 1e-6):
        true_amplitudes = np.array([DECAY_AMPLITUDE, floor])
        with_floor_variance = np.broadcast_to(
            true_amplitudes @ dictionary, (N_BINS, times_s.size)
        )
        coefficients = sample_complex_gaussian(with_floor_variance, rng=rng)
        power = np.abs(coefficients) ** 2
        result = fit_amplitudes(
            power,
            dictionary,
            initial_amplitudes=np.tile([0.25, 0.01], (N_BINS, 1)),
            max_iter=100,
            tol=1e-12,
        )

        weighted_dictionary = dictionary / (
            true_amplitudes @ dictionary
        )
        information = weighted_dictionary @ weighted_dictionary.T
        fisher_sd = np.sqrt(np.diag(np.linalg.inv(information)))

        assert result.converged
        assert np.max(np.diff(result.loss_history)) <= 1e-10
        assert np.all(
            np.abs(np.mean(result.amplitudes, axis=0) - true_amplitudes)
            <= 4 * fisher_sd / np.sqrt(N_BINS)
        )
        np.testing.assert_allclose(
            np.std(result.amplitudes, axis=0, ddof=1),
            fisher_sd,
            rtol=0.15,
        )

        fitted_variance = result.amplitudes @ dictionary
        variance_derivative = (
            1.0 / fitted_variance - power / fitted_variance**2
        )
        log_amplitude_score = result.amplitudes * (
            variance_derivative @ dictionary.T
        )
        assert (
            np.max(np.abs(log_amplitude_score)) / times_s.size < 5e-6
        )
