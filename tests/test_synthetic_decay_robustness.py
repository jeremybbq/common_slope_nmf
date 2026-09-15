import numpy as np

from experiments.synthetic_decay_robustness import (
    DIRICHLET_ALPHA,
    METHOD_LABELS,
    METHOD_POWERS,
    run_identifiability_comparison,
)


def test_identifiability_comparison_uses_only_two_cw_sage_orders():
    assert METHOD_POWERS == (1.0, 2.0)
    assert METHOD_LABELS == (r"CW-SAGE ($p=1$)", r"CW-SAGE ($p=2$)")
    assert DIRICHLET_ALPHA == 0.5


def test_small_identifiability_comparison_preserves_result_dimensions():
    times_s = np.arange(100, dtype=np.float64) * (128.0 / 24_000.0)
    result = run_identifiability_comparison(
        np.random.default_rng(11),
        times_s,
        n_pairs=2,
        n_rirs=32,
        batch_size=1,
        max_iter=1,
        tol=0.0,
        decay_tol=0.0,
        rate_method="newton",
    )

    assert result["true_t60_s"].shape == (2, 2)
    assert result["estimated_t60_s"].shape == (2, 2, 2)
    assert result["estimated_amplitudes"].shape == (2, 32, 2, 2)
    assert result["frequency_n_iter"].shape == (2, 2)
    assert result["objective_history"].shape == (2, 2, 2)
