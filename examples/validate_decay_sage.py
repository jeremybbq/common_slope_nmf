"""Validate profiled SAGE decay-rate recovery on exact synthetic power."""

from __future__ import annotations

import numpy as np

from common_slope_nmf import (
    decay_sage,
    exponential_variance,
    rate_to_t60,
    t60_to_rate,
)


def main() -> None:
    """Recover known independent-frequency decay rates and amplitudes."""

    times_s = np.arange(151, dtype=np.float64) * 0.01
    true_t60_s = np.array([[0.35], [0.55], [0.80], [1.10]])
    true_rates_per_s = t60_to_rate(true_t60_s)
    true_amplitudes = np.array(
        [
            [[0.4], [0.7], [1.0], [1.3]],
            [[1.4], [1.1], [0.8], [0.5]],
            [[0.9], [0.6], [1.2], [0.75]],
        ]
    )
    power = exponential_variance(
        times_s, true_rates_per_s, true_amplitudes
    )

    result = decay_sage(
        power,
        times_s,
        np.full_like(true_rates_per_s, t60_to_rate(0.65)),
        rate_bounds_per_s=(t60_to_rate(1.5), t60_to_rate(0.2)),
        initial_amplitudes=np.full_like(true_amplitudes, 2.0),
        estimate_noise_floor=False,
        max_iter=5,
        tol=0.0,
    )

    estimated_t60_s = rate_to_t60(result.rates_per_s)
    print(f"true_t60_s={true_t60_s.ravel()}")
    print(f"estimated_t60_s={estimated_t60_s.ravel()}")
    print(
        "maximum_relative_rate_error="
        f"{np.max(np.abs(result.rates_per_s / true_rates_per_s - 1.0)):.3e}"
    )
    print(
        "maximum_relative_amplitude_error="
        f"{np.max(np.abs(result.amplitudes / true_amplitudes - 1.0)):.3e}"
    )
    print(
        f"sweeps={result.n_iter}, converged={result.converged}, "
        f"final_is={result.objective_history[-1]:.3e}"
    )


if __name__ == "__main__":
    main()
