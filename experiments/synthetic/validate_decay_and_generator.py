"""Visual validation of the decay convention and observation generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    exponential_variance,
    sample_complex_gaussian,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir

SEED = 20260724


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for the timestamped run directory.",
    )
    return parser.parse_args()


def run_validation(output_root: Path) -> Path:
    """Generate generator-validation arrays and write them to NPZ."""

    output_dir = create_run_output_dir(output_root)
    t60_s = 0.6
    times_s = np.arange(121, dtype=np.float64) * 0.01
    variance = exponential_variance(times_s, t60_to_rate(t60_s), amplitudes=2.5)
    normalized_db = 10.0 * np.log10(variance / variance[0])

    target_variances = np.array([1.0, 1e-1, 1e-3, 1e-6])
    coefficients = sample_complex_gaussian(
        target_variances,
        rng=np.random.default_rng(SEED),
        n_realizations=100_000,
    )
    powers = np.abs(coefficients) ** 2
    normalized_power = powers / target_variances
    empirical_mean_power = np.mean(powers, axis=0)
    archive_path = output_dir / "generator_validation_results.npz"
    np.savez_compressed(
        archive_path,
        seed=np.asarray(SEED),
        t60_s=np.asarray(t60_s),
        times_s=times_s,
        normalized_decay_db=normalized_db,
        normalized_power=normalized_power,
        target_variances=target_variances,
        empirical_mean_power=empirical_mean_power,
    )
    print(f"seed={SEED}")
    print(f"T60 crossing={normalized_db[60]:.12f} dB")
    print(f"mean(Y/V)={np.mean(normalized_power, axis=0)}")
    print(f"saved={archive_path.resolve()}")
    return archive_path


def main() -> None:
    run_validation(_arguments().output_root)


if __name__ == "__main__":
    main()
