"""Visual validation of the decay convention and observation generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_slope_nmf import (
    exponential_variance,
    sample_complex_gaussian,
    t60_to_rate,
)
from common_slope_nmf import plotting as decay_plots
from examples._run_output import create_run_output_dir

SEED = 20260724


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for the timestamped run directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    """Run both validation experiments and save their diagnostic figure."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    output_path = output_dir / "experiment_01_02_validation.png"

    t60_s = 0.6
    times_s = np.arange(121, dtype=np.float64) * 0.01
    variance = exponential_variance(
        times_s, t60_to_rate(t60_s), amplitudes=2.5
    )
    normalized_db = 10.0 * np.log10(variance / variance[0])

    target_variances = np.array([1.0, 1e-1, 1e-3, 1e-6])
    coefficients = sample_complex_gaussian(
        target_variances,
        rng=np.random.default_rng(SEED),
        n_realizations=100_000,
    )
    powers = np.abs(coefficients) ** 2
    normalized_power = powers / target_variances

    figure = decay_plots.plot_generator_validation(
        times_s, normalized_db, t60_s, normalized_power,
        target_variances, np.mean(powers, axis=0),
    )
    decay_plots.save_figure(figure, output_dir, output_path.name, show=args.show)

    print(f"seed={SEED}")
    print(f"T60 crossing={normalized_db[60]:.12f} dB")
    print(f"mean(Y/V)={np.mean(normalized_power, axis=0)}")
    print(f"saved={output_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
