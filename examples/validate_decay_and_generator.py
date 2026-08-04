"""Visual validation of the decay convention and observation generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multislope_nmf import (
    exponential_variance,
    sample_complex_gaussian,
    t60_to_rate,
)

SEED = 20260724


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_01_02_validation.png"),
        help="Destination for the validation figure.",
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

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].plot(times_s, normalized_db, linewidth=2)
    axes[0].axvline(t60_s, color="tab:red", linestyle="--", linewidth=1)
    axes[0].axhline(-60.0, color="tab:red", linestyle="--", linewidth=1)
    axes[0].scatter([t60_s], [-60.0], color="tab:red", zorder=3)
    axes[0].set(
        title="Energy-decay convention",
        xlabel="Elapsed time (s)",
        ylabel="Relative variance (dB)",
        ylim=(-125.0, 5.0),
    )
    axes[0].grid(alpha=0.25)

    histogram_values = normalized_power[:, 0]
    axes[1].hist(
        histogram_values,
        bins=100,
        range=(0.0, 7.0),
        density=True,
        alpha=0.65,
        label="Empirical",
    )
    z = np.linspace(0.0, 7.0, 400)
    axes[1].plot(z, np.exp(-z), linewidth=2, label="Exp(1)")
    axes[1].set(
        title=r"Normalized power $Y/V$",
        xlabel=r"$Y/V$",
        ylabel="Density",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    empirical_mean = np.mean(powers, axis=0)
    axes[2].loglog(
        target_variances,
        empirical_mean,
        "o",
        markersize=7,
        label="Empirical mean",
    )
    axes[2].loglog(
        target_variances,
        target_variances,
        linestyle="--",
        label="Identity",
    )
    axes[2].set(
        title="Power calibration",
        xlabel="Requested variance",
        ylabel="Mean sampled power",
    )
    axes[2].legend()
    axes[2].grid(alpha=0.25, which="both")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)

    print(f"seed={SEED}")
    print(f"T60 crossing={normalized_db[60]:.12f} dB")
    print(f"mean(Y/V)={np.mean(normalized_power, axis=0)}")
    print(f"saved={args.output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
