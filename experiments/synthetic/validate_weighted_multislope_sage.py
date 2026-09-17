"""Run the seeded two-slope experiment with weighted pseudo-SAGE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.synthetic.validate_multislope_sage import run_experiment


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-weight-power",
        type=float,
        default=1.0,
        help=(
            "Non-negative exponent p for fixed decay M-step weights rho_k**p "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2_000,
        help="Maximum number of complete pseudo-SAGE component sweeps.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Relative outer observed-objective decrease tolerance.",
    )
    parser.add_argument(
        "--rate-method",
        choices=("newton", "bisection"),
        default="newton",
        help="Weighted profile-rate solver (default: newton).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for the timestamped run directory.",
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.component_weight_power)
        or args.component_weight_power < 0.0
    ):
        parser.error("--component-weight-power must be finite and non-negative")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    return args


def main() -> None:
    """Generate, fit, and summarize the weighted experiment."""

    args = _arguments()
    run_experiment(
        args, component_weight_power=args.component_weight_power
    )


if __name__ == "__main__":
    main()
