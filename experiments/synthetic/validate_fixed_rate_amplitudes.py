"""Validate supplied-atom IS-SAGE amplitude and floor estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from common_slope_nmf import (
    exponential_atoms,
    amplitude_sage,
    gaussian_variance_nll,
    is_divergence,
    sample_power,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir

SEED = 20260724
COMPONENT_LABELS = ("fast decay", "slow decay", "noise floor")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=200,
        help="Number of independent train/test power realizations.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5_000,
        help="Maximum SAGE sweeps for the stochastic batch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for the timestamped run directory.",
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    return args


def _experiment_case():
    times_s = np.arange(121, dtype=np.float64) * 0.01
    decay_atoms = exponential_atoms(
        times_s, t60_to_rate(np.array([0.25, 0.8]))
    )
    dictionary = np.vstack(
        [decay_atoms, np.ones((1, times_s.size))]
    )
    amplitudes = np.array(
        [
            [1.00, 0.08, 1e-5],
            [0.10, 1.00, 3e-5],
            [0.80, 0.30, 1e-4],
            [0.25, 0.70, 3e-6],
        ]
    )
    return times_s, dictionary, amplitudes


def _scipy_reference(observed_power, dictionary):
    n_rirs = observed_power.shape[0]
    n_components = dictionary.shape[0]

    def objective_and_gradient(log_amplitudes):
        amplitudes = np.exp(
            log_amplitudes.reshape(n_rirs, n_components)
        )
        variance = amplitudes @ dictionary
        derivative_variance = (
            1.0 / variance - observed_power / variance**2
        )
        gradient = amplitudes * (derivative_variance @ dictionary.T)
        objective = np.sum(
            np.log(variance) + observed_power / variance
        )
        return float(objective), gradient.ravel()

    optimization = minimize(
        objective_and_gradient,
        np.zeros(n_rirs * n_components),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-40.0, 10.0)] * (n_rirs * n_components),
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 10_000},
    )
    amplitudes = np.exp(
        optimization.x.reshape(n_rirs, n_components)
    )
    return optimization, amplitudes, amplitudes @ dictionary


def main() -> None:
    """Run deterministic and stochastic supplied-atom validation."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    times_s, dictionary, true_amplitudes = _experiment_case()
    true_variance = true_amplitudes @ dictionary

    single_dictionary = dictionary[1:2]
    single_amplitudes = np.array([[0.2], [0.5], [1.0], [2.0]])
    single_result = amplitude_sage(
        single_amplitudes @ single_dictionary,
        single_dictionary,
        initial_amplitudes=np.full_like(single_amplitudes, 7.0),
        max_iter=5,
        tol=0.0,
    )

    deterministic_results = {}
    for scale in (0.1, 1.0, 10.0):
        deterministic_results[scale] = amplitude_sage(
            true_variance,
            dictionary,
            initial_amplitudes=np.full_like(true_amplitudes, scale),
            max_iter=70_000,
            tol=1e-13,
        )
    deterministic_result = deterministic_results[1.0]

    rng = np.random.default_rng(SEED)
    train_power = sample_power(
        true_variance, rng=rng, n_realizations=args.seeds
    )
    test_power = sample_power(
        true_variance, rng=rng, n_realizations=args.seeds
    )
    stochastic_result = amplitude_sage(
        train_power.reshape(-1, times_s.size),
        dictionary,
        max_iter=args.max_iter,
        tol=1e-10,
    )
    estimated_amplitudes = stochastic_result.amplitudes.reshape(
        args.seeds, true_amplitudes.shape[0], dictionary.shape[0]
    )
    estimated_variance = stochastic_result.variance.reshape(
        args.seeds, true_amplitudes.shape[0], times_s.size
    )

    heldout_excess_nll = np.mean(
        gaussian_variance_nll(
            test_power, estimated_variance, reduction="none"
        )
        - gaussian_variance_nll(
            test_power, true_variance, reduction="none"
        ),
        axis=(1, 2),
    )
    log_parameter_ratio = np.log10(
        estimated_amplitudes / true_amplitudes
    )

    scipy_result, scipy_amplitudes, scipy_variance = _scipy_reference(
        train_power[0], dictionary
    )
    first_sage_variance = estimated_variance[0]
    scipy_objective_gap = is_divergence(
        train_power[0], first_sage_variance
    ) - is_divergence(train_power[0], scipy_variance)

    histories = [result.objective_history for result in deterministic_results.values()]
    max_len = max(history.size for history in histories)
    padded = np.full((len(histories), max_len), np.nan)
    for index, history in enumerate(histories):
        padded[index, : history.size] = history
    archive_path = output_dir / "experiment_03_fixed_rate_sage.npz"
    np.savez_compressed(
        archive_path,
        deterministic_histories=padded,
        deterministic_labels=np.array(
            [f"initial amplitude {scale:g}" for scale in deterministic_results]
        ),
        true_amplitudes=true_amplitudes,
        fitted_amplitudes=deterministic_result.amplitudes,
        log_parameter_ratio=log_parameter_ratio,
        component_labels=np.array(COMPONENT_LABELS),
        heldout_excess_nll=heldout_excess_nll,
    )

    single_error = np.max(
        np.abs(single_result.amplitudes - single_amplitudes)
        / single_amplitudes
    )
    deterministic_error = np.max(
        np.abs(deterministic_result.amplitudes - true_amplitudes)
        / true_amplitudes
    )
    print(f"seed={SEED}")
    print(f"stochastic_realizations={args.seeds}")
    print(f"single_component_max_relative_error={single_error:.3e}")
    print(
        "two_component_floor_max_relative_error="
        f"{deterministic_error:.3e}"
    )
    for scale, result in deterministic_results.items():
        print(
            f"initial_scale={scale:g}, sweeps={result.n_iter}, "
            f"final_is={result.objective_history[-1]:.6e}"
        )
    print(
        "median_log10_parameter_ratio="
        f"{np.median(log_parameter_ratio, axis=(0, 1))}"
    )
    print(
        "mean_heldout_excess_nll="
        f"{np.mean(heldout_excess_nll):.6e}"
    )
    print(
        f"stochastic_sweeps={stochastic_result.n_iter}, "
        f"converged={stochastic_result.converged}"
    )
    print(
        f"scipy_success={scipy_result.success}, "
        f"scipy_is_gap={scipy_objective_gap:.6e}"
    )
    print(
        "first_seed_scipy_amplitudes="
        f"{np.array2string(scipy_amplitudes, precision=4)}"
    )
    print(f"saved={archive_path.resolve()}")


if __name__ == "__main__":
    main()
