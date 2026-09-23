"""Measure decay-estimate robustness across independently sampled T60 pairs."""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    DecayFit,
    exponential_variance,
    init_decay,
    fit_decay,
    rate_to_t60,
    sample_power,
    sample_t60,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir

SEED = 20260907

N_RIRS = 512

N_PAIRS = 100

N_COMPONENTS = 2

N_FRAMES = 512

HOP_S = 128.0 / 24_000.0

T60_RANGE_S = (0.5, 3.0)

NOISE_MEAN_DB = -40.0

NOISE_STD_DB = 2.0 / 3.0

METHOD_KEYS = ("p1", "p2")

METHOD_LABELS = (r"CW-SAGE ($p=1$)", r"CW-SAGE ($p=2$)")

METHOD_POWERS = (1.0, 2.0)

def frequency_batches(
    n_frequencies: int, batch_size: int
) -> list[slice]:
    """Return contiguous frequency slices covering ``range(n_frequencies)``.

    Parameters
    ----------
    n_frequencies
        Positive number of independent frequency bins ``F``.
    batch_size
        Positive maximum number of bins fitted in one SAGE call.

    Returns
    -------
    list of slice
        Non-overlapping slices that cover every frequency exactly once.
    """

    if n_frequencies <= 0 or batch_size <= 0:
        raise ValueError("n_frequencies and batch_size must be positive.")
    return [
        slice(start, min(start + batch_size, n_frequencies))
        for start in range(0, n_frequencies, batch_size)
    ]


def order_decay_components(
    estimated_t60_s: np.ndarray,
    estimated_amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order fitted components by increasing energy-``T60`` in every bin.

    Parameters
    ----------
    estimated_t60_s
        Fitted decay times in seconds, shape ``(F,K)``.
    estimated_amplitudes
        Fitted unit-origin variance amplitudes, shape ``(R,F,K)``.

    Returns
    -------
    ordered_t60_s, ordered_amplitudes, order
        Decay times ``(F,K)``, consistently permuted amplitudes ``(R,F,K)``,
        and the integer component permutation ``(F,K)``.
    """

    t60_s = np.asarray(estimated_t60_s, dtype=np.float64)
    amplitudes = np.asarray(estimated_amplitudes, dtype=np.float64)
    if t60_s.ndim != 2:
        raise ValueError("estimated_t60_s must have shape (F,K).")
    if amplitudes.ndim != 3 or amplitudes.shape[1:] != t60_s.shape:
        raise ValueError("estimated_amplitudes must have shape (R,F,K).")
    order = np.argsort(t60_s, axis=1)
    ordered_t60_s = np.take_along_axis(t60_s, order, axis=1)
    amplitude_order = np.broadcast_to(order[np.newaxis, :, :], amplitudes.shape)
    ordered_amplitudes = np.take_along_axis(
        amplitudes, amplitude_order, axis=2
    )
    return ordered_t60_s, ordered_amplitudes, order


def _rate_bounds() -> tuple[float, float]:
    """Return energy-decay-rate bounds in inverse seconds."""

    return (
        float(t60_to_rate(T60_RANGE_S[1])),
        float(t60_to_rate(T60_RANGE_S[0])),
    )


def _fit_method(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    initial_rates_per_s: np.ndarray,
    initial_amplitudes: np.ndarray,
    initial_noise_floor: np.ndarray,
    component_weight_power: float,
    *,
    max_iter: int,
    tol: float,
) -> DecayFit:
    """Fit one contribution-weighted SAGE order to power ``(R,F,N)``."""

    return fit_decay(
        observed_power,
        times_s,
        initial_rates_per_s,
        rate_bounds_per_s=_rate_bounds(),
        component_weight_power=component_weight_power,
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=True,
        max_iter=max_iter,
        tol=tol,
    )


def _ordered_fit(
    result: DecayFit,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted T60 ``(F,2)`` and correspondingly ordered amplitudes."""

    t60_s, amplitudes, _ = order_decay_components(
        rate_to_t60(result.rates_per_s), result.amplitudes
    )
    return t60_s, amplitudes


def _per_frequency_is(
    observed_power: np.ndarray, fitted_variance: np.ndarray
) -> np.ndarray:
    """Return observed IS divergence per frequency, shape ``(F,)``."""

    ratio = observed_power / fitted_variance
    return np.sum(ratio - np.log(ratio) - 1.0, axis=(0, 2))


def run_identifiability_comparison(
    rng: np.random.Generator,
    times_s: np.ndarray,
    *,
    n_pairs: int,
    n_rirs: int,
    batch_size: int,
    max_iter: int,
    tol: float,
) -> dict[str, np.ndarray]:
    """Fit contribution-weighted SAGE p=1 and p=2 to independent random T60 pairs.

    Parameters
    ----------
    rng
        Seeded random generator controlling all synthetic quantities.
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    n_pairs
        Number of independently parameterized frequency-bin labels ``F``.
    n_rirs
        Number of independent RIR realizations ``R`` per pair.
    batch_size
        Number of frequency bins per estimator call. Use one for independent
        per-bin stopping decisions.
    max_iter
        Maximum complete component iterations per fit.
    tol
        Dimensionless loss stopping tolerance.

    Returns
    -------
    dict
        True/fitted T60 values in seconds, amplitudes and floors in variance
        units, per-frequency losses, sweep counts, and stopping diagnostics.
    """

    true_t60_s = sample_t60(
        n_pairs, N_COMPONENTS, T60_RANGE_S, rng=rng
    )
    true_rates_per_s = np.asarray(t60_to_rate(true_t60_s), dtype=np.float64)
    first_amplitude = rng.uniform(0.0, 1.0, size=(n_rirs, n_pairs, 1))
    true_amplitudes = np.concatenate((first_amplitude, 1.0 - first_amplitude), axis=2)
    noise_level_db = rng.normal(
        NOISE_MEAN_DB, NOISE_STD_DB, size=(n_rirs, n_pairs)
    )
    true_noise_floor = 10.0 ** (noise_level_db / 10.0)

    batches = frequency_batches(n_pairs, batch_size)
    n_methods = len(METHOD_KEYS)
    estimated_t60_s = np.empty((n_methods, n_pairs, N_COMPONENTS))
    estimated_amplitudes = np.empty(
        (n_methods, n_rirs, n_pairs, N_COMPONENTS)
    )
    estimated_noise_floor = np.empty((n_methods, n_rirs, n_pairs))
    final_is = np.empty((n_methods, n_pairs))
    batch_n_iter = np.empty((n_methods, len(batches)), dtype=np.int64)
    batch_converged = np.empty_like(batch_n_iter, dtype=bool)
    frequency_n_iter = np.empty((n_methods, n_pairs), dtype=np.int64)
    frequency_converged = np.empty((n_methods, n_pairs), dtype=bool)
    objective_history = np.full(
        (n_methods, len(batches), max_iter + 1), np.nan
    )

    rate_history_per_s = np.full((n_methods, max_iter + 1, n_pairs, N_COMPONENTS), np.nan)

    for batch_index, frequency_slice in enumerate(batches):
        exact_variance = exponential_variance(
            times_s,
            true_rates_per_s[frequency_slice],
            true_amplitudes[:, frequency_slice, :],
            noise_floor=true_noise_floor[:, frequency_slice],
        )
        observed_power = sample_power(exact_variance, rng=rng)
        init_rate_per_s, init_amplitudes, init_noise_floor = init_decay(
            observed_power,
            times_s,
            n_head_frames=8,
            n_tail_frames=8,
            floor_margin_db=6.0,
        )
        lower_per_s, upper_per_s = _rate_bounds()
        init_rates_per_s = np.repeat(
            np.clip(init_rate_per_s, lower_per_s, upper_per_s)[:, np.newaxis],
            N_COMPONENTS,
            axis=1,
        )
        init_amplitudes = np.repeat(
            (init_amplitudes / N_COMPONENTS)[:, :, np.newaxis],
            N_COMPONENTS,
            axis=2,
        )
        for method_index, power in enumerate(METHOD_POWERS):
            print(
                f"pair {batch_index + 1}/{len(batches)}: "
                f"{METHOD_LABELS[method_index]}",
                flush=True,
            )
            result = _fit_method(
                observed_power,
                times_s,
                init_rates_per_s,
                init_amplitudes,
                init_noise_floor,
                power,
                max_iter=max_iter,
                tol=tol,
            )
            n_iter = result.loss_history.size - 1
            rate_history_per_s[method_index, : n_iter + 1, frequency_slice] = result.rate_history_per_s
            ordered_t60_s, ordered_amplitudes = _ordered_fit(result)
            estimated_t60_s[method_index, frequency_slice] = ordered_t60_s
            estimated_amplitudes[
                method_index, :, frequency_slice, :
            ] = ordered_amplitudes
            estimated_noise_floor[
                method_index, :, frequency_slice
            ] = result.noise_floor
            final_is[method_index, frequency_slice] = _per_frequency_is(
                observed_power,
                exponential_variance(
                    times_s,
                    result.rates_per_s,
                    result.amplitudes,
                    result.noise_floor,
                ),
            )
            batch_n_iter[method_index, batch_index] = n_iter
            batch_converged[method_index, batch_index] = result.converged
            frequency_n_iter[method_index, frequency_slice] = n_iter
            frequency_converged[method_index, frequency_slice] = result.converged
            objective_history[
                method_index, batch_index, : result.loss_history.size
            ] = result.loss_history

    return {
        "true_t60_s": true_t60_s,
        "true_amplitudes": true_amplitudes,
        "true_noise_floor": true_noise_floor,
        "estimated_t60_s": estimated_t60_s,
        "estimated_amplitudes": estimated_amplitudes,
        "estimated_noise_floor": estimated_noise_floor,
        "final_is": final_is,
        "batch_n_iter": batch_n_iter,
        "batch_converged": batch_converged,
        "frequency_n_iter": frequency_n_iter,
        "frequency_converged": frequency_converged,
        "objective_history": objective_history,
        "rate_history_per_s": rate_history_per_s,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n-rirs", type=int, default=N_RIRS)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Bins per fit; one gives independent per-bin stopping.",
    )
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Relative observed-loss stopping tolerance.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()
    for name in ("n_pairs", "n_rirs", "n_frames", "batch_size", "max_iter"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_frames < 17:
        parser.error("--n-frames must be at least 17")
    for name in ("tol",):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def main() -> None:
    """Fit sampled T60 pairs and save numerical archives without plotting."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    times_s = np.arange(args.n_frames, dtype=np.float64) * HOP_S
    plane_seed = np.random.SeedSequence(args.seed).spawn(2)[1]
    result = run_identifiability_comparison(
        np.random.default_rng(plane_seed),
        times_s,
        n_pairs=args.n_pairs,
        n_rirs=args.n_rirs,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    archive_path = output_dir / "t60_identifiability_results.npz"
    np.savez_compressed(
        archive_path,
        seed=np.asarray(args.seed),
        times_s=times_s,
        method_keys=np.asarray(METHOD_KEYS),
        method_labels=np.asarray(METHOD_LABELS),
        method_weight_powers=np.asarray(METHOD_POWERS),
        t60_range_s=np.asarray(T60_RANGE_S),
        true_t60_s=result["true_t60_s"],
        true_amplitudes=result["true_amplitudes"],
        estimated_t60_s=result["estimated_t60_s"],
        estimated_amplitudes=result["estimated_amplitudes"],
        estimated_noise_floor=result["estimated_noise_floor"],
        frequency_n_iter=result["frequency_n_iter"],
        frequency_converged=result["frequency_converged"],
        objective_history=result["objective_history"],
    )
    for method_index, key in enumerate(METHOD_KEYS):
        print(
            f"{key}: converged="
            f"{int(np.sum(result['frequency_converged'][method_index]))}/"
            f"{args.n_pairs}, median_iterations="
            f"{np.median(result['frequency_n_iter'][method_index]):.1f}"
        )
    print(f"saved={archive_path.resolve()}")


if __name__ == "__main__":
    main()
