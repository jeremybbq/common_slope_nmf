"""Refine saved coupled-room fits with ordinary, unweighted decay SAGE.

Each input is a ``full_results.npz`` artifact produced by
``examples.fit_coupled_rooms``.  Its fitted rates and per-RIR amplitudes are
used together as a warm start, while all other coupled-room settings remain
unchanged.  Frequencies are fitted independently and without a noise floor.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_slope_nmf import decay_sage, rate_to_t60, t60_to_rate
from examples._run_output import create_run_output_dir
from examples.fit_coupled_rooms import (
    CACHE_METADATA_PATH,
    CACHE_PATH,
    T60_BOUNDS_S,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "starts",
        nargs="+",
        type=Path,
        help="Saved full_results.npz files to use as warm starts.",
    )
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument(
        "--cache-metadata", type=Path, default=CACHE_METADATA_PATH
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument(
        "--highest-bins",
        type=int,
        default=0,
        help="Use only this many highest-frequency bins from each warm start.",
    )
    args = parser.parse_args()
    if args.max_iter <= 0 or args.jobs <= 0:
        parser.error("--max-iter and --jobs must be positive")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    if args.highest_bins < 0:
        parser.error("--highest-bins must be non-negative")
    return args


def load_warm_start(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frequencies, T60s, and amplitudes from one saved result.

    Returns
    -------
    frequencies_hz
        Frequency-bin centers, shape ``(F,)``, in Hz.
    t60_s
        Ordered energy-decay times, shape ``(F,K)``, in seconds.
    amplitudes
        Unit-origin variance amplitudes, shape ``(R,F,K)``.
    """

    with np.load(path, allow_pickle=False) as result:
        frequencies_hz = np.asarray(result["frequencies_hz"], dtype=np.float64)
        t60_s = np.asarray(result["estimated_t60_s"], dtype=np.float64)
        amplitudes = np.asarray(
            result["estimated_amplitudes"], dtype=np.float64
        )
    if frequencies_hz.ndim != 1:
        raise ValueError("saved frequencies_hz must have shape (F,).")
    if t60_s.ndim != 2 or t60_s.shape[0] != frequencies_hz.size:
        raise ValueError("saved estimated_t60_s must have shape (F,K).")
    if amplitudes.ndim != 3 or amplitudes.shape[1:] != t60_s.shape:
        raise ValueError(
            "saved estimated_amplitudes must have shape (R,F,K)."
        )
    if (
        np.any(~np.isfinite(t60_s))
        or np.any(t60_s <= 0.0)
        or np.any(~np.isfinite(amplitudes))
        or np.any(amplitudes < 0.0)
    ):
        raise ValueError("saved warm-start parameters must be finite and valid.")
    return frequencies_hz, t60_s, amplitudes


def _fit_one(
    cache_path: str,
    metadata_path: str,
    frequency_index: int,
    warm_t60_s: np.ndarray,
    warm_amplitudes: np.ndarray,
    max_iter: int,
    tol: float,
    result_path: str,
) -> str:
    power_cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    observed_power = np.asarray(
        power_cache[:, frequency_index : frequency_index + 1, :],
        dtype=np.float64,
    )
    with np.load(metadata_path, allow_pickle=False) as metadata:
        times_s = np.asarray(metadata["times_s"], dtype=np.float64)
        frequency_hz = float(metadata["frequencies_hz"][frequency_index])

    result = decay_sage(
        observed_power,
        times_s,
        np.asarray(t60_to_rate(warm_t60_s))[np.newaxis, :],
        rate_bounds_per_s=(
            t60_to_rate(T60_BOUNDS_S[1]),
            t60_to_rate(T60_BOUNDS_S[0]),
        ),
        initial_amplitudes=warm_amplitudes[:, np.newaxis, :],
        estimate_noise_floor=False,
        max_iter=max_iter,
        tol=tol,
        rate_method="newton",
    )
    fitted_t60_s = np.asarray(rate_to_t60(result.rates_per_s[0]))
    order = np.argsort(fitted_t60_s)
    fitted_t60_s = fitted_t60_s[order]
    fitted_amplitudes = result.amplitudes[:, 0, order]
    t60_history_s = np.sort(
        np.asarray(rate_to_t60(result.rate_history_per_s[:, 0, :])), axis=1
    )
    np.savez_compressed(
        result_path,
        frequency_index=np.asarray(frequency_index),
        frequency_hz=np.asarray(frequency_hz),
        warm_t60_s=warm_t60_s,
        estimated_t60_s=fitted_t60_s,
        estimated_amplitudes=fitted_amplitudes,
        objective_history=result.objective_history,
        t60_history_s=t60_history_s,
        n_iter=np.asarray(result.n_iter),
        converged=np.asarray(result.converged),
    )
    return result_path


def _run_branch(
    label: str,
    start_path: Path,
    cache_path: Path,
    metadata_path: Path,
    max_iter: int,
    tol: float,
    jobs: int,
    highest_bins: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    start_frequencies, warm_t60, warm_amplitudes = load_warm_start(start_path)
    if highest_bins:
        if highest_bins > start_frequencies.size:
            raise ValueError(
                "--highest-bins exceeds the frequency count in a warm start."
            )
        start_frequencies = start_frequencies[-highest_bins:]
        warm_t60 = warm_t60[-highest_bins:]
        warm_amplitudes = warm_amplitudes[:, -highest_bins:, :]
    with np.load(metadata_path, allow_pickle=False) as metadata:
        cache_frequencies = np.asarray(
            metadata["frequencies_hz"], dtype=np.float64
        )
    frequency_indices = []
    for frequency_hz in start_frequencies:
        matches = np.flatnonzero(cache_frequencies == frequency_hz)
        if matches.size != 1:
            raise ValueError(
                f"saved frequency {frequency_hz:g} Hz is absent from the cache."
            )
        frequency_indices.append(int(matches[0]))

    branch_dir = output_dir / label
    branch_dir.mkdir()
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(jobs, len(frequency_indices))) as pool:
        futures = {}
        for local_index, frequency_index in enumerate(frequency_indices):
            result_path = branch_dir / f"frequency_{frequency_index:03d}.npz"
            future = pool.submit(
                _fit_one,
                str(cache_path),
                str(metadata_path),
                frequency_index,
                warm_t60[local_index],
                warm_amplitudes[:, local_index, :],
                max_iter,
                tol,
                str(result_path),
            )
            futures[future] = local_index
        for count, future in enumerate(as_completed(futures), start=1):
            local_index = futures[future]
            path = future.result()
            with np.load(path, allow_pickle=False) as result:
                row: dict[str, object] = {
                    "label": label,
                    "frequency_hz": float(result["frequency_hz"]),
                    "warm_t60_s": result["warm_t60_s"].copy(),
                    "estimated_t60_s": result["estimated_t60_s"].copy(),
                    "estimated_amplitudes": result[
                        "estimated_amplitudes"
                    ].copy(),
                    "objective_history": result["objective_history"].copy(),
                    "t60_history_s": result["t60_history_s"].copy(),
                    "n_iter": int(result["n_iter"]),
                    "converged": bool(result["converged"]),
                }
            rows.append(row)
            print(
                f"{label} {count:02d}/{len(frequency_indices):02d}: "
                f"{row['frequency_hz']:7.1f} Hz, sweeps={row['n_iter']:4d}, "
                f"converged={row['converged']}, "
                f"T60={np.round(row['estimated_t60_s'], 3)} s",
                flush=True,
            )
    return sorted(rows, key=lambda row: float(row["frequency_hz"]))


def _safe_label(path: Path, index: int) -> str:
    parent = path.parent.name or path.stem
    return f"start_{index + 1}_{parent}".replace(" ", "_")


def _save_summary(
    branches: list[list[dict[str, object]]], output_dir: Path
) -> None:
    flat = [row for branch in branches for row in branch]
    with (output_dir / "unweighted_warm_start_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "start",
                "frequency_hz",
                "warm_t60_1_s",
                "warm_t60_2_s",
                "warm_t60_3_s",
                "final_t60_1_s",
                "final_t60_2_s",
                "final_t60_3_s",
                "sweeps",
                "converged",
                "initial_is",
                "final_is",
                "final_over_initial_is",
            ]
        )
        for row in flat:
            objective = np.asarray(row["objective_history"])
            writer.writerow(
                [
                    row["label"],
                    row["frequency_hz"],
                    *np.asarray(row["warm_t60_s"]),
                    *np.asarray(row["estimated_t60_s"]),
                    row["n_iter"],
                    row["converged"],
                    objective[0],
                    objective[-1],
                    objective[-1] / objective[0],
                ]
            )

    labels = [str(branch[0]["label"]) for branch in branches]
    frequencies = np.asarray(
        [float(row["frequency_hz"]) for row in branches[0]]
    )
    final_t60 = np.stack(
        [
            np.stack([np.asarray(row["estimated_t60_s"]) for row in branch])
            for branch in branches
        ]
    )
    final_objective = np.stack(
        [
            [np.asarray(row["objective_history"])[-1] for row in branch]
            for branch in branches
        ]
    )
    initial_objective = np.stack(
        [
            [np.asarray(row["objective_history"])[0] for row in branch]
            for branch in branches
        ]
    )
    n_iter = np.asarray(
        [[int(row["n_iter"]) for row in branch] for branch in branches]
    )
    converged = np.asarray(
        [[bool(row["converged"]) for row in branch] for branch in branches]
    )
    amplitudes = np.stack(
        [
            np.stack(
                [np.asarray(row["estimated_amplitudes"]) for row in branch],
                axis=1,
            )
            for branch in branches
        ]
    )
    np.savez_compressed(
        output_dir / "unweighted_warm_start_results.npz",
        labels=np.asarray(labels),
        frequencies_hz=frequencies,
        estimated_t60_s=final_t60,
        estimated_amplitudes=amplitudes,
        initial_objective=initial_objective,
        final_objective=final_objective,
        n_iter=n_iter,
        converged=converged,
        component_weight_power=np.asarray(0.0),
        estimate_noise_floor=np.asarray(False),
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    line_styles = ("-", "--", ":", "-.")
    for branch_index, label in enumerate(labels):
        for component_index in range(final_t60.shape[2]):
            axes[0].plot(
                frequencies,
                final_t60[branch_index, :, component_index],
                marker="o",
                linestyle=line_styles[branch_index % len(line_styles)],
                color=f"C{component_index}",
                label=(
                    f"{label}, component {component_index + 1}"
                    if component_index == 0
                    else None
                ),
            )
    axes[0].set_ylabel("Final energy T60 (s)")
    axes[0].set_title("Ordinary SAGE from weighted-fit warm starts")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for branch_index, label in enumerate(labels):
        axes[1].plot(
            frequencies,
            final_objective[branch_index],
            marker="o",
            linestyle=line_styles[branch_index % len(line_styles)],
            label=label,
        )
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Final observed IS objective")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "unweighted_warm_start_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Run unweighted refinements from all supplied saved estimates."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    branches = []
    for index, start in enumerate(args.starts):
        label = _safe_label(start, index)
        branches.append(
            _run_branch(
                label,
                start,
                args.cache,
                args.cache_metadata,
                args.max_iter,
                args.tol,
                args.jobs,
                args.highest_bins,
                output_dir,
            )
        )
    reference_frequencies = [
        float(row["frequency_hz"]) for row in branches[0]
    ]
    for branch in branches[1:]:
        if [float(row["frequency_hz"]) for row in branch] != reference_frequencies:
            raise ValueError("all warm starts must contain the same frequencies.")
    _save_summary(branches, output_dir)
    print(f"outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
