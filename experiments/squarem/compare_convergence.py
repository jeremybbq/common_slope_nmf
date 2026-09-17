"""Compare ordinary, weighted, and SQUAREM two-slope convergence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from common_slope_nmf import (
    decay_sage,
    decay_squarem,
    pseudo_decay_sage,
    rate_to_t60,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir
from experiments.synthetic.validate_multislope_sage import (
    INITIAL_NOISE_DB,
    INITIAL_T60_S,
    N_COMPONENTS,
    N_FREQUENCIES,
    N_RIRS,
    T60_RANGE_S,
    MultislopeData,
    generate_multislope_data,
)


@dataclass(frozen=True)
class ConvergenceRun:
    """One method's loss trajectory on the fixed two-slope dataset.

    ``sweep_evaluations`` and ``objective_history`` have shape ``(H,)``;
    ``t60_history_s`` has shape ``(H, 2)`` in seconds. ``elapsed_s`` is wall
    time in seconds, or NaN when loading a previously saved reference run.
    """

    label: str
    sweep_evaluations: np.ndarray
    objective_history: np.ndarray
    t60_history_s: np.ndarray
    elapsed_s: float
    converged: bool | None
    accepted_extrapolations: int = 0
    rejected_extrapolations: int = 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-sweep-evaluations",
        type=int,
        default=2_000,
        help="Common budget of complete ordinary-SAGE sweep evaluations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Relative observed-objective tolerance shared by all methods.",
    )
    parser.add_argument(
        "--fixed-point-tol",
        type=float,
        default=1e-6,
        help="Additional dimensionless fixed-point tolerance for SQUAREM.",
    )
    parser.add_argument(
        "--reference-diagnostics",
        type=Path,
        action="append",
        default=[],
        help=(
            "Previously saved validate_multislope_sage diagnostics to reuse. "
            "Repeat for multiple methods. Without references, ordinary SAGE "
            "and pseudo-SAGE with p=1 and p=2 are rerun."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for the timestamped comparison output.",
    )
    args = parser.parse_args()
    if args.max_sweep_evaluations <= 0:
        parser.error("--max-sweep-evaluations must be positive")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and non-negative")
    if not np.isfinite(args.fixed_point_tol) or args.fixed_point_tol < 0.0:
        parser.error("--fixed-point-tol must be finite and non-negative")
    return args


def _initial_parameters(data: MultislopeData) -> tuple[np.ndarray, np.ndarray]:
    """Return the historical amplitude and floor initialization.

    The amplitudes have shape ``(512, 1, 2)`` in variance units and the
    floors have shape ``(512, 1)`` in variance units.
    """

    initial_power = np.mean(data.observed_power[:, :, :8], axis=2)
    initial_amplitudes = np.repeat(
        (initial_power / N_COMPONENTS)[:, :, np.newaxis],
        N_COMPONENTS,
        axis=2,
    )
    initial_noise_floor = np.full(
        (N_RIRS, N_FREQUENCIES), 10.0 ** (INITIAL_NOISE_DB / 10.0)
    )
    return initial_amplitudes, initial_noise_floor


def _rate_bounds() -> tuple[float, float]:
    """Return lower and upper energy-decay-rate bounds in inverse seconds."""

    return (
        float(t60_to_rate(T60_RANGE_S[1])),
        float(t60_to_rate(T60_RANGE_S[0])),
    )


def _run_sage(
    data: MultislopeData,
    max_sweep_evaluations: int,
    tol: float,
    component_weight_power: float | None,
) -> ConvergenceRun:
    initial_amplitudes, initial_noise_floor = _initial_parameters(data)
    estimator = decay_sage if component_weight_power is None else pseudo_decay_sage
    extra = (
        {}
        if component_weight_power is None
        else {"component_weight_power": component_weight_power}
    )
    label = (
        "ordinary SAGE"
        if component_weight_power is None
        else f"pseudo-SAGE (rho^{component_weight_power:g})"
    )
    print(f"running {label}...", flush=True)
    start = perf_counter()
    result = estimator(
        data.observed_power,
        data.times_s,
        t60_to_rate(INITIAL_T60_S),
        rate_bounds_per_s=_rate_bounds(),
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=True,
        max_iter=max_sweep_evaluations,
        tol=tol,
        rate_method="newton",
        **extra,
    )
    return ConvergenceRun(
        label=label,
        sweep_evaluations=np.arange(result.objective_history.size),
        objective_history=result.objective_history,
        t60_history_s=np.sort(
            rate_to_t60(result.rate_history_per_s[:, 0, :]), axis=1
        ),
        elapsed_s=perf_counter() - start,
        converged=result.converged,
    )


def _run_squarem(
    data: MultislopeData,
    max_sweep_evaluations: int,
    tol: float,
    fixed_point_tol: float,
) -> ConvergenceRun:
    initial_amplitudes, initial_noise_floor = _initial_parameters(data)
    print("running safeguarded SQUAREM...", flush=True)
    start = perf_counter()
    result = decay_squarem(
        data.observed_power,
        data.times_s,
        t60_to_rate(INITIAL_T60_S),
        rate_bounds_per_s=_rate_bounds(),
        initial_amplitudes=initial_amplitudes,
        initial_noise_floor=initial_noise_floor,
        estimate_noise_floor=True,
        max_sweep_evaluations=max_sweep_evaluations,
        objective_tol=tol,
        fixed_point_tol=fixed_point_tol,
        rate_method="newton",
    )
    return ConvergenceRun(
        label="SQUAREM (ordinary-SAGE map)",
        sweep_evaluations=result.sweep_evaluation_history,
        objective_history=result.objective_history,
        t60_history_s=np.sort(
            rate_to_t60(result.rate_history_per_s[:, 0, :]), axis=1
        ),
        elapsed_s=perf_counter() - start,
        converged=result.converged,
        accepted_extrapolations=result.n_accepted_extrapolations,
        rejected_extrapolations=result.n_rejected_extrapolations,
    )


def _load_reference(path: Path, data: MultislopeData) -> ConvergenceRun:
    """Load one historical run and verify that it used the fixed dataset."""

    with np.load(path) as archive:
        objective = np.asarray(archive["objective_history"], dtype=np.float64)
        t60_history = np.asarray(archive["t60_history_s"], dtype=np.float64)
        true_t60 = np.asarray(archive["true_t60_s"], dtype=np.float64)
        if "component_weight_power" in archive:
            power = float(archive["component_weight_power"])
            label = f"pseudo-SAGE (rho^{power:g})"
        else:
            label = "ordinary SAGE"
    if not np.allclose(true_t60, data.t60_s, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path} does not contain the fixed seeded dataset.")
    return ConvergenceRun(
        label=label,
        sweep_evaluations=np.arange(objective.size),
        objective_history=objective,
        t60_history_s=t60_history,
        elapsed_s=float("nan"),
        converged=None,
    )


def _save_summary(
    output_dir: Path, runs: list[ConvergenceRun], true_t60_s: np.ndarray
) -> Path:
    """Write one CSV row per method and return its absolute path."""

    path = output_dir / "multislope_convergence_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "sweep_evaluations",
                "retained_states",
                "initial_is",
                "final_is",
                "final_t60_short_s",
                "final_t60_long_s",
                "t60_absolute_error_short_s",
                "t60_absolute_error_long_s",
                "elapsed_s",
                "converged",
                "accepted_extrapolations",
                "rejected_extrapolations",
            ),
        )
        writer.writeheader()
        for run in runs:
            final_t60 = run.t60_history_s[-1]
            writer.writerow(
                {
                    "method": run.label,
                    "sweep_evaluations": int(run.sweep_evaluations[-1]),
                    "retained_states": run.objective_history.size,
                    "initial_is": f"{run.objective_history[0]:.12g}",
                    "final_is": f"{run.objective_history[-1]:.12g}",
                    "final_t60_short_s": f"{final_t60[0]:.12g}",
                    "final_t60_long_s": f"{final_t60[1]:.12g}",
                    "t60_absolute_error_short_s": (
                        f"{abs(final_t60[0] - true_t60_s[0]):.12g}"
                    ),
                    "t60_absolute_error_long_s": (
                        f"{abs(final_t60[1] - true_t60_s[1]):.12g}"
                    ),
                    "elapsed_s": (
                        "" if np.isnan(run.elapsed_s) else f"{run.elapsed_s:.9g}"
                    ),
                    "converged": (
                        "unknown" if run.converged is None else run.converged
                    ),
                    "accepted_extrapolations": run.accepted_extrapolations,
                    "rejected_extrapolations": run.rejected_extrapolations,
                }
            )
    return path.resolve()


def _save_histories(output_dir: Path, runs: list[ConvergenceRun]) -> Path:
    """Save each variable-length numerical trajectory without interpolation."""

    path = output_dir / "multislope_convergence_histories.npz"
    arrays: dict[str, np.ndarray] = {}
    for index, run in enumerate(runs):
        prefix = f"method_{index}"
        arrays[f"{prefix}_label"] = np.asarray(run.label)
        arrays[f"{prefix}_sweep_evaluations"] = run.sweep_evaluations
        arrays[f"{prefix}_objective_history"] = run.objective_history
        arrays[f"{prefix}_t60_history_s"] = run.t60_history_s
    np.savez_compressed(path, **arrays)
    return path.resolve()


def main() -> None:
    """Run or load baselines, fit SQUAREM, and save the comparison."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    data = generate_multislope_data()
    if args.reference_diagnostics:
        runs = [
            _load_reference(path, data) for path in args.reference_diagnostics
        ]
    else:
        runs = [
            _run_sage(data, args.max_sweep_evaluations, args.tol, None),
            _run_sage(data, args.max_sweep_evaluations, args.tol, 1.0),
            _run_sage(data, args.max_sweep_evaluations, args.tol, 2.0),
        ]
    squarem = _run_squarem(
        data,
        args.max_sweep_evaluations,
        args.tol,
        args.fixed_point_tol,
    )
    if any(
        not np.isclose(
            run.objective_history[0],
            squarem.objective_history[0],
            rtol=1e-13,
            atol=0.0,
        )
        for run in runs
    ):
        raise ValueError("comparison runs do not share the same initial state.")
    runs.append(squarem)

    summary_path = _save_summary(output_dir, runs, data.t60_s[0])
    histories_path = _save_histories(output_dir, runs)

    print(f"true_t60_s={data.t60_s[0].tolist()}")
    for run in runs:
        print(
            f"{run.label}: evaluations={run.sweep_evaluations[-1]}, "
            f"final_is={run.objective_history[-1]:.9f}, "
            f"final_t60_s={run.t60_history_s[-1].tolist()}, "
            f"elapsed_s={run.elapsed_s:.6g}, "
            f"accepted={run.accepted_extrapolations}, "
            f"rejected={run.rejected_extrapolations}"
        )
    print(f"saved={summary_path}")
    print(f"saved={histories_path}")


if __name__ == "__main__":
    main()
