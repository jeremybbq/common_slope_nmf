"""Compare SAGE loss paths on a profiled two-decay loss surface."""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    AmplitudeSAGEResult,
    DecaySAGEInit,
    DecaySAGEResult,
    amplitude_sage,
    decay_sage,
    exponential_variance,
    init_decay_sage,
    cw_decay_sage,
    rate_to_t60,
    sample_power,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir

SEED = 20260907

N_RIRS = 256

N_COMPONENTS = 2

N_FRAMES = 256

HOP_S = 128.0 / 24_000.0

T60_RANGE_S = (0.5, 3.0)

TRUE_T60_S = np.array([0.80, 1.40], dtype=np.float64)

SHORT_T60_RANGE_S = (0.6, 1.4)

LONG_T60_RANGE_S = (1.0, 1.8)

SURFACE_GRID_SIZE = 21

NOISE_MEAN_DB = -40.0

NOISE_STD_DB = 2.0 / 3.0

DECAY_LOSS_TOL = 0.0

SURFACE_LOSS_TOL = 1e-6

METHOD_KEYS = ("ordinary", "p1", "p2")

METHOD_LABELS = (
    r"SAGE ($p=0$)",
    r"CW-SAGE ($p=1$)",
    r"CW-SAGE ($p=2$)",
)

METHOD_POWERS: tuple[float | None, ...] = (None, 1.0, 2.0)

METHOD_COLORS = ("C2", "C0", "C1")

def _rate_bounds() -> tuple[float, float]:
    """Return energy-decay-rate bounds in inverse seconds."""

    return (
        float(t60_to_rate(T60_RANGE_S[1])),
        float(t60_to_rate(T60_RANGE_S[0])),
    )


def _init_from_data(
    observed_power: np.ndarray, times_s: np.ndarray
) -> DecaySAGEInit:
    """Return pooled-regression decay and equal-amplitude initialization."""

    return init_decay_sage(
        observed_power,
        times_s,
        N_COMPONENTS,
        rate_bounds_per_s=_rate_bounds(),
        n_head_frames=8,
        n_tail_frames=8,
        floor_margin_db=6.0,
    )


def _fit_method(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    init: DecaySAGEInit,
    component_weight_power: float | None,
    *,
    max_iter: int,
    tol: float,
    rate_method: str,
) -> DecaySAGEResult:
    """Fit one method to power ``(R,1,N)`` from the shared warm start."""

    common = dict(
        rate_bounds_per_s=_rate_bounds(),
        initial_amplitudes=init.amplitudes,
        initial_noise_floor=init.noise_floor,
        estimate_noise_floor=True,
        max_iter=max_iter,
        tol=tol,
        rate_method=rate_method,
    )
    if component_weight_power is None:
        return decay_sage(observed_power, times_s, init.rates_per_s, **common)
    return cw_decay_sage(
        observed_power,
        times_s,
        init.rates_per_s,
        component_weight_power=component_weight_power,
        **common,
    )


def run_loss_comparison(
    rng: np.random.Generator,
    times_s: np.ndarray,
    *,
    n_rirs: int,
    max_iter: int,
    tol: float,
    rate_method: str,
) -> dict[str, object]:
    """Fit the three SAGE orders to one controlled two-decay dataset.

    Parameters
    ----------
    rng
        Seeded random generator controlling amplitudes, floors, and powers.
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    n_rirs
        Number of independent RIR realizations ``R``.
    max_iter
        Maximum number of complete component sweeps.
    tol
        Relative observed IS-loss stopping tolerance.
    rate_method
        Profile-rate solver, ``"newton"`` or ``"bisection"``.

    Returns
    -------
    dict
        Ground truth, common warm initialization, observations, estimates,
        objective histories, and T60 trajectories. Component amplitudes have
        shape ``(R,1,2)`` and sum to one at time zero.
    """

    first_amplitude = rng.uniform(0.0, 1.0, size=(n_rirs, 1, 1))
    true_amplitudes = np.concatenate((first_amplitude, 1.0 - first_amplitude), axis=2)
    noise_level_db = rng.normal(NOISE_MEAN_DB, NOISE_STD_DB, size=(n_rirs, 1))
    true_noise_floor = 10.0 ** (noise_level_db / 10.0)
    exact_variance = exponential_variance(
        times_s,
        t60_to_rate(TRUE_T60_S[np.newaxis, :]),
        true_amplitudes,
        noise_floor=true_noise_floor,
    )
    observed_power = sample_power(exact_variance, rng=rng)
    init = _init_from_data(observed_power, times_s)

    histories: list[np.ndarray] = []
    rate_histories: list[np.ndarray] = []
    estimated_t60_s = np.empty((len(METHOD_KEYS), N_COMPONENTS))
    estimated_rates_per_s = np.empty((len(METHOD_KEYS), 1, N_COMPONENTS))
    estimated_amplitudes = np.empty((len(METHOD_KEYS), n_rirs, 1, N_COMPONENTS))
    estimated_noise_floor = np.empty((len(METHOD_KEYS), n_rirs, 1))
    n_iter = np.empty(len(METHOD_KEYS), dtype=np.int64)
    converged = np.empty(len(METHOD_KEYS), dtype=bool)
    for method_index, power in enumerate(METHOD_POWERS):
        print(f"loss path: {METHOD_LABELS[method_index]}", flush=True)
        result = _fit_method(
            observed_power,
            times_s,
            init,
            power,
            max_iter=max_iter,
            tol=tol,
            rate_method=rate_method,
        )
        estimated_t60_s[method_index] = np.sort(
            rate_to_t60(result.rates_per_s[0])
        )
        estimated_rates_per_s[method_index] = result.rates_per_s
        estimated_amplitudes[method_index] = result.amplitudes
        estimated_noise_floor[method_index] = result.noise_floor
        n_iter[method_index] = result.n_iter
        converged[method_index] = result.converged
        histories.append(result.objective_history.copy())
        rate_histories.append(
            np.sort(rate_to_t60(result.rate_history_per_s[:, 0, :]), axis=1)
        )

    return {
        "histories": [histories],
        "t60_trajectories_s": rate_histories,
        "true_amplitudes": true_amplitudes,
        "true_noise_floor": true_noise_floor,
        "observed_power": observed_power,
        "exact_variance": exact_variance,
        "init_rates_per_s": init.rates_per_s,
        "init_amplitudes": init.amplitudes,
        "init_noise_floor": init.noise_floor,
        "initial_t60_s": np.sort(rate_to_t60(init.rates_per_s[0])),
        "estimated_t60_s": estimated_t60_s,
        "estimated_rates_per_s": estimated_rates_per_s,
        "estimated_amplitudes": estimated_amplitudes,
        "estimated_noise_floor": estimated_noise_floor,
        "n_iter": n_iter,
        "converged": converged,
    }


def _fixed_rate_atoms(times_s: np.ndarray, t60_s: np.ndarray) -> np.ndarray:
    """Return two exponential atoms and one constant-floor atom, shape ``(3,N)``."""

    decay_atoms = np.exp(
        -t60_to_rate(t60_s)[:, np.newaxis] * times_s[np.newaxis, :]
    )
    return np.concatenate((decay_atoms, np.ones((1, times_s.size))), axis=0)


def _profile_one_point(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    t60_s: np.ndarray,
    initial_joint_amplitudes: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> AmplitudeSAGEResult:
    """Profile amplitudes and the floor for one supplied T60 pair."""

    return amplitude_sage(
        observed_power,
        _fixed_rate_atoms(times_s, t60_s),
        initial_amplitudes=initial_joint_amplitudes,
        max_iter=max_iter,
        tol=tol,
    )


def profile_fixed_rate_loss_surface(
    observed_power: np.ndarray,
    times_s: np.ndarray,
    short_t60_s: np.ndarray,
    long_t60_s: np.ndarray,
    initial_amplitudes: np.ndarray,
    initial_noise_floor: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    cold_check_t60_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Profile nuisance amplitudes over a two-dimensional T60 grid.

    Parameters
    ----------
    observed_power
        Strictly positive observed power, shape ``(R,N)``.
    times_s
        Elapsed frame times in seconds, shape ``(N,)``.
    short_t60_s, long_t60_s
        Strictly increasing decay grids in seconds, shapes ``(X,)`` and
        ``(Y,)``.
    initial_amplitudes
        Common data-derived decay-amplitude start, shape ``(R,2)`` in
        variance units.
    initial_noise_floor
        Common data-derived floor start, shape ``(R,)`` in variance units.
    max_iter
        Maximum fixed-rate amplitude SAGE sweeps per grid point.
    tol
        Relative total IS-loss stopping tolerance.
    cold_check_t60_s
        Optional additional T60 pair in seconds, shape ``(2,)``. Its nearest
        grid point is included in the independent-start diagnostics.

    Returns
    -------
    dict
        Profiled total IS loss ``(Y,X)``, fitted decay amplitudes
        ``(Y,X,R,2)``, fitted floors ``(Y,X,R)``, iteration/convergence maps,
        and continuation-versus-independent diagnostics. Every point is fit
        once from a serpentine continuation and once from the common pooled
        initialization; the lower-loss converged state defines the reported
        profile. Sparse corner/center/check-location gaps are returned as a
        compact traversal-dependence summary.
    """

    power = np.asarray(observed_power, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    short = np.asarray(short_t60_s, dtype=np.float64)
    long = np.asarray(long_t60_s, dtype=np.float64)
    amplitudes = np.asarray(initial_amplitudes, dtype=np.float64)
    floor = np.asarray(initial_noise_floor, dtype=np.float64)
    if power.ndim != 2 or times.shape != (power.shape[1],):
        raise ValueError("observed_power and times_s must have shapes (R,N) and (N,).")
    if amplitudes.shape != (power.shape[0], 2) or floor.shape != (power.shape[0],):
        raise ValueError("initial amplitudes/floor must have shapes (R,2) and (R,).")
    if (
        short.ndim != 1
        or long.ndim != 1
        or short.size < 2
        or long.size < 2
        or np.any(np.diff(short) <= 0.0)
        or np.any(np.diff(long) <= 0.0)
    ):
        raise ValueError("T60 grids must be strictly increasing and non-trivial.")

    n_long, n_short, n_rirs = long.size, short.size, power.shape[0]
    loss = np.empty((n_long, n_short))
    fitted_amplitudes = np.empty((n_long, n_short, n_rirs, 2))
    fitted_floor = np.empty((n_long, n_short, n_rirs))
    n_iter = np.empty((n_long, n_short), dtype=np.int64)
    converged = np.empty((n_long, n_short), dtype=bool)
    continuation_loss = np.empty((n_long, n_short))
    independent_loss = np.empty((n_long, n_short))
    selected_independent = np.empty((n_long, n_short), dtype=bool)
    common_start = np.concatenate((amplitudes, floor[:, np.newaxis]), axis=1)
    continuation = common_start.copy()

    for long_index, long_value in enumerate(long):
        short_indices = (
            range(n_short) if long_index % 2 == 0 else range(n_short - 1, -1, -1)
        )
        for short_index in short_indices:
            continuation_result = _profile_one_point(
                power,
                times,
                np.array([short[short_index], long_value]),
                continuation,
                max_iter=max_iter,
                tol=tol,
            )
            independent_result = _profile_one_point(
                power,
                times,
                np.array([short[short_index], long_value]),
                common_start,
                max_iter=max_iter,
                tol=tol,
            )
            continuation_value = continuation_result.objective_history[-1]
            independent_value = independent_result.objective_history[-1]
            use_independent = (
                independent_result.converged and not continuation_result.converged
            ) or (
                independent_result.converged == continuation_result.converged
                and independent_value < continuation_value
            )
            result = independent_result if use_independent else continuation_result
            continuation = result.amplitudes
            continuation_loss[long_index, short_index] = continuation_value
            independent_loss[long_index, short_index] = independent_value
            selected_independent[long_index, short_index] = use_independent
            loss[long_index, short_index] = result.objective_history[-1]
            fitted_amplitudes[long_index, short_index] = result.amplitudes[:, :2]
            fitted_floor[long_index, short_index] = result.amplitudes[:, 2]
            n_iter[long_index, short_index] = result.n_iter
            converged[long_index, short_index] = result.converged

    cold_index_list = [
        [0, 0],
        [0, n_short - 1],
        [n_long // 2, n_short // 2],
        [n_long - 1, 0],
        [n_long - 1, n_short - 1],
    ]
    if cold_check_t60_s is not None:
        check_pair = np.asarray(cold_check_t60_s, dtype=np.float64)
        if check_pair.shape != (2,) or not np.all(np.isfinite(check_pair)):
            raise ValueError("cold_check_t60_s must be finite with shape (2,).")
        cold_index_list.append(
            [
                int(np.argmin(np.abs(long - check_pair[1]))),
                int(np.argmin(np.abs(short - check_pair[0]))),
            ]
        )
    cold_indices = np.unique(np.asarray(cold_index_list, dtype=np.int64), axis=0)
    check_long = cold_indices[:, 0]
    check_short = cold_indices[:, 1]
    cold_loss = independent_loss[check_long, check_short]
    warm_check_loss = continuation_loss[check_long, check_short]

    return {
        "loss": loss,
        "amplitudes": fitted_amplitudes,
        "noise_floor": fitted_floor,
        "n_iter": n_iter,
        "converged": converged,
        "continuation_loss": continuation_loss,
        "independent_loss": independent_loss,
        "selected_independent": selected_independent,
        "cold_check_indices": cold_indices,
        "cold_check_loss": cold_loss,
        "cold_check_relative_gap": np.abs(cold_loss - warm_check_loss)
        / np.maximum(1.0, np.abs(cold_loss)),
    }


def _padded_histories(
    histories: list[list[np.ndarray]], max_iter: int
) -> np.ndarray:
    """Pad nested histories to shape ``(case,method,max_iter+1)``."""

    padded = np.full((len(histories), len(METHOD_KEYS), max_iter + 1), np.nan)
    for case_index, case in enumerate(histories):
        for method_index, history in enumerate(case):
            padded[case_index, method_index, : history.size] = history
    return padded


def _padded_trajectories(
    trajectories: list[np.ndarray], max_iter: int
) -> np.ndarray:
    """Pad T60 paths to shape ``(method,max_iter+1,2)`` in seconds."""

    padded = np.full((len(trajectories), max_iter + 1, 2), np.nan)
    for method_index, trajectory in enumerate(trajectories):
        padded[method_index, : trajectory.shape[0]] = trajectory
    return padded


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rirs", type=int, default=N_RIRS)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument(
        "--tol",
        type=float,
        default=DECAY_LOSS_TOL,
        help="Relative total observed-loss tolerance; zero uses all sweeps.",
    )
    parser.add_argument("--surface-grid-size", type=int, default=SURFACE_GRID_SIZE)
    parser.add_argument("--surface-max-iter", type=int, default=2_000)
    parser.add_argument(
        "--surface-tol",
        type=float,
        default=SURFACE_LOSS_TOL,
        help="Relative IS-loss tolerance for fixed-rate amplitude profiling.",
    )
    parser.add_argument(
        "--rate-method", choices=("newton", "bisection"), default="newton"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()
    for name in (
        "n_rirs",
        "n_frames",
        "max_iter",
        "surface_grid_size",
        "surface_max_iter",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_frames < 17:
        parser.error("--n-frames must be at least 17")
    if args.surface_grid_size < 2:
        parser.error("--surface-grid-size must be at least two")
    for name in ("tol", "surface_tol"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def main() -> None:
    """Fit one decay pair and save numerical archives without plotting."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    times_s = np.arange(args.n_frames, dtype=np.float64) * HOP_S
    result = run_loss_comparison(
        np.random.default_rng(args.seed),
        times_s,
        n_rirs=args.n_rirs,
        max_iter=args.max_iter,
        tol=args.tol,
        rate_method=args.rate_method,
    )
    histories = result["histories"]
    trajectories = result["t60_trajectories_s"]
    if not isinstance(histories, list) or not isinstance(trajectories, list):
        raise RuntimeError("internal error: histories have the wrong type")

    short_t60_s = np.linspace(
        *SHORT_T60_RANGE_S, args.surface_grid_size, dtype=np.float64
    )
    long_t60_s = np.linspace(
        *LONG_T60_RANGE_S, args.surface_grid_size, dtype=np.float64
    )
    print(
        f"profiling {args.surface_grid_size}x{args.surface_grid_size} T60 grid",
        flush=True,
    )
    surface = profile_fixed_rate_loss_surface(
        np.asarray(result["observed_power"])[:, 0, :],
        times_s,
        short_t60_s,
        long_t60_s,
        np.asarray(result["init_amplitudes"])[:, 0, :],
        np.asarray(result["init_noise_floor"])[:, 0],
        max_iter=args.surface_max_iter,
        tol=args.surface_tol,
        cold_check_t60_s=TRUE_T60_S,
    )
    archive_path = output_dir / "loss_convergence_results.npz"
    np.savez_compressed(
        archive_path,
        seed=np.asarray(args.seed),
        times_s=times_s,
        method_keys=np.asarray(METHOD_KEYS),
        method_labels=np.asarray(METHOD_LABELS),
        method_weight_powers=np.asarray([0.0, 1.0, 2.0]),
        true_t60_s=TRUE_T60_S,
        objective_history=_padded_histories(histories, args.max_iter),
        t60_trajectory_s=_padded_trajectories(trajectories, args.max_iter),
        true_amplitudes=result["true_amplitudes"],
        true_noise_floor=result["true_noise_floor"],
        observed_power=result["observed_power"],
        exact_variance=result["exact_variance"],
        init_rates_per_s=result["init_rates_per_s"],
        init_amplitudes=result["init_amplitudes"],
        init_noise_floor=result["init_noise_floor"],
        initial_t60_s=result["initial_t60_s"],
        estimated_t60_s=result["estimated_t60_s"],
        estimated_rates_per_s=result["estimated_rates_per_s"],
        estimated_amplitudes=result["estimated_amplitudes"],
        estimated_noise_floor=result["estimated_noise_floor"],
        rate_method=np.asarray(args.rate_method),
        n_iter=result["n_iter"],
        converged=result["converged"],
        surface_short_t60_s=short_t60_s,
        surface_long_t60_s=long_t60_s,
        surface_loss=surface["loss"],
        surface_amplitudes=surface["amplitudes"],
        surface_noise_floor=surface["noise_floor"],
        surface_n_iter=surface["n_iter"],
        surface_converged=surface["converged"],
        surface_continuation_loss=surface["continuation_loss"],
        surface_independent_loss=surface["independent_loss"],
        surface_selected_independent=surface["selected_independent"],
        surface_cold_check_indices=surface["cold_check_indices"],
        surface_cold_check_loss=surface["cold_check_loss"],
        surface_cold_check_relative_gap=surface["cold_check_relative_gap"],
        max_iter=np.asarray(args.max_iter),
        tolerance=np.asarray(args.tol),
        surface_max_iter=np.asarray(args.surface_max_iter),
        surface_tolerance=np.asarray(args.surface_tol),
        amplitude_sampling=np.asarray("Uniform(0, 1); complementary second component"),
    )
    print(f"initial_t60_s={np.asarray(result['initial_t60_s']).tolist()}")
    print(
        "surface_converged="
        f"{int(np.sum(surface['converged']))}/{surface['converged'].size}"
    )
    print(
        "max_cold_start_relative_gap="
        f"{float(np.max(surface['cold_check_relative_gap'])):.3e}"
    )
    print(f"saved={archive_path.resolve()}")


if __name__ == "__main__":
    main()
