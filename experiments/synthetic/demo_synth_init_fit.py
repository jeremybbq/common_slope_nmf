"""Demonstrate package synthesis, preprocessing, initialization, and fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_slope_nmf import (
    init_decay_sage,
    pseudo_decay_sage,
    rate_to_t60,
    sample_multislope_data,
    t60_to_rate,
)
from experiments._run_output import create_run_output_dir

SEED = 20260817
N_RIRS = 512
N_FREQUENCIES = 1
N_COMPONENTS = 2
N_FRAMES = 374
HOP_S = 128.0 / 24_000.0
T60_RANGE_S = (0.5, 3.0)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    return args


def main() -> None:
    """Generate an exact dataset, initialize it, fit it, and save diagnostics."""

    args = _arguments()
    output_dir = create_run_output_dir(args.output_root)
    times_s = np.arange(N_FRAMES, dtype=np.float64) * HOP_S
    data = sample_multislope_data(
        times_s,
        N_RIRS,
        N_FREQUENCIES,
        N_COMPONENTS,
        t60_range_s=T60_RANGE_S,
        amplitude_concentration=0.25,
        noise_mean_db=-40.0,
        noise_std_db=2.0 / 3.0,
        min_t60_separation_s=0.4,
        rng=np.random.default_rng(SEED),
    )
    init = init_decay_sage(
        data.observed_power,
        data.times_s,
        N_COMPONENTS,
        rate_bounds_per_s=(
            t60_to_rate(T60_RANGE_S[1]),
            t60_to_rate(T60_RANGE_S[0]),
        ),
        n_head_frames=8,
        n_tail_frames=8,
        floor_margin_db=6.0,
    )
    result = pseudo_decay_sage(
        data.observed_power,
        data.times_s,
        init.rates_per_s,
        rate_bounds_per_s=(
            t60_to_rate(T60_RANGE_S[1]),
            t60_to_rate(T60_RANGE_S[0]),
        ),
        component_weight_power=2.0,
        initial_amplitudes=init.amplitudes,
        initial_noise_floor=init.noise_floor,
        max_iter=args.max_iter,
    )

    true_t60_s = data.t60_s[0]
    estimated_t60_s = np.sort(rate_to_t60(result.rates_per_s)[0])
    initial_t60_s = rate_to_t60(init.rates_per_s)[0]
    t60_history_s = np.sort(rate_to_t60(result.rate_history_per_s[:, 0, :]), axis=1)
    archive_path = output_dir / "synth_init_fit_results.npz"
    np.savez_compressed(
        archive_path,
        seed=np.asarray(SEED),
        times_s=times_s,
        true_t60_s=true_t60_s,
        initial_t60_s=initial_t60_s,
        estimated_t60_s=estimated_t60_s,
        true_amplitudes=data.amplitudes,
        initial_amplitudes=init.amplitudes,
        true_noise_floor=data.noise_floor,
        initial_noise_floor=init.noise_floor,
        coarse_rate_per_s=init.fit.rate_per_s,
        regression_r_squared=init.fit.r_squared,
        regression_frame_mask=init.fit.frame_mask,
        objective_history=result.objective_history,
        rate_history_per_s=result.rate_history_per_s,
        t60_history_s=t60_history_s,
        observed_power=data.observed_power,
        true_variance=data.variance,
        fitted_variance=result.variance,
        n_iter=np.asarray(result.n_iter),
        converged=np.asarray(result.converged),
    )

    print(f"true_t60_s={true_t60_s.tolist()}")
    print(f"coarse_t60_s={rate_to_t60(init.fit.rate_per_s).tolist()}")
    print(f"estimated_t60_s={estimated_t60_s.tolist()}")
    print(f"sweeps={result.n_iter}, converged={result.converged}")
    print(f"final_is={result.objective_history[-1]:.9g}")
    print(f"saved={archive_path.resolve()}")


if __name__ == "__main__":
    main()
