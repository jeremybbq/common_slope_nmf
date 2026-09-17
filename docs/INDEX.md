# Project index

## Purpose

`common_slope_nmf` develops a physically constrained IS-NMF/SAGE method for common,
multi-slope RIR decay estimation. It estimates frequency-dependent rate trajectories
shared across RIRs, with non-negative per-RIR amplitudes and noise floors.

## Reading order

1. [Model](MODEL.md) defines the observation model, units, and identifiability.
2. [Synthetic decay data](SYNTHESIS.md) defines reproducible parameter and observation draws.
3. [Preprocessing](PREPROCESSING.md) defines coarse data summaries and SAGE initialization.
4. [Plotting](PLOTTING.md) defines experiment plot scripts that load saved NPZ results.
5. [Algorithm](ALGORITHM.md) separates supplied-atom amplitude updates, profiled
   decay-rate SAGE, wideband search, pruning, and smooth trajectories.
6. [Experiments](EXPERIMENTS.md) defines the validation sequence before algorithmic claims.
7. [Design findings](DESIGN_FINDINGS.md) records measured numerical behavior and
   unvalidated solver ideas for future experiments.
8. [References](REFERENCES.md) records the source material and relationship to the LINEX
   predecessor.

## Code map

| Area | Status | Responsibility |
| --- | --- | --- |
| `common_slope_nmf/model.py` | Implemented | Exponential variance model and parameter transforms |
| `common_slope_nmf/synth.py` | Implemented | Dirichlet decay parameters and complex-Gaussian observations |
| `common_slope_nmf/preprocess.py` | Implemented | Head/tail power summaries and pooled coarse decay fitting |
| `common_slope_nmf/rir.py` | Implemented | Array DSP: pooled global onset, resampling, STFT power, and frame selection |
| `common_slope_nmf/is_objective.py` | Implemented | IS divergence and Gaussian variance criterion |
| `common_slope_nmf/sage/` | Implemented | Supplied-atom amplitudes, profiled decay-rate SAGE, and experimental weighted pseudo-SAGE |
| `common_slope_nmf/squarem.py` | Implemented | Safeguarded full-sweep SQUAREM acceleration for fixed-rate and joint decay SAGE maps |
| `experiments/datasets/` | Implemented | SOFA and MATLAB SRIR readers that return RIR arrays |
| `common_slope_nmf/plotting.py` | Removed | Experiment plot scripts load NPZ; Matplotlib is not in the numerical package |
| `common_slope_nmf/wideband.py` | Planned | Interval-integrated decay atoms and subdivision |
| `common_slope_nmf/refine.py` | Planned | Component pruning and ordering |
| `common_slope_nmf/smooth.py` | Planned | Frequency-trajectory parameterization and penalties |

Runnable questions live under `experiments/`:

| Experiment | Fit CLI | Plot CLI |
| --- | --- | --- |
| Synthetic validation and sweeps | `experiments.synthetic.*` | `experiments.synthetic.plot_*` |
| Room-to-hallway SOFA | `experiments.room_to_hallway.omni` | `experiments.room_to_hallway.plot_omni` |
| Coupled-room SRIR | `experiments.coupled_rooms.fit`, `.refine` | `experiments.coupled_rooms.plot_fit`, `.plot_refine` |
| SQUAREM comparisons | `experiments.squarem.compare_convergence`, `.warm_start` | matching `plot_*` modules |

The implemented modules support the validation experiments through the independent-
frequency decay-detection sweep and three-coupled-room real-RIR experiment, including the
independent-frequency joint rate/amplitude step, seeded stochastic two-slope run, and a
controlled weighted pseudo-SAGE comparison, decay-initialization basin scan, and one
equal-base-sweep SQUAREM convergence comparison with a weighted-to-ordinary warm-start
continuation. The weighted variant is implemented for experimentation rather than claimed
as a monotone-likelihood method. Wideband
initialization, component selection, and smooth frequency trajectories remain research
plans.

The raw SOFA room-transition experiment is implemented in
`experiments/room_to_hallway/omni.py`. Dataset-format readers live in
`experiments/datasets`. The package starts from already-read RIR arrays: it caches the
agreed 512-sample Hann/512-point FFT powers in the experiment, and keeps pilot and full
weighted pseudo-SAGE stages separate so numerical behavior can be reviewed before the
full launch.
