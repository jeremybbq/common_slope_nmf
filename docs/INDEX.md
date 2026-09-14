# Project index

## Purpose

`multi_slope_NMF` develops a physically constrained IS-NMF/SAGE method for common,
multi-slope RIR decay estimation. It estimates frequency-dependent rate trajectories
shared across RIRs, with non-negative per-RIR amplitudes and noise floors.

## Reading order

1. [Model](MODEL.md) defines the observation model, units, and identifiability.
2. [Synthetic decay data](SYNTHESIS.md) defines reproducible parameter and observation draws.
3. [Preprocessing](PREPROCESSING.md) defines coarse data summaries and SAGE initialization.
4. [Plotting](PLOTTING.md) describes experiment-local diagnostics without rerunning estimators.
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
| `common_slope_nmf/rir.py` | Implemented | MATLAB/SOFA RIR loading, pooled global onset, resampling, STFT power, and frame selection |
| `common_slope_nmf/georg_baselines.py` | Implemented | External DecayFitNet ONNX adapter and Georg-style common-slope EDC clustering/amplitude fit |
| `common_slope_nmf/is_objective.py` | Implemented | IS divergence and Gaussian variance criterion |
| `common_slope_nmf/sage.py` | Implemented | Supplied-atom amplitudes, profiled decay-rate SAGE, and experimental weighted pseudo-SAGE |
| `common_slope_nmf/wideband.py` | Planned | Interval-integrated decay atoms and subdivision |
| `common_slope_nmf/refine.py` | Planned | Component pruning and ordering |
| `common_slope_nmf/smooth.py` | Planned | Frequency-trajectory parameterization and penalties |

## Experiment map

| Script | Question |
| --- | --- |
| `experiments/synthetic_convergence.py` | One T60 pair: total IS loss and profiled surface |
| `experiments/synthetic_decay_robustness.py` | Sampled T60 pairs: decay-estimate robustness |
| `experiments/synthetic_amplitude_identifiability.py` | Known T60 pair: amplitude masking and uncertainty |
| `experiments/fit_coupled_rooms.py` | Three-coupled-room RIR fits |
| `experiments/refine_coupled_rooms_unweighted.py` | Ordinary-SAGE continuation from saved coupled-room fits |
| `experiments/roomToHallway_omni.py` | Room-transition SOFA RIR fits |
| `experiments/roomToHallway_georg_benchmarks.py` | External DecayFitNet/CommonSlopeAnalysis comparison |

Plot functions live in their experiments; `_run_output.py` only creates run directories.
See [Experiments](EXPERIMENTS.md) for configurations, commands, and interpretation.
SQUAREM and historical scripts are preserved on `feat/squarem`.

The weighted variant remains experimental, without a monotone-likelihood claim.
Wideband initialization, component selection, and smooth frequency trajectories remain
research plans. Existing result objects provide total IS and rate histories and optional
intermediate responsibilities/profile moments. General package checkpoint APIs remain
separate from the experiment-specific numerical archives.
