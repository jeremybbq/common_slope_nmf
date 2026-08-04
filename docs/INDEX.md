# Project index

## Purpose

`multi_slope_NMF` develops a physically constrained IS-NMF/SAGE method for common,
multi-slope RIR decay estimation. It estimates frequency-dependent rate trajectories
shared across RIRs, with non-negative per-RIR amplitudes and noise floors.

## Reading order

1. [Model](MODEL.md) defines the observation model, units, and identifiability.
2. [Algorithm](ALGORITHM.md) separates the fixed-rate SAGE solver from wideband search,
   continuous relocation, pruning, and smooth trajectories.
3. [Experiments](EXPERIMENTS.md) defines the validation sequence before algorithmic claims.
4. [Design findings](DESIGN_FINDINGS.md) records measured numerical behavior and
   unvalidated solver ideas for future experiments.
5. [References](REFERENCES.md) records the source material and relationship to the LINEX
   predecessor.

## Code map

| Area | Status | Responsibility |
| --- | --- | --- |
| `multislope_nmf/model.py` | Implemented | Exponential variance model and parameter transforms |
| `multislope_nmf/synth.py` | Implemented | Complex-Gaussian coefficients and instantaneous power |
| `multislope_nmf/is_objective.py` | Implemented | IS divergence and Gaussian variance criterion |
| `multislope_nmf/sage.py` | Implemented | Fixed-dictionary componentwise SAGE amplitude updates |
| `multislope_nmf/wideband.py` | Planned | Interval-integrated decay atoms and subdivision |
| `multislope_nmf/refine.py` | Planned | Continuous rate relocation, pruning, and ordering |
| `multislope_nmf/smooth.py` | Planned | Frequency-trajectory parameterization and penalties |

The implemented modules support the first three validation experiments in
[Experiments](EXPERIMENTS.md). Joint rate estimation remains a research plan.
