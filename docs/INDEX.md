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
4. [References](REFERENCES.md) records the source material and relationship to the LINEX
   predecessor.

## Planned code map

| Area | Intended responsibility |
| --- | --- |
| `multislope_nmf/model.py` | Exponential variance model and parameter transforms |
| `multislope_nmf/is_objective.py` | IS likelihood and numerically stable derivatives |
| `multislope_nmf/sage.py` | Latent-component E-step and component M-steps |
| `multislope_nmf/wideband.py` | Interval-integrated decay atoms and subdivision |
| `multislope_nmf/refine.py` | Continuous rate relocation, pruning, and ordering |
| `multislope_nmf/smooth.py` | Frequency-trajectory parameterization and penalties |
| `multislope_nmf/synth.py` | Synthetic STFT/RIR data with known ground truth |

Only the package placeholder exists today. Create modules together with focused tests and
documented numerical conventions.
