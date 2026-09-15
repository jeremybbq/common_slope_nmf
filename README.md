# multi_slope_NMF

Research package for estimating multiple, frequency-dependent room-acoustic decay
processes from sets of room impulse responses (RIRs). The target method is a
wideband-initialized, gridless IS-SAGE estimator: an Itakura-Saito (IS) likelihood
with physically constrained exponential variance components.

This is a successor to `multislope_linex`. That project estimates per-RIR component
amplitudes for fixed common decay times using a LINEX loss. This project instead
jointly estimates shared, frequency-dependent decay-rate trajectories and non-negative
per-RIR amplitudes from STFT power.

## Research model

For RIR `r`, frequency bin `f`, and time frame `n`, observed STFT power is
`Y[r,f,n] = |X[r,f,n]|^2`, with variance

```text
V[r,f,n] = b[r,f] + sum_k a[r,f,k] exp(-lambda[f,k] tau[n]).
```

The rates `lambda[f,k]` are shared across RIRs; amplitudes `a[r,f,k]` and noise
floors `b[r,f]` are non-negative and RIR-dependent. Fitting minimizes IS divergence,
equivalently the working complex-Gaussian variance likelihood.

## Documentation

The documentation map, model definition, algorithm plan, experimental protocol, and
references are collected in [docs/INDEX.md](docs/INDEX.md). The forward model and synthetic
generator are validated, together with supplied-atom amplitude estimation and
independent-frequency joint decay-rate estimation.

## Repository layout

```text
common_slope_nmf/  Importable model, objective, synthesis, and SAGE utilities
tests/             Deterministic convention, recovery, and statistical tests
experiments/       Synthetic and real-data experiments with local plots
docs/              Research specification and project index
```

Every experiment that writes artifacts creates a unique timestamped directory under
`output/YYYY-MM-DD_HH-MM-SS-ffffff/`, keeping figures and numerical diagnostics from one
run together.

## Package workflow

```python
import numpy as np

from common_slope_nmf import (
    init_decay_sage,
    cw_decay_sage,
    sample_multislope_data,
    t60_to_rate,
)

times_s = np.arange(374) * 128 / 24_000
data = sample_multislope_data(
    times_s,
    n_rirs=512,
    n_frequencies=1,
    n_components=2,
    t60_range_s=(0.5, 3.0),
    amplitude_concentration=0.25,
    noise_mean_db=-40.0,
    noise_std_db=2 / 3,
    min_t60_separation_s=0.4,
    rng=np.random.default_rng(20260817),
)
rate_bounds = (t60_to_rate(3.0), t60_to_rate(0.5))
init = init_decay_sage(
    data.observed_power,
    data.times_s,
    n_components=2,
    rate_bounds_per_s=rate_bounds,
)
result = cw_decay_sage(
    data.observed_power,
    data.times_s,
    init.rates_per_s,
    rate_bounds_per_s=rate_bounds,
    initial_amplitudes=init.amplitudes,
    initial_noise_floor=init.noise_floor,
    component_weight_power=2.0,
)
```

Run the three synthetic experiments from the repository root:

```bash
python -m experiments.synthetic_convergence
python -m experiments.synthetic_decay_robustness
python -m experiments.synthetic_amplitude_identifiability
```

They cover one-pair convergence/loss geometry, pooled-pair decay robustness, and
known-rate amplitude identifiability. Each supports `--plot-results PATH` to regenerate
figures from saved numerical results. Full configurations and real-data commands are
in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The paper's main recorded-RIR application is the coupled-room transition dataset:

```bash
python -m experiments.roomToHallway_omni --stage pilot
python -m experiments.roomToHallway_omni --stage full
```

`experiments/roomToHallway_georg_benchmarks.py` provides the corresponding external
baselines. Dataset preparation and external model requirements are documented in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Install the numerical package and optional experiment/test dependencies:

```bash
python -m pip install -e '.[experiments,test]'
python -m pytest
```

## Status

The numerical package implements synthesis, initialization, fixed-rate amplitude SAGE,
and independent-frequency decay SAGE, including contribution-weighted CW-SAGE.
Results include total IS-loss and decay-rate histories, final parameter estimates, and
optional intermediate responsibilities and profile moments. Weighted updates do not
guarantee a monotone observed objective.

Experiment archives provide saved results and replotting. A general package-level
checkpoint API is separate future work. Wideband localization, component selection,
and smooth frequency trajectories remain research plans.

SQUAREM and the historical synthetic experiments are preserved on `feat/squarem`.
The three-room Treble simulation scripts, tests, and findings are preserved on
`feat/treble-simulation`.
