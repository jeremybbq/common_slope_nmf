# multi_slope_NMF

Research scaffold for estimating multiple, frequency-dependent room-acoustic decay
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
examples/          Reproducible validation analyses and figures
docs/              Research specification and project index
```

Every example that writes artifacts creates a unique timestamped directory under
`output/YYYY-MM-DD_HH-MM-SS-ffffff/`, keeping figures and numerical diagnostics from one
run together.

## Package workflow

```python
import numpy as np

from common_slope_nmf import (
    init_decay_sage,
    pseudo_decay_sage,
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
result = pseudo_decay_sage(
    data.observed_power,
    data.times_s,
    init.rates_per_s,
    rate_bounds_per_s=rate_bounds,
    initial_amplitudes=init.amplitudes,
    initial_noise_floor=init.noise_floor,
    component_weight_power=2.0,
)
```

Run `python -m examples.demo_synth_init_fit` for the corresponding executable
demonstration and timestamped diagnostic outputs.

## Status

The energy-decay convention, complex-Gaussian generator, IS objective, supplied-atom
amplitude stage, and independent-frequency profiled decay-rate SAGE are implemented with
focused tests. Safeguarded full-sweep SQUAREM wrappers accelerate the fixed-rate and joint
decay SAGE maps while retaining the original IS objective and an ordinary-SAGE fallback.
An optional component-strength-weighted pseudo-SAGE M-step is also implemented as an
experimental comparison; unlike exact SAGE, it does not assume a monotone observed
objective. One controlled fixed-data SQUAREM convergence comparison is recorded in
`docs/EXPERIMENTS.md`; broader acceleration benchmarks remain pending. The next
model-development milestones are small-component-count enumeration, wideband localization,
pruning, and smooth frequency trajectories.
