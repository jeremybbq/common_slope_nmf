# Common-slope joint decay and amplitude estimation using parameterized nonnegative matrix factorization

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
common_slope_nmf/     Importable model, objective, synthesis, array DSP, and SAGE
experiments/          Fit CLIs (NPZ/CSV), sibling plot scripts, and dataset readers
tests/package/        Deterministic package tests
tests/experiments/    Experiment, plot, and dataset-reader tests
docs/                 Research specification and project index
```

Every experiment that writes artifacts creates a unique timestamped directory under
`output/YYYY-MM-DD_HH-MM-SS-ffffff/`, keeping figures and numerical diagnostics from one
run together.

## Package workflow

```python
import numpy as np

from common_slope_nmf import (
    init_decay,
    fit_decay,
    sample_stft_power,
    t60_to_rate,
)

frame_time_s = np.arange(374) * 128 / 24_000
data = sample_stft_power(
    frame_time_s,
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
rate_per_s, amplitudes, noise_floor = init_decay(
    data.observed_power,
    data.frame_time_s,
    n_head_frames=8,
    n_tail_frames=8,
)
result = fit_decay(
    data.observed_power,
    data.frame_time_s,
    np.repeat(
        np.clip(rate_per_s, *rate_bounds)[:, np.newaxis], 2, axis=1
    ),
    rate_bounds_per_s=rate_bounds,
    initial_amplitudes=np.repeat(
        (amplitudes / 2.0)[:, :, np.newaxis], 2, axis=2
    ),
    initial_noise_floor=noise_floor,
    component_weight_power=2.0,
)
```

Run the three synthetic experiments from the repository root. Fit CLIs write NPZ/CSV
only; sibling plot scripts read those archives:

```bash
python -m experiments.synthetic.convergence
python -m experiments.synthetic.decay_robustness
python -m experiments.synthetic.amplitude_identifiability
python -m experiments.synthetic.plot_convergence --results output/RUN/loss_convergence_results.npz
```

They cover one-pair convergence/loss geometry, pooled-pair decay robustness, and
known-rate amplitude identifiability. The numerical package starts from already-read
RIR arrays; SOFA/SRIR readers live in `experiments.datasets`. Full configurations and
real-data commands are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The paper's main recorded-RIR application is the coupled-room transition dataset:

```bash
python -m experiments.room_to_hallway.omni --stage pilot
python -m experiments.room_to_hallway.omni --stage full
python -m experiments.room_to_hallway.plot_omni --results output/RUN
```

`experiments.room_to_hallway.georg_benchmarks` provides the corresponding external
baselines. Dataset preparation and external model requirements are documented in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Install the numerical package from PyPI:

```bash
python -m pip install common-slope-nmf
```

From a checkout, install the package with optional experiment and test dependencies:

```bash
python -m pip install -e '.[experiments,test]'
python -m pytest
```

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
Register a pending publisher for project `common-slope-nmf`, owner `jeremybbq`,
repository `common_slope_nmf`, workflow `publish.yml`, and environment `pypi`.
Require a manual approval on that GitHub environment. Pushing a `v*` tag, such
as `v0.1.0`, runs the tests, builds the sdist and wheel, and uploads them.

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
