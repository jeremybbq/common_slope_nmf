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
generator are validated, together with fixed-rate amplitude and floor estimation.

## Repository layout

```text
multislope_nmf/    Importable model, objective, synthesis, and SAGE utilities
tests/             Deterministic convention, recovery, and statistical tests
examples/          Reproducible validation analyses and figures
docs/              Research specification and project index
```

## Status

The energy-decay convention, complex-Gaussian observation generator, IS objective, and
fixed-rate SAGE baseline are implemented with focused tests. The next milestone is
single-frequency rate profiling and small-component-count enumeration, followed by
wideband localization and continuous rate refinement.
