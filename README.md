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
references are collected in [docs/INDEX.md](docs/INDEX.md). These documents define the
research direction; they are not yet a validated implementation.

## Repository layout

```text
multislope_nmf/    Future importable implementation
tests/             Future deterministic and recovery tests
examples/          Future reproducible analyses and figures
docs/              Research specification and project index
```

## Status

The repository intentionally contains only the project structure and research index.
The first implementation milestone is a fixed-rate IS baseline with synthetic recovery
tests, followed by wideband localization and continuous rate refinement.
