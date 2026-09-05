# Design findings and open numerical questions

This note records observations that should influence later solver design. Measured
results are separated from explanations and candidate improvements that still require
experiments.

## Fixed-rate baseline evidence

Experiment 3 uses two fixed exponential atoms with energy `T60 = [0.25, 0.8] s`,
a constant floor atom, 121 frames from `0` to `1.2 s`, and four amplitude rows.

- A one-component deterministic problem recovers its amplitudes to floating-point
  precision after the first component update.
- With two components and floors from `3e-6` to `1e-4`, deterministic recovery from
  uniform initial amplitudes `0.1`, `1`, and `10` took approximately `53,600`,
  `61,400`, and `61,600` complete SAGE sweeps. The final summed IS divergence was
  about `4.1e-10`; the maximum relative parameter error was about `2.4e-4`.
- In the 200-realization stochastic run, SAGE reached the configured 5,000-sweep
  limit. For the first realization, its summed IS divergence was about `0.0195`
  above the independently optimized SciPy solution.
- Across the stochastic run, the median base-10 amplitude errors for the fast
  decay, slow decay, and floor were approximately `[-0.090, -0.005, 0.002]`.
  The mean held-out negative-log-likelihood difference from the generating
  variance was about `0.0126` per coefficient.

These values describe the current synthetic configuration and seed `20260724`.
They are regression evidence rather than general performance claims.

## Known-decay STFT-shaped evidence

Experiment 4 uses 512 independent abstract bins, 374 frames over a 2 s signal
configuration at 24 kHz with a 256-sample frame and 128-sample hop, and an energy
`T60 = 1 s` decay with unit variance amplitude. The exact-model draws use seed
`20260724`.

- Without a floor, SAGE agreed exactly with the analytic maximum-likelihood estimate
  `mean_n(Y/D)` to the displayed floating-point precision. Across bins, the estimated
  amplitude mean was `0.998835` and its standard deviation was `0.052411`, compared
  with the exact values `1` and `1/sqrt(374) = 0.051709`. Exact per-bin 95% confidence
  intervals covered the generating amplitude in `0.953125` of the bins.
- With an unknown `-30 dB` variance floor (`b = 0.001`), the fitted decay amplitudes
  had mean `1.003392` and standard deviation `0.109196`. The fitted floors had mean
  `0.00100408` and standard deviation `0.00006119`. Inverse Fisher information at
  the generating parameters predicts standard deviations `0.111794` and `0.00006156`,
  respectively.
- With an unknown `-60 dB` variance floor (`b = 1e-6`), the fitted decay amplitudes
  had mean `0.997027` and standard deviation `0.075861`. The fitted floors had mean
  `9.93182e-7` and standard deviation `7.63553e-8`; Fisher-predicted standard
  deviations were `0.075950` and `7.63888e-8`.
- Every decay amplitude was initialized at `0.25`; the fitted floors in both noisy
  conditions were initialized at `0.01`. The no-floor, `-30 dB`, and `-60 dB` runs met
  the objective stopping rule after 2, 56, and 34 complete sweeps, respectively. Their
  summed IS objectives were non-increasing at every recorded sweep.

This experiment validates fixed-rate amplitude and floor inference when data are drawn
from the working likelihood. It does not validate rate estimation, STFT independence for
real waveforms, or robustness to model mismatch.

## Why weak components are slow

For component variance `C`, total variance `V`, and SAGE gain `G = C / V`, a weak
component has `G` close to zero over much of the observation window. Its posterior
power and subsequent amplitude correction are then small. This is the familiar
slow-boundary behavior of EM-like fixed-point iterations.

The following effects are plausible contributors and should be tested separately:

- A slow exponential and a constant floor become similar over a short or noisy
  tail, producing an ill-conditioned amplitude subproblem.
- Nearby exponential rates produce coherent dictionary rows and slow redistribution
  of power between components.
- A weak component can have a large relative parameter error while contributing
  negligibly to the total variance and IS objective.

The experiments currently establish the symptom and the proximity to an independent
optimum. They do not yet isolate the convergence factor of each mechanism.

## Consequences for stopping and reporting

On the saved one-frequency `rho**2` trajectory from experiment 7, the first relative
objective-decrease gates at `1e-4`, `1e-5`, and `1e-6` occurred at sweeps 31, 45, and 67.
The corresponding sorted absolute `T60` errors were approximately `[0.110, 0.075] s`,
`[0.014, 0.037] s`, and `[0.019, 0.017] s`. On this run, `1e-4` was visibly too loose,
while `1e-6` avoided the old `1e-10` setting's impractically long tail without materially
changing the recovered rates. The decay-estimator API therefore uses `1e-6` as its
pragmatic default. This single trajectory does not make it a universal accuracy
guarantee; experiment 10 records parameter errors and rate histories at that gate over
100 newly sampled pairs.

A small change in the summed objective is insufficient as the only stopping rule.
Future solvers should record:

1. summed and per-RIR objective changes;
2. maximum relative parameter change, with an absolute scale for tiny components;
3. gradient or bound-constrained stationarity residuals;
4. the number of completed sweeps and whether the iteration cap was reached;
5. held-out likelihood and the fitted-variance error where ground truth exists.

When many independent RIRs or Monte Carlo trials are fitted as one batch, a global
objective can be dominated by well-behaved rows. Per-row diagnostics are needed to
identify the slowest fits.

The same issue matters for rate estimation. A profile-likelihood search can compare
rate candidates unfairly if their inner amplitude problems stop at different
optimization errors. Warm starts across nearby candidates and a stationarity-based
inner stopping condition should be tested before interpreting profile shapes.

## Candidate solver designs to test

These are experiment candidates, not validated replacements:

- Use SAGE for its simple non-negative component updates, then polish amplitudes
  with a bound-constrained optimizer and analytic gradients.
- Compare log-amplitude L-BFGS-B with a direct non-negative parameterization.
  Log parameters remain positive and cannot represent an exact zero without a
  lower bound, so active-set behavior needs explicit testing.
- Test safeguarded fixed-point acceleration such as SQUAREM, Anderson acceleration,
  or over-relaxation, accepting accelerated steps only when the IS objective does
  not increase.
- Compare the published IS-NMF multiplicative update as another supplied-atom
  baseline.
- Precondition the amplitude coordinates using atom means or norms, while converting
  results back to unit-origin physical amplitudes.
- Use a positive least-squares estimate as a warm start, then optimize the IS
  criterion. Its benefit under stochastic power observations remains unverified.

For the expected small component counts, a hybrid SAGE-plus-polish solver is a
particularly practical baseline: it retains the latent-component interpretation and
provides an independent stationarity check at modest cost.

## Required ablations

Before selecting an accelerated amplitude solver, sweep:

- true floor and weak-component levels, including exact zeros;
- rate separation and slow-rate/floor similarity;
- decay-window length and tail truncation;
- uniform, random, least-squares, and warm-start initializations;
- individual versus batched stopping rules;
- SAGE, multiplicative updates, direct optimization, and hybrid polishing.

Report runtime and completed sweeps together with objective gap, parameter error,
variance error, and held-out likelihood. Real-RIR experiments should additionally
vary the decay origin and usable tail because those choices directly affect
slow-rate/floor confounding.
