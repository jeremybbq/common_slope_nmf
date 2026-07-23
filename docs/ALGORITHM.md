# Algorithm plan

## 1. Fixed-rate IS-SAGE baseline

With rates fixed, treat the exponential components and noise floor as latent independent
Gaussian components. Use the SAGE E-step to form expected component powers, then update
non-negative amplitudes and floors. Profile each component's rate only after the fixed-rate
baseline has been validated.

## 2. Wideband localization

Avoid a dense point-rate grid. For each active interval `[lambda_minus, lambda_plus]`, use
the normalized integrated atom

```text
psi(tau) = [exp(-lambda_minus tau) - exp(-lambda_plus tau)]
           / [(lambda_plus - lambda_minus) tau],
psi(0) = 1.
```

Fit a fixed wideband dictionary, retain intervals with support across RIRs, and subdivide
only those intervals. Wideband coefficients localize rates; they are not final point-rate
amplitudes.

## 3. Continuous refinement and pruning

Replace surviving intervals by point rates. Alternate posterior assignment, continuous
rate relocation using the profiled component objective, exact amplitude updates, and
pruning of inactive or duplicate components. Preserve rate ordering to avoid label swaps.

## 4. Smooth frequency trajectories

Parameterize `log(lambda[f,k])` with smooth trajectories only after independent-frequency
recovery is reliable. Tune smoothness against held-out data and report its effect on bias.

## Implementation order

1. Model and IS objective with finite-value safeguards.
2. Fixed-rate amplitudes/floors and SAGE recovery tests.
3. Single-frequency rate profiling and small-K enumeration.
4. Wideband subdivision, continuous relocation, and pruning.
5. Cross-frequency smoothing and real-RIR experiments.
