# Experimental protocol

## Synthetic recovery

Generate complex Gaussian STFT coefficients from known exponential variances. Sweep rate
separation, component strength, noise floor, observation length, RIR count, and frequency
trajectory smoothness. Evaluate rate error, amplitude error, missed/extra components,
likelihood, and sensitivity to initialization.

## Ablations

Compare a coarse point grid, dense point grid, wideband-only localization, wideband plus
continuous refinement, and the optional smooth-trajectory model. Keep data, component-count
selection, and stopping criteria fixed across variants.

## Real RIRs

Use held-out likelihood and stability across receiver subsets as primary evidence. Compare
with the LINEX predecessor only for the fixed-rate amplitude-estimation subproblem; the
objectives and observations differ, so direct headline comparisons require care.

## Reproducibility

Every experiment should record a seed, STFT configuration, decay origin, units, parameter
bounds, initial intervals, stopping rules, and software environment.
