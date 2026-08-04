# Experimental protocol

## Implemented foundational validation

### 1. Energy-decay convention

Use `T60 = 0.6 s`, `a = 2.5`, `b = 0`, and elapsed times from `0` to `1.2 s`
in `0.01 s` increments. With `lambda = 6 log(10) / T60`, verify in float64
that the unit-origin exponential reaches `10^-6`, or `-60 dB` in power, at
`T60`. Also verify the linear relation
`10 log10(V(tau) / V(0)) = -60 tau / T60`.

### 2. Complex-Gaussian observation generator

For target variances `[1, 10^-1, 10^-3, 10^-6]`, generate `50,000`
independent complex coefficients per variance with a fixed seed. Verify that
the real and imaginary parts each have variance `V / 2`, while normalized
instantaneous power `Y / V` has the mean, variance, median, and 95th percentile
of a unit-mean exponential distribution. The deterministic test tolerances are
predeclared in `tests/test_synth.py`.

Run `examples/validate_decay_and_generator.py` to produce the corresponding
decay, distribution, and calibration plots. These experiments validate the
forward conventions and generator; they make no estimator-recovery claim.

### 3. Fixed-rate amplitude and floor recovery

This experiment fixes the rate dictionary so it tests the IS objective and SAGE
amplitude updates without mixing in rate-search errors.

The algebraic check uses one `T60 = 0.6 s` atom, amplitudes `[0.2, 0.5, 1, 2]`,
no floor, and deterministic `Y = V`. One component sweep must recover every
amplitude to floating-point precision.

The multi-component check uses `T60 = [0.25, 0.8] s`, a row-of-ones floor atom,
and four RIR parameter rows:

```text
fast       slow       floor
1.00       0.08       1e-5
0.10       1.00       3e-5
0.80       0.30       1e-4
0.25       0.70       3e-6
```

With deterministic `Y = V`, initialize every coefficient at `0.1`, `1`, and
`10`. Verify a non-increasing objective and recovery of the amplitudes and
variance. This deliberately weak floor is also a convergence diagnostic:
componentwise SAGE needs tens of thousands of sweeps near the non-negative
boundary.

Finally, generate 200 independent training and held-out power realizations from
the same true variance with seed `20260724`. Record log-amplitude error by
component, held-out complex-Gaussian variance likelihood relative to the true
variance, iteration-limit behavior, and initialization sensitivity. Compare the
first stochastic fit with an independent log-amplitude L-BFGS-B optimization
from SciPy. Sampling recovery is reported as a distribution; exact recovery is
only expected for deterministic `Y = V`.

Run:

```text
python -m examples.validate_fixed_rate_amplitudes
```

The focused assertions are in `tests/test_is_objective.py` and
`tests/test_fixed_rate_sage.py`. Numerical findings and future solver ablations are
maintained in [Design findings](DESIGN_FINDINGS.md).

## Future joint-rate synthetic recovery

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

Real RIR validation follows synthetic rate recovery. It is necessary evidence for the
overall method, while experiment 3 isolates the optimizer with exactly known fixed atoms.

## Reproducibility

Every experiment should record a seed, STFT configuration, decay origin, units, parameter
bounds, initial intervals, stopping rules, and software environment.
