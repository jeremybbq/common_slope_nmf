# Algorithm

## 1. SAGE amplitude step

With rates fixed, treat the exponential components and noise floor as latent independent
Gaussian components. Use the SAGE E-step to form expected component powers, then update
non-negative amplitudes and floors. For fixed positive atom `D[q,n]`, current component
variance `C[r,n] = A[r,q] D[q,n]`, residual `R = V - C`, and Wiener gain `G = C / V`,
the implemented component update is

```text
P_hat[r,n] = G[r,n] (G[r,n] Y[r,n] + R[r,n])
A[r,q] = mean_n(P_hat[r,n] / D[q,n]).
```

Each updated component is immediately folded into `V`, giving a componentwise SAGE sweep.
A row of ones in `D` estimates the time-invariant variance floor. Atoms remain fixed and
unit-normalized at the decay origin, so no unconstrained NMF scale normalization is needed.

`common_slope_nmf.sage` exposes this stage as `amplitude_sage` and records the summed
IS divergence after each sweep. Tests cover exact recovery, monotonicity, nearby rates,
short windows, weak components, weak floors, and agreement with an independent SciPy
optimizer. Weak near-boundary components can require many SAGE sweeps; iteration count and
initialization sensitivity remain required diagnostics. The measured behavior, stopping
implications, and acceleration candidates are recorded in
[Design findings](DESIGN_FINDINGS.md).

The component loop is intentional. After updating component `q`, SAGE immediately inserts
its new contribution into `V` before forming the posterior for `q + 1`. Vectorizing this
axis into a simultaneous update would change the schedule from Gauss--Seidel/SAGE to a
Jacobi/EM-like iteration. The implementation instead vectorizes all row and frame work
within each component. With the intended small component counts, this retains the exact
algorithm while keeping the expensive operations in NumPy.

The acceleration implementation and its derivation are preserved on `feat/squarem`.

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

## 3. Profiled decay-rate SAGE step

Before fitting, `init_decay_sage` can construct equal component amplitudes from leading
frame power, a tail-average floor, and one pooled log-power rate repeated across all
components. The reusable summaries and regression assumptions are specified in
[Preprocessing](PREPROCESSING.md).

`decay_sage` implements the full independent-frequency model with arrays
`Y: (R,F,N)`, `lambda: (F,K)`, `a: (R,F,K)`, and `b: (R,F)`. For component `k`, it forms
posterior powers

```text
S = (C / V)^2 Y + C (1 - C / V)
```

and profiles out its amplitudes. For each frequency, the rate derivative is

```text
g(lambda) = sum_r [N E_{w(r,f,:;lambda)}[tau] - sum_n tau[n]],
w proportional to S exp(lambda tau).
```

The derivative is monotone because its derivative is a non-negative weighted variance:

```text
h(lambda) = N sum_r Var_{w(r,f,:;lambda)}[tau] >= 0.
```

The default rate solver is safeguarded Newton: it proposes `lambda - g/h` inside a valid
gradient-sign bracket and falls back to that bracket's midpoint when the Newton proposal
is outside the bracket or the curvature is numerically unusable. Bracketed bisection is
also exposed as a separate solver and can be selected with
`decay_sage(..., rate_method="bisection")`. Both solvers are vectorized over all
frequencies. After finding the bounded profile minimum, they use the exact amplitude
update `a = mean_n(S exp(lambda tau))`. Log-sum-exp calculations prevent overflow in this
profile step. A constant component receives the corresponding exact floor update.

Components remain sequential for the same SAGE reason as in the amplitude stage, while
RIRs, frequencies, frames, and all frequencywise scalar rate solves are vectorized.
The returned diagnostics contain both the observed IS objective and the complete rate
array before the first sweep and after every completed sweep, allowing raw-objective and
rate-trajectory plots without altering the update schedule.

### Experimental contribution-weighted profile

`cw_decay_sage` keeps the same E-step and sequential component schedule, but changes
the decay-component M-step. Immediately before updating component `k`, evaluate its
current Wiener strength `rho_k = C_k / V`, choose a non-negative exponent `p`, and freeze

```text
w[r,f,n] = rho_k[r,f,n]**p
```

for the entire amplitude-and-rate M-step. For one candidate rate, define

```text
W[r,f]             = sum_n w[r,f,n],
T_w[r,f]           = sum_n w[r,f,n] tau[n],
M[r,f](lambda)     = sum_n w[r,f,n] S[r,f,n] exp(lambda[f] tau[n]),
a[r,f](lambda)     = M[r,f](lambda) / W[r,f].
```

If `q` is the normalized frame distribution proportional to
`w S exp(lambda tau)`, the weighted profile derivatives are

```text
g[f](lambda) = sum_r (W[r,f] E_q[tau] - T_w[r,f]),
h[f](lambda) = sum_r W[r,f] Var_q[tau] >= 0.
```

Both the safeguarded Newton and bracketed-bisection solvers use these expressions. The
weights are recomputed from the current pre-update model for the next component, but are
not differentiated or refreshed inside one M-step. The noise-floor update stays
unweighted. `p = 0` gives unit decay-component weights and recovers the ordinary update;
`p = 1` and `p = 2` select `rho` and `rho**2` weighting.

CW-SAGE is an experimental local reweighting and has not been established as an
auxiliary function for the original
observed likelihood. Therefore the total observed IS objective is still recorded at each
sweep, but monotonic decrease is not assumed or enforced. Numerical experiments must
report any increases rather than presenting this variant as a validated acceleration.

Optional sweep diagnostics expose the scaled total error and every Wiener-style component
weight at a common post-sweep state:

```text
epsilon[r,f,n] = Y[r,f,n] / V[r,f,n] - 1,
rho_k[r,f,n]   = v_k[r,f,n] / V[r,f,n].
```

The weight axis contains all decay components followed by the constant noise component,
and therefore sums to one at every `(r,f,n)`. For the same selected sweep, the diagnostics
also retain the actual weighted profile moment and its denominator used by each
sequential component update,

```text
M[r,f,k](lambda_new) = sum_n w[r,f,n,k] S[r,f,n,k]
                             exp(lambda_new[f,k] tau[n]),
W[r,f,k]             = sum_n w[r,f,n,k],
a_new[r,f,k]         = M[r,f,k](lambda_new) / W[r,f,k].
```

For ordinary SAGE, `w = 1` and `W = N`.

The moments therefore retain their Gauss--Seidel timing, while `epsilon` and the complete
weight vector are evaluated after the sweep so their denominators are consistent.
Collection is opt-in through `diagnostic_interval`; the first sweep is retained and
subsequent multiples of the interval are retained. Each recorded `epsilon` is a full
`(R,F,N)` field and the weights have shape `(R,F,K+1,N)`, so recording every sweep can be
memory-intensive.

## 4. Continuous refinement and pruning

Replace surviving intervals by point rates. Alternate posterior assignment, continuous
rate relocation using the profiled component objective, exact amplitude updates, and
pruning of inactive or duplicate components. Preserve rate ordering to avoid label swaps.

## 5. Smooth frequency trajectories

Parameterize `log(lambda[f,k])` with smooth trajectories only after independent-frequency
recovery is reliable. Tune smoothness against held-out data and report its effect on bias.

## Implementation order

1. Model and IS objective with finite-value safeguards. Implemented.
2. Supplied-atom amplitudes/floors and SAGE recovery tests. Implemented.
3. Independent-frequency profiled decay-rate SAGE. Implemented.
4. Wideband subdivision, component enumeration, and pruning.
5. Cross-frequency smoothing and real-RIR experiments.
