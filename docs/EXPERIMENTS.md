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

This experiment supplies the decay atoms so it tests the IS objective and the SAGE
amplitude stage without mixing in rate-search errors.

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

### 4. Known-decay STFT-shaped inference

Use 512 abstract, independent frequency bins and the exact complex-Gaussian variance
model. The array is STFT-shaped but is not obtained by transforming a time-domain
waveform, so the number of abstract bins is independent of the 256-sample frame length.
For a 2 s duration at 24 kHz, a 256-sample frame, and a 128-sample hop, retain the 374
complete unpadded frames. Set frame 0 as the decay origin:

```text
tau[n] = n * 128 / 24000 seconds,  n = 0, ..., 373.
D[n] = exp(-lambda tau[n]),  lambda = 6 log(10) / 1 second.
```

Interpret dB in variance/power units. The decay amplitude is 0 dB, hence `a = 1`.
Generate one coefficient at every bin and frame under each of three conditions:

```text
no floor:       X[f,n] ~ CN(0, a D[n])
-30 dB floor:   X[f,n] ~ CN(0, a D[n] + b),  b = 10^(-30/10) = 0.001.
-60 dB floor:   X[f,n] ~ CN(0, a D[n] + b),  b = 10^(-60/10) = 0.000001.
```

For the no-floor condition, fitting the exact single decay atom has a closed form. The
SAGE gain is one, so one component update gives

```text
a_hat[f] = mean_n(|X[f,n]|^2 / D[n]).
```

Because `|X[f,n]|^2 / (a D[n])` is unit-mean exponential,
`a_hat[f] / a ~ Gamma(shape=N, scale=1/N)`. Thus the estimator is unbiased, its standard
deviation is `a / sqrt(N)`, and an exact chi-square confidence interval is available per
bin. Verify that SAGE agrees with this expression to floating-point tolerance and that
the mean, spread, and 95% interval coverage across the 512 independent bins agree with
the exact sampling law.

For both floor conditions, fit the two fixed atoms `[D, 1]`, estimating both the decay
amplitude and the unknown floor in every bin. Initialize every decay amplitude at `0.25`
and every fitted floor at `0.01`; these are deliberately fixed across conditions rather
than initialized at the generating values. Verify a non-increasing IS objective and
compare empirical parameter spread across bins with the inverse Fisher information

```text
I[i,j] = sum_n D_i[n] D_j[n] / V[n]^2,
```

where `D_0 = D`, `D_1 = 1`, and `V = a D + b`. This comparison is asymptotic and should
be reported as a calibration diagnostic, not an exact finite-sample identity.

Run:

```text
python -m examples.validate_known_decay_stft
```

The script saves every plot as a separately named PNG under a timestamped
`./output/YYYY-MM-DD_HH-MM-SS-ffffff/` run directory: one generated power/estimated-
variance comparison for each floor condition, one convergence plot, one three-condition
estimator-distribution plot, linear and dB binwise decay-amplitude plots, and a dB
noise-floor plot. Only the no-floor distribution receives an exact Gamma-law overlay.
The focused regressions for all three conditions are in
`tests/test_known_decay_stft.py`.

### 5. Independent-frequency profiled decay recovery

Use four frequencies with one exact exponential component each, three RIRs with varied
unit-origin amplitudes, 151 frames spaced by `0.01 s`, no floor, and energy
`T60 = [0.35, 0.55, 0.80, 1.10] s`. Initialize every rate at `T60 = 0.65 s` and every
amplitude at `2`. Bound the profile search to the rates corresponding to
`T60 in [0.2, 1.5] s`. With deterministic `Y = V`, verify recovery of every rate,
amplitude, and fitted variance to floating-point tolerance and a non-increasing IS
objective.

Also verify that a known two-decay-plus-floor model is a numerical fixed point of every
complete SAGE step. This covers sequential decay components and the constant floor update
without describing the characteristically slow EM-like convergence near weak-component
boundaries as fast recovery.

Run:

```text
python -m examples.validate_decay_sage
```

Focused assertions are in `tests/test_decay_sage.py`. Stochastic joint-rate recovery,
initialization basins, nearby-rate separation, and model selection remain future
experiments.

### 6. Seeded stochastic two-slope recovery

Use the exact circular complex-Gaussian variance model with `R = 512`, `F = 1`, two
rates shared by all RIRs, and the same 374 complete frames used in experiment 4. With
seed `20260724`, draw two energy `T60` values once from `Uniform(0.5, 3.0) s` and sort
them by increasing `T60`.

Draw each RIR's two unit-origin variance amplitudes jointly in power dB. The marginal
means are `(-10, -10) dB` and marginal standard deviations are `10/3 dB`. Thus each
marginal lies in `(-20, 0) dB` with approximately 99.7% probability before finite-sample
variation. Drawing in dB, followed by `a = 10**(a_db/10)`, avoids negative linear
variance amplitudes. Independently draw each RIR's time-invariant floor as
`Normal(-40, (2/3)^2) dB`, placing it within two dB of its mean with approximately 99.7%
probability.

The first run initializes energy `T60` at `[0.75, 2.75] s`, splits the mean power in the
first eight frames equally between the two decay components, initializes every floor at
`-35 dB`, and uses safeguarded Newton rate updates. It deliberately does not initialize
at the generating parameters. After 1,000 SAGE sweeps, the measured result is:

```text
true T60:                  [1.041716, 2.680267] s
estimated T60:             [0.760496, 2.638224] s
absolute T60 error:        [0.281220, 0.042043] s
amplitude RMSE:             [3.221917, 0.578090] dB
noise-floor RMSE:           0.537152 dB
fitted-variance log RMSE:   0.431673 dB
outer stopping condition:   not reached
observed IS objective:      non-increasing
```

This first run used amplitude correlation `-0.95`. It is a reproducible
incomplete-convergence result, not evidence of successful recovery of both rates. In
particular, the short component remains substantially biased after the declared sweep
budget even though the fitted total variance is visually close to the generating
variance.

Repeat with amplitude correlation `-0.8`, keeping the same seed, realized shared rates,
noise distribution, frame setup, bounds, and initialization, but increasing the sweep cap
to 2,000. The realized amplitude correlation is `-0.807243`. The result is:

```text
true T60:                  [1.041716, 2.680267] s
estimated T60:             [0.774465, 2.638962] s
absolute T60 error:        [0.267251, 0.041305] s
amplitude RMSE:             [3.738781, 0.555650] dB
noise-floor RMSE:           0.535067 dB
fitted-variance log RMSE:   0.424126 dB
outer stopping condition:   not reached
observed IS objective:      non-increasing
```

The shorter-slope estimate improves only modestly and is still substantially biased.
Because both correlation and sweep budget changed between runs, their individual effects
cannot be inferred from this comparison. Initialization sensitivity and controlled
one-factor ablations require separate experiments rather than post-hoc tuning.

As an initialization stress test, repeat the `-0.8` condition with both components
initialized at `T60 = 2 s`. The two atoms and their initial per-RIR amplitude shares are
then identical. Sequential SAGE nevertheless breaks the symmetry because the second
component is conditioned on the total variance containing the first component's updated
value. After 2,000 sweeps:

```text
true T60:                  [1.041716, 2.680267] s
estimated T60:             [1.527472, 2.841984] s
absolute T60 error:        [0.485756, 0.161717] s
amplitude RMSE:             [3.605376, 2.123692] dB
noise-floor RMSE:           0.658215 dB
fitted-variance log RMSE:   0.476596 dB
final observed IS objective: 110286.091
outer stopping condition:   not reached
observed IS objective:      non-increasing
```

Over the final 500 sweeps, the two estimates move by `[-0.071779, -0.030685] s`,
toward their generating values, while the final per-sweep objective decrease remains
`0.171787`. This start is therefore still evolving substantially, but at the 2,000-sweep
checkpoint it is worse than the separated `[0.75, 2.75] s` start, whose objective is
`110095.774`. The original separated initialization was a manually chosen bracket-spanning
condition, not a data-derived initialization rule.

Run:

```text
python -m examples.validate_multislope_sage
```

The current script defaults to correlation `-0.8`, equal `2 s` initialization, and 2,000
sweeps. It saves separate spatial observed-power and fitted-variance maps over `(t,r)`, an
observed/true/estimated variance comparison, marginal generating-versus-estimated
amplitude distributions, a joint-amplitude comparison, per-RIR parameters, and
raw-objective and `T60` trajectories. It also saves animated `(r,n)` maps of
`log(epsilon + 1) = log(Y/V)` and a ternary color mixture of the short-decay, long-decay,
and noise weights. The marginal distribution plot shows the known
`Normal(-10, (10/3)^2) dB` generating law. The correlation is included in every filename.
The numerical objective and rate histories are retained in a compressed NumPy archive so
new diagnostic views do not require rerunning the estimator. The current multislope run
also requests complete-sweep scaled-error, all-component Wiener weights, and sequential
profile-moment diagnostics at sweep 1 and every 100 sweeps thereafter; these fields are
written into the same archive. All example outputs are grouped under a timestamped
`./output/YYYY-MM-DD_HH-MM-SS-ffffff/` run directory so separate runs cannot mix. The
deterministic forward-model checks are in `tests/test_multislope_experiment.py`.

### 7. Component-strength-weighted pseudo-SAGE comparison

Reuse exactly the seeded dataset, initialization, rate bounds, stopping controls, plots,
and saved intermediate diagnostics from experiment 6. Change only the decay-component
M-step: freeze `w = rho**p` from the pre-update model, then use it in the profiled
amplitude, weighted time sums, rate gradient, and Hessian. Keep the noise-floor update
unweighted. The default comparison is `p = 1`; use `--component-weight-power 2` for the
`rho**2` version and `0` as a unit-weight implementation check.

Run:

```text
python -m examples.validate_weighted_multislope_sage
python -m examples.validate_weighted_multislope_sage --component-weight-power 2
```

The script writes the same plot family and animations as experiment 6 to a new
timestamped directory. Its compressed diagnostics additionally save the selected power,
the weighted moment `M`, and its denominator `sum_n w`. The total observed IS objective
and `T60` trajectories remain the primary checks. Since this weighting is not an exact
SAGE auxiliary function, objective monotonicity, convergence speed, and recovery quality
are experimental outcomes; no improvement is claimed before the full controlled runs are
compared.

### 8. Fixed-data decay-initialization basin

Hold the exact dataset from experiment 6 fixed and vary only the two initial decay times.
This isolates optimization-basin and sequential component-order sensitivity from sampling
variation. At every start, retain the existing amplitude initialization without any scale
or allocation ablation:

```text
P[r,f]       = mean over the first 8 frames of Y[r,f,n],
a0[r,f,1]    = P[r,f] / 2,
a0[r,f,2]    = P[r,f] / 2,
b0[r,f]      = -35 dB in variance units.
```

The default script evaluates all 36 ordered pairs from the six-point grid
`T60 = [0.5, 1, 1.5, 2, 2.5, 3] s`. Ordered pairs intentionally retain both `(x,y)` and
`(y,x)`, since
the sequential component schedule can make those trajectories differ even though the
final physical model is permutation-invariant. It uses `rho**2` pseudo-SAGE, safeguarded
Newton updates, a 500-sweep cap, and the same stopping tolerance as experiment 7. All are
command-line controls.

Run:

```text
python -m examples.sweep_multislope_decay_initialization
```

The timestamped output contains separate basin-arrow, final-error, convergence-sweep,
objective-excess, and convergence-status plots, plus CSV summaries and padded NumPy
objective/decay trajectories for every initialization. This is an empirical initialization
landscape, not a proof of convexity. Statistical robustness across newly sampled datasets
is a separate experiment.

### 9. Package synthesis-to-fit workflow

`examples/demo_synth_init_fit.py` exercises the reusable package path without changing
the fixed historical datasets above. It samples separated decay times, unit-sum
Dirichlet amplitudes, Gaussian power-dB floors, and exact complex-Gaussian observations;
constructs a pooled-log SAGE starting point; and fits with `rho**2` pseudo-SAGE.

Run:

```text
python -m examples.demo_synth_init_fit
```

The example saves its simplex-share histogram, observed/exact/fitted variance map,
decay trajectories, objective, and numerical initialization diagnostics in one
timestamped directory. It demonstrates API composition; recovery claims still belong to
predeclared controlled experiments and tests.

### 10. Independent-frequency decay-detection sweep

Test stochastic two-slope detection on 100 independent frequency bins. Frequency index
is an arbitrary experiment label in this experiment: there is no imposed ordering,
cross-frequency smoothness, or shared decay time between bins. For each bin, independently
draw and sort two energy-decay times from `Uniform(0.5, 3.0) s`. For every `(R,F)` location,
draw the two unit-origin variance amplitudes from a symmetric
`Dirichlet(alpha = 1/2)` distribution, so they are positive, sum to one (0 dB in power),
and favor regions where one component is locally strong. Draw the time-invariant floor
independently as `Normal(-40, (2/3)^2) dB`, and generate exact circular
complex-Gaussian observations with `R = 512` and the usual 374 frames.

Initialize each frequency from the pooled log-power decay fit, split its leading-frame
power equally between the two slopes, and initialize its floor from the final eight
frames. Fit with `rho**2` pseudo-SAGE and safeguarded Newton rate updates. Do not enforce a
minimum true slope separation: error versus realized separation is a primary diagnostic.

Run:

```text
python -m examples.sweep_decay_detection
```

The default outer stopping tolerance is `1e-6`, matching the decay-estimator API default.
This is a pragmatic choice rather than a validated universal convergence certificate. The
script processes bins in configurable batches to bound memory use and records batch
membership because the relative stopping rule is applied to the batch-summed observed
objective. Use `--batch-size 1` to make the gate strictly per-frequency. It saves a CSV
row per frequency, a compressed numerical archive, true-versus-estimated and pair-plane
plots, error versus true separation, an empirical maximum-error CDF, error-evolution
maps, batch objective curves, and one observed/exact/fitted variance map. The experiment
must be run before any recovery threshold is described as validated.

### 11. Three-coupled-room simulated RIRs

Fit the public *Dataset of simulated room impulse responses in three coupled rooms*,
Zenodo record `13338346`, using its omnidirectional channel. The checksum-verified
`srirs.mat` file contains 838 receivers, 9 channels, and 128001 samples at 32 kHz. The
responses already share a global onset at sample zero: a pooled-energy onset threshold
of -40 dB returns zero, while per-RIR first-energy locations span only 0--0.31 ms. Apply
one common decay origin 50 ms later without shifting individual RIRs.

Resample the remaining responses to 24 kHz and compute complex Hann STFT coefficients
with a 256-sample frame, 128-sample hop, 384-point FFT, no boundary extension, and no end
padding. The FFT zero-padding gives the exact 62.5 Hz grid. Retain the 128 bins from
62.5 through 8000 Hz and the 739 complete frames from 0 through 3.936 s elapsed time.
Squared magnitudes are clipped only at a relative floating-point floor to satisfy the
strictly positive IS observation convention; no physical noise floor is included or
estimated.

Use `K = 3`, pseudo-SAGE weights `w = rho`, safeguarded Newton rate updates, energy-
`T60` bounds `[0.2, 6] s`, an outer relative tolerance of `1e-6`, and a 2000-sweep cap.
Initialize every bin at `[0.73, 1.43, 3.48] s` and split the mean first-eight-frame power
equally across components. This triplet comes from the LINEX repository's comparison-
script default; it is an initialization, not a published broadband ground truth. Stop
every frequency independently so one difficult bin does not control the others.

First gate the run with 100 evenly spaced receivers at 250, 500, 1000, and 2000 Hz. The
pilot converged in 32--105 sweeps with fitted triplets `[0.550, 1.709, 4.053]`,
`[0.498, 1.722, 4.909]`, `[0.533, 1.469, 3.465]`, and
`[0.643, 1.690, 3.765] s`, respectively. The full run is:

```text
python -m examples.fit_coupled_rooms --stage full --reuse-cache --jobs 8
```

The full 838-receiver fit converged in 126 of 128 bins. The 6062.5 and 6187.5 Hz bins
reached the 2000-sweep cap. Across all bins the final/initial observed IS-objective ratio
ranged from 0.131 to 0.487 with median 0.214. Eleven bins had at least one objective
increase, with maximum relative single-sweep increase `2.55e-5`, consistent with the
fact that the weighted update is pseudo-SAGE rather than a monotone auxiliary-function
method.

The fitted frequency regions have median energy-`T60` triplets:

| Frequency interval | Bins | Median fitted `T60` (s) |
| --- | ---: | --- |
| 62.5--750 Hz | 12 | `[0.512, 1.691, 4.213]` |
| 812.5--1375 Hz | 10 | `[0.571, 1.491, 3.475]` |
| 1437.5--3000 Hz | 26 | `[0.703, 1.698, 3.765]` |
| 3312.5--5937.5 Hz | 43 | `[0.536, 1.254, 2.428]` |

At 6375 Hz and at every bin from 6437.5 through 8000 Hz, the longest component reaches
the 6 s upper bound. Its median unit-origin share is only `3.9e-11` to `2.9e-10`, so
these are inactive-component/non-identifiability flags, not evidence of a physical 6 s
high-frequency decay. Component-count selection or pruning is required before making a
three-slope claim in that range.

To test initialization sensitivity, rerun the highest five bins with one pooled
log-power decay fit per frequency repeated across all three components. Keep the equal
amplitude split and every other setting unchanged:

```text
python -m examples.fit_coupled_rooms --stage full --reuse-cache \
    --highest-bins 5 --initialization equal-log-linear --jobs 5
```

All five equal-start fits converged. They required 80--89 sweeps rather than 48--56 for
the distinct start and reached a different active-component basin:

| Frequency (Hz) | Equal initial `T60` (s) | Distinct-start final (s) | Equal-start final (s) | Equal-start IS excess |
| ---: | ---: | --- | --- | ---: |
| 7750.0 | 1.4290 | `[0.633, 1.018, 6.000]` | `[0.825, 1.078, 6.000]` | 0.292% |
| 7812.5 | 1.3964 | `[0.638, 1.013, 6.000]` | `[0.824, 1.069, 6.000]` | 0.221% |
| 7875.0 | 1.3553 | `[0.644, 1.011, 6.000]` | `[0.817, 1.059, 6.000]` | 0.187% |
| 7937.5 | 1.3872 | `[0.650, 1.017, 6.000]` | `[0.817, 1.068, 6.000]` | 0.226% |
| 8000.0 | 1.4498 | `[0.651, 1.022, 6.000]` | `[0.829, 1.086, 6.000]` | 0.315% |

The equal start therefore does not remove the inactive third component, whose median
unit-origin share remains about `6e-11`. Its final observed IS objective is consistently
but only 0.19--0.32% above the distinct-start result. This is evidence of initialization-
basin sensitivity, not proof that either solution is globally optimal.

Refine both weighted solutions with ordinary SAGE (`w = 1`, equivalently pseudo-SAGE
`p = 0`), using the saved fitted rates and amplitudes together as warm starts:

```text
python -m examples.refine_coupled_rooms_unweighted \
    output/2026-08-24_17-53-05-079735/full_results.npz \
    output/2026-08-25_09-13-41-528023/full_results.npz \
    --highest-bins 5 --jobs 5
```

All ten refinements converged. Ordinary SAGE reduced each warm-start objective by only
0.041--0.052%, but it did not merge the two solutions. The distinct branch finished at
`[0.633--0.651, 1.003--1.017, 6.000] s`; the equal branch finished at
`[0.818--0.829, 1.046--1.078, 6.000] s`. The equal branch remained 0.181--0.313% above
the distinct branch in final observed IS objective. Its median unit-origin shares were
approximately `[0.934, 0.066, 6.4e-11]`, versus `[0.881, 0.119, 3.9e-11]` for the
distinct branch. Thus removing the experimental `rho` weighting locally refines both
fits but does not erase the basin dependence inherited from the weighted warm starts.

The companion `Common_Slope_Analysis_Results.zip` contains frequency-dependent published
octave-band results, not one broadband triplet:

| Band (Hz) | Published common decay times (s) |
| ---: | --- |
| 63 | `[1.425, 1.675, 2.025]` |
| 125 | `[0.725, 1.375, 3.425]` |
| 250 | `[0.775, 1.625, 3.825]` |
| 500 | `[0.775, 1.525, 3.925]` |
| 1000 | `[0.725, 1.625, 3.725]` |
| 2000 | `[0.675, 1.575, 3.325]` |
| 4000 | `[0.825, 1.475, 2.175]` |
| 8000 | `[0.525, 0.925, 1.225]` |

Those values are plotted as comparison markers, not treated as ground truth for the
direct-bin STFT fit. The published analysis uses octave bands and a different estimation
pipeline, so disagreement does not isolate an optimizer error.

### 12. Fixed-data SQUAREM convergence comparison

Reuse the exact observations, equal `[2, 2] s` initialization, rate bounds, amplitude and
floor initialization, safeguarded Newton rate solver, and `1e-10` outer objective tolerance
from experiments 6 and 7. Compare ordinary SAGE, pseudo-SAGE with `rho` and `rho**2`
weights, and safeguarded SQUAREM applied to the complete ordinary-SAGE map. Give every
method at most 2,000 complete ordinary-SAGE sweep evaluations. Count both construction
sweeps and every stabilization sweep against SQUAREM's budget; plotting SQUAREM cycle
number as if it were one SAGE sweep would understate its work.

Run from scratch with:

```text
python -m examples.compare_multislope_convergence
```

The script can instead accept repeated `--reference-diagnostics` arguments to reuse the
saved deterministic ordinary and weighted histories while running only SQUAREM. The
controlled run on 2026-09-06 measured:

| Method | Sweep evaluations | Final observed IS | Final sorted `T60` (s) |
| --- | ---: | ---: | --- |
| Ordinary SAGE | 2,000 | 110286.091 | `[1.527472, 2.841984]` |
| Pseudo-SAGE, `rho` | 2,000 | 109934.741 | `[1.025453, 2.687106]` |
| Pseudo-SAGE, `rho**2` | 258 | 109977.015 | `[1.013123, 2.682803]` |
| SQUAREM on ordinary SAGE | 2,000 | 109910.115 | `[1.073279, 2.701824]` |

SQUAREM accepted 636 extrapolated states and rejected 29 through its feasibility and
original-objective safeguards. Its retained observed objective was non-increasing. It
reached the ordinary-SAGE 2,000-sweep endpoint by sweep evaluation 283, the `rho**2`
endpoint by evaluation 616, and the `rho` endpoint by evaluation 883. The weighted
methods descended substantially faster during the first few hundred sweeps, while
SQUAREM continued below their recorded endpoints later in the budget.

The generating sorted energy-decay times were `[1.041716, 2.680267] s`. The lowest
in-sample observed objective therefore did not give the smallest error in both decay
times: sampling variation and optimization target must be kept distinct. This is one
fixed stochastic dataset and one initialization, so it supports an implementation-level
comparison only. Runtime conclusions require fresh timed runs of every method, and
recovery conclusions require multiple datasets, slope separations, and initializations.

The timestamped output contains the raw-loss and best-retained-loss-gap curves, a CSV
summary, and un-interpolated numerical histories. SQUAREM's objective archive includes
the cumulative base-sweep evaluation index for every retained state.

### 13. SQUAREM continuation from weighted pseudo-SAGE

Repeat the experiment-7 `rho` pseudo-SAGE fit for 2,000 sweeps so its complete final
state is available, including the shared rates, all 512-by-2 amplitudes, and all 512
floors. Use that complete state—not only its two decay times—to initialize safeguarded
SQUAREM on the ordinary-SAGE fixed-point map. Give SQUAREM a fresh 2,000 base-sweep
evaluation budget with objective tolerance `1e-10` and fixed-point tolerance `1e-6`.

Run:

```text
python -m examples.warm_start_squarem_from_pseudo_sage
```

The deterministic rerun exactly reproduced the saved pseudo-SAGE endpoint. The measured
two-stage result on 2026-09-06 was:

| Stage | Additional sweeps | Final observed IS | Final sorted `T60` (s) | Converged |
| --- | ---: | ---: | --- | --- |
| Pseudo-SAGE, `rho` | 2,000 | 109934.741 | `[1.025453, 2.687106]` | No |
| SQUAREM continuation | 2,000 | 109908.855 | `[1.056690, 2.696954]` | No |

SQUAREM reduced the pseudo-SAGE endpoint by `25.886`, or `0.02355%`, and retained a
non-increasing observed objective. It accepted 663 extrapolations and rejected 3. It
passed the cold-start SQUAREM 2,000-evaluation endpoint (`109910.115`) after 398 additional
evaluations and finished `1.260` objective units below it.

The continuation still exhausted its budget. Its final dimensionless fixed-point
residual was `1.22e-4`, above the requested `1e-6`, so the endpoint is not declared
converged. Relative to the generating `[1.041716, 2.680267] s`, the short-decay absolute
error improved slightly from `0.01626` to `0.01497 s`, while the long-decay error increased
from `0.00684` to `0.01669 s`. This again shows that a lower in-sample stochastic loss
need not minimize realized parameter error.

The timestamped output saves both complete final parameter states, objective and rate
histories, SQUAREM fixed-point residuals, a CSV summary, and a two-stage convergence plot.
This single continuation demonstrates that the weighted pseudo-SAGE endpoint is not an
ordinary-SAGE fixed point; it does not establish a generally superior optimizer sequence.

## Future stochastic joint-rate recovery

Generate complex Gaussian STFT coefficients from unknown exponential rates. Sweep rate
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
Every executable example that writes artifacts creates one timestamped directory under
`output/YYYY-MM-DD_HH-MM-SS-ffffff/`; all files from that invocation remain within that
directory.
