# Experiments

All runnable scripts live in `experiments/`. Synthetic experiments are organized
around three questions. Each script owns its plotting functions, writes numerical
results to a unique timestamped `output/` directory, and accepts `--plot-results`
to regenerate figures without fitting. All decay times below are energy `T60` in
seconds; amplitudes and floors are variance/power quantities.

## 1. Convergence for one decay pair

```bash
python -m experiments.synthetic_convergence
```

Fix the true pair at `[0.80, 1.40]` s, generate 256 RIR realizations with 256
frames at a 128/24000 s hop, and compare ordinary SAGE with responsibility-weighted
pseudo-SAGE (`rho` and `rho**2`) on exactly the same observations and initialization.
Unit-sum amplitudes follow Dirichlet(0.5); floors have mean -40 dB and standard
deviation 2/3 dB. The default trajectory budget is 1,000 sweeps.

Save raw total IS-loss curves, excess-loss curves, and trajectories over a profiled
21-by-21 loss surface. At each supplied rate pair, amplitudes and the floor are
fit from both continuation and independent starts. Save the selected loss, full
profiled amplitudes/floors, convergence flags, and discrepancies between starts.
The grid covers short T60 0.6–1.4 s and long T60 1.0–1.8 s; trajectory rate bounds
correspond to T60 0.5–3.0 s. The default profile budget is 2,000 sweeps per start.

The archive also retains observations, generating parameters, initial and final
states, trajectories, and solver settings. A finite-budget surface is an approximate
profile, not proof of a global minimum. Read `surface_converged` before interpreting
its geometry. Raster interpolation is display-only; numerical values stay unchanged.
The weighted methods do not guarantee monotone observed IS loss.

## 2. Decay-estimate robustness across sampled pairs

```bash
python -m experiments.synthetic_decay_robustness
```

Draw 100 independent T60 pairs uniformly over 0.5–3.0 s, including nearby rates.
Each pair has 512 RIR realizations, 374 frames at a 128/24000 s hop, Dirichlet(0.5)
unit-sum amplitudes, and -40 dB mean floors with 2/3 dB standard deviation.
Compare `rho` and `rho**2` on matched observations with pooled-log initialization.
Default batching is one pair per fit, so stopping is independent across pairs.
Larger batches share a summed-loss stopping decision; batch size is saved.

Save signed rate-error scatter, T60/amplitude error histograms, T60 error versus
pair separation, and amplitude RMSE/bias versus separation. All cases are retained;
unfinished fits are marked in the T60 separation plot. Numerical output includes
truth, estimates, total IS and rate histories, iteration counts, and separate
loss/decay stopping flags. Reaching a stopping tolerance is not a recovery guarantee.
Pooled amplitude errors share estimated rates within a pair and are not independent
across RIRs. These experiments measure empirical robustness, not universal identifiability.

## 3. Amplitude identifiability for known decay times

```bash
python -m experiments.synthetic_amplitude_identifiability
```

Hold `[0.50, 1.50]` s fixed. Estimate only the two variance amplitudes and a
constant floor for fast-to-slow ratios `[+10, 0, -10, -15, -20]` dB. Use 100
matched circular complex-Gaussian realizations per case, 250 frames at a
256/48000 s hop, total decay amplitude one, and a -50 dB floor.

Ordinary `amplitude_sage` runs for at most 20,000 sweeps per case. The loss stop
uses the case-summed IS objective. One extra diagnostic sweep checks the maximum
amplitude change normalized by each realization's mean observed power, without
changing the saved estimate. `converged` requires both the loss stop and this
stability check. The probe does not extend a loss-based early stop; reduce
`--objective-tol` or increase the budget when diagnostics remain unsatisfactory.

Save observations, coefficients, initialization, estimates, full total IS histories,
solver controls, and convergence diagnostics. CSVs report bias, spread, quantiles,
and fast/slow error correlations. Relative-error plots mark cases that fail the
combined convergence check with `*`; optimizer error must be separated from
statistical uncertainty before making identifiability claims.

## Replot saved results

Use the matching experiment module and its saved NPZ, for example:

```bash
python -m experiments.synthetic_convergence --plot-results output/RUN/loss_convergence_results.npz
python -m experiments.synthetic_decay_robustness --plot-results output/RUN/t60_identifiability_results.npz
python -m experiments.synthetic_amplitude_identifiability --plot-results output/RUN/amplitude_inference_results.npz
```

Historical numerical filenames are retained. The amplitude loader expects the new
ordinary-SAGE format; archived accelerated runs belong to `feat/squarem`.

## Foundational validation

Deterministic package tests retain energy/power conventions, Gaussian synthesis,
exact recovery, statistical fixed-rate checks, independent optimizer comparisons,
nearby rates, short windows, weak components, and noise floors. These checks live
in `tests/`; they no longer have separate publication experiment scripts.

## Archived research

`feat/squarem` preserves the pre-cleanup source, including acceleration code,
its tests, historical synthetic scripts, and their recorded results. It includes
uncommitted source additions present at the start of this cleanup. The initialization
basin sweep and standalone demonstrations are outside the three current questions.
Generated data and outputs remain in their existing locations.

## Georg Götz baselines on the room transition

`experiments/roomToHallway_georg_benchmarks.py` runs two fixed-order baselines
on channel-zero RIRs from all four raw v1.3 Meeting Room to Hallway conditions.
The agreed comparison uses two slopes and band centers at 250, 500, 1000,
2000, 4000, and 8000 Hz. It deliberately follows the baseline's full-RIR
Schroeder-EDC preprocessing, causal fifth-order Butterworth filters with
`sqrt(1.5)` band edges, 0.5% filtered-tail removal, and 100-point DecayFitNet
representation instead of reusing the proposed method's STFT frames.

DecayFitNet first produces independent per-RIR decay times, normalized EDC
amplitudes, and noise estimates. For each band, CommonSlopeAnalysis applies
one-dimensional L1 clustering and a 0.05 s histogram mode to obtain two shared
decay times, then refits every EDC using non-negative dB-domain least squares.
Run the benchmark with an external DecayFitNet checkout:

```bash
python -m experiments.roomToHallway_georg_benchmarks \
  --model-dir ~/Documents/DecayFitNet/model
```

The combined NPZ preserves normalized and absolute EDC coefficients as well as
an explicitly labeled discrete-time RIR-power equivalent. The latter remains
band-filter dependent and must not be treated as numerically identical to an
STFT-bin variance amplitude. Per-band checkpoints make the run recoverable.
DecayFitNet model paths and SHA-256 hashes are recorded; the external weights
and transforms are not copied into this repository.

## Raw room-to-hallway transition experiment

`experiments/roomToHallway_omni.py` operates on the four raw v1.3 Meeting Room to
Hallway SOFA files. It pools the four source/visibility conditions only for the common
decay rates; amplitudes and constant noise floors remain specific to every RIR and
frequency. Channel zero is the ACN omnidirectional response.

At 48 kHz, use a 512-sample Hann window, 256-sample hop, and 512-point FFT without
boundary or end padding. Retain bins 2 through 85 (187.5 through 7968.75 Hz). Of the 280
complete frames in each 1.5 s RIR, discard frames 0 through 9 and the final 20 frames,
then reset the first retained frame to elapsed time zero. This keeps 250 frames, starts
the first retained window at 53.33 ms, and leaves approximately 108 ms of raw samples
after the final retained window, excluding the dataset's published fade-out region.

The default pilot uses 25 evenly spaced listener positions from each of the four
conditions and the FFT bins nearest 250, 500, 1000, 2000, 4000, and 8000 Hz. Compare
K=1,2,3 under equal pooled-log-linear and log-spaced rate starts; K=1 has only the equal
start because both rules coincide. Decay amplitude is divided equally after subtracting
the initialized tail-average floor from the leading-frame power. Fit a per-RIR,
per-frequency constant floor and use `rho` as the pseudo-SAGE surrogate weight. The
weighted method remains experimental, so a finite-result and objective-growth gate is a
numerical diagnostic rather than a validation claim.

Run the pilot before the full stage:

```text
python -m experiments.roomToHallway_omni --stage pilot
python -m experiments.roomToHallway_omni --stage full
```

## Three-coupled-room simulated RIRs

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
python -m experiments.fit_coupled_rooms --stage full --reuse-cache --jobs 8
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
python -m experiments.fit_coupled_rooms --stage full --reuse-cache \
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
python -m experiments.refine_coupled_rooms_unweighted \
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

Record seeds, time axes, decay origins, units, bounds, initialization, stopping rules,
and solver settings. Synthetic scripts save numerical archives for replotting.
Independent Gaussian frames do not reproduce correlations caused by overlapping
STFT windows in real RIRs. Full scientific runs are distinct from small software
smoke checks. Real datasets and external model assets are supplied separately.
