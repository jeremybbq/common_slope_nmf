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
For each RIR, the first unit-sum amplitude is Uniform(0, 1) and the second is its
complement; floors have mean -40 dB and standard
deviation 2/3 dB. The default trajectory budget is 1,000 sweeps.

Save log-scale excess-loss curves and trajectories over a profiled
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

`feat/treble-simulation` preserves the three-room Treble simulation experiment,
its ordinary-SAGE continuation script, experiment-specific tests, and the full
protocol and recorded findings in that branch's `docs/EXPERIMENTS.md`.
These simulation scripts are excluded from the paper workflow on `main`.

## Main recorded-RIR experiment: coupled-room transition

The coupled-room transition recordings are the paper's main real-RIR application.

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
