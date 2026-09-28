# Common-slope joint decay and amplitude estimation using parameterized nonnegative matrix factorization

Estimate shared room-decay times, and how strong each decay is in every recording, from room impulse responses (RIRs).

Several exponential decays share one set of rates across a set of RIRs. Each recording keeps its own amplitudes and noise floor. The fit uses an Itakura–Saito likelihood (the working complex-Gaussian variance model) and a SAGE optimizer.

This follows `multislope_linex`, which estimated amplitudes for decay times that were already fixed. Here the shared, frequency-dependent rates and the non-negative amplitudes are estimated together.

For RIR `r`, frequency `f`, and time frame `n`, the model variance is

```text
V[r, f, n] = b[r, f] + sum_k a[r, f, k] * exp(-lambda[f, k] * tau[n])
```

`lambda` is shared. Amplitudes `a` and floors `b` stay non-negative and can differ by recording.

## Install

Python 3.10 or newer.

```bash
python -m pip install common-slope-nmf
```

From a checkout, install the package with the experiment and test extras:

```bash
python -m pip install -e '.[experiments,test]'
python -m pytest
```

NumPy and SciPy are required. The `experiments` extra adds h5py and Matplotlib for dataset readers and plots. The `decayfitnet` extra is only for the external DecayFitNet benchmark.

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/). Register a pending publisher for project `common-slope-nmf`, owner `jeremybbq`, repository `common_slope_nmf`, workflow `publish.yml`, and environment `pypi`, and require a manual approval on that environment. Pushing a `v*` tag, such as `v0.1.0`, runs the tests, builds the sdist and wheel, and uploads them.

## Quick start

```python
import numpy as np
from common_slope_nmf import fit_decay, init_decay, sample_stft_power, t60_to_rate

times = np.arange(256) * 128 / 24_000  # 128-sample hop at 24 kHz
data = sample_stft_power(
    times,
    n_rirs=64,
    n_frequencies=1,
    n_components=2,
    t60_range_s=(0.5, 3.0),
    amplitude_concentration=0.25,
    noise_mean_db=-40.0,
    noise_std_db=2 / 3,
    min_t60_separation_s=0.4,
    rng=np.random.default_rng(0),
)

rate, amplitude, floor = init_decay(data.observed_power, data.frame_time_s)
# A longer T60 is a smaller rate, so the bound pair is (slow, fast).
bounds = (float(t60_to_rate(3.0)), float(t60_to_rate(0.5)))
k = data.rates_per_s.shape[-1]
result = fit_decay(
    data.observed_power,
    data.frame_time_s,
    np.clip(rate, *bounds)[:, None].repeat(k, axis=1),
    rate_bounds_per_s=bounds,
    initial_amplitudes=(amplitude / k)[:, :, None].repeat(k, axis=2),
    initial_noise_floor=floor,
)
```

`result.rates_per_s` is `(F, K)`, `result.amplitudes` is `(R, F, K)`, and `result.noise_floor` is `(R, F)`. `rate_to_t60` converts rates back to T60 in seconds.

For measured RIRs that already start at a common onset, use `rir_stft_power` and then the same fit. SOFA and MATLAB SRIR readers are in `experiments.datasets`.

## Experiments

Each fit writes one timestamped folder, `output/YYYY-MM-DD_HH-MM-SS-ffffff/`. Plot scripts read that folder; they do not refit.

Synthetic checks:

```bash
python -m experiments.synthetic.convergence
python -m experiments.synthetic.decay_robustness
python -m experiments.synthetic.amplitude_identifiability
python -m experiments.synthetic.plot_convergence --results output/RUN/loss_convergence_results.npz
```

These cover loss paths for one two-slope case, decay estimates across many T60 pairs, and amplitude recovery when the rates are known. The other plot scripts follow the same `--results` pattern.

Recorded coupled-room RIRs (Meeting Room to Hallway, v1.3). Start with the pilot, then the full fit:

```bash
python -m experiments.room_to_hallway.omni --stage pilot
python -m experiments.room_to_hallway.omni --stage full
python -m experiments.room_to_hallway.plot_omni --results output/RUN
```

`experiments.room_to_hallway.georg_benchmarks` compares those fits with DecayFitNet and CommonSlopeAnalysis. Both stay in an external checkout; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Layout

```text
common_slope_nmf/    model, loss, synthesis, STFT power, and SAGE
experiments/         fit scripts, plot scripts, and dataset readers
tests/package/       package tests
tests/experiments/   experiment and reader tests
```

## Status

Synthesis, initialization, fixed-rate amplitude SAGE, and per-frequency decay SAGE are implemented, including the experimental contribution-weighted (CW-SAGE) updates. A fit returns the IS-loss history, the rate history, and the final parameters. Weighted updates record that loss but do not guarantee it decreases every step.

Still planned: a general checkpoint API, wideband localization, automatic component selection, and smooth decay trajectories across frequency.

Older experiments are kept on other branches: SQUAREM on `feat/squarem`, and the three-room Treble simulation on `feat/treble-simulation`.

## License

MIT. Notices for adapted third-party code are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
