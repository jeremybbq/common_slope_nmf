# Common-slope joint estimation of decay and amplitude using parameterized nonnegative matrix factorization

Estimate shared frequency dependent decay rates (or reverberation time, RT), and the amplitudes of the decays in each recording of room impulse responses (RIRs).

![A fixed-frequency slice of the stacked RIR spectrograms forms a space–time power matrix, which is factored into per-position amplitudes and exponential decays shared by every position](https://raw.githubusercontent.com/jeremybbq/common_slope_nmf/main/assets/nmf-overview.png)

At one frequency, the powers of all RIRs form a space–time matrix (blue frame). It is factored into per-position amplitudes and a few exponential decays shared by every position; the right panel shows one RIR (green frame) with its two fitted decays.

The exponential decay fit uses an Itakura–Saito divergence loss (the corresponding negative log-likelihood from the complex-Gaussian variance model) and a SAGE optimizer.

For RIRs indexed with $r$, frequency $f$, and time frame $n$ at elapsed time $\tau_n$, the model variance is

```math
V_{rfn} = b_{rf} + \sum_{k=1}^{K} a_{rfk} \, e^{-\lambda_{fk} \tau_n}
```

The rates $\lambda_{fk}$ are shared. The amplitudes $a_{rfk}$ and floors $b_{rf}$ stay non-negative and can differ by recording.

## Install

Python 3.10 or newer.

```bash
pip install common-slope-nmf
```

The experiments from the paper are not in the standalone PyPI package. To run them, clone the repository and set it up with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra experiments --extra test
uv run pytest
```

`uv sync` creates `.venv` and installs the package in editable mode.

NumPy and SciPy are required.

## Quick start

The fit takes STFT power of shape `(R, F, N)` (RIRs, frequencies, frames) and the elapsed time of each frame in seconds. Frame zero is the decay origin, so the RIRs must already start at a common onset.

From waveforms, with `rirs` a real array of shape `(R, L)`:

```python
from common_slope_nmf import rir_stft_power

stft = rir_stft_power(
    rirs,
    sample_rate_hz,
    frame_size_samples=512,
    hop_size_samples=256,
    fft_size_samples=512,
    minimum_frequency_hz=125.0,
    maximum_frequency_hz=8_000.0,
)
observed_power, frame_time_s = stft.observed_power, stft.frame_time_s
```

The values shown are the defaults. The window is Hann, sizes are in samples, and `stft.frequencies_hz` holds the bins kept between the two frequency limits.

Then estimate `n_components` shared decays per frequency:

```python
import numpy as np
from common_slope_nmf import fit_decay, init_decay, rate_to_t60

n_components = 2

# One pooled decay rate per frequency, a few-frame estimate of the total amplitude, and the noise floor.
rate, amplitude, floor = init_decay(observed_power, frame_time_s)
result = fit_decay(
    observed_power,
    frame_time_s,
    initial_rates_per_s=np.repeat(rate[:, None], n_components, axis=1),
    initial_amplitudes=np.repeat(amplitude[:, :, None] / n_components, n_components, axis=2),
    initial_noise_floor=floor,
)
t60_s = rate_to_t60(result.rates_per_s)
```

`result.rates_per_s` is of size `(F, K)`, `result.amplitudes` is `(R, F, K)`, and `result.noise_floor` is `(R, F)`.

## Experiments

Run all experiments from the repository root. The `plot_*` scripts draw figures from those saved results. For example,

```bash
uv run python -m experiments.synthetic.convergence
uv run python -m experiments.synthetic.plot_convergence --results output/[PATH]/loss_convergence_results.npz
```

Measured coupled-room RIRs ([Meeting Room to Hallway, v1.3](https://doi.org/10.5281/zenodo.6375520)), read from `--dataset-dir`:

```bash
uv run python -m experiments.room_to_hallway.sage_fit --dataset-dir PATH
uv run python -m experiments.room_to_hallway.plot_sage_fit --results output/RUN
```

`experiments.room_to_hallway.decayfitnet_benchmarks` compares these fits with DecayFitNet and CommonSlopeAnalysis. It needs `uv sync --extra decayfitnet` and an external DecayFitNet checkout; see [THIRD_PARTY_NOTICES.md](https://github.com/jeremybbq/common_slope_nmf/blob/main/THIRD_PARTY_NOTICES.md).

## License

MIT. Notices for adapted third-party code are in [THIRD_PARTY_NOTICES.md](https://github.com/jeremybbq/common_slope_nmf/blob/main/THIRD_PARTY_NOTICES.md).
