"""Research package for constrained multi-slope IS-NMF decay estimation."""

from .loss import is_divergence
from .georg_baselines import (
    CommonSlopeEDCFit,
    DecayFitNetEstimate,
    ExternalDecayFitNet,
    band_energy_decay_curves,
    determine_common_decay_times,
    edc_to_equivalent_rir_power_amplitudes,
    fit_common_slope_edcs,
)
from .model import (
    exponential_features,
    exponential_variance,
    rate_to_t60,
    t60_to_rate,
)
from .preprocess import fit_linear_decay, head_power, tail_power
from .rir import (
    STFTPower,
    global_energy_onset,
    rir_stft_power,
    select_stft_frames,
)
from .sage import (
    AmplitudeFit,
    DecayFit,
    fit_amplitudes,
    fit_decay,
    init_decay,
    update_rates_newton,
)
from .synth import (
    SynthData,
    sample_complex_gaussian,
    sample_stft_power,
    sample_power,
    sample_simplex_amplitudes,
    sample_t60,
)

__all__ = [
    "AmplitudeFit",
    "CommonSlopeEDCFit",
    "SynthData",
    "DecayFit",
    "DecayFitNetEstimate",
    "ExternalDecayFitNet",
    "STFTPower",
    "band_energy_decay_curves",
    "determine_common_decay_times",
    "edc_to_equivalent_rir_power_amplitudes",
    "exponential_features",
    "exponential_variance",
    "fit_amplitudes",
    "fit_decay",
    "fit_linear_decay",
    "fit_common_slope_edcs",
    "global_energy_onset",
    "head_power",
    "init_decay",
    "is_divergence",
    "rate_to_t60",
    "rir_stft_power",
    "select_stft_frames",
    "sample_complex_gaussian",
    "sample_stft_power",
    "sample_power",
    "sample_simplex_amplitudes",
    "sample_t60",
    "tail_power",
    "t60_to_rate",
    "update_rates_newton",
]
