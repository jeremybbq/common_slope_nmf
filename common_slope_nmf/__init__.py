"""Research package for constrained multi-slope IS-NMF decay estimation."""

from .is_objective import gaussian_variance_nll, is_divergence
from .model import (
    exponential_atoms,
    exponential_variance,
    rate_to_t60,
    t60_to_rate,
)
from .preprocess import DecayFit, fit_coarse_decay, head_power, tail_power
from .rir import (
    SRIRDatasetInfo,
    STFTPower,
    global_energy_onset,
    inspect_srir_dataset,
    load_srir_channel,
    resample_rirs,
    rir_stft_power,
)
from .sage import (
    AmplitudeSAGEResult,
    DecaySAGEInit,
    DecaySAGEResult,
    DecaySAGEUpdateDiagnostics,
    amplitude_sage,
    decay_sage,
    init_decay_sage,
    pseudo_decay_sage,
    update_rates_bisection,
    update_rates_newton,
)
from .squarem import (
    AmplitudeSQUAREMResult,
    DecaySQUAREMResult,
    amplitude_squarem,
    decay_squarem,
)
from .synth import (
    DecayData,
    sample_complex_gaussian,
    sample_multislope_data,
    sample_power,
    sample_simplex_amplitudes,
    sample_t60,
)

__all__ = [
    "AmplitudeSAGEResult",
    "AmplitudeSQUAREMResult",
    "DecayData",
    "DecayFit",
    "DecaySAGEInit",
    "DecaySAGEResult",
    "DecaySAGEUpdateDiagnostics",
    "DecaySQUAREMResult",
    "SRIRDatasetInfo",
    "STFTPower",
    "amplitude_sage",
    "amplitude_squarem",
    "decay_sage",
    "decay_squarem",
    "exponential_atoms",
    "exponential_variance",
    "gaussian_variance_nll",
    "fit_coarse_decay",
    "global_energy_onset",
    "head_power",
    "init_decay_sage",
    "inspect_srir_dataset",
    "is_divergence",
    "load_srir_channel",
    "pseudo_decay_sage",
    "rate_to_t60",
    "resample_rirs",
    "rir_stft_power",
    "sample_complex_gaussian",
    "sample_multislope_data",
    "sample_power",
    "sample_simplex_amplitudes",
    "sample_t60",
    "tail_power",
    "t60_to_rate",
    "update_rates_bisection",
    "update_rates_newton",
]
