# Experiment plots

Plotting lives in sibling `plot_*.py` modules next to each fit CLI, outside the
numerical package. There is no central plotting module. Fit scripts write NPZ/CSV
only. Plot scripts read those archives, write PDF/PNG beside the saved results, and
accept `--show` to display figures.

- `experiments/synthetic/plot_convergence.py`: total IS loss, excess loss, profiled
  loss surface with T60 trajectories. Surface interpolation is display-only.
- `experiments/synthetic/plot_decay_robustness.py`: rate-error scatter, error
  histograms, and decay/amplitude errors versus true pair separation.
- `experiments/synthetic/plot_amplitude_identifiability.py`: amplitude-error
  distributions with unfinished cases marked. The fit module's `load_run`
  reconstructs a saved ordinary-SAGE run.
- `experiments/room_to_hallway/plot_omni.py`: two-slope T60 curves, contribution-weight
  space-time maps, and amplitude-mixture images from a run directory.
- `experiments/room_to_hallway/plot_georg_benchmarks.py`: DecayFitNet/CommonSlopeAnalysis
  overlay on a proposed two-slope T60 result.

See [Experiments](EXPERIMENTS.md) for commands.

Decay times are energy T60 in seconds; rates are inverse seconds. Amplitude ratios
use `10 log10`, since amplitudes represent variance/power. Preserve component ordering
when comparing amplitudes and rates. Numerical convergence flags remain distinct from
statistical recovery. Saved arrays retain outliers even if a display clips them.
