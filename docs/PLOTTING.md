# Experiment plots

Plotting functions live with their owning scripts in `experiments/`, outside the
numerical package. There is no central plotting module. Functions accept arrays
or the owning experiment's saved run and return figures or saved paths.

- `synthetic_convergence.py`: total IS loss, excess loss, profiled loss surface
  with T60 trajectories. Surface interpolation is display-only.
- `synthetic_decay_robustness.py`: rate-error scatter, error histograms, and
  decay/amplitude errors versus true pair separation.
- `synthetic_amplitude_identifiability.py`: amplitude-error distributions with
  unfinished cases marked. `load_run` reconstructs a saved ordinary-SAGE run.
- Real-data scripts retain their existing local plotting functions.

Each synthetic CLI accepts `--plot-results PATH` to plot an existing NPZ without
calling an estimator, and `--show` to display figures. PDF and PNG outputs go to
a new timestamped run directory. See [Experiments](EXPERIMENTS.md) for commands.

Decay times are energy T60 in seconds; rates are inverse seconds. Amplitude ratios
use `10 log10`, since amplitudes represent variance/power. Preserve component ordering
when comparing amplitudes and rates. Numerical convergence flags remain distinct from
statistical recovery. Saved arrays retain outliers even if a display clips them.
