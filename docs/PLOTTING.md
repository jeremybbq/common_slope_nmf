# Plotting result diagnostics

Matplotlib is not part of the numerical package. Fit CLIs write NPZ archives
(and CSV summaries when they already did). Sibling `plot_*.py` scripts load those
archives and write PNG/PDF figures. The package can be imported without the
optional Matplotlib extra.

```text
python -m experiments.synthetic.sweep_decay_detection
python -m experiments.synthetic.plot_sweep_decay_detection --results output/RUN/decay_detection_results.npz
```

`--plot-results` is not a fit-CLI flag. Use the matching plot module and pass
`--results` to a saved NPZ file, or, for room-to-hallway, a timestamped run
directory:

```text
python -m experiments.room_to_hallway.plot_omni --results output/RUN
```

A shared `save_figure` helper lives in `experiments._run_output` for PNG+PDF
file I/O. Plot content belongs in the experiment that owns it; there is no
shared experiment plotting package and no `common_slope_nmf.plotting` module.

Plot families implemented in the experiment scripts include:

- observed, generating, and fitted variance maps over `(R,N)`;
- complete-sweep objectives and energy-`T60` trajectories;
- true-versus-fitted decay pairs, error CDFs, and errors versus true separation;
- unsigned power-dB amplitude RMSE and signed power-dB amplitude bias;
- simplex shares, amplitude distributions, joint amplitudes, and per-RIR parameters;
- initialization-basin arrows and metric heatmaps;
- known-decay amplitude/floor calibration plots; and
- animated scaled fitting error and three-component Wiener-style weights.

Units remain explicit in each public docstring. Amplitude and floor dB use the
power convention `10 log10(value)`. Animation builders return both the figure
and `FuncAnimation`; the caller selects GIF/video encoding and filename.
