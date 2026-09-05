# Plotting result diagnostics

Reusable result visualization lives in `common_slope_nmf.plotting`. The module
is imported explicitly because Matplotlib is an optional dependency:

```python
from common_slope_nmf import plotting
```

The plotting functions accept numerical arrays rather than objects defined by
an example script. They return Matplotlib figures without writing files;
`save_figure` handles output paths and optional figure closing. This separates
result visualization from synthesis and optimization, and allows saved NPZ
results to be replotted without rerunning SAGE.

Implemented plot families include:

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
