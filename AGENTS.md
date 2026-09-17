# Repository Guidelines

Keep the repository research-first. The source notes in `docs/` define the intended model;
do not describe untested optimization ideas as validated methods.

Use Python with NumPy and SciPy. Add public functions with array shapes, units, and return
values in their docstrings. Keep energy/power, amplitude, rate, and `T60` conventions
explicit. Prefer vectorized operations and stable exponential/logarithmic calculations.

Write deterministic pytest tests with known synthetic ground truth before extending the
estimator. Numerical assertions should use stated tolerances and include difficult cases:
nearby rates, short decay windows, weak components, and noise floors.

Place reusable theory and assumptions in `docs/`, runnable questions in `experiments/`,
and implementation-specific tests in `tests/`. Update `docs/INDEX.md` when a planned module
becomes real.
