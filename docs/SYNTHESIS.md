# Synthetic decay data

## Fixed-sum component amplitudes

For every RIR and frequency, draw component shares from a symmetric Dirichlet law,

```text
p[r,f,:] ~ Dirichlet(alpha, ..., alpha),
a[r,f,k] = A_total[r,f] p[r,f,k].
```

The shares are strictly positive and sum to one. With the default total amplitude
`A_total = 1`, every RIR-frequency pair has 0 dB total unit-origin decay variance:

```text
sum_k a[r,f,k] = 1.
```

The concentration controls the simplex geometry. `alpha < 1` concentrates samples near
the `K` corners, `alpha = 1` is uniform over the simplex, and `alpha > 1` favors equal
component shares. For two components this is the symmetric Beta distribution. Component
dominance remains exchangeable; `dominant_component` in the generated data records the
realized winning component for coverage checks.

## Decays, floors, and observations

`sample_t60` draws and sorts `K` energy-decay times independently at every frequency from
a uniform range. An optional minimum adjacent separation rejects accidentally
unidentifiable draws. `sample_multislope_data` combines these rates with Dirichlet
amplitudes and Gaussian power-dB floor levels, then evaluates

```text
V[r,f,n] = b[r,f] + sum_k a[r,f,k] exp(-lambda[f,k] tau[n])
X[r,f,n] ~ CN(0, V[r,f,n]),
Y[r,f,n] = abs(X[r,f,n])**2.
```

The returned `DecayData` retains all generating parameters, exact variances, complex
coefficients, powers, and dominant-component labels. Passing one seeded NumPy generator
controls every draw reproducibly.

Minimum decay separation and corner concentration define a benchmark distribution; they
should be reported with results. A separation-constrained simulation is a controlled
identifiability experiment rather than an unconditional sample from the uniform model.
