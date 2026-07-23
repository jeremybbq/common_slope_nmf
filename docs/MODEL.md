# Statistical model

For RIR `r`, STFT frequency bin `f`, and frame `n`, let `X[r,f,n]` be a complex STFT
coefficient and `Y[r,f,n] = |X[r,f,n]|^2` its observed power. The working model is

```text
X[r,f,n] ~ CN(0, V[r,f,n])
V[r,f,n] = b[r,f] + sum_k a[r,f,k] exp(-lambda[f,k] tau[n]).
```

`tau[n]` is elapsed time from the selected decay origin. The rate trajectories
`lambda[f,k] > 0` are shared over RIRs; amplitudes `a[r,f,k] >= 0` and floors
`b[r,f] >= 0` are RIR-dependent. Since each atom equals one at the decay origin, the
amplitudes have an identifiable extrapolated-variance interpretation.

The fitting criterion is the IS divergence

```text
J = sum_{r,f,n} [Y[r,f,n] / V[r,f,n] - log(Y[r,f,n] / V[r,f,n]) - 1].
```

Up to terms independent of the model, this is `sum(log(V) + Y / V)`, the negative
working likelihood for complex Gaussian coefficients. Rates map to energy `T60` through
`T60[f,k] = 6 log(10) / lambda[f,k]`.

## Identifiability

Shared rates gain support from amplitude diversity over RIRs. Closely spaced rates,
short usable tails, weak components, and slow-rate/noise-floor confounding remain hard
cases. Model selection and recovery claims therefore require controlled synthetic tests.
