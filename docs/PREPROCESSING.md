# Preprocessing and SAGE initialization

## Frame summaries

`head_power` averages the first requested frames of `Y` and `tail_power` averages the
last requested frames. The tail mean initializes the constant variance floor. Residual
long-decay energy can bias it upward, so it is an optimization starting point rather than
an unbiased floor estimate for arbitrary observation lengths.

## Pooled coarse decay

`fit_coarse_decay` fits one log-power slope at each frequency. For a coarse single-decay
description,

```text
log Y[r,f,n] = c[r,f] - lambda[f] tau[n] + error[r,f,n].
```

All RIRs use the same selected frame times. Consequently their different intercepts do
not bias a pooled slope: for each RIR,

```text
sum_n (tau[n] - mean(tau)) c[r,f] = 0.
```

The implementation therefore fits the pooled slope directly and discards the intercept.
It evaluates its diagnostic R-squared after removing each RIR's temporal mean. The
instantaneous powers are not floor-subtracted before taking logarithms. Instead, an
optional common mask retains frames whose across-RIR mean power lies a chosen dB margin
above the mean initialized floor, and the tail frames are excluded.

This is deliberately a coarse effective decay for initializing a multislope model, not a
claim that a sum of exponentials is log-linear.

## SAGE starting point

`init_decay_sage` performs the algorithm-specific assembly:

```text
b0[r,f]       = mean over the final N_tail frames of Y[r,f,n],
P0[r,f]       = mean over the first N_head frames of Y[r,f,n],
a0[r,f,k]     = P0[r,f] / K,
lambda0[f,k]  = clipped pooled coarse rate at f.
```

The leading-frame amplitude is not corrected for the initialized floor. Every component
starts at the same coarse rate, clipped to the common intersection of its supplied rate
bounds. The returned `DecaySAGEInit` contains arrays ready for `decay_sage` or
`pseudo_decay_sage`, together with the unclipped `DecayFit` diagnostics.
