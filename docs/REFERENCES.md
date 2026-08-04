# References and provenance

## Source notes supplied for this repository

- *Common-Slope Decay Estimation with IS-NMF'SAGE*: model, IS objective, multiplicative
  updates, SAGE derivation, profiled rates, and smooth rate trajectories.
- *Decay Refinement with Wideband-Dictionary and Gridless Estimation*: division of roles
  between IS-SAGE, wideband interval localization, continuous relocation, pruning, and
  smooth trajectories.

## Original IS-NMF implementation

C. Févotte, N. Bertin, and J.-L. Durrieu's original MATLAB archive for their 2009
IS-NMF paper contains multiplicative-update, SAGE/EM, and inverse-gamma-prior variants:
<https://www.irit.fr/~Cedric.Fevotte/extras/neco09/code.zip>. Treat it as a primary
algorithmic and reproducibility reference. The archive states the required paper citation
but contains no explicit software license, so derive repository code from the published
equations rather than copying the MATLAB source.

## Predecessor

`multislope_linex` provides multi-slope amplitude estimation for fixed decay times from
instantaneous RIR energy using a LINEX loss and alpha continuation. Reuse its numerical
discipline, synthesis conventions, tests, and documentation style; retain the distinction
between its fixed-rate energy-domain objective and this repository's STFT-power IS model.

## Literature to cite before publication

- C. Févotte, N. Bertin, and J.-L. Durrieu, “Nonnegative Matrix Factorization with the
  Itakura-Saito Divergence: With Application to Music Analysis,” *Neural Computation*, 2009.
- The wideband dictionary and gridless damped-mode works identified in the supplied
  refinement note, including Butsenko, Swärd, Jakobsson; and Jälmby, Swärd, Elvander,
  Jakobsson. Verify complete bibliographic details before formal citation.
