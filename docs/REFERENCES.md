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

## Fixed-point acceleration

R. Varadhan and C. Roland, “Simple and Globally Convergent Methods for Accelerating the
Convergence of Any EM Algorithm,” *Scandinavian Journal of Statistics*, vol. 35, no. 2,
pp. 335--353, 2008. <https://doi.org/10.1111/j.1467-9469.2007.00585.x>.

The SQUAREM functions preserved on `feat/squarem` use the paper's first-order S3 squared extrapolation with
ordinary SAGE sweeps as the fixed-point map. Feasibility checks, a stabilizing sweep, and
strict original-IS-objective acceptance retain an ordinary two-sweep SAGE fallback.

## Predecessor

`multislope_linex` provides multi-slope amplitude estimation for fixed decay times from
instantaneous RIR energy using a LINEX loss and alpha continuation. Reuse its numerical
discipline, synthesis conventions, tests, and documentation style; retain the distinction
between its fixed-rate energy-domain objective and this repository's STFT-power IS model.

## DecayFitNet and CommonSlopeAnalysis benchmarks

- G. Götz, R. Falcón Pérez, S. J. Schlecht, and V. Pulkki, “Neural network for
  multi-exponential sound energy decay analysis,” *JASA*, vol. 152, no. 2,
  pp. 942–953, 2022. <https://doi.org/10.1121/10.0013416>.
- G. Götz, S. J. Schlecht, and V. Pulkki, “Common-slope modeling of late
  reverberation,” *IEEE/ACM TASLP*, vol. 31, pp. 3945–3957, 2023.
  <https://doi.org/10.1109/TASLP.2023.3317572>.

The Python benchmark port is limited to preprocessing, common-time clustering,
and log-EDC amplitude fitting. DecayFitNet ONNX weights and transforms remain
external. Copyright and MIT-license attribution are in
`THIRD_PARTY_NOTICES.md`.

## Literature to cite before publication

- C. Févotte, N. Bertin, and J.-L. Durrieu, “Nonnegative Matrix Factorization with the
  Itakura-Saito Divergence: With Application to Music Analysis,” *Neural Computation*, 2009.
- The wideband dictionary and gridless damped-mode works identified in the supplied
  refinement note, including Butsenko, Swärd, Jakobsson; and Jälmby, Swärd, Elvander,
  Jakobsson. Verify complete bibliographic details before formal citation.
