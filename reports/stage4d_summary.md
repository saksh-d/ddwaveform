# Stage 4D — Learned-Basin Geometry and Relative-Unitary Decomposition

## Outcome D

**Outcome D — the exponential-generator coordinate is branch-unstable at the OFDM-to-Learned displacement.**

The five Learned solutions do form a tight unitary basin. However, the aligned center's OFDM-relative spectrum places eigenvalues immediately beside the logarithm branch cut. The principal logarithm reconstructs the matrix numerically, but it is not a unique stable structural coordinate. One-sided valid branches agree on the coarse noncompact conclusion, yet no branch is canonical and the Cayley alternative is severely ill-conditioned. Therefore Stage 4D does not promote `W = W_OFDM exp(K_theta)` as a credible compact law.

## Frozen protocol and gauges

- Native convention uses checkpoint column order and phases exactly.
- Canonical Learned convention aligns every repetition once to `LEARNED_0` using only column permutation/phase equivalence. Controls align once to OFDM and are reported separately.
- The center is an intrinsic Karcher mean initialized by the polar-projected Euclidean centroid; no BER, Q, or structural score enters alignment or centering.
- Aligned center converged in 6 iterations with unitarity residual 1.226e-14.

## Basin geometry

Aligned member-to-center geodesic distance spans 0.10729-0.22686; native distance spans 0.10942-0.23169. The aligned basin radius is 12.432% of the center-to-OFDM displacement.
Within-basin tangent norms span 0.8583-1.8149. Centered observed rank is 4 (necessarily at most four); this is descriptive and is not an intrinsic manifold-dimension claim.

## Logarithm validity and branch sensitivity

The aligned-center principal branch margin is only 0.002659 rad (native center 0.019149 rad), below the predeclared 0.05-rad gate. Skew residual is 0.000e+00 and exponential reconstruction residual is 8.327e-15; numerical exactness does not remove branch ambiguity.
Generator 95% spectral-energy ranks are principal/positive-cut/negative-cut = 41/41/41. Their rank-64 Weyl residuals are 0.8101/0.8151/0.8173; coarse noncompactness is robust to these choices.
The Cayley `(R+I)` condition number is 752, versus the tolerance-derived admissible limit 20.0; its near-minus-one amplification makes it unsuitable as a compact replacement here.

## Principal-coordinate decomposition (diagnostic only)

The principal generator requires ranks 35/41/50/53 for 90/95/99/99.5% spectral energy. It is not low rank.
Principal Weyl rank-64 residual is 0.8101; full rank-4096 returns the exact principal generator. Fixed physical-dictionary residual is 0.9454 with 19 real coefficients, and dominant normalized commutator mean/max are 0.000/0.000.
Rank-64 Weyl reconstruction has d_B=1.4429, residual Q mass=0.00870, and G_D/G_S=1.539. Full Weyl reconstruction returns d_B=1.330e-28 and the center structurally, but stores the complete 4,096-coefficient operator and needs a dense exponential.

## Learned versus control generators

Learned principal-log 95% spectral ranks span 41-41 and rank-64 Weyl residuals span 0.8170-0.8385. Stage-3B values are rank 36-39 and residual 0.5499-0.6758; Haar residual is 0.9171-0.9199.
Top-64 coefficient overlap, entropy/effective support, all nested Weyl residuals, and the same 19-term physical fit are reported for every Learned, Stage-3B, classical, and Haar member in `population_generator_comparison.csv`. No compact Learned-specific separation survives these controls.

## Center and reconstruction communication

At 18 dB Fixed-Q, the center gives QPSK/16-QAM BER 0.00104523/0.025074; the five Learned members span 0.00104618-0.00114632 / 0.0250196-0.025403. Center deviations from the population medians are 1.08%/0.95%, so it is communication-comparable without BER-based redefinition.
All QPSK/16-QAM, 12/15/18/20 dB, Full/Oracle-K24/Fixed-Q-K24 results for every predeclared frozen reconstruction are in `communication_summary.csv`.

## Interpretation gate

1. **Yes under both reported conventions.** Native and canonical centers give similarly small population radii; the canonical convention is retained because the literal relative generator remains gauge-dependent.
2. **No as a stable coordinate.** The principal log is numerically exact but fails the branch-margin gate.
3. **No.** Principal spectral 95% rank is 41 of 64.
4. **No at genuinely low complexity.** DD/Weyl fidelity requires hundreds to thousands of terms.
5. **No predeclared alternative is compact.** Time/subcarrier sparsity, bandedness, Toeplitz/circulant, 8x8 block/Kronecker, and the 19-term physical dictionary leave large residuals.
6. **Locally yes.** Dominant within-basin directions and Weyl overlaps are reproducible descriptively, subject to only five samples.
7. **Yes.** Aligned basin radius/displacement ratio is 0.124.
8. **Yes within each fixed branch.** Nested projections monotonically recover that branch's generator by construction.
9. **Only at high complexity.** Low ranks do not enter the center's complex-B basin.
10. **No monotone functional approach.** Residual Q mass worsens sharply at intermediate Weyl ranks and recovers only at the exact dense endpoint.
11. **Approaches at intermediate-to-high complexity, nonmonotonically.** Full reception improves before exact matrix recovery, but supplies no compact structural law.
12. **No monotone finite-K approach.** Oracle/Fixed-Q communication degrades badly at intermediate Weyl ranks and recovers only at the exact dense endpoint.
13. **No robust learned-specific compact law.** The same nested compression pipeline on Stage-3B and Haar controls prevents interpreting generic dense-log behavior as Learned structure.
14. **Reported without selection.** The center's full paired curves are frozen above and in the machine-readable table.
15. **No.** `W_OFDM exp(K_theta)` remains branch-sensitive and noncompact at N=64.

## Parameter and computation accounting

A retained complex Weyl coefficient is counted as two real stored scalars before skew projection. Sparse coefficient storage is descriptive: every evaluated `exp(K_r)` was formed as a dense 64x64 exponential with O(N^3) offline cost, then stored/applied as a dense unitary with O(N^2) per vector. No tested truncation supplies a fast exponential/action implementation.

## Preservation

- All five Learned and all six Stage-3B hashes verified; 394 prior artifacts remained byte-identical.
- Production banks: 2048 Q atoms, 512 complex-B atoms, 512 paired channels, at least 1000000 bits per waveform/point.
- Stage 2E absent. No training, BER/Q/geometry fitting, truncation reselection, receiver change, or new waveform-family guess.
- STOP. Proceed only after review; Stage 4D itself does not start multi-N work.
