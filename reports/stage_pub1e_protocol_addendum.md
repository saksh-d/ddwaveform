# Pub-1E protocol addendum

This addendum corrects interpretation and provenance wording without changing the frozen experiment or rerunning simulations. The original protocol remains the historical pre-production record.

## Plot and label corrections

- The first E1 budget rendering mixed all waveforms into every nominal waveform curve because the plot selector omitted `waveform`. The active rendering now uses condition/modulation/SNR/receiver/waveform/K keys and asserts one row per K. The old file is retained under `outputs/stage_pub1e/superseded/`.
- Every E1-E4 plotting selection now specifies all relevant data keys and asserts uniqueness. Companion plots expose the all-waveform receiver ladder and named AFDM behavior.
- `boundary_joint_0.2` remains an immutable historical CSV/config key. It means predeclared intermediate/probe point, not an empirically discovered boundary.

## Actual random-stream derivations

- E1 condition index `c`: test `17500+10c`; receiver validation `17110+10c`; residual statistics `17300+10c`; symbols `17800+10c`; noise `17900+10c`.
- E2: physical channel seed `17400` at every delay/Doppler cell to pair the underlying realization; symbols `17840`; noise `17940`.
- E3 family index `f`: test `17700+100f`; shifted support `17600+100f`; nominal support `17200`; symbols `17880+f`; noise `17980+f`. Severities share base randomness within family.
- E4 anchored conditions: physical `18000`, symbols `17920`, noise `18020`. Nominal full-information: physical `18100`, symbols `17930`, noise `18030`.
- Bootstrap base `18000` and E4 physical `18000` reuse the same integer across different generator roles. This does not couple arrays or leak evaluation data. Therefore “all numeric seeds are disjoint” is not a valid description; generator roles and produced banks are the meaningful provenance.
- Within each K, waveforms are paired. Across E1 K values, common-random-number pairing is absent: K=24 evaluates the complete SNR list so 18 dB uses index 2, while other K values evaluate only 18 dB at index 0. Historical K=64 and full-information K=64 are likewise separate receiver calls and must not be presented as paired duplicates.

## Receiver and support scope

Residual-aware E1 statistics estimate the omitted matrix mean and receive covariance separately for each waveform, condition, and K. Statistics construction and evaluation both use realization-wise oracle row-top-K. The bank is independent of the E1 test bank. The receiver does not condition the moments on the realized retained values or support, so it is a moment-matched approximation rather than conditional-optimal LMMSE. It stores a dense complex 64x64 mean and covariance per waveform/condition/K and performs a direct dense solve; equal K is not equal information, storage, or compute.

E3 evaluates fixed-support policies with the historical receiver only. No E3 residual-aware result exists, so there is no hidden fixed-support/residual-bank policy mismatch.

## AFDM and full-information interpretation

Guarded AFDM selection searched odd `2Nc1` values 3/5/7/9. Historical slope 1 was retained as a named comparator but excluded from that engineering search. All these odd-slope transforms are numerically unitary; the lower bound is an overlap/guard design rule, not a mathematical unitarity requirement. The validation-selected slope 3 remains frozen. Saved E4 rows show slope 1 can be stronger in some regimes; post hoc paired intervals are descriptive and do not trigger test-set reselection. Slope 1 with the improved receiver ladder remains untested.

With full A and aggregate LMMSE MSE, unitary waveform transforms are invariant under the stated model. Finite-constellation BER after symbol slicing need not be invariant. Retained energy alone is also insufficient to determine BER.
