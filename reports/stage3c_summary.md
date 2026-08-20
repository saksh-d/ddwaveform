# Stage 3C — Saturated-Q Communication Geometry

## Result: Outcome B — constellation-sensitive geometry identified

No single modulation-agnostic scalar explains the residual ranking. Q fixes representation energy but not the joint coefficient cross-products governing the finite receiver's error distribution. The simplest robust formulation is therefore a two-statistic LMMSE error profile: aggregate error plus an adverse stream/channel tail.

## Discrepancy localization at 18 dB

| Modulation | Receiver | Learned BER | mean Stage-3B BER |
|---|---|---:|---:|
| QPSK | Full | 0.00103188 | 0.00244522 |
| QPSK | Oracle | 0.00114346 | 0.0024724 |
| QPSK | Q | 0.00121307 | 0.00249036 |
| 16-QAM | Full | 0.0219936 | 0.0225031 |
| 16-QAM | Oracle | 0.0238247 | 0.0227768 |
| 16-QAM | Q | 0.0243769 | 0.0228907 |

The QPSK separation is already visible with full reception at every tested SNR and persists through oracle and fixed-Q reception, so it does not enter first at support selection. The 16-QAM ranking is SNR- and receiver-sensitive: Stage-3B is better at 12/15 dB even under full reception, Learned is slightly better under full reception at 18/20 dB, and truncation shifts 18/20 dB back toward Stage-3B.

## Receiver geometry

- Fixed-Q retained energy is higher for Stage-3B on average (0.999682) than Learned (0.998082); support/energy loss cannot explain Learned's QPSK advantage.
- Median condition number is effectively unchanged: Learned 11.997, Stage-3B mean 11.791. Aggregate conditioning and equalizer norm are not sufficient separators.
- Median lower-tail post-LMMSE SINR is healthier for Learned (7.778) than the Stage-3B mean (5.247); median worst-stream MSE tail is 0.1205 versus 0.1694.
- Paired QPSK channel results place the BER differences in the same adverse reliability and decision-margin tails identified by Stage 1A.1/1A.2. This reappearance was tested, not assumed.

## Why modulation ranks differ

At 18 dB fixed-Q, Stage-3B has lower aggregate/scaled error variance and higher mean normalized margin, but worse p05 margin and lower-tail SINR. QPSK is dominated by the adverse sign-decision tail and therefore favors Learned. For 16-QAM, Stage-3B has fewer negative-margin decisions overall despite a worse extreme margin tail, so it wins after truncation. Because mean and tail point in opposite directions, neither one alone explains both constellations.

## What Q omits

`Q_ij = E|A_ij|^2` contains elementwise second moments. It does not determine joint products such as `E[A_ij A_kl*]`. The finite receiver depends on these cross-products through `A_K^H A_K`, the equalizer, and the exact mismatched error covariance `(F A-I)(F A-I)^H + sigma^2 F F^H`. Nearly equal eta and retained energy therefore do not imply equal stream reliability.

## Representation equivalence

- Pairwise support overlap among Stage-3B solutions spans 0.384-0.761; they are not one identical support mask.
- OFDM-start solutions retain more Stage-2B locality than Haar-start solutions, while all reach nearly identical concentration. They are functionally concentration-equivalent but genuinely distinct in atom and support geometry.

## Interpretation gate

1. The QPSK gap first appears at full-information reception and persists through oracle-K and fixed-Q reception.
2. It is not caused by support or retained-energy loss.
3. Aggregate condition number/noise enhancement does not explain it.
4. Robust lower-tail post-LMMSE reliability explains the QPSK gap more directly, but not the complete two-modulation ranking.
5. The Stage-1A.2 rare channel/stream-tail mechanism reappears.
6. 16-QAM values aggregate variance and multi-threshold decision geometry differently, producing an SNR- and receiver-dependent reversal.
7. No single tested modulation-agnostic scalar explains both; the simplest robust formulation retains both mean and adverse-tail error statistics.
8. Q omits joint coefficient phase/correlation and thus Gram/error-covariance geometry.
9. Stage-3B solutions share saturated concentration but not identical supports or locality.
10. Simplest later candidate: Q concentration plus a two-statistic receiver-geometry criterion that controls aggregate LMMSE error and its robust stream/channel tail. Do not optimize it in Stage 3C.

## Protocol and preservation

- Frozen paired bank: 512 independent normalized P=3 DD-v1 channels; SNRs (12, 15, 18, 20); K=24; common symbols and physical noise.
- Authoritative Learned SHA-256: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`; all six Stage-3B hashes are recorded in the diagnostics JSON.
- All 206 prior artifacts remained byte-identical. IDENTITY is numerical-only and Stage 2E was not used.
- No waveform was trained, tuned, or modified in Stage 3C. STOP before Stage 3D.
