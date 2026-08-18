# Stage 1A.1 — Frozen Communication-Error Forensic Audit

## A. Executive diagnosis

The BER/MSE discrepancy is diagnosed as an **extreme-tail QPSK decision-geometry effect**, not a mean noise-enhancement tradeoff.

The frozen Learned waveform reduces QPSK analytic MSE relative to IDENTITY at every SNR by reorganizing the post-equalizer operator: mean off-diagonal interference falls by `46%–72%`, and its noise term is also slightly lower. This bulk improvement raises mean/median effective SINR and mean/median nearest-neighbor margin. However, Learned accepts more desired-stream diagonal gain distortion and has a worse extreme stream tail. At 12 dB its diagnostic SINR mean/median improve from `8.12/7.35` to `10.73/9.59`, while its 5th/1st percentiles worsen from `3.22/1.95` to `2.49/1.34`. The same crossing persists more weakly at 15–20 dB.

QPSK hard errors expose that tail exactly. At 12 dB the nearest-neighbor margin mean/median improve from `0.901/0.964` to `0.929/1.004`, but the 1st percentile worsens from `-0.130` to `-0.190` and SER rises from `1.819%` to `2.091%`. At 15 dB the corresponding 1st percentile changes from `0.079` to `0.053`, and SER rises from `0.698%` to `0.823%`. Learned becomes competitive at 18 dB and significantly better at 20 dB once the improved bulk/lower tail outweighs the rare extreme failures.

This is not a general finite-constellation failure. With normalized Gray 16-QAM, Learned improves BER and SER over IDENTITY at all four SNRs; every paired bootstrap interval is entirely below zero. The next loss should therefore target generic decision-margin tails without discarding the exact mean communication-MSE objective.

## B. Frozen experiment configuration

- Checkpoint: `LEARNED_FROM_HAAR_2`, loaded from `outputs/stage1a/checkpoints/haar_2_best.pt`.
- Checkpoint SHA-256: `3e55e338e031600f7e83e0b26f6432226a6924dec1a03990a6096e35f39fbaad`.
- The loaded tensor exactly matches the HAAR-2 best matrix in `training.pt`; unitarity residual is `2.18e-14`, and the tensor was unchanged after evaluation.
- Channel: validated DD-v1, `N=64`, `P=3`, continuous delays `U[0,4]`, continuous Dopplers `U[-0.75,0.75]`, per-realization Frobenius normalization.
- Receiver: exact row-wise hard `K=24`; `G=(A_K^H A_K+sigma^2 I)^{-1}A_K^H` applied to the true `A` observation.
- Frozen waveforms: IDENTITY, OFDM, OTFS, AFDM_K24_TUNED, and Learned.
- Forensic bank: 512 new channels; seed `11020`. The final test bank was not loaded.
- Primary SNRs: `12, 15, 18, 20` dB.
- Modulations: unit-power QPSK and unit-power Gray square 16-QAM; both use generic nearest-neighbor detection.
- Symbol/noise/bootstrap seeds: `11021`, `11022`, and `11023`.
- QPSK: 16 blocks/channel, 524,288 symbols and 1,048,576 bits per waveform/SNR.
- 16-QAM: 8 blocks/channel, 262,144 symbols and 1,048,576 bits per waveform/SNR.
- All comparisons use paired channels, symbols, and physical time-domain noise.

IDENTITY is retained in every numerical comparison below and in all CSV data. Per project convention it is absent from every figure.

## C. Exact MSE decomposition

The maximum numerical residual between the validated receiver MSE and

`L_diagonal + L_offdiagonal + L_noise`

is `1.11e-16`.

| SNR | Waveform | Total | Diagonal gain/phase | Off-diagonal interference | Noise |
|---:|---|---:|---:|---:|---:|
| 12 | IDENTITY | 0.132093 | 0.017334 | 0.031660 | 0.083099 |
| 12 | Learned | 0.120775 | 0.021177 | 0.017129 | 0.082469 |
| 15 | IDENTITY | 0.090654 | 0.007303 | 0.030479 | 0.052872 |
| 15 | Learned | 0.075079 | 0.009580 | 0.013160 | 0.052339 |
| 18 | IDENTITY | 0.067222 | 0.003318 | 0.031796 | 0.032108 |
| 18 | Learned | 0.046848 | 0.004580 | 0.010574 | 0.031694 |
| 20 | IDENTITY | 0.058618 | 0.002167 | 0.033824 | 0.022626 |
| 20 | Learned | 0.034731 | 0.002988 | 0.009461 | 0.022283 |

Learned does not amplify more aggregate noise. Its noise term is `0.8%–1.5%` lower. It trades slightly larger desired-stream diagonal distortion for a large off-diagonal reduction. Post-training representation metrics on this bank are consistent: Learned versus IDENTITY mean retained energy is `0.999303` versus `0.992456`, median limiting distortion is `0.002491` versus `0.044057`, and median `cond(T_24(A))` is `12.45` versus `14.49`.

## D. Per-stream reliability

The diagnostic `gamma_i` is second-order only; residual interference is not assumed Gaussian.

| SNR | Waveform | Gamma mean | Gamma median | Gamma p05 | Gamma p01 | `|B_ii|` p05 | `|B_ii|` p01 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 12 | IDENTITY | 8.123 | 7.354 | 3.225 | 1.952 | 0.780 | 0.676 |
| 12 | Learned | 10.734 | 9.588 | 2.486 | 1.339 | 0.715 | 0.574 |
| 15 | IDENTITY | 13.565 | 11.972 | 4.537 | 2.759 | 0.845 | 0.760 |
| 15 | Learned | 20.348 | 17.993 | 3.914 | 2.016 | 0.799 | 0.670 |
| 18 | IDENTITY | 21.833 | 17.693 | 5.692 | 3.371 | 0.888 | 0.813 |
| 18 | Learned | 38.536 | 33.758 | 5.935 | 2.789 | 0.858 | 0.739 |
| 20 | IDENTITY | 29.438 | 22.393 | 6.101 | 3.485 | 0.907 | 0.836 |
| 20 | Learned | 58.713 | 51.030 | 7.607 | 3.312 | 0.888 | 0.775 |

Learned has far less phase error: at 12 dB the signed phase 5th/1st percentiles are `-0.00191/-0.00445` rad versus IDENTITY `-0.00947/-0.01769` rad. The reliability weakness is therefore principally a rare magnitude/gain tail, not a phase-rotation tail. Learned's interference and output-noise medians are lower at every SNR, but its rare small `|B_ii|` streams lower extreme SINR.

## E. Channel-tail analysis

| SNR | Learned lower MSE | Learned lower BER | Lower MSE, higher BER | Lower MSE, lower BER | Worst 10% share of positive BER penalty |
|---:|---:|---:|---:|---:|---:|
| 12 | 96.68% | 32.23% | 56.25% | 32.03% | 44.34% |
| 15 | 96.48% | 30.08% | 42.58% | 29.69% | 54.45% |
| 18 | 96.29% | 27.34% | 31.05% | 27.15% | 64.90% |
| 20 | 95.51% | 27.15% | 24.80% | 26.95% | 70.22% |

Mean Learned-minus-IDENTITY MSE deltas are `-0.01132, -0.01557, -0.02037, -0.02389`. Mean QPSK BER deltas are `+0.001534, +0.000674, -0.000026, -0.000329`. At 12 and 15 dB the paired BER 95% CIs are `[+0.001017,+0.002054]` and `[+0.000290,+0.001053]`; at 18 dB the CI crosses zero; at 20 dB it is `[-0.000581,-0.000092]`.

The penalty is tail-concentrated, increasingly so with SNR, but not caused by only a tiny pathological set at 12 dB: over half the channels have lower MSE and higher observed BER there. Channel-level BER is based on 2,048 QPSK bits/channel, so individual zero/tie differences are coarse; paired aggregate inference remains well resolved.

## F. QPSK decision margins

The signed-bit margin uses `sign(Re{s}) Re{s_hat}` and its imaginary analogue. The fixed near-boundary threshold is `epsilon=0.1` normalized coordinate units, 14.1% of the nominal QPSK coordinate magnitude `1/sqrt(2)`; it was not tuned.

| SNR | Waveform | Signed mean | Signed median | Signed p05 | Signed p01 | `P(m<0)` | `P(0<=m<0.1)` |
|---:|---|---:|---:|---:|---:|---:|---:|
| 12 | IDENTITY | 0.6241 | 0.6304 | 0.2151 | 0.0091 | 0.9257% | 1.1580% |
| 12 | Learned | 0.6223 | 0.6350 | 0.2236 | -0.0108 | 1.0791% | 1.0880% |
| 15 | IDENTITY | 0.6573 | 0.6635 | 0.3085 | 0.1169 | 0.3561% | 0.5033% |
| 15 | Learned | 0.6553 | 0.6662 | 0.3353 | 0.1175 | 0.4235% | 0.4589% |
| 18 | IDENTITY | 0.6784 | 0.6830 | 0.3765 | 0.1933 | 0.1768% | 0.2590% |
| 18 | Learned | 0.6755 | 0.6844 | 0.4226 | 0.2246 | 0.1742% | 0.2086% |
| 20 | IDENTITY | 0.6875 | 0.6914 | 0.4093 | 0.2253 | 0.1503% | 0.2023% |
| 20 | Learned | 0.6840 | 0.6918 | 0.4695 | 0.2785 | 0.1174% | 0.1417% |

Generic QPSK nearest-neighbor margins show the same crossover:

| SNR | Waveform | Mean | Median | p05 | p01 | Negative probability = SER |
|---:|---|---:|---:|---:|---:|---:|
| 12 | IDENTITY | 0.901 | 0.964 | 0.240 | -0.130 | 1.819% |
| 12 | Learned | 0.929 | 1.004 | 0.241 | -0.190 | 2.091% |
| 15 | IDENTITY | 1.007 | 1.071 | 0.437 | 0.079 | 0.698% |
| 15 | Learned | 1.048 | 1.119 | 0.475 | 0.053 | 0.823% |
| 18 | IDENTITY | 1.081 | 1.144 | 0.576 | 0.222 | 0.347% |
| 18 | Learned | 1.139 | 1.202 | 0.672 | 0.265 | 0.340% |
| 20 | IDENTITY | 1.118 | 1.181 | 0.645 | 0.277 | 0.295% |
| 20 | Learned | 1.185 | 1.243 | 0.777 | 0.371 | 0.228% |

On channels where Learned has lower MSE but higher BER, the negative-margin penalty becomes much sharper. At 12 dB the QPSK NN negative fraction is `2.524%` for Learned versus `1.626%` for IDENTITY, and the 1st percentile is `-0.242` versus `-0.098`. At 15 dB it is `1.316%` versus `0.593%`, with p01 `-0.067` versus `+0.093`. Thus the bulk margin improvement and extreme-tail degradation coexist within the same difficult-channel subset.

## G. 16-QAM generalization probe

| SNR | Waveform | BER | SER | Analytic MSE | Empirical MSE | NN mean | NN p05 | NN negative |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 12 | IDENTITY | 0.09406 | 0.33109 | 0.13209 | 0.13181 | 0.104 | -0.464 | 33.109% |
| 12 | Learned | 0.08233 | 0.28503 | 0.12078 | 0.12095 | 0.136 | -0.456 | 28.503% |
| 15 | IDENTITY | 0.05739 | 0.20841 | 0.09065 | 0.09070 | 0.199 | -0.314 | 20.841% |
| 15 | Learned | 0.04403 | 0.15656 | 0.07508 | 0.07482 | 0.246 | -0.270 | 15.656% |
| 18 | IDENTITY | 0.03607 | 0.13271 | 0.06722 | 0.06738 | 0.267 | -0.200 | 13.271% |
| 18 | Learned | 0.02264 | 0.08155 | 0.04685 | 0.04711 | 0.332 | -0.106 | 8.155% |
| 20 | IDENTITY | 0.02841 | 0.10471 | 0.05862 | 0.05877 | 0.301 | -0.150 | 10.471% |
| 20 | Learned | 0.01493 | 0.05389 | 0.03473 | 0.03510 | 0.379 | -0.016 | 5.389% |

Learned-minus-IDENTITY 16-QAM BER differences are `-0.01173, -0.01335, -0.01343, -0.01349`; all paired 95% CIs exclude zero. SER differences are `-0.04606, -0.05185, -0.05117, -0.05082`, also significant. The MSE improvement transfers cleanly to a denser constellation. The Stage-1A discrepancy is therefore QPSK-specific in this frozen experiment, not modulation-general across QPSK and 16-QAM.

## H. Analytic-versus-empirical MSE

Empirical unit-power symbol MSE agrees with the modulation-independent analytic result for both constellations. Across all waveform/SNR/modulation points, the largest relative aggregate discrepancy is `1.25%` (IDENTITY QPSK at 20 dB); most discrepancies are below `0.6%`. For the headline IDENTITY/Learned pairs, discrepancies have no consistent sign and do not explain BER ordering. This validates the physical-noise simulation, constellation normalization, and use of the analytic decomposition.

## I. Hypothesis assessment

| Hypothesis | Assessment | Evidence |
|---|---|---|
| H1 — noise-enhancement tradeoff | **NOT SUPPORTED** | Learned's aggregate noise term is slightly lower at every SNR; the major MSE change is reduced off-diagonal interference. |
| H2 — tail-reliability problem | **SUPPORTED** | Mean/median gamma improve while p05/p01 worsen at 12/15 dB; p01 remains worse through 20 dB. Desired-magnitude lower tails show the same behavior. |
| H3 — decision-geometry mismatch | **SUPPORTED** | QPSK bulk NN margins improve but negative probability and extreme tail worsen exactly where BER worsens; conditional discordant channels amplify this pattern. |
| H4 — QPSK-specific effect | **SUPPORTED** | Frozen Learned significantly improves 16-QAM BER/SER at every SNR while QPSK degrades significantly at 12/15 dB. |
| H5 — general finite-constellation effect | **NOT SUPPORTED** | The discrepancy does not persist under normalized Gray 16-QAM. |
| H6 — channel-tail problem | **PARTIALLY SUPPORTED** | The worst 10% of channels contribute 44%–70% of positive penalty, but discordance is spread across 56% of channels at 12 dB. |

### Council cross-examination

- **First-principles lens:** exact second-moment equality and decomposition eliminate normalization, receiver substitution, and mean noise enhancement as causes.
- **Contrarian lens:** 16-QAM and empirical-MSE controls falsify a generic simulator or finite-constellation failure. The remaining diagnosis is distributional and QPSK-tail-specific.
- **Expansionist lens:** the remedy should accept arbitrary constellations, rather than encoding QPSK quadrants.
- **Outsider lens:** this is a ranking-tail problem: average squared regression error improves while rare nearest-class ordering worsens.
- **Executor lens:** retain the trusted MSE objective and add one validation-calibrated generic tail term; do not change receiver, K, or channel family simultaneously.

## J. Recommended Stage-1A.2 loss

Recommend **mean exact communication MSE plus channel-CVaR of a generic normalized nearest-neighbor margin loss**.

For arbitrary unit-power constellation `C` with minimum distance `d_min(C)`, define

`m_bar = (min_{c in C, c != s}|s_hat-c| - |s_hat-s|) / d_min(C)`

and a smooth per-symbol ranking loss

`ell_margin = T log(1 + exp((m0-m_bar)/T))`.

For a channel `H`, average this over the locked SNR distribution, sampled symbols, physical noise, and streams:

`R_C(H) = E[ell_margin | H]`.

The proposed Stage-1A.2 objective is

`L_1A2 = E[L_comm] + lambda * CVaR_beta,H(R_C(H))`,

with

`CVaR_beta(R) = min_eta { eta + E[(R-eta)_+] / (1-beta) }`.

This preserves the exact modulation-independent mean MSE and directly penalizes the measured failure: rare incorrect/near-boundary constellation rankings, with extra weight on difficult channels. It is not QPSK-specific; any constellation is supplied by its point set and `d_min`. The normalized margin makes temperature comparable across constellations.

Validation-only calibration is required for `lambda`, tail level `beta`, temperature `T`, target margin `m0`, and the constellation/SNR sampling mixture. A minimal first calibration should keep `m0=0`, compare a small grid around `beta in {0.9,0.95}`, and use both QPSK and 16-QAM definitions while preserving QPSK as the primary selection modulation. Do not add a separate SINR-tail penalty initially: gamma is only second-order, whereas NN margin sign exactly identifies symbol errors.

The proposal is falsified if, at matched training budget and frozen evaluation protocol, it fails to improve paired QPSK BER at 12/15 dB over the mean-MSE learner, loses the existing 20 dB or 16-QAM gains, materially erodes the analytic-MSE advantage, or violates full-information/normalization invariants. Stop calibration if no validation setting improves QPSK lower-tail margins and BER without one of those failures.

Confidence: **high** that the frozen discrepancy is a QPSK extreme-tail decision-margin effect; **moderate** that the proposed CVaR margin term is the best trainable remedy, because its causal training effect has not yet been tested.

## K. Files, tests, and git boundaries

- Added `src/diagnostics.py`, `scripts/stage1a1.py`, `tests/test_stage1a1.py`, and this report.
- Extended `src/metrics.py` additively with generic constellations and `src/config.py` with independent forensic seeds/count.
- Generated all data and seven required figures under ignored `outputs/stage1a1/`.
- Production forensic runtime: `26.2 s`.
- Final test command: `conda run -n pytorch python -m pytest -q`.
- Final deterministic result: `52 passed`.
- `outputs/` remains ignored; no generated output is tracked.
- `researchPlan.MD` and validated Stage-0 channel/receiver/waveform implementations are unchanged.
- The final test bank was not generated, loaded, or inspected.
- No waveform was retrained or mutated. Stage 1A.2 and Stage 1B were not implemented.

STAGE 1A.1 COMPLETE — BER/MSE DISCREPANCY DIAGNOSED
