# Stage 1A — Universal Fixed Unitary Waveform Existence Test

## A. Executive scientific verdict

The selected universal checkpoint is the `HAAR_2` initialization run. Its pooled MSE is `0.07115105` versus `0.09018911` for the strongest fixed comparator, `IDENTITY`. The paired difference is `-0.01903806` (95% CI `[-0.02012959, -0.01796669]`).

Learned improves BER at `1/4` primary SNRs, multiple-start reproducibility is `True`, and mean un-clipped finite-budget gap closure is `77.94%`. The result is classified **Outcome B**.

## B. Training protocol

DD-v1 (`N=64`, `P=3`, continuous delays `U[0,4]`, continuous Dopplers `U[-0.75,0.75]`, normalized per realization), fresh online channels, continuous SNR `U[12,20]` dB, exact hard `K=24`, batch `16`, `2000` projected-Adam steps, learning rates `1e-2/3e-3/1e-3` over steps `1–1000/1001–1600/1601–2000`, validation every `25` steps on the frozen 256-channel bank at `12/15/18/20` dB. Complex128; every update receives SVD polar projection `UV^H`. Training seeds: channel `11010`, SNR `11011`, Haar `11012..11014`.

## C. Per-initialization results

| Initialization | Initial val MSE | Best val MSE | Best step | Final val MSE | Beats init | Held-out pooled MSE | Beats fixed held-out | Max unitarity residual | Runtime (shared s) |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| IDENTITY | 0.09176463 | 0.07568695 | 100 | 0.07575475 | True | 0.07440022 | True | 2.511e-14 | 571.4 |
| OFDM | 0.09848872 | 0.07314295 | 1125 | 0.07342656 | True | 0.07191693 | True | 2.535e-14 | 571.4 |
| OTFS | 0.12344083 | 0.07586493 | 1975 | 0.07586923 | True | 0.07461083 | True | 2.495e-14 | 571.4 |
| AFDM_K24_TUNED | 0.10704711 | 0.07466948 | 1075 | 0.07507583 | True | 0.07339181 | True | 2.481e-14 | 571.4 |
| HAAR_1 | 0.85483877 | 0.07263276 | 875 | 0.07308260 | True | 0.07137367 | True | 2.472e-14 | 571.4 |
| HAAR_2 | 0.85762626 | 0.07237389 | 1050 | 0.07270051 | True | 0.07115105 | True | 2.456e-14 | 571.4 |
| HAAR_3 | 0.86440912 | 0.07267858 | 1275 | 0.07301971 | True | 0.07144298 | True | 2.478e-14 | 571.4 |

Analytically indistinguishable checkpoints retained for BER: `LEARNED_FROM_HAAR_2`.

## D. Held-out analytic results

Means use the independent 1,024-channel Stage-1A bank. The strongest fixed entry is selected by aggregate MSE at each SNR. The pooled strongest principal baseline is `OFDM` at `0.09829901` MSE; Learned improves it by `27.62%`.

| SNR dB | IDENTITY | OFDM | OTFS | AFDM K24 | Learned | Strongest fixed | R_MSE | C_gap | Delta 95% CI |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 12 | 0.13489694 | 0.13728567 | 0.15734926 | 0.14402712 | 0.12327644 | IDENTITY | 8.7004% | 79.7124% | [-0.01219754, -0.01104231] |
| 15 | 0.09340169 | 0.09877644 | 0.12185998 | 0.10707914 | 0.07709329 | IDENTITY | 16.9483% | 78.5857% | [-0.01720491, -0.01542666] |
| 18 | 0.07027642 | 0.08042768 | 0.10499171 | 0.08951497 | 0.04833732 | IDENTITY | 27.9907% | 77.2878% | [-0.02326203, -0.02064915] |
| 20 | 0.06218141 | 0.07670623 | 0.10141088 | 0.08569702 | 0.03589715 | IDENTITY | 35.8205% | 76.3431% | [-0.02798226, -0.02461064] |

## E. BER

All entries use paired channels, QPSK bits, and physical time-domain AWGN.

| SNR dB | Waveform | BER | Errors | Bits | Paired delta vs best fixed |
|---:|---|---:|---:|---:|---:|
| 12 | IDENTITY | 9.56726074e-03 | 10032 | 1048576 | +0.00000000e+00 |
| 12 | OFDM | 1.48010254e-02 | 15520 | 1048576 | +5.23376465e-03 |
| 12 | OTFS | 1.26228333e-02 | 13236 | 1048576 | +3.05557251e-03 |
| 12 | AFDM_K24_TUNED | 1.09062195e-02 | 11436 | 1048576 | +1.33895874e-03 |
| 12 | Learned | 1.14250183e-02 | 11980 | 1048576 | +1.85775757e-03 |
| 15 | IDENTITY | 3.54194641e-03 | 3714 | 1048576 | +0.00000000e+00 |
| 15 | OFDM | 7.71713257e-03 | 8092 | 1048576 | +4.17518616e-03 |
| 15 | OTFS | 6.06060028e-03 | 6355 | 1048576 | +2.51865387e-03 |
| 15 | AFDM_K24_TUNED | 4.77313995e-03 | 5005 | 1048576 | +1.23119354e-03 |
| 15 | Learned | 4.71687317e-03 | 4946 | 1048576 | +1.17492676e-03 |
| 18 | IDENTITY | 2.00080872e-03 | 2098 | 1048576 | +0.00000000e+00 |
| 18 | OFDM | 5.49507141e-03 | 5762 | 1048576 | +3.49426270e-03 |
| 18 | OTFS | 4.58717346e-03 | 4810 | 1048576 | +2.58636475e-03 |
| 18 | AFDM_K24_TUNED | 3.09276581e-03 | 3243 | 1048576 | +1.09195709e-03 |
| 18 | Learned | 2.09331512e-03 | 2195 | 1048576 | +9.25064087e-05 |
| 20 | IDENTITY | 1.75380707e-03 | 1839 | 1048576 | +0.00000000e+00 |
| 20 | OFDM | 5.44929504e-03 | 5714 | 1048576 | +3.69548798e-03 |
| 20 | OTFS | 4.56714630e-03 | 4789 | 1048576 | +2.81333923e-03 |
| 20 | AFDM_K24_TUNED | 3.20434570e-03 | 3360 | 1048576 | +1.45053864e-03 |
| 20 | Learned | 1.33609772e-03 | 1401 | 1048576 | -4.17709351e-04 |

No aggregate point with zero errors is represented as exact zero; channel-level zero-error rows include the rule-of-three bound in CSV.

## F. Gap closure

The pooled un-clipped mean `C_gap` is `0.779389`; the per-SNR values are reported in Section D. Full-MSE references are computed independently as `tr[(I+rho H^H H)^-1]/N` for every channel/SNR.

## G. Representation diagnostics

These post-training quantities were not used for optimization or checkpoint selection.

| Waveform | Mean retained energy | Mean truncation error | Median condition | Median limiting distortion |
|---|---:|---:|---:|---:|
| IDENTITY | 0.99241932 | 0.00758068 | 16.266709 | 0.05160540 |
| OFDM | 0.99185173 | 0.00814827 | 17.647358 | 0.11641034 |
| OTFS | 0.97658443 | 0.02341557 | 18.255693 | 0.12689155 |
| AFDM_K24_TUNED | 0.98647743 | 0.01352257 | 16.962882 | 0.13334622 |
| Learned | 0.99933162 | 0.00066838 | 14.356268 | 0.00290872 |

## H. Full-information invariant

Maximum per-channel learned/fixed versus independent trace discrepancy: `1.221e-15`. Maximum cross-waveform aggregate discrepancy: `2.776e-17`.

## I. Training stability

Non-finite runs: `0`. Best-seen checkpointing changed the selected step for `7/7` runs. Validation traces record hard-support switching directly; no smoothing or relaxed mask was introduced.

## J. Interpretation relative to Stage 0F

Stage 0F found 20.4% mean channel-specific K=24 improvement versus its principal comparator, 12.7% versus its IDENTITY-inclusive comparator, and 69.5% principal-to-full gap closure. Stage 1A obtains 19.69% IDENTITY-inclusive relative MSE improvement and 77.94% gap closure on a different 1,024-channel bank. These numbers are not a direct fraction-of-oracle estimate because Stage 0F used 24 channel-specific calibration channels and a different comparator construction. The conceptual distinction remains: Stage 0F adapted one matrix per channel, while Stage 1A freezes one matrix for every unseen channel.

## K. Files/tests/git

Added `src/learning.py`, `scripts/stage1a.py`, and `tests/test_stage1a.py`; extended `src/config.py` and `HISTORY.md`; created this report. Generated data and six required figures are under ignored `outputs/stage1a/`. Final deterministic result: `45 passed`. No output is tracked. `researchPlan.MD` is unchanged; the final test bank was never generated or inspected. Total Stage-1A runner time: `619.0` s.

STAGE 1A COMPLETE — OUTCOME B: WEAK/PARTIAL UNIVERSAL HEADROOM
