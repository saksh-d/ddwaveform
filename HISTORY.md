# Project History

## 2026-08-17 19:45:35 EDT — Initial Stage 0 harness

- Added DD-v1 channel, classical waveform, finite-budget receiver, and QPSK metric modules in CPU double precision.
- Added deterministic core tests covering operators, normalization, transforms, receiver MSE, and full-information invariance.
- Initial `pytest -q` stopped at import collection because repository root was absent from pytest's import path; no scientific test ran.
- Added minimal pytest root-path configuration; numerical implementation unchanged.

## 2026-08-17 19:46:12 EDT — DD commutation diagnostic correction

- Core suite reached the scientific checks: 22 passed and one DD commutation assertion failed.
- Confirmed circular delay plus fractional sampled Doppler has a wrap-boundary phase term, so the stated scalar Weyl identity is globally exact only for integer Doppler bins.
- Retained the mandated delay and Doppler operators; changed only the sign/order diagnostic to an integer Doppler bin.
- Flagged the fractional-boundary qualification for the Stage 0 anomalies report.

## 2026-08-17 19:49:10 EDT — AFDM validation selection

- Focused deterministic suite passed: 23 tests in 0.71 s.
- Swept all eight AFDM `c1=m/(2N)` candidates on 256 validation-only DD-v1 channels, `K=3`, and 12-20 dB.
- Selected and locked `m=1`, hence `c1=1/128=0.0078125`; mean analytic MSE was 0.602568632.
- Final test seed/bank and Stage-0 evaluation bank were not used for parameter selection.

## 2026-08-17 19:50:08 EDT — First complete Stage 0 run

- Complete Stage 0A-E campaign passed in 13.0 s; all 23 deterministic tests passed.
- Full-information receiver/trace discrepancy was `3.61e-16`; cross-waveform spread was `1.11e-16`.
- Receiver Monte Carlo relative gaps were at most 0.51%; DD-v1 normalization error was `4.44e-16`.
- Generated paired BER/NMSE curves and the tracked scientific summary; visual inspection found clear, unclipped plots and plausible mismatch-limited high-SNR behavior.
- Audit found random-unitary QR phase canonicalization could use the standard positive-diagonal convention; corrected without changing its unitarity or role.

## 2026-08-17 19:50:48 EDT — Final Stage 0 verification

- Re-ran the final code: 23 tests passed, followed by `STAGE 0 PASS` in 11.6 s.
- Maximum waveform unitarity residual was `7.18e-14`; full receiver/trace and cross-waveform discrepancies remained `3.61e-16` and `1.11e-16`.
- Confirmed generated artifacts are ignored and none are tracked; `researchPlan.MD` is unchanged from `origin/main`.
- Confirmed both figures render cleanly and the tracked summary matches final numerical outputs.

## 2026-08-17 20:14:16 EDT — Stage 0E audit primitives

- Renamed the untransformed control from TIME to IDENTITY in source, tests, future plots, and README; preserved historical Stage-0 artifacts.
- Added independent two-stage OTFS and direct-sum AFDM references, canonical AFDM parameter/support rules, retained-energy metrics, and controlled channel-family samplers.
- Added audit tests for transform equivalence, clean reconstruction, AFDM path locations, retained-energy identities, and full-CSI BER decline.
- First expanded run had 30 passes and one assertion-only float32/float64 mismatch; corrected the reference tensor dtype without changing implementation math.

## 2026-08-17 20:24:56 EDT — Stage 0E corrective audit complete

- Final audit run passed 31 tests and completed all paired experiments in 61.5 s; each principal full-CSI point used 1,048,576 bits.
- Validated OTFS two-stage/Kronecker equivalence (`4.08e-15`) and AFDM direct/factored equivalence (`7.04e-14`); all 20 canonical AFDM path offsets matched.
- Established root cause: DD-v1 top-3 energy is OFDM 0.916, tuned AFDM 0.821, OTFS 0.707; full-CSI curves waterfall while K=3 mismatch floors/rises.
- GRID-DD is exactly top-3 representable for all baselines; fractional delay materially changes ordering, and the guarded canonical AFDM variant is worse than K3-tuned AFDM on DD-v1.
- N=256 reference recovered qualitative OTFS/AFDM advantage over OFDM; audit status requires scientific review of DD-v1/K=3 before Stage 1A.

## 2026-08-17 21:56:17 EDT — Stage 0F oracle harness

- Added fixed Stage-0F calibration/control/bootstrap seeds and validation-only AFDM tuning for K=16/24/32/multi-K; all four selected `c1=1/128`.
- Added separable batched projected-Adam oracle optimization with exact hard top-K, six deterministic starts, per-restart best-seen tracking, and checkpointed 300-step runs.
- Added paired analytic/BER, bootstrap, representation, multi-K, GRID-DD, and fractional-Doppler diagnostics plus required plotting/output paths.
- Expanded deterministic suite to 37 passing tests, including polar projection, finite gradients, reproducibility, best-seen safety, and the GRID-DD full-MSE negative control.
- A 24-channel 10-step smoke run improved mean K=16 objective from 0.10063 to 0.09306 while retaining `7.18e-14` unitarity.

## 2026-08-17 22:32:00 EDT — Stage 0F production diagnostic

- Completed the 24-channel, six-start, 300-step production campaign in 1735.5 s with no non-finite runs and maximum unitarity residual `7.18e-14`.
- Pooled oracle-minus-principal MSE confidence intervals exclude zero at K=16/24/32; mean relative gains are 31.3%, 20.4%, and 11.2%.
- Paired million-bit BER checks transfer the analytic gain at 15/20 dB; K=24 is the recommended primary discovery budget while K=16 has the largest but more mismatch-driven headroom.
- GRID-DD and fractional-Doppler controls match full-information MSE to `6.94e-17`; one multi-K basis nearly spans K=24/32 with a 4.6% mean K=16 penalty.
- Added the tracked A–K scientific report, README run guidance, and visually corrected log-axis labels on required figures.

## 2026-08-17 22:32:37 EDT — Stage 0F final verification

- Final deterministic suite passed: 37 tests in 2.09 s.
- Confirmed all required Stage-0F artifacts exist, render cleanly, and remain ignored with no generated output tracked.
- Confirmed `researchPlan.MD` remains identical to `origin/main` and the report ends with the locked headroom-confirmed status.

## 2026-08-17 23:05:00 EDT — Stage 1A learning core

- Added complex128 universal projected-Adam learning with seven independent starts sharing fresh paired DD-v1/SNR streams.
- Added exact hard-K communication loss, frozen-bank validation, best-seen checkpoint retention, and Stage-1A seed/protocol constants.
- Added eight focused Stage-1A checks covering projection, gradients, reproducibility, checkpointing, immutability, seed separation, and full-information invariance.
- Initial full suite reached 44 passes; one exact assertion exposed only float32 reconstruction of a float64 trace in the test.

## 2026-08-17 23:16:00 EDT — Stage 1A runner smoke verification

- Added the complete training/evaluation runner with paired analytic MSE, bootstrap, BER, representation, invariance, figures, checkpoints, and report generation.
- Corrected the checkpoint test to preserve float64 trace precision; the full suite passes 45 tests in 1.77 s.
- A two-step production-shape smoke completed in 11.9 s with finite objectives and checkpoint output.
- Fixed validation objectives reproduce the locked K=24 calibration; no protocol reduction or precision change was made.

## 2026-08-17 23:35:00 EDT — Stage 1A production result and evaluation audit

- Completed seven 2,000-step complex128 runs in 571.4 s; all starts improved, no non-finite run occurred, and max unitarity residual was 2.54e-14.
- Corrected an evaluation-only name collision where learned-from-classical checkpoints shadowed fixed baselines; regenerated all held-out results from saved checkpoints.
- Learned pooled MSE is 0.071151 versus fixed IDENTITY 0.090189; paired 95% CI is [-0.020130, -0.017967], with 77.94% mean gap closure.
- Million-bit paired BER improves only at 20 dB and loses to IDENTITY at 12/15/18 dB; classified Outcome B despite strong analytic/reproducibility evidence.
- Final full-information discrepancy is 1.22e-15; six required figures were visually inspected and the full suite passes 45 tests.

## 2026-08-18 09:12:00 EDT — Stage 1A.1 forensic primitives

- Added exact receiver-MSE decomposition, per-stream reliability quantities, and frozen representation diagnostics.
- Added generic unit-power constellation handling with existing-compatible QPSK and Gray-coded normalized 16-QAM.
- Verified the standalone HAAR-2 checkpoint exactly equals the selected tensor in the Stage-1A training payload and remains immutable.
- Added seven focused forensic test groups; the complete suite passes 52 tests in 2.23 s.

## 2026-08-18 09:28:00 EDT — Stage 1A.1 frozen forensic diagnosis

- Completed the 512-channel paired QPSK/16-QAM audit in 26.2 s; exact MSE decomposition closes to 1.11e-16.
- Falsified noise enhancement: Learned lowers noise and off-diagonal terms but raises diagonal gain distortion and worsens rare stream tails.
- QPSK negative decision margins worsen significantly at 12/15 dB despite better bulk margins; 16-QAM BER/SER improve significantly at all SNRs.
- Diagnosed a QPSK extreme-tail decision-geometry effect with a partial channel-tail contribution; all seven allowed figures exclude IDENTITY.
- Recommended exact mean MSE plus generic normalized nearest-neighbor margin channel-CVaR for Stage 1A.2; no loss or waveform was trained.
