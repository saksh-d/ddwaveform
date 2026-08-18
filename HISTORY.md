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

## 2026-08-18 09:17:00 EDT — Stage 1A.2 workspace recovery and baseline

- Recovered only the four required ignored Stage-1A checkpoint artifacts from the sibling Conductor worktree.
- Verified all copied SHA-256 hashes; HAAR-2 matches the documented `3e55…baad` checkpoint and exact training-payload tensor.
- Locked the brief interpretation to eight positive-tail candidates plus a separate lambda-zero continuation control.
- Pre-change deterministic suite passes: 52 tests in 2.27 s.

## 2026-08-18 09:20:00 EDT — Stage 1A.2 tail-learning core

- Added exact per-stream communication MSE and deterministic upper-stream CVaR with 7/4-stream tails.
- Added modulation-independent mean-plus-tail objective, paired projected Adam, and 2% feasibility-aware validation checkpointing.
- Added independent Stage-1A.2 seeds and locked calibration/production protocol constants.
- Expanded deterministic suite to 62 passing tests in 2.15 s, including objective identity, finite gradients, unitarity, gate selection, and constellation independence.

## 2026-08-18 09:24:00 EDT — Stage 1A.2 validation-only calibration

- Completed the full eight-candidate positive-tail grid plus matched lambda-zero control from the exact HAAR-2 checkpoint.
- Selected beta 0.95 and lambda 0.40 at step 50 using only frozen-bank mean MSE and stream CVaR 0.95.
- Selected mean MSE increased 1.973% (inside the 2% gate); CVaR 0.95 fell to 0.196725 versus 0.205838 for mean-only continuation.
- All candidates stayed finite and unitary to at most 2.53e-14; the preregistered production stop gate did not trigger.

## 2026-08-18 09:34:00 EDT — Stage 1A.2 production result and verification

- Selected the OFDM-start tail refinement at step 50; pooled held-out mean MSE rose 1.79% while stream CVaR 0.95 fell 21–23% versus frozen Stage 1A.
- QPSK BER improved significantly at all four SNRs, including 19.6%/27.5% reductions at 12/15 dB and a 47.0% reduction at 20 dB.
- Matched HAAR-2 tail refinement significantly beat matched mean-only continuation in QPSK at every SNR; all three starts improved their validation tail.
- 16-QAM remains materially better than IDENTITY but loses some frozen Stage-1A performance; full-information discrepancy is 1.44e-15.
- Final suite passes 62 tests in 1.92 s; seven figures inspected, outputs ignored/untracked, research plan unchanged, and final test bank untouched.

## 2026-08-18 10:49:00 EDT — Stage 2A frozen-evaluation baseline

- Confirmed clean `cambridge` worktree at `origin/main`; Stage 2A is evaluation-only with no optimizer or validation selection authorized.
- Audited unchanged DD-v1 sampling and `Delta(nu) Pi(tau)` ordering plus the frozen K=24 Stage-1A.2 evaluation path.
- Baseline suite reached 61 passes; one checkpoint-preservation test failed only because ignored Stage-1A artifacts were absent from this fresh worktree.
- Located the production artifacts in the sibling source worktree; documented HAAR-2 hash `3e55...baad` and final OFDM-start tail checkpoint hash `31e4...b2cc` before recovery.

## 2026-08-18 11:06:00 EDT — Stage 2A diagnostic core

- Added frozen-checkpoint loading with an exact SHA-256 gate and the fixed Learned/OFDM/OTFS/AFDM/IDENTITY comparison set.
- Added paired QPSK/16-QAM evaluation, analytic and K-sweep summaries, BER bootstrap intervals, median-control channel selection, and basis/operator structural diagnostics.
- Added Stage-2A-only seeds and the preregistered full/BER K grids; no training interface or physical convention changed.
- Focused checks pass 4/4 and the complete unchanged-plus-new suite passes 66/66; diff whitespace check passes.

## 2026-08-18 11:25:00 EDT — Stage 2A frozen qualification result

- Exact 1,024-channel Stage-1A.2 reproduction passed for all reported Learned BER counts and analytic MSE means; checkpoint remained byte-identical and unitary to `2.23e-14`.
- Frozen K sweep shows a useful Learned QPSK advantage across K=20–32, not only K=24; it disappears and reverses for large K despite a persistent analytic-MSE edge through K=48.
- At K=64 aggregate analytic MSE is waveform-invariant while QPSK BER remains strongly waveform-dependent, reaffirming that the full-information invariant is MSE rather than BER.
- Effective Learned operators retain 0.99856 mean energy at K=24 with 0.00518 median limiting distortion, but form dense frequency-group blocks rather than a simple diagonal/band.
- Basis forensics show frequency-ordered sub-populations, moderate spectral localization, and no strong global cyclic-shift, modulation, permutation, or low-order chirp prototype.

## 2026-08-18 11:30:00 EDT — Stage 2A notebook and figure verification

- Full analysis completed in 158.3 s and the clean notebook independently executed top-to-bottom without a cell failure.
- Generated Tables A/B, paired confidence intervals, eight figures, per-channel CSVs, NPZ arrays, and JSON diagnostics under ignored `outputs/stage2a/`.
- Visually inspected every figure; corrected only shared-colorbar/title layout in the two multi-panel operator previews.
- Focused Stage-2A checks remain 4/4; final complete-suite, diff, checkpoint, and repository-boundary checks follow this iteration.

## 2026-08-18 11:35:00 EDT — Stage 2A final preservation audit

- Final complete suite passes 66 tests in 2.55 s; notebook JSON and `git diff --check` pass.
- Confirmed channel, receiver, waveform, and research-plan files are unchanged from `origin/main`.
- Reconfirmed all five recovered checkpoint hashes, ignored-output policy, frozen tensor equality, and absence of any checkpoint overwrite.

## 2026-08-18 11:44:00 EDT — Stage 2B preservation baseline

- Locked Stage 2B to additive diagnostics with no waveform, checkpoint, receiver, or physical-model edits.
- Captured SHA-256 for every Stage-2A artifact; frozen Learned hash remains `31e4...b2cc` and unitarity residual `2.23e-14`.
- Confirmed the project AFDM integer-path law uses offset `(2Nc1 delay + Doppler) mod N`; the tuned baseline has `2Nc1=1`.
- Pre-change complete suite passes 66 tests; channel, receiver, waveform, and research-plan files match `origin/main`.

## 2026-08-18 11:51:00 EDT — Stage 2B forensic-core smoke correction

- Added orientation, dense-DTFT, path decomposition, AFDM-center, ensemble, support, and spectral-community primitives.
- Initial focused run passed three checks; lazy-conjugate NumPy conversion and a k-means distance-axis bug stopped two utility checks before experiment execution.
- Corrected only tensor materialization and deterministic clustering dimensions; scientific conventions and inputs remain unchanged.

## 2026-08-18 12:03:00 EDT — Stage 2B full forensic run

- AFDM integer-path hard gate passed all five offsets exactly with row peaks equal to one numerically.
- Completed 1,024-channel ensemble energy/support analysis and all controlled path galleries in 5.9 s.
- Unsupervised coupling selected three Learned groups of sizes 20/21/23; 96.46% of energy is within groups and spectral/operator rank association is 0.950.
- Visual QA found shared-colorbar overlap in three figure templates; corrected layout only and retained all numerical outputs.

### 2026-08-18 12:17 EDT — Stage 2B final verification

- Executed the diagnostic notebook top-to-bottom and visually rechecked every figure after colorbar/layout corrections.
- Final suite passes: 71 tests; repository diff whitespace check also passes.
- Confirmed all 23 Stage 2A artifacts remain byte-for-byte unchanged and the frozen checkpoint retains SHA-256 `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`.
- Outputs remain ignored; protected baseline files unchanged; no commit, push, training, or checkpoint mutation performed.

### 2026-08-18 12:31 EDT — Stage 2C preservation and protocol audit

- Locked Stage 2C to nine raw-start runs: OFDM plus Haar seeds 11012/11013 at K=16/24/32, with identical online DD-v1/SNR streams.
- Retained the exact beta 0.95, lambda 0.40 objective and per-run Stage-1A.2 validation rule; neither clustering nor BER can affect selection.
- Verified the authoritative checkpoint hash `31e4...b2cc`, protected physical/receiver modules, and captured hashes for all 51 Stage-2A/2B artifacts.
- Raw-start validation confirmed materially different initial scales, requiring the existing 2% mean-MSE gate to be applied relative to each independent start.

### 2026-08-18 12:46 EDT — Stage 2C additive core

- Added raw-start multi-budget training with the exact Stage-1A.2 loss, projected-Adam schedule, independent per-run feasibility gates, and validation-only selection.
- Added all-run unconstrained clustering, selected operator/support/spectral maps, matched QPSK/16-QAM BER, and analytic/BER cross-budget evaluation.
- Stage-2C communication figures reuse exact Stage-2A numerical tables; no legacy Monte Carlo is rerun.
- Five focused checks pass in 1.78 s, including one-step projected training, deterministic starts, structural accounting, and the authoritative checkpoint hash.

### 2026-08-18 12:48 EDT — Stage 2C production launch correction

- Direct script invocation stopped before training because the repository root was absent from Python's import path.
- No training checkpoint or scientific result was produced by the failed launch.
- Switched to the established module invocation under the `pytorch` environment; experiment protocol is unchanged.

### 2026-08-18 13:03 EDT — Stage 2C production and controlled evaluation

- Completed nine 1,000-step runs in 371.1 s; all stayed finite and unitary below 2.52e-14, with OFDM selected by validation at every K.
- Unconstrained counts are K16 6/2/3, K24 5/2/4, and K32 3/2/2 across OFDM/Haar-1/Haar-2; no start follows the predicted 4/3/2 law.
- Selected OFDM-start counts fall 6/5/3 with 91–93% within-group energy, but spectral/operator ARI ranges 0.273–0.922.
- Same-seed K16 retraining reduces 18 dB QPSK/16-QAM BER by 47.6%/42.3% versus the frozen K24 waveform; K32 changes are modulation-dependent.

### 2026-08-18 13:10 EDT — Stage 2C interpretation and visual QA

- Qualified Outcome C: partitions vary by initialization, but worse Haar objectives do not support communication-equivalent symmetry solutions.
- Corrected overlapping shared colorbars and a low-BER heatmap annotation; numerical results and selections unchanged.
- Added same-seed frozen-versus-retrained tables and the repository-convention Stage-2C notebook.
- Stage-2A/2B artifacts and the authoritative checkpoint remained byte-identical through the 159.3 s evaluation rerun.

### 2026-08-18 13:16 EDT — Stage 2C final verification

- Notebook executed top-to-bottom with zero error outputs; cleared embedded outputs afterward to retain the lightweight repository convention.
- Final suite passes 76 tests in 2.19 s and repository diff whitespace checks pass.
- Rechecked all 51 Stage-2A/2B hashes, frozen checkpoint `31e4...b2cc`, and protected channel/receiver/waveform/research-plan files.
- Final classification remains qualified Outcome C; no commit, push, merge, P-variation, formula fitting, or Stage 3 action occurred.

### 2026-08-18 13:32 EDT — Stage 2D preservation baseline

- Continued in the clean `moroni` worktree at `origin/main`; the brief's `cambridge` reference was a stale workspace name.
- Recovered only ignored Stage-1A/1A.2 and Stage-2A/2B/2C artifacts from the sibling worktree for frozen evaluation.
- Authoritative Stage-1A.2 SHA-256 matches `31e42c4...b2cc`; all nine Stage-2C best checkpoints plus three training payloads are present.
- Pre-change complete suite passes 76 tests in 2.71 s; no training, commit, push, or protected-module change occurred.

### 2026-08-18 13:43 EDT — Stage 2D diagnostic core

- Added frozen empirical `C`, Sobol single-path `Q`, spectral `S`, continuous-locality, and fixed-support primitives.
- Added deterministic Q/S/centroid/prior masks that use actual coefficients only after positions are fixed.
- Added off-diagonal and row-wise agreement metrics plus paired QPSK/16-QAM support-receiver evaluation.
- Four focused checks pass; complete suite passes 80 tests in 1.96 s.

### 2026-08-18 14:02 EDT — Stage 2D production runner

- Added three disjoint banks, 2,048-point Sobol ambiguity integration, and full checkpoint/preserved-artifact gates.
- Added oracle/Q/S/centroid/prior structural, analytic, QPSK, and 16-QAM comparisons plus all-checkpoint exploration.
- Added eleven publication figures and tracked report generation; legacy communication curves reuse frozen Stage-2A tables.
- Focused Stage-2D checks and Python compilation pass; diff whitespace check passes before production launch.

### 2026-08-18 14:17 EDT — Stage 2D production result

- Completed 2,048-point QMC ambiguity, 512-channel structural/prior/test banks, and million-bit paired BER in 97.5 s; Outcome A.
- Learned Q-to-C off-diagonal Spearman is 0.9998; fixed-Q support retains 99.809% energy with 96.35% oracle-support recall.
- At 18 dB, Learned fixed-Q QPSK BER is 0.001120 versus 0.001045 oracle and 0.005167 for the best classical fixed-Q baseline.
- Learned fixed-Q 16-QAM BER is 0.02485 versus 0.02426 oracle; all nine checkpoints confirm Q-to-C structure, while weak Haar runs have diffuse support and large BER penalties.

### 2026-08-18 14:27 EDT — Stage 2D notebook and visual QA

- Added the frozen evaluation notebook and executed all 10 cells with zero error outputs.
- Inspected all eleven required figures; corrected only axes/title/legend spacing and CSV-reload typing in plotting code.
- Final figures connect frozen K=24 communication, continuous DD locality, fixed-support removal, and checkpoint-level communication evidence.
- No numerical table, checkpoint, prior artifact, or scientific classification changed during layout corrections.

### 2026-08-18 14:31 EDT — Stage 2D final preservation audit

- Final complete suite passes 80 tests in 1.91 s; notebook has 10 cells and zero execution errors; diff whitespace check passes.
- Authoritative and all nine Stage-2C checkpoint hashes match; maximum learned unitarity residual remains 2.29e-14.
- All 88 Stage-2A/2B/2C artifacts match the source worktree byte-for-byte; Stage-2D outputs remain ignored.
- Channel, receiver, waveform, and research-plan files are unchanged from `origin/main`; no commit, push, merge, training, or Stage 3 action occurred.
