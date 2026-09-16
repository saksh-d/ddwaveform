# Project History

## 2026-09-16 12:00 EDT — ICC result summary audit

- Consolidated production nominal, unchanged-support stress, and five-checkpoint reproducibility CSVs into exact BER/count/energy/MSE summaries.
- Confirmed 12,288 nominal, 8,192 stress, and 5,120 reproducibility rows; per-channel error range, mean, median, and zero-error counts were independently recomputed.
- Preserved the interpretation boundary: Learned leads fractional stress conditions at fixed nominal support, while integer/full-information controls favor classical waveforms; no confidence intervals inferred.

## 2026-09-16 11:47 EDT — ICC N=64 production confirmation complete

- Completed the fresh campaign in 9.29 s: 12,288 nominal, 8,192 stress, and 5,120 reproducibility rows; all 25,600 rows retain raw counts, energy, and analytic MSE.
- Nominal Learned fixed/realized BER is `0.001163/0.001113` QPSK and `0.024457/0.023884` 16-QAM; full-information controls favor classical OTFS/AFDM and reject universal superiority.
- Unchanged nominal Learned support is descriptively best across wider-delay, wider-Doppler, and five-path conditions; classical waveforms lead the exact integer-grid control.
- Five fixed-support tail checkpoints span `0.000839-0.001148` QPSK and `0.025213-0.025780` 16-QAM on the identical 512-channel bank.
- Independent aggregation, exact schemas, source/artifact hashes, K semantics, full-MSE invariance, compilation, and focused tests pass; no confidence intervals were inferred.

## 2026-09-16 11:46 EDT — ICC smoke provenance completion

- Added tensor hashes for Learned, OFDM, OTFS, and slope-3 AFDM to the generated runtime config.
- Re-ran the focused suite and complete reduced campaign; 6/6 tests and every evaluation path pass.
- Reconfirmed exact CSV schemas and checkpoint provenance; code whitespace validation passes.
- Production is authorized only against the now-frozen config/runner/core hashes captured at launch.

## 2026-09-16 11:44 EDT — ICC confirmation end-to-end smoke

- Exercised authoritative, four-waveform, four-stress, and five-seed paths through direct complex128 LMMSE without retraining.
- Generated all seven requested smoke data/config files plus a hash manifest: 96 nominal, 96 stress, and 40 reproducibility per-channel rows.
- Verified exact requested per-channel headers, raw count aggregation, seed ledger, and six live checkpoint hashes.
- Smoke completed in 0.10 s at reduced channel counts; values are integration-only and carry no scientific interpretation.

## 2026-09-16 11:42 EDT — ICC input and focused-test gate

- Restored only the authoritative N=64 checkpoint and five selected Pub-1F tail checkpoints from the preserved producer workspace.
- Verified all six file hashes exactly, including authoritative `31e42c...b2cc`; no checkpoint was modified.
- Focused channel, support, receiver, summary, and frozen-config tests pass 6/6 in the `pytorch` environment.
- Scientific production remains unstarted pending end-to-end smoke validation.

## 2026-09-16 11:40 EDT — ICC N=64 confirmation implementation

- Froze a fresh `icc27-final-v1` evaluation-only campaign at N=64, K=24, 18 dB, QPSK/16-QAM; no training or test-BER receiver selection is exposed.
- Added nominal fixed/realized/full receivers, unchanged-support delay/Doppler/path/integer stress controls, and common-bank five-checkpoint reproducibility.
- Added exact per-channel and aggregate CSV contracts with raw bit counts, retained energy, analytic MSE, provenance, seeds, and hashes.
- Python compilation and whitespace validation pass; production inputs remain hash-gated before smoke or scientific execution.

## 2026-09-06 20:00 EDT — Pub-1F five-seed training complete

- Completed five independent 2,000-step OFDM-start parents plus matched 1,000-step mean/tail continuations in 786 s aggregate recorded training time; no non-finite state occurred.
- Parent validation selected steps 275-1375; every mean-only continuation selected the cloned parent-equivalent step 0, while tail selected step 975/1000.
- All five tail arms reduced validation stream CVaR from `0.1713-0.1943` to `0.1392-0.1639` while remaining inside the seed-relative 2% mean-MSE gate.
- Preserved every selected checkpoint, fixed endpoint, optimizer state, trace, tensor hash, and parent-equality record; communication reproducibility remains unevaluated at this boundary.

## 2026-09-06 19:47 EDT — Pub-1F continuation end-to-end smoke

- Added separate authorization/reuse, resumable training, waveform-specific statistics, receiver/operating validation, fresh communication, matched-target, scaling, figure, report, and manifest phases.
- Smoke exercised five seeds, both continuation arms, oracle/fixed supports, residual statistics, direct/dense-CG paths, target confirmation, and N=64 scaling without promoting smoke state.
- Added six continuation figure families with saved-table replotting and analytical N=512/1024 rows; original gate remains byte-preserved and failed.
- Continuation-focused invariants pass 13/13, including namespace separation, direct/CG equivalence, fixed-support covariance, exact scaling ledgers, and live checkpoint hashes.

## 2026-09-06 19:18 EDT — Pub-1F continuation amendment

- Preserved the original failed gate and froze a distinct user-authorized continuation; primary scope is fractional propagation plus limited interactions, with 16-QAM primary and QPSK/integer/full-A retained as non-universal controls.
- Authorized five-seed replication, matched objective ablation, matched-quality cost, and N=64/128/256 measured scaling without changing the historical gate record.
- Verified live N=128/N=256 checkpoint hashes `3254...a3a1` and `1338...c5d`; retained Pub-1B normalized channel law, OTFS grids, and dimension-specific AFDM provenance.
- Froze fresh continuation banks, target/selection rules, scaling caps/workloads, separate representation/compute claims, and a no-retraining rule for N=128/256.

## 2026-09-06 19:05 EDT — Pub-1F early-gate production result

- Completed fresh 256/1,024/512-channel validation/statistics/test receiver qualification in 128.7 s; slope-1 residual-aware AFDM was the favorable operational choice.
- Gate failed only favorable QPSK Learned-AFDM: BER `0.001204` versus `0.001461`, paired difference CI `[-5.43e-4, 2.48e-5]`; all favorable 16-QAM and OTFS/OFDM contrasts supported Learned.
- Stopped five-seed production training per protocol; preserved two-step five-seed smoke solely as integration evidence and reported replication as an explicit follow-up decision.
- Audited 64-step sparse cost/storage and one-thread latency: K=8 saves 83.6% modeled MACs versus K=64 but only 22.1% sparse latency; sparse CG remained slower than dense direct and had 98/768 nonconvergences.
- Completed setup/amortization ledgers, three visually checked figure families, comprehensive report, seed/provenance and artifact manifests; all protected hashes remained unchanged.

## 2026-09-06 18:42 EDT — Pub-1F baseline and training smoke

- Added semantic role-keyed banks, slope-1-inclusive receiver validation, paired early-gate evaluation, and resumable five-seed parent/continuation machinery.
- Added direct receiver, residual-statistics, sparse-CG correctness, operation/storage, timing, plotting, report, and manifest paths; smoke traversed all implemented phases.
- Pub-1F invariants plus Pub-1B/Pub-1E regressions pass 39/39; smoke sizes were 6-8 channels and 256 bits/channel and carry no scientific interpretation.
- Corrected a seed-field namespace collision and timing callback return defect found by smoke; historical checkpoint remains protected.

## 2026-09-06 18:20 EDT — Pub-1F prerequisite and protocol freeze

- Verified Denpasar's corrected 48-file Pub-1E package, active/superseded figure split, handoffs, and exact stream derivations; restored only manifest-backed ignored inputs.
- Reconfirmed protected Learned checkpoint `31e42c4...b2cc`, Pub-1E config `2ab9c3...3488`, and all four protected raw-table hashes before Pub-1F work.
- Froze five-seed OFDM-start replication, matched fresh-stream continuations, per-waveform support/statistics, slope-1-inclusive receiver validation, and direct-solve authority.
- Predeclared a paired joint-(0.5,0.5) early gate plus 18 dB BER targets; expensive training remains contingent on the historical-checkpoint gate.

## 2026-09-05 19:02 EDT — Pub-1E production qualification complete

- Completed frozen production: E1 424.4 s, E2 19.9 s, E3 74.6 s, E4 63.7 s; 512-channel confirmatory and 128-channel exploratory sizes remained unchanged.
- Corrected E1 shows off-grid gain shrinking across historical, tuned-ridge, and residual-aware receivers; nominal QPSK becomes inconclusive under residual awareness, while joint-fractionality gains survive.
- E2 localizes gains to joint fractionality; isolated axes retain losses. E3 finds partial support-recalibration recovery under doubled delay/Doppler spread despite high retained energy.
- Locked E4 confirms favorable/boundary gains and integer/full-information reversals; recommendation is a narrower claim, with Pub-1F not executed.
- Final figures pass visual QA; authoritative checkpoint hash remains `31e42c4...b2cc`.

## 2026-09-05 18:59 EDT — Protected prerequisite restoration

- Full suite initially reached 133 passes with six failures caused only by absent ignored historical artifacts.
- Rejected later-augmented Los Angeles Pub package copies recoverably; restored authoritative Pub-1A/Pub-1B trees from clean stage producers.
- Restored exact Stage-1A, Stage-2C, and Stage-2E prerequisites with verified source/destination hashes.
- Complete repository suite passes 139/139 after restoration; protected Pub-1A tree hash is `a127...2344`.

## 2026-09-05 18:48 EDT — E1 receiver-label audit correction

- Report audit found the validation-selected ridge map was passed to both historical and tuned E1 calls, making their saved labels identical.
- Corrected historical evaluation to force the original unit regularization; tuned scaling now applies only to receiver 2.
- Added a regression gate proving supplied tuning maps cannot affect historical receiver rows.
- E2–E4 are unaffected; E1 production artifacts are being regenerated before interpretation.

## 2026-09-05 18:15 EDT — Pub-1E protocol and baseline freeze

- Confirmed `HEAD` equals `origin/pub`; Pub-1E remains additive and evaluation-only on the frozen N=64 waveform.
- Baseline suite: 118 passed, 11 failed only because this fresh worktree lacks ignored historical checkpoints/Pub artifacts.
- Froze SNR, K, bank separation, pairing, AFDM validation, receiver, mismatch, uncertainty, and confirmatory rules before new test evaluation.
- Derived residual-aware receiver as a separately banked mean/covariance approximation; explicitly not conditional-optimal because realized support/residual dependence is neglected.

## 2026-09-05 18:23 EDT — Pub-1E evaluation core and resumable runner

- Added frozen 2-D fractionality and one-factor prior-shift banks, waveform-specific oracle/fixed supports, receiver ladder, channel-paired BER counts, and channel-bootstrap utilities.
- Added phase-selectable provenance, validation, E1–E4, saved-data plotting, report, Pub-1F handoff, and artifact-manifest flow; no optimizer or training interface exists.
- Scientific gates cover full-A observation semantics, deterministic pairing, exact row budgets, receiver direct references, residual covariance PSD, and K=N collapse.
- Focused Pub-1E suite passes 9/9; Python compilation and diff whitespace checks pass.

## 2026-09-05 18:29 EDT — Pub-1E end-to-end smoke

- Complete reduced-shape E1–E4 path, saved-data replot, report guard, handoff guard, and manifest completed without error.
- Measured smoke times: E1 7.6 s, E2 1.1 s, E3 1.0 s, E4 1.0 s; production remains at the frozen bank/bit sizes.
- Added paired channel-bootstrap tables for receiver and support comparisons; E3 now shares base randomness across mismatch severity.
- Smoke report is explicitly non-scientific and no Pub-1F machine handoff is emitted until production completes.

## 2026-09-05 18:38 EDT — Pub-1E validation and E1 production

- Production AFDM validation selected canonical `2Nc1=3` over guarded candidates 5/7/9 using 25,600 validation channel-condition scores; no BER/test/Learned result entered selection.
- E1 completed 512 paired test channels with separate 256-channel receiver tuning and 1,024-channel residual-statistics banks at the frozen K/SNR/modulation grid.
- Saved 430,080 per-channel rows plus 324 paired channel-bootstrap contrasts; direct complex128 runtime was 401.2 s.
- Authoritative checkpoint remains byte-identical at SHA-256 `31e42c4...b2cc`.

## 2026-09-05 18:18 EDT — Pub-1E scientific core iteration

- Added deterministic 2-D anchored fractionality and one-factor delay/Doppler/power-prior shift banks with unchanged normalization.
- Added per-waveform fixed supports, tuned-ridge selection, residual-statistics receiver, full-observation communication, paired channel intervals, and finite zero-error ratios.
- Restored only the verified authoritative checkpoint from the clean Stage-1A.2 producer; SHA-256 matches `31e42c4...b2cc`.
- Initial focused run reached 8/9; sole failure was an assertion-only float32/float64 expected-tensor mismatch, now corrected without changing scientific math.

## 2026-08-31 23:02 EDT — Pub-1C P4 dual-render preservation

- Preserved both P4 color renderings for N=64/N=128/N=256 under `outputs/stage_pub1c/figures/`: current `viridis` and prior-style `_magma_legacy` PNG/PDF pairs.
- Legacy copies use the same saved thresholded-energy maps and the former no-colorbar layout; no statistic, checkpoint, or evaluation input changed.
- Regenerated the Pub-1C manifest and confirmed focused tests pass 13/13; N=128/N=256 checkpoint hashes remain unchanged.

## 2026-08-31 22:54 EDT — Pub-1C P4 colorway alignment

- Re-rendered N=64/N=128/N=256 P4 threshold graphs with the shared purple-to-yellow `viridis` colorway used by the common-scale effective-A figures.
- Retained P4's thresholded ensemble-energy statistic and common `[-50, 0]` dB scale; added a labeled shared colorbar without altering structural data or checkpoints.
- Visual inspection passed after dedicated shared-colorbar layout; Pub-1C focused tests pass 13/13.

## 2026-08-31 22:50 EDT — Pub-1C integration and verification

- Integrated the N-scaled threshold graphs as the Pub-1C `scaling-structure` phase; figures, data, provenance, report, and manifest now live under `outputs/stage_pub1c/`.
- Removed the mistaken Pub-1D-named runner/test; its ten transient ignored artifacts were moved recoverably to `.context/obsolete-stage_pub1d-20260831/`.
- Regenerated N=128/N=256 1,024-channel outputs and visual figure pairs; Pub-1C focused tests pass 12/12 and protected scaling-checkpoint hashes are unchanged.

## 2026-08-31 22:40 EDT — Pub-1C N-scaled P4 structural extension

- Added Pub-1C evaluation-only N=128/N=256 P4-style threshold-graph phase using preserved Pub-1B Learned checkpoints and the frozen normalized scaling channel law; no training or checkpoint mutation.
- Matched the N=64 P4 rho grid exactly: `K/N=0.125, 0.1875, ..., 1`, with one unthresholded 1,024-channel ensemble energy map per dimension.
- N=128 stabilizes at three spectral communities from rho `0.1875` (within energy `0.9612`); N=256 has four across the tested grid (rho `0.375`: within energy `0.9379`).
- Focused P4 extension plus Pub-1C tests pass 12/12; N=128/N=256 checkpoint SHA-256 values remain `3254...a3a1` and `1338...b2cc`.

## 2026-08-24 17:31 EDT — Pub-1B final verification

- Compute and scaling claims independently classify `SUPPORTED`; combined report, re-plottable tables, provenance, notebook, and 9 PNG/PDF figure pairs are complete under ignored `outputs/stage_pub1b/`.
- Visually inspected B1/B3/B4 and S1–S4; layouts are readable, scaling panels remain separated by dimension/modulation, and zero/low-error ratios use the declared fallback.
- Final focused/complete suites pass 18/116; notebook executes, Python compilation and `git diff --check` pass.
- N=64 checkpoint and all 71 Pub-1A artifacts retain hashes `31e4...b2cc` and tree `a127...2344`; 50 Pub-1B artifacts pass their manifest, smoke checkpoints were removed, and outputs remain ignored.

## 2026-08-24 17:28 EDT — N256 production result and scaling claim

- Completed the exact 2,000-step mean and 1,000-step tail replica in 2,431.5 s; validation selected mean step 1025 and tail step 1000 with no non-finite state.
- Learned-N256 is unitary to `6.44e-14`; tail CVaR fell from `0.14726` to `0.12551` with a 0.77% mean-MSE increase inside the fixed gate.
- AFDM slopes 7/9 passed predicted offsets; AFDM-only validation selected slope 7 (`0.51213` versus `0.55888`).
- At K=96/18 dB, Learned BER is `0.000687/0.03569` versus best classical `0.05432/0.15995`; all gates pass and scaling claim classifies `SUPPORTED`.

## 2026-08-24 16:44 EDT — N128 production result and hard gate

- Completed the exact OFDM-start 2,000-step mean stage and 1,000-step tail refinement in 542.7 s; validation selected mean step 900 and tail step 1000 with no non-finite state.
- Learned-N128 is unitary to `3.79e-14`; tail validation CVaR fell from `0.16664` to `0.13403` with mean MSE `0.07832`, inside the frozen 2% gate.
- AFDM slopes 5/7 passed all predicted offsets; AFDM-only validation selected slope 5 (`0.17087` versus `0.21880`).
- At K=48/18 dB, Learned BER is `0.000954/0.03259` versus best classical `0.009480/0.08944` for QPSK/16-QAM; every N128 hard gate passed.

## 2026-08-24 16:38 EDT — N128 execution-policy corrections

- Initial launch exposed the one-thread runtime policy unintentionally covering training; interrupted after preserving resumable validation-selected state, before test BER.
- Scoped single-thread control to runtime benchmarking and resumed exact optimizer state under normal CPU parallelism.
- Increased exact validation chunks from 4 to 32 channels at N=128 and 8 at N=256 while retaining all channels, SNRs, values, cadence, and selection.

## 2026-08-24 16:34 EDT — Pub-1B production compute result

- Tight numerical gate passed nine Learned/OTFS/AFDM systems: maximum solution error `1.90e-12`, residual `9.64e-13`, exact K=24 nnz `1536`.
- Completed 792 paired operating points over K=8–64 and 1–64 CG iterations with 262,144 bits per modulation; saved disaggregated MACs, residuals, convergence, direct references, and controlled runtimes.
- At K=24/16 iterations, Learned BER is `0.000977/0.02271/0.03464` for QPSK/16-QAM/64-QAM versus best classical `0.01123/0.09244/0.18155` at identical `56,896` mean total MACs.
- Learned has multiple strict common-or-lower-MAC Pareto wins in all three modulations; compute claim classified `SUPPORTED` under the predeclared rule.

## 2026-08-24 16:29 EDT — Pub-1B end-to-end smoke

- Added resumable compute validation/run/plot and scaling config/N128/N256/plot/report phases with a literal persisted N128 gate.
- Reused exact Pub-1A N=64 banks and support construction; scaling training mirrors OFDM-start mean-MSE then fixed tail-aware projected optimization.
- Complete reduced-shape `--phase all --smoke` executed both claims, all figures/data/report paths, and N=128/N=256 checkpoint flows without error.
- Smoke gate outcomes are integration-only and will be overwritten by production results; no scientific claim is taken from them.

## 2026-08-24 16:25 EDT — Pub-1B numerical core smoke

- Added exact-K sparse row storage and matrix-free complex128 CG for the unchanged mismatched LMMSE normal system; measured path forms no dense Gram matrix.
- Defined explicit sparse, vector/dot, preconditioner, and setup complex-MAC categories with analytic toy-count checks.
- Froze normalized DD support scaling and deterministic OTFS grids; added N=128/256 channel, AFDM, unitarity, reconstruction, and full-MSE gates.
- First focused run passed 13 scientific checks; one assertion-only float32/float64 expected-scalar mismatch was corrected without changing implementation math.

## 2026-08-24 16:16 EDT — Pub-1B provenance and preservation gate

- Started dedicated `stage-pub-1b` at Pub-1A parent `737c720`, exactly matching `origin/pub`; worktree was clean and publication lineage remains separate from `origin/main`.
- Verified the clean Pub-1A producer tree is content-identical to the merged parent and restored its 71-file result package plus its exact Stage-1A/Stage-2E prerequisites.
- Reconfirmed checkpoint SHA-256 `31e42c4...b2cc`; every restored Pub-1A artifact and prerequisite matches the producer byte-for-byte.
- Locked scope to common sparse CG compute validation and N=128/N=256 scaling replicas, with the N=128 hard gate and no Stage-3/4 interaction.

## 2026-08-24 — Pub-1A PR preparation

- Focused Pub-1A regression suite passes: 10 tests in 1.35 s.
- Reviewed the scoped source, tests, runner, notebook, README, and history changes for publication.
- Generated outputs and Python caches remain ignored and unstaged.

## 2026-08-24 12:43 EDT — Pub-1A provenance and scope baseline

- Started independent `stage-pub-1a` at `6d99f42`, exactly matching `origin/pub`; branch is 1 commit ahead and 8 behind current `origin/main` by identity.
- Locked Pub-1A to frozen N=64 evaluation: no Learned training, no Stage 3/4 changes, new outputs only under ignored `outputs/stage_pub1a/`.
- Authoritative checkpoint was absent locally; verified producing Stage-2E worktree copy has required SHA-256 `31e42c4...b2cc` and 78 reusable Stage-2E files.
- AFDM candidate remains Stage-2E slope `2Nc1=1`, with existing `c2=sqrt(2)/(100N)`; structural gate and validation-only audit precede BER.

## 2026-08-24 13:06 EDT — Pub-1A core and AFDM smoke gate

- Added frozen Pub-1A core, normalized Gray 64-QAM, centralized regimes, exact fixed statistical supports, paired communication, analytic MSE, and practicality utilities.
- AFDM smoke gate passed for admissible slopes 3/5; all four predicted grid offsets matched and stayed distinct, with validation-only smoke selecting slope 3.
- Added resumable phase runner, PNG/PDF publication figures, saved-data report/notebook generation, and nine focused preservation/normalization/pairing gates.
- Focused suite passes 9 tests; complete suite passes 97 tests after restoring verified ignored Stage-1A prerequisites (`HAAR-2 3e55...baad`, authoritative `31e4...b2cc`).

## 2026-08-24 13:42 EDT — Pub-1A production result and visual QA

- Production AFDM gate selected admissible slope `2Nc1=3` by validation MSE `0.126729` versus `0.143175` for guarded slope 5; all four grid offsets were exact/distinct.
- At fixed K=24, classical waveforms win GRID-DD, while Learned wins QPSK/16-QAM/64-QAM representative points in fractional-Doppler, fractional-delay, and DD-v1; DD-v1 BER is `0.001260/0.02501/0.03597`.
- Dense K sweep localizes Learned's advantage to constrained reception; full K=64 returns QPSK/16-QAM leadership to classical waveforms, while 64-QAM remains a finite-test exception.
- Generated 20 PNG/PDF figure pairs, machine-readable raw counts/manifest, saved-result notebook, and detailed `SUPPORTED` report; inspected communication, operator, waveform, stress, and practicality figures after colorbar repair.
- Final focused/complete suites pass 9/97; notebook executes, diff check passes, checkpoint hash and all 78 copied Stage-2E artifacts remain byte-identical.

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

## 2026-08-19 19:22 EDT — Stage 2E preservation baseline

- Confirmed merged Stage 2D commit `9b78e6b` is both `HEAD` and `origin/main`; current isolated branch/worktree is clean.
- Verified Stage 2A–2D tracked code, reports, notebooks, and tests are present; historical outputs are ignored and absent in this fresh worktree.
- Baseline suite reached 78 passes; two preservation tests failed only because ignored Stage-1A/1A.2 checkpoint prerequisites were not copied into the workspace.
- Located three byte-identical authoritative checkpoints with expected SHA-256 `31e4...b2cc`; no scientific experiment ran.

## 2026-08-19 19:25 EDT — Stage 2E ignored-artifact recovery

- Recovered prior ignored Stage-1A/1A.2 prerequisites and all available Stage-2A–2D outputs from the merged Stage-2D worktree.
- Authoritative Learned checkpoint hash is exactly `31e4...b2cc`; HAAR-2 and its training payload match the documented hashes.
- Complete pre-change suite now passes 80 tests in 2.32 s; generated outputs remain ignored and only HISTORY is tracked as modified.
- Established a 110-file Stage-2A–2D preservation set for before/after production comparison.

## 2026-08-19 19:39 EDT — Stage 2E forensic-core first iteration

- Derived DD-v1 AFDM integer support from nearest-integer Doppler decomposition: alpha-max 1, integer canonical slope 3, guarded slope 5.
- Added deterministic stress regimes, constrained AFDM validation sweep, fixed-Q receiver evaluation, path/spectral/time/ambiguity/DD-map/PAPR primitives, and simple explanatory regression.
- Focused suite reached 6 passes; one sampler assertion failed only from float64-versus-default-float32 expected dtype.
- Corrected the test tensor dtype without changing sampler normalization or scientific behavior.

## 2026-08-19 20:03 EDT — Stage 2E production runner and test closure

- Added one evaluation-only runner covering AFDM/OTFS audits, nominal full/fixed-Q/oracle receivers, seven frozen stress regimes, and N=256 reference behavior.
- Added all-atom spectrum/time/STFT, auto/cross-ambiguity, DD concentration, explanatory coupling, PAPR, and consistent operator-regeneration outputs.
- Focused Stage-2E tests pass 7/7 and the complete suite passes 87/87 in 2.00 s.
- Pre-smoke inspection found the N=256 reference path inherited global N=64 symbol shapes; made BER dimensions explicit from the supplied channel without changing N=64 behavior.

## 2026-08-19 20:05 EDT — Stage 2E smoke forensic-array correction

- Smoke completed AFDM/OTFS, nominal, N=256, and seven-regime stress phases before stopping during physical-array serialization.
- Failure was PyTorch's lazy-conjugate NumPy guard on the Learned phase-increment diagnostic; no scientific result or protected artifact changed.
- Materialized the conjugate before NumPy conversion; waveform values and phase definition remain identical.

## 2026-08-19 20:13 EDT — Stage 2E complete smoke and visual QA

- Complete reduced pipeline passed in 18.2 s after 87/87 tests; all preservation and canonical AFDM/OTFS hard gates passed.
- Added explicit per-atom PAPR, random-block PSD/99%-occupied bandwidth, AFDM multipath overlap, easy/median/hard, and S/Q/C outputs.
- Visually inspected all 21 figures in a contact sheet; layouts, axes, orientation, labels, and color scales are readable with no clipping.
- Smoke selected AFDM slope 1 on only 16 validation channels; this is non-scientific and must be replaced by the 256-channel production calibration.

## 2026-08-19 20:18 EDT — Stage 2E production limiting-floor correction

- Full 256-channel validation selected AFDM slope 1 with mean K=24 analytic MSE 0.106405; canonical guarded slope 5 scored 0.146480.
- Production reached stress structural evaluation before one ill-conditioned batched pseudoinverse failed SVD convergence.
- Replaced only the zero-noise floor calculation with an explicitly reported 1e-10 regularized LMMSE limit; BER and support definitions are unchanged.
- Authoritative checkpoint remains exact and no prior artifact changed.

## 2026-08-19 20:25 EDT — Stage 2E production result and reporting audit

- Production completed in 161.4 s; AFDM K=24 validation retained historical slope 1, while canonical guarded slope 5 remained a support-law reference.
- At 18 dB fixed-Q K=24, Learned QPSK BER is 0.001260 versus AFDM 0.005190; 16-QAM is 0.02501 versus 0.06474, with paired CIs excluding zero.
- Frozen stress evaluation shows wider-delay is hardest for Learned; QPSK/16-QAM BER rise about 170%/77% versus the separate stress DD-v1 bank.
- Found isolated historical-AFDM BER used a different SNR-index seed despite selecting the identical waveform; aligned seeds so the true recalibration change is exactly zero.

## 2026-08-19 20:32 EDT — Stage 2E final production verification

- Final production completed in 164.4 s after 88/88 tests; matched historical/principal AFDM BER is exactly 0.00518990 at 18 dB.
- Notebook executed all six cells with zero errors; added stable cell IDs to remove the nbformat forward-compatibility warning.
- Verified all 110 prior Stage-2 artifacts and authoritative checkpoint unchanged, zero Learned training steps, and protected modules byte-identical to main.
- `git diff --check` passes; only six Stage-2E tracked files plus README/HISTORY are modified or added, and all generated outputs remain ignored.

## 2026-08-19 20:36 EDT — Stage 2E canonical-BER scope correction

- Final scope audit found canonical AFDM had analytic/path audits but was absent from the required full-information BER sanity table.
- Added guarded canonical slope 5 as a secondary numerical-only nominal baseline with its own Q masks and paired confidence differences.
- Headline figures and the validation-selected principal AFDM label remain unchanged; no selection or Learned result is affected.

## 2026-08-19 20:41 EDT — Stage 2E canonical-inclusive final run

- Canonical-inclusive production completed in 169.0 s with principal AFDM selection unchanged at slope 1.
- Canonical AFDM now appears in nominal full/fixed-Q tables and paired diagnostics but remains excluded from headline plots.
- Restored append-only HISTORY chronology; final notebook, test, preservation, and diff gates follow this run.

## 2026-08-19 20:49 EDT — Stage 2E machine-readable figure export

- Added a deterministic exporter covering all 21 Stage-2E figures with exact axes, records, series, matrices, units, and provenance.
- Existing CSV/NPZ values are serialized directly; operator galleries are reconstructed from their recorded seeds and unchanged physical conventions.
- Export supports one self-contained `all_figures.json` plus optional per-figure JSON files and a SHA-256 manifest; no pixel digitization or downsampling.

## 2026-08-19 20:54 EDT — Stage 2E concise semantic export

- Added a compact all-figure JSON that retains every communication point and PAPR summary while replacing dense arrays with declared DSP statistics.
- Dense reductions include spectral/time centroids and spreads, STFT marginals, ambiguity peaks/spreads, DD concentration, dominant offsets, path overlap, and Q/S/C correlations.
- The concise file preserves figure-by-figure traceability and explicitly labels every reduction instead of presenting summaries as raw plotted arrays.

## 2026-08-24 13:00 EDT — Pub-1A publication figure refresh

- Added five supplementary C1–C5 BER views using only saved measurements through 20 dB; full-range C1–C5 and C8 remain byte-identical.
- Reworked F1/F2 into four-row OFDM/OTFS/AFDM/Learned layouts with shared matched axes and unchanged spectral data.
- Added the `figure-updates` saved-data-only runner phase and documented the capped views in the report/README.
- Visual QA passed for all seven new/changed PNGs; output set is now 25 PNG/PDF pairs with no simulation, AFDM, or checkpoint change.

## 2026-08-24 13:12 EDT — Pub-1A spectral normalization correction

- Peak-normalized spectra once per waveform for presentation, preserving atom-to-atom peak offsets and the absolute saved DTFT arrays.
- Re-rendered vertical F1/F2 with a 0 dB ceiling and added Learned-only F1L/F2L supplements from saved spectra.
- Numerical checks confirm each waveform maximum is 0 dB with no positive values; focused tests and visual QA passed.
- Output set is now 27 PNG/PDF pairs; communication figures, simulations, AFDM selection, and checkpoint remain untouched.

## 2026-08-24 13:15 EDT — Pub-1A figure-update preservation gate

- Restricted the saved-data figure refresh to spectral figures so communication outputs are never rewritten during layout-only updates.
- Restored five capped-view PDF metadata timestamps after the preservation check exposed non-deterministic renderer metadata.
- Re-running `figure-updates` now leaves every C figure byte-identical while regenerating F1/F2/F1L/F2L.

## 2026-08-29 21:41 EDT — Pub-1C provenance and reuse baseline

- Scoped Phase 1 to frozen N=64 evaluation: per-waveform oracle row-top-K, controlled fractionality, constellations, and separate threshold/training-K forensics; Phase 2 remains excluded.
- Restored verified Pub-1A/Pub-1B prerequisites and three validation-selected Stage-2C OFDM-start K=16/24/32 checkpoints from reviewed producer worktrees.
- Authoritative Learned SHA-256 is `31e42c4...b2cc`; selected K16/K24/K32 hashes agree across three independent worktrees.
- Pre-change suite passes 116 tests; restored outputs remain ignored and the tracked worktree changed only in HISTORY.

## 2026-08-29 21:44 EDT — Pub-1C controlled-axis and graph core

- Added exact paired epsilon sweeps for delay-only, Doppler-only, and joint fractionality within the existing circular DD operator.
- Added per-waveform realization-wise oracle retained-energy accounting over K/N and fixed-map threshold graph diagnostics.
- Added input-hash, exact-offset, monotonic-energy, nested-support, and no-training gates; corrected one test-only ChannelBatch field name.
- Focused/complete suites pass 5/121 tests; no scientific experiment or checkpoint mutation occurred.

## 2026-08-29 21:50 EDT — Pub-1C end-to-end smoke and visual QA

- Added resumable provenance, fractionality, rho-budget, raw-constellation, two-part structure, report, and manifest phases plus a saved-result notebook.
- Complete reduced pipeline generated six PNG/PDF figure pairs and 28 manifested artifacts; all layouts were visually inspected and readable.
- Reused the exact Stage-2B authoritative map: detected groups change 6 to 3 by K=16, then remain 20/21/23 through K=64; no exact K=24-only law was imposed.
- Notebook initially resolved a copied `.context` path incorrectly; upward git-root discovery corrected it, then all six cells and 121 tests passed.

## 2026-08-29 21:53 EDT — Pub-1C production claim and bootstrap gate

- Production completed in 11.1 s with 64 paired communication channels, 524,288 bits per DD-v1 budget point, and 1,024-channel structural banks.
- At joint epsilon 0.5, 16-QAM BER is 0.02150 Learned versus 0.11343 AFDM and 0.13406 OTFS; retained energy is 0.99834 versus 0.95285/0.92917.
- Added 10,000-resample paired channel intervals: Learned-minus-AFDM/OTFS BER upper bounds are -0.08694/-0.10645; energy lower bounds are +0.04212/+0.06410.
- Claim classifies `SUPPORTED`; on-grid and full-K results favor classical methods, preserving the explicit regime-specific boundary.

## 2026-08-29 21:55 EDT — Pub-1C publication-layout refinement

- Re-rendered retained-energy panels on declared informative ranges, making the joint off-grid concentration loss visible without changing saved values.
- Relabeled training-K spectra in normalized frequency and time maps by sample index; full untruncated operator maps remain unchanged.
- Re-ran deterministic production fractionality, budget, and structure phases; numerical results and `SUPPORTED` classification are identical.
- Final visual recheck passed for controlled fractionality, rho budget, and training-K structure figures.

## 2026-08-29 21:57 EDT — Pub-1C final verification

- Final complete suite passes 122 tests; the six-cell production notebook executes with zero errors.
- All 30 generated artifacts pass the SHA-256 manifest; six PNG/PDF figure pairs passed visual QA and outputs remain ignored.
- Authoritative and training-K checkpoints, reused Stage-2B map, protected source modules, and complete Pub-1A/Pub-1B packages remain byte-identical.
- `git diff --check` passes; tracked scope is the additive Pub-1C core, runner, tests, notebook, summary, README, and concise history only.

## 2026-08-29 22:38 EDT — Pub-1C 15 dB constellation core

- Added waveform-invariant physical-median channel selection and balanced per-symbol display sampling for named GRID-DD, fractional-Doppler, and DD-v1 regimes.
- Added an additive raw 16-QAM K=24 phase with one shared 8,192-symbol/noise stream, exact support-count gates, and no post-receiver alignment.
- Saved selected channels, candidate full-information scores, paired inputs, raw estimates, metrics, and channel-law provenance for reproduction.
- Ten focused tests pass, covering selector determinism, legacy regime laws, balanced classes, shared randomness, and no-alignment source gates.

## 2026-08-29 22:40 EDT — Pub-1C 15 dB constellation smoke

- Reduced named-regime phase completed all channel selection, raw receiver, saved-data, and PNG/PDF paths with exact K=24 rows.
- The 3x3 layout, transmitted-symbol coloring, ideal markers, decision boundaries, common axes, EVM, and BER annotations render cleanly.
- Existing P3 data and PNG/PDF hashes remained byte-identical before and after the additive phase.
- Complete suite passes 126 tests; smoke metrics remain integration-only and will be replaced by 256-channel-selection production output.

## 2026-08-29 22:43 EDT — Pub-1C 15 dB constellation production

- Selected physical-median channels 176/11/32 from independent 256-channel GRID-DD/fractional-Doppler/DD-v1 banks using full-information MSE only.
- GRID-DD BER is similar across Learned/OTFS/AFDM at 0.0575/0.0555/0.0535; fractional-Doppler is 0.0589/0.0555/0.0771.
- DD-v1 raw clusters favor Learned: EVM/BER 0.284/0.0500 versus OTFS 0.363/0.0931 and AFDM 0.380/0.0992.
- Production notebook executes seven cells, complete suite passes 126 tests, and the regenerated manifest covers 35 artifacts.

## 2026-08-29 22:45 EDT — Pub-1C 15 dB constellation final verification

- Recomputed all three physical-median selections from saved seeds and confirmed exact indices 176/11/32 without waveform or finite-K input.
- Verified 200 displayed observations for each of 16 symbols, zero plotted clipping, exact K=24 support rows, and paired saved bits/noise.
- All 35 manifest hashes pass; P3 data/figures, four frozen checkpoints, and protected channel/receiver/waveform/learning modules remain unchanged.
- Final PNG/PDF visual QA, notebook, 126 tests, ignored-output policy, and `git diff --check` pass.

## 2026-08-29 23:29 EDT — Pub-1C constellation styling refinement

- Reordered P7 columns to Learned, AFDM, OTFS and reduced headings to waveform names only.
- Replaced transmitted-symbol class colors with one established color per waveform; ideal 16-QAM centers now use larger matching-color crosses.
- Regenerated only P7 from saved production arrays and refreshed the manifest; saved samples and metrics remain byte-identical to the producer workspace.
- Visual QA passes for constellation readability; complete suite passes 126 tests and `git diff --check` passes.

## 2026-08-29 23:38 EDT — Pub-1C compact constellation variant

- Added a separate 2x3 P8 view containing only Integer DD and Off-Grid DD rows with Learned, AFDM, OTFS columns.
- Retained one point color per waveform, changed ideal 16-QAM centers to black crosses, and set both axes to `[-1.5, 1.5]`.
- Regenerated P8 only from byte-identical production samples and metrics; existing P7 PNG/PDF hashes remain unchanged.
- Visual QA passes after increasing inter-column padding; complete suite passes 126 tests and `git diff --check` passes.

## 2026-08-29 23:50 EDT — Pub-1C controlled contrast constellation

- Added P9 at the existing supported 18 dB operating point using paired integer/joint-half-bin channels and no SNR or waveform-outcome search.
- Selected channel 2 as the off-grid physical-MSE median from the established 64-channel bank; the paired integer row reuses its anchors, gains, symbols, and noise.
- Off-grid EVM/BER is 0.208/0.0232 Learned versus 0.392/0.0905 AFDM and 0.369/0.0918 OTFS; the integer control does not favor Learned.
- Saved raw samples, metrics, and selection provenance; visual QA passes, P7/P8 remain byte-identical, all 42 manifest entries verify, and 127 tests pass.

## 2026-08-30 00:06 EDT — Pub-1C PR validation

- Reviewed the complete additive Pub-1C source, tests, report, notebook, README, and history scope against `origin/pub`.
- Executed the seven-cell saved-result notebook to a temporary copy with no errors; no tracked notebook outputs changed.
- Complete suite passes 127 tests, Python compilation passes, and all 42 ignored production artifacts match the saved manifest.
- Generated outputs and caches remain ignored; PR staging is restricted to the reviewed publication deliverables.

## 2026-09-05 22:10 EDT — Pub-1E targeted correction implementation

- Preserved the frozen checkpoint, config, raw E1-E4 tables, prior reports/handoff, manifest, protocol snapshot, and original figures before editing.
- Fixed E1 budget selection to key by waveform; added shared uniqueness-checked selectors across all E1-E4 figures and saved-data-only receiver/AFDM companions.
- Added protocol addendum, correction log, strengthened Pub-1F handoff content, and explicit receiver/AFDM/seed/pairing limitations; no simulation rerun.
- Focused Pub-1E suite passes 14 tests, including raw-table hash locks and plot-key regressions; Python compilation passes.

## 2026-09-05 22:16 EDT — Pub-1E saved-data correction verification

- Regenerated only plots, post hoc slope-1 intervals, reports, handoffs, and manifests from frozen production CSVs; no E1-E4 simulation phase ran.
- Visual QA passes for corrected E1 budget, all-waveform receiver ladder, and named-AFDM companion; superseded E1 PNG/PDF retain original hashes.
- Checkpoint/config and all four per-channel production tables remain byte-identical; correction manifest records before/after provenance and active inventory.
- Complete suite passes 143 tests; Python compilation and `git diff --check` pass.

## 2026-09-05 22:18 EDT — Pub-1E correction final audit

- Reverified 48 artifact-manifest entries with zero mismatches; inventory contains 20 active figure files and three clearly superseded files.
- Confirmed checkpoint/config before-after hashes match and correction metadata records no simulation rerun.
- Complete suite remains 143 passing; Python compilation and diff whitespace validation pass.

## 2026-09-05 22:41 EDT — Pub-1E PR validation

- Reviewed the complete additive Pub-1E source, frozen protocol/config, tests, reports, README, and history against current `origin/pub`.
- Confirmed production artifacts remain ignored and PR staging contains only the 12 intended publication files.
- Complete suite passes 143 tests; Python compilation and cached-diff whitespace validation pass.

## 2026-09-06 20:19 EDT — Pub-1F continuation operating validation

- Completed the frozen production selection grid: 23,920 rows over four channel controls, both modulations, SNR/K sweeps, support policies, receiver variants, and all seed checkpoints.
- Used 256 validation channels and 512 bits/channel only for configuration selection; confirmatory channels remain untouched.
- At the primary validation point, the strongest classical candidate is OFDM at BER 0.032608; AFDM selects slope 1 at 0.040886 and OTFS gives 0.092834.
- Run completed in 2,605.7 s with no non-finite BER rows.

## 2026-09-06 20:39 EDT — Pub-1F continuation confirmation and scaling

- Fresh N=64 confirmation evaluates 910,336 channel/configuration rows; all five tail seeds beat strongest classical OFDM at the frozen joint-fractional 16-QAM point (0.023963–0.024352 versus 0.032200).
- Matched-quality selection confirms 150/150 validation-attained cases on fresh data; two strict-QPSK OFDM cases fail validation, and equal 4,096-MAC dense applications provide no Learned detector-cost saving.
- Measured N=64/128/256 scaling completes in 626.2 s with verified historical checkpoints; Learned has the lowest rho=0.375 16-QAM BER at each measured N under the Pub-1B law.
- Sparse-CG remains 269–718x slower than precomputed dense application at rho=0.375; the fixed-64 nonconvergence rate rises from 115/384 at N=64 to 85/96 at N=256.

## 2026-09-06 21:01 EDT — Pub-1F matched-grid completeness correction

- Corrected the matched-quality search to exercise all frozen convergence tolerances; reran only matched selection/confirmation, with no training or operating-grid change.
- Added validation-selected target searches at N=64/128/256 across the frozen rho, iteration, tolerance, direct, precomputed, and iterative alternatives.
- Fresh scaled confirmation attains 46/46 validation-attained targets; two targets are validation failures and remain absent from confirmation.
- Every selected scaled configuration uses precomputed dense application, so recurring detector costs tie classical alternatives at each N and no Learned computational saving is supported.

## 2026-09-06 21:15 EDT — Pub-1F continuation interpretation and figure QA

- Added 10,000-resample paired channel intervals from saved fresh counts; all five primary 16-QAM tail-versus-OFDM intervals exclude zero in Learned's favor.
- Separated tail-risk success from BER: all five tails improve validation CVaR, while four significantly worsen BER against matched mean-only continuation.
- Expanded F2–F6 with fixed endpoints, full-information controls, measured latency, validation failures, storage, convergence, profiling, and reuse amortization; inspected all seven PNGs.
- Corrected full-run phase ordering and documented the separate amended runner; report classifies representation as supported and computational saving as unsupported.

## 2026-09-06 21:19 EDT — Pub-1F continuation final validation

- Complete repository suite passes 157 tests; targeted Python compilation and diff whitespace checks pass.
- Standalone saved-data replot completes, producing seven PNG/PDF figure pairs without simulation.
- Original failed gate and N=64/128/256 protected checkpoints retain their recorded SHA-256 hashes.
- Production phases, tracked report/protocol/config/code/tests, and ignored artifacts are ready for manifest verification; no commit, push, or PR performed.

## 2026-09-06 21:21 EDT — Pub-1F continuation preservation closeout

- Final production manifest contains 469 entries; independent rehash verifies all 469 with zero mismatches.
- Rechecked 31 inherited Pub-1E/Pub-1B/Stage protected artifacts and directory trees against the original Pub-1F prerequisite ledger; zero mismatches.
- Manifest records the unchanged failed-gate hash and all three unchanged scaling-checkpoint hashes.

## 2026-09-06 21:22 EDT — Pub-1F PR validation

- Reviewed and explicitly staged the complete 16-file Pub-1F package against current `origin/pub`; generated outputs and caches remain ignored.
- Complete suite passes 157 tests; Python compilation and cached-diff whitespace checks pass.
- Reconfirmed the scoped representation conclusion, negative compute result, protected hashes, and 469-entry artifact manifest before publication.
