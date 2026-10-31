# Stage 5A — Learned delay–Doppler representation investigation

**Status: bounded investigation executed; compact fitter unqualified; historical population incomplete.**

## 1. Executive findings

No validated useful compact representation was established. An exact 2,016-rotation Givens reference reconstructs the available N=64 teacher to numerical precision. Independent multiscale mixers improve attained native reconstruction with depth, but the best four-pass fit still has **38.31% relative matrix error**, mean normalized squared complex-DD error **0.24364**, and mean multipath error **0.24302**. Its cached PyTorch implementation is substantially slower than dense multiplication at N=64.

The decisive limitation is numerical: **all 12 cold fits to exactly representable synthetic targets fail the predeclared 1e-4 relative-error gate**. Four deeper fits also fail when initialized only 1e-3 away from their known generating parameters. The data cannot establish that the architecture lacks capacity, or that 1,600 parameters are intrinsically necessary. The supported interpretation is primarily **Outcome E, optimization prevents a capacity conclusion**, with an unfavorable attained fidelity/runtime tradeoff. Outcome A is unsupported; Outcomes B/C are not established jointly across the declared conditions; Outcome D cannot be inferred from an unqualified optimizer; Outcome F is not supported by the attained Learned–Haar difference, but optimal compressibility remains unknown.

The historical teacher is not the strongest full-receiver waveform. At 18 dB both OTFS and AFDM have lower QPSK and 16-QAM BER. The best matrix fit does not establish QPSK preservation at the provisional 10% relative margin. Some less matrix-faithful fits preserve selected communication points, demonstrating why matrix, DD response, BER and implementation must remain distinct.

Only **LEARNED_0** was recovered. Four Stage-4B repeats and Stage-3B Q-only checkpoints remain unavailable after searching the user-specified archive tree. This is a one-teacher exploratory result, not the requested five-teacher/Q-control population study. No missing waveform was silently retrained.

## 2. Prior evidence, provenance and decisions

Initial worktree: clean `e6406f6beaad3130f532f2745f31de52919a559b`, equal to the local `origin/main`; branch initially `phoenix/pasted-text`, subsequently observed as `phoenix/stage-5a`. This agent did not rename it. Historical scientific files were preserved; no push, merge or remote job occurred.

The original outputs were absent. Recovered `learned_w.npy` from the sealed prior `review_bundle.zip` recorded in [archive provenance](../outputs/stage5a/prior_evidence/archive_provenance.json). Its SHA-256 is `11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f`, matching the archived input manifest. This verifies the exported tensor, **not the absent original `.pt` serialization**. The historic report identifies the original checkpoint as `31e42c4d...d5b2cc`; this study does not claim a fresh byte-hash verification of that missing file.

| Prior evidence | Consequence for Stage 5A |
|---|---|
| Stage 4A low-parameter polynomial/chirp/Gabor fits failed | Do not repeat those families. |
| Stage 4B archived five-member comparison | Useful historical reference; unavailable tensors prevent fresh population validation. Inspection of `src/tail_learning.py` confirms `base+step` streams and overlapping Stage-4B seed ranges, so the old five-member independence claim needs qualification. |
| Stage 4C polynomial phase cascades failed; its Arm-M overlap ignores column phases | Use native complex Frobenius loss, with nonzero phase-gradient check. No polynomial-depth rerun. |
| Stage 4D branch-sensitive principal logarithm | No logarithmic truncation. Branch sensitivity is a coordinate/gauge issue, not proof that compact coordinates are impossible. |
| Prior direct phase/IFFT/SU(2) action/adjoint package | Inspected before extending its gate algebra to independently parameterized mixers and batched differentiation. The previous seven-descriptor restriction is absent here. |
| Prior constant-modulus obstruction | Reproduced lower bound 0.4327657 for one phase–Fourier–phase transform. Amplitude-forming mixers are justified. |
| Prior matrix-free full receiver and exact labeled-generator recovery | Reused the physical full-LMMSE identity and identifiability argument; no grouped receiver or Top-K acceptance criterion. |

Used prior implementations/reports are preserved under [prior evidence](../outputs/stage5a/prior_evidence/), including `lead/waveform_construction.py`, `implementation/identifiability.py`, `implementation/REPORT.md`, and `structure/compact_family.py`. Rejected/obsolete archive branches were not promoted as authoritative. [Recovery search](../outputs/stage5a/recovery_search.json) records the missing population.

## 3. Mathematical assessment

With `x=Ws`, `y=Hx+n`, the labeled effective family is

\[
B_W(\tau,\nu)=W^H\Delta(\nu)\Pi(\tau)W,\qquad A_W(H)=W^HHW.
\]

**Exact identifiability.** Suppose unitary `V,W` produce the same complete labeled family. Then `C=VW^H` commutes with every physical atom. The allowed family contains `Delta(0.5)`, with distinct diagonal eigenvalues, and `Pi(1)`, a cyclic shift. Commuting with the former makes `C` diagonal; commuting with the latter forces all diagonal entries equal. Thus `V=exp(i phi) W`. Equality almost everywhere under the continuous atom law extends to these points by continuity. The N=4 numerical commutant check has nullity one; the proof applies to N=64. Matching only magnitudes or Q has much greater ambiguity.

Global phase leaves every labeled B exactly unchanged and preserves the physical matched receiver's distribution under circular Gaussian noise. Right column permutation/phase transforms instead conjugate/relabel B and do not preserve its native labels. Under iid fixed square-QAM symbols, permutations and quarter-turn symbol rotations preserve the constellation distribution with the appropriate known stream/bit mapping; arbitrary column phases generally do not. None was introduced into fitting or communication. Phase/permutation-aligned overlap is descriptive only; its metadata is not deployed.

Exact identifiability does **not** give a stable converse for approximate matching on a finite, distribution-weighted atom bank. For unitary atoms a one-way bound is `d_B <= 4 ||V-W||_F^2/N`; narrow DD support can make some perturbations weakly visible. Operator fitting could therefore have a different approximation tradeoff. Its planned arm was not launched because matrix-fitting qualification failed; the exact commutant result is not used to rule out useful approximate operator fitting. Held-out complex fidelity was still measured directly.

**Representation restriction.** Reverse triangle inequality gives

\[
\frac{\|W-Q\|_F}{\sqrt N}\ge
\frac{\||W|-1/\sqrt N\|_F}{\sqrt N}=0.4327657
\]

for every constant-modulus `Q`, including phase–Fourier–phase transforms and their row/column permutations. This is a necessary lower bound, not an attainable optimum. The best tested mixer fit crosses it slightly, confirming that it can form amplitudes; it remains inaccurate.

**Full reception.** `G_A=W^H G_H W`, where `G_H=(H^H H+sigma² I)^-1 H^H`. Raw mean full-LMMSE MSE and channel singular values are invariant to square-unitary W. Per-stream reliability and fixed-QAM slicing are not invariant. Numerical mean-MSE invariance is verified to `6.15e-16`; it is a receiver check, not compression evidence.

## 4. Structural signatures and ranked hypotheses

| Rank / hypothesis | Supporting / opposing evidence | Cost and minimal test |
|---|---|---|
| 1. Independent carrier mixers capture some teacher organization | `F W` has 40.04% energy within circular bandwidth 3, versus 10.61–10.68% for Haar. Its eight strongest entries per column hold 77.02%, versus 37.38–37.40%; participation mean is 9.89 versus 33.25–33.35. This is target-specific structure, not an exact block law. | O(p N log N) nominal gate work plus one FFT; test nested independent mixers at 1/2/4 passes with Haar and synthetic controls. Executed. |
| 2. Local carrier topology may matter more than blind extra passes | Reversed scales improve the attained four-pass native error from 0.416 to 0.383, but restart spread and failed synthetic recovery prevent a topology-optimality claim. | Same parameter budget and fitting resources; compare only ascending/descending here. Frequency-neighbor topology remains untested. |
| 3. Low numerical displacement rank supplies a compact law | Opposed: cyclic displacement rank for 95% energy is 45/64 for Learned, 44/64 for Haar. Constant-modulus failure also excludes a simple single-FFT phase model. | Cheap displacement singular-value diagnostic, executed; no displacement-rank model fit justified. |

The new evidence concerns one historical matrix. It cannot establish shared operations across five learned solutions, or distinguish communication-trained Learned from Stage-3B Q-only. A single learned–Haar contrast is insufficient for population significance.

## 5. Factorization, exact reference and fitting results

Implemented `W_hat = F^H D L_last ... L_first`. Each layer contains 32 independent two-angle complex SU(2) gates. Six deterministic XOR pair scales form a pass; the second topology reverses scale order. Diagonal D supplies 64 real phases. Native `apply_W` and `apply_W_hermitian` use cached coefficients and index gathers, with no dense matrix needed. Dense materialization is restricted to fitting and diagnostics.

For each of six available targets (Learned 0, two Haar, OFDM, OTFS, AFDM), both topologies, both restarts and all three depths used 250 Adam steps plus up to 200 L-BFGS iterations. Checkpoint selection minimized native squared Frobenius/N, including step zero. A larger model copies the preceding solution and makes added layers identity; every one of 24 nested sequences retained its incumbent. Training/BER was not used to choose any fit. The 72 target fits consumed 80.3 s of recorded numerical fitting time on one CPU thread.

| Nominal real parameters | Passes / mixers | Best attained Learned relative error | Haar 1 | Haar 2 |
|---:|---:|---:|---:|---:|
| 448 | 1 / 192 | 0.987802 | 1.115552 | 1.116455 |
| 832 | 2 / 384 | 0.710912 | 0.915015 | 0.917011 |
| 1,600 | 4 / 768 | 0.383131 | 0.584948 | 0.589091 |

These are minima over the **same two topologies and two restarts**, not capacity optima. Native/global-phase errors are essentially identical. OFDM reaches `1.08e-7`; OTFS `4.36e-6`; AFDM `3.40e-6` within the tested ladder. Classical success does not rescue failed generic synthetic qualification. All unsuccessful runs, gradients, initial/final/best losses and timings remain in [fit results](../outputs/stage5a/fit_results.csv) and [traces](../outputs/stage5a/traces/).

![Attained reconstruction versus nominal parameter count](../outputs/stage5a/figures/compressibility.png)

Lines show best of two restarts within each topology; shaded ranges show the two runs. Synthetic targets differ by depth and are deliberately exactly representable at that depth; their increasing residual is a qualification failure, not a synthetic capacity curve. Dotted line is the frozen 1e-4 relative-error gate.

The complete QR/Givens reference uses exactly 2,016 rotations and 64 phases. Maximum relative reconstruction error across seven available matrices is `6.50e-15`. Forward/adjoint, noncontiguous tensors, identity nesting and physical/effective equalizer identities pass. The exact reference has 4,096 nominal real coordinates, equal to dim U(64); it provides no low-complexity structural discovery. Exact reconstructed-matrix communication reproduces every teacher error count; native exact action is separately checked algebraically rather than used for expensive full-bank transmission.

Nominal parameter counts are **upper bounds on identifiable dimension**. At generic N=8 parameters, numerical Jacobian ranks are 32/32, 56/56 and 64/104 for 1/2/4 passes. Zero-angle gate phases are unidentifiable; deep factorizations can be redundant. N=64 full Jacobian rank was not computed. Do not equate 1,600 stored coordinates with 1,600 proven independent physical degrees of freedom.

## 6. Optimization diagnosis and stopping decision

Cold known-family relative errors span `1.265e-4` to `3.322e-2`; **0/12 qualify**. Exact planted parameters reconstruct identically. With independent 1e-3 perturbations of generating parameters, both one-pass cases qualify, whereas neither two-pass nor either four-pass case qualifies within the same budget. Their final relative errors remain approximately `6.1e-4` to `1.39e-3`. The initial neighborhood itself is not below the qualification gate.

This demonstrates an optimization limitation beyond insufficient architecture capacity, including difficulty refining known deep solutions locally. It does not isolate polar-coordinate conditioning, phase degeneracies, local minima or iteration budget as the sole cause. Larger cold target errors and marked restart variation cannot be turned into architectural lower bounds.

At the selected four-pass Learned fits, median column errors are only 0.191–0.195, but 95th-percentile errors reach 0.720–0.833, with maxima 0.870–0.941. The residual is uneven across pulses; aggregate loss hides adverse columns. Detailed distributions are saved. Their carrier participation suggests a useful future local-topology hypothesis, but the failed fitter prevents attributing these columns to a missing physical mechanism.

The [diagnosis protocol](stage5a_diagnosis_protocol.json) and [evaluation clarification](stage5a_evaluation_amendment.json) were written before final-bank use. The latter permits diagnostic evaluation of attained fixed waveforms while explicitly withholding successful-compression/capacity conclusions. No extra architecture, additional passes, large search or Stage-5B implementation was launched.

## 7. Held-out complex fidelity

Final banks: 512 continuous fractional DD atoms, and 512 separate normalized DD-v1 multipath channels. Neither entered fitting or candidate selection. Statistics below are **normalized squared discrepancies**, unlike the relative matrix norm.

| Candidate | Matrix relative error | DD mean / p95 / worst-5%-mean | Multipath mean / p95 / worst-5%-mean | Column error median / p95 |
|---|---:|---:|---:|---:|
| Four-pass ascending | 0.415656 | 0.254871 / 0.361759 / 0.369945 | 0.254925 / 0.407633 / 0.471194 | 0.1913 / 0.8334 |
| Four-pass descending | 0.383131 | 0.243643 / 0.316758 / 0.322449 | 0.243025 / 0.370686 / 0.418490 | 0.1953 / 0.7199 |

Both models are unitary below `6e-15` Frobenius residual, yet neither reproduces the complete complex response accurately. The descending fit's Q support overlap is 0.9447 and eta24 is 0.9915; apparently strong magnitude concentration does not imply small complex error. Aligned squared atom overlap 0.8683 is descriptive only. No empirical Stage-4B aligned range is promoted to a native-order equivalence tolerance.

![Complete complex fidelity](../outputs/stage5a/figures/complex_operator_fidelity.png)

Raw atom/multipath discrepancies, banks, medians, adverse tails and classical/Haar comparisons are saved in [representation summary](../outputs/stage5a/representation_summary.csv), [operator discrepancies](../outputs/stage5a/operator_discrepancies.npz) and [operator banks](../outputs/stage5a/operator_banks.npz). Neither similar Q nor a heatmap substitutes for these complex measurements.

## 8. Communication preservation

Used normalized DD-v1, N=64, P=3, continuous delay [0,4], Doppler [-0.75,0.75], iid complex gains of variance 1/P before realization normalization, repository negative Fourier/DD signs and delay-first ordering. W remains fixed with no transmitter CSI. Complete channel-aware LMMSE retains all interactions. `H`-domain filtering followed by `W^H` is algebraically the full effective-coordinate receiver, independently replayed below.

Frozen evaluation uses **1,024 independent channels and 16 blocks/channel**, with shared physical H, bits and physical noise across waveforms. Each QPSK point has 2,097,152 bits; each 16-QAM point 4,194,304 bits. Evaluated all six preselected fits, teacher, exact reconstruction, OFDM/OTFS/AFDM and IDENTITY; IDENTITY remains a numerical control outside figures. Primary slicing is the existing nearest-constellation rule. The second diagnostic divides each full-LMMSE stream by its positive diagonal signal gain; it does not cancel residual cross-stream interference or implement joint detection.

Uncertainty: 5,000 paired **channel-cluster** bootstrap resamples. The estimator averages random finite-block error rates across independent channel/bit/noise clusters; empirical cluster variability includes their realized finite-block variability. It is not an iid-bit binomial interval or an exact rare-event bound. Noninferiority uses the prospectively fixed statistic `BER_candidate - 1.10 BER_teacher`, requiring its one-sided 95% upper bootstrap bound below zero. Intervals are pointwise; no simultaneous eight-condition claim is made. A failed noninferiority test can be unresolved, and is not automatically proof that the 10% margin is exceeded.

| 18 dB, legacy full receiver | QPSK errors / BER | 16-QAM errors / BER |
|---|---:|---:|
| Teacher | 2,132 / 0.00101662 | 93,601 / 0.02231622 |
| Four-pass ascending | 2,253 / 0.00107431 | 93,486 / 0.02228880 |
| Four-pass descending | 2,316 / 0.00110435 | 93,816 / 0.02236748 |
| OFDM | 4,518 / 0.00215435 | 95,329 / 0.02272820 |
| OTFS | 251 / 0.00011969 | 86,029 / 0.02051091 |
| AFDM | 1,078 / 0.00051403 | 89,729 / 0.02139306 |

For descending four-pass QPSK at 18 dB, the paired BER increase is `8.7738e-5`, two-sided 95% CI `[4.4823e-5, 1.3113e-4]`; the one-sided noninferiority upper bound is **+2.3131e-5**, so 10% preservation is unconfirmed. Ascending four-pass has a positive BER-difference CI `[1.0490e-5,1.0349e-4]` but a negative noninferiority upper bound `-3.9101e-6`: a detectable increase can still fit the engineering margin.

At 20 dB both four-pass candidates fail QPSK noninferiority; ascending's point estimate is about 11.6% worse, descending about 17.7% worse. Ascending passes the pointwise QPSK margin at 12/15/18 dB; descending only at 12 dB. Both pass the four pointwise 16-QAM margins with either slicing diagnostic. This is partial communication preservation despite substantial matrix/operator differences, not a validated reproduction of Learned.

At 18 dB OTFS/AFDM beat the teacher for both modulations with paired difference intervals below zero. The lower-SNR 16-QAM ordering differs: teacher is better at 12 and 15 dB. Do not convert these results into universal classical or Learned superiority. High-SNR OTFS QPSK includes only 71 errors at 20 dB, so tail precision remains limited even though no reported aggregate point has zero errors.

![Full-receiver BER point estimates](../outputs/stage5a/figures/full_receiver_ber.png)

Figure shows highest-budget candidates and classical controls; all lower-budget fits and both slicers remain in [raw channel counts](../outputs/stage5a/communication_raw.csv), [summaries](../outputs/stage5a/communication_summary.csv), [paired intervals](../outputs/stage5a/communication_paired.csv), and [stream reliability](../outputs/stage5a/stream_geometry.csv). Positive scalar gain correction leaves every QPSK error count unchanged; 16-QAM changes are retained separately.

## 9. Real implementation cost

| Passes | Float64 parameter payload | Cached coefficients | Cached int64 indices | Compiled numeric cache total |
|---:|---:|---:|---:|---:|
| 1 | 3,584 B | 5,632 B | 6,144 B | 11,776 B |
| 2 | 6,656 B | 10,240 B | 12,288 B | 22,528 B |
| 4 | 12,800 B | 19,456 B | 24,576 B | 44,032 B |

Dense complex128 W occupies 65,536 B. Four-pass serialized parameter payload is 5.12x smaller, while the deployed numeric cache is only **1.49x smaller**; keeping parameters as well gives 56,832 B. Tensor/Python object overhead and FFT backend workspace are additional and not measured as resident bytes. Deterministic topology could avoid stored indices in a redesigned kernel, but this implementation caches them and counts them. Serialized `.pt` archive sizes include further metadata and are not equated to numeric payload bytes.

Four passes apply 768 gates, one 64-point FFT, 1,600 complex multiplies, 3,072 real scalings and 1,536 complex additions outside the FFT, per vector and per forward/adjoint action. Nominal cost is O(p N log N); index gathers, copies, allocations and dispatch are additional. Temporary working arrays scale O(batch*N), plus FFT workspace. Dense offline fitting is distinct from native deployment. The analytic exact reference's 4,096 angle/phase coordinates are quadratic; cached c/s/diagonal coefficients alone need 49,408 B, and explicitly stored pair indices would add 32,256 B before Python overhead. It is not a fast transform.

Actual CPU timings use one PyTorch thread, float64/complex128, five warmups, 41 samples, batches 1 and 32, direct cached factor actions versus dense BLAS. For descending four-pass, batch-1 forward medians are **424.083 µs native versus 4.500 µs dense** for the same reconstructed W; the original teacher dense action is 4.208 µs. Parameter compilation is separately measured at 25.94 µs median. These are implementation/host-specific observations, not a lower bound on optimized mixer kernels.

![Measured native and dense application latency](../outputs/stage5a/figures/native_application_cost.png)

A complete physical chain benchmark includes synthesis, H propagation, complete equalizer application, analysis and legacy 16-QAM slicing with one reused channel. For descending four-pass, batch-1 median is 906.083 µs native versus 44.625 µs dense; batch-32 is 1,531.167 µs versus 266.375 µs. Receiver solve/setup is recorded separately. Physical H generation/acquisition is excluded, so this is not an end-to-end radio benchmark. `W` compression does not reduce H's singular-value condition number, remove full solver setup, or establish receiver FLOP savings. Exact setup and raw action/chain samples are in the [cost ledger](../outputs/stage5a/cost_ledger.json), [action timings](../outputs/stage5a/latency_raw.csv), and [chain timings](../outputs/stage5a/complete_chain_latency_raw.csv).

N=128/256 checks only validate the same native action/adjoint rule against its dense materialization: maximum action error <`1.95e-16`, unitarity residual <`1.04e-14`. No larger-N learned target was fit, and no waveform/generalization/scaling-law advantage is claimed.

## 10. Independent review, verification and reproduction

An actual separate Codex process reviewed the mathematics, source, protocol and partial results read-only. Its original response and a resolution ledger are retained in [review artifacts](../outputs/stage5a/review/). It correctly identified failed positive controls, missing population, missing completeness guards and misleading compilation timing. These are incorporated. Its claim that nesting was unprotected was disproved: the initial closure retains step zero before any update, in both restarts, and all 24 sequences were checked. Its recommendation to defer communication was reconciled by the prospective evaluation clarification: the final data diagnose fixed attained matrices, without claiming fitter optimality or capacity rejection.

Independent numerical verification used a separately implemented effective-coordinate LMMSE solve rather than the production physical-coordinate solve. It reproduced **512 per-channel error counts exactly** across two waveforms, both modulations and both slicers; audited all **196,608 raw rows / 192 summary groups**; reproduced a critical paired noninferiority bound; and verified exact-reconstruction counts, QPSK gain invariance, mean-MSE invariance and 42 protected historical files. See [verification receipt](../outputs/stage5a/independent_verification.json). This is independent numerical implementation, not an independently sampled scientific replication.

Focused Stage-5A tests: **9/9 pass**. Whole repository: **141 pass, 2 fail**, both due to pre-existing absent historical checkpoint files (`outputs/stage1a/...haar_2_best.pt` and `outputs/stage1a2/...tail_from_ofdm_best.pt`). No tests were disabled and no substitute serialization was fabricated. Full tests and whitespace checks are recorded alongside the final provenance manifest. Four PNGs were rendered and visually checked, with DejaVu Sans, 18-point labels, 15-point ticks, 16-point legends and 300 DPI.

Executed fitting/evaluation source snapshots, protocol, selected checkpoint hashes, raw traces, role seeds, banks and result hashes are preserved under [outputs/stage5a](../outputs/stage5a/). Historical results and source were not edited. Generated data stay ignored by Git. Native loss uses the entire known matrix, as intended; there is no artificial matrix-entry holdout. Atom and communication banks have separate hashed seed namespaces and are not selection inputs. The exact fitting-runner hash was recovered and checked after later evaluation-only source edits; snapshots identify what actually ran.

Commands, from repository root:

```bash
/Users/saksh/miniforge3/envs/pytorch/bin/python -m pytest tests/test_stage5a.py -q
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5a_verify
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5a_figures
# Full bounded fresh reproduction; original output namespace is never overwritten:
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5a fit --output outputs/stage5a_replay --teacher outputs/stage5a/teacher.npy
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5a evaluate --output outputs/stage5a_replay --teacher outputs/stage5a/teacher.npy
```

The main reproduction commands rerun fitting/evaluation; diagnostic protocol/source snapshots specify the six planted-neighborhood checks. `stage5a_diagnose` refuses overwrite of the completed diagnostic. Source numerical/environment drift may change optimization and timing; checkpoint and input hashes distinguish exact saved-state replay from a new fit. The report is not a claim that a full fresh reproduction has already been run.

# Recommended Stage 5B Research Direction

## 1. Qualify optimization for the existing independent-mixer family — recommended

**Hypothesis:** coordinate degeneracy/conditioning or optimization basin selection, rather than capacity alone, limits the current fits. **Evidence:** 12/12 cold known-family failures and 4/6 deeper near-truth failures, despite exact planted reconstruction and correct actions. **Open uncertainty:** polar-angle conditioning versus true local traps versus finite iteration budget; none was isolated.

**Minimum experiment:** freeze a small synthetic suite at N=64, depths two/four, compare the current coordinates with one nonsingular local SU(2) coordinate chart or a deterministic gate-coordinate solver, using equal evaluation budgets and cold/near-truth starts. No Learned BER is needed until this gate passes. **Implementation:** reuse exact Givens oracle, native actions, checkpoints, gradient traces and incumbent logic; add just the coordinate/solver diagnostic. **Cost:** roughly tens of seconds to a few minutes locally based on 1–2 s per existing full-size fit, depending on solver; cap the pilot before extending it.

**Success/failure:** recover all declared synthetic cases below 1e-4 relative error from the specified starts, preserving unitarity/action checks; reject a solver if it only works from planted truth or needs uncontrolled restarts. If qualified, repeat the same Learned/Haar ladder and add recovered historical population tensors. **Publication significance:** a reproducible optimization/representation result needs subsequent learned-specific separation, fidelity and practical cost evidence; numerical rescue alone is not a new waveform. **Risks:** poor conditioning may persist, extra solver complexity may overwhelm savings, and a better generic fitter may eliminate the apparent Learned-specific contrast.

## 2. Test frequency-local, data-independent sparse-factor topology after qualification

**Hypothesis:** the target's concentrated `F W` carrier organization permits a smaller useful factor network than fixed XOR scales. **Evidence:** 77.0% top-eight carrier energy and participation 9.89 versus Haar's 37.4% and 33.3; uneven adverse columns and scale-order dependence. **Open uncertainty:** amplitude concentration may not imply easy complex factorization and may not generalize beyond this one teacher.

**Minimum experiment:** one frozen frequency-neighbor pairing schedule versus the existing XOR rule, matched parameter/evaluation budget, all available Learned/Haar/Q-only targets, synthetic topology controls, then held-out complex fidelity. **Implementation:** reuse the qualified fitter and ledger; change only deterministic pair scheduling. **Cost:** approximately the current 80 s target-fitting budget per matched grid, plus diagnostics, rather than a broad architecture search. **Success/failure:** predeclare a substantial reduction in native/DD error at equal deployed bytes and no worse measured action cost; no gain on Haar alone is a learned-specific result. Stop if synthetic qualification or matched advantage fails. **Publication significance:** distribution-informed carrier factorization could be useful if it explains reproducible learned structure beyond established universal unitary decompositions. **Risks:** missing population, target-specific index metadata, optimizer bias and Python gather overhead.

## 3. Conditional direct compact full-receiver waveform design

**Hypothesis:** a compact independently designed basis can attain useful finite-alphabet/full-receiver behavior without reproducing the historical finite-K teacher. **Evidence:** some matrix-different fits pass pointwise 16-QAM margins while exact complex fidelity remains poor; OTFS/AFDM beat the teacher in important full-receiver conditions. **Open uncertainty:** any compact learned design may simply rediscover an established classical transform, and low-SNR modulation tradeoffs may persist.

**Minimum experiment:** only after a separate protocol decision, compare one compact family trained on a finite-alphabet full-receiver objective against OFDM/OTFS/AFDM and dense controls, with a frozen validation rule and fresh test bank. **Implementation:** reuse the physical model, qualified full receiver, slicing diagnostics and paired count pipeline; new objective/training would belong to Stage 5B. **Cost:** greater than this post-hoc study; profile a bounded gradient pilot first, then set a fixed budget. **Success/failure:** beat the strongest matched classical control with predeclared uncertainty and cost criteria; matching this teacher alone is insufficient. **Publication significance:** would be a new waveform-design investigation, not compression of Learned. **Risks:** weak headroom, modulation-dependent gains, training cost, and no receiver/runtime benefit.

**Recommend direction 1 first.** Numerical qualification currently blocks both a capacity conclusion and a credible topology comparison. A new architecture or direct-training campaign would add ambiguity before resolving this demonstrated failure. Stage 5B has not been implemented.

## Literature boundary

Exact arbitrary-unitary constructions are established; the QR reference is a qualification tool, not a novelty claim. See [Reck et al., Experimental realization of any discrete unitary operator](https://link.aps.org/doi/10.1103/PhysRevLett.73.58) and [Clements et al., An Optimal Design for Universal Multiport Interferometers](https://arxiv.org/abs/1603.08788). A future contribution must demonstrate something beyond writing a unitary as elementary mixers: reproducible learned-specific structure, a qualified fidelity/complexity tradeoff, or matched communication/implementation benefit.
