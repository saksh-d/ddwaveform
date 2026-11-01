# Stage 5B-A — Optimization qualification

## Executive summary

The independent mixer implementation is mathematically correct. Replacing polar angle/phase coordinates with Cartesian Lie coordinates materially improves recovery **at unchanged topology, parameter count and original optimizer schedule**. The replacement passes the strict 1e-4 relative-error gate in all 36 frozen near-truth tests. The unchanged baseline passes 12/36. However, Cartesian cold fitting passes only 12/18: all six four-pass cold targets fail. The single additional solver intervention (direct Cartesian L-BFGS) passes36/36 local and14/18 cold cases; four deep failures remain. Practical optimization remains unqualified, and the Learned pilot is skipped. This is primarily **Outcome B**, with unresolved practical fitting, not evidence of insufficient representation capacity.

[Complete recovery ledger](stage5b_a_results.md) reports every declared fit, initial distance, endpoint, closure count and aggregate fitting cost. [Raw artifacts](../outputs/stage5b_a/) retain every trace, selected checkpoint, target, input state, seeds, source snapshots, verification receipts and failures. No Stage 5B-B work, waveform training, BER campaign, remote job, push or merge occurred.

## Starting evidence and preservation

Worktree `/Users/saksh/.t3/worktrees/delayDopplerWaveform/t3-12e6c015`, clean initial branch `t3/qualify-factor-optimization`, revision `a12d24f07c03a1319ee5fa8773efcf3e84086747`; not the earlier e6406f6 revision. Read project HISTORY and research-record README and delay-Doppler evidence map. Inspected Stage5A source, protocol, diagnosis, tests and final report. Ignored artifacts were absent here; copied six required prior files from the completed phoenix worktree into the new ignored namespace, preserving originals. Teacher SHA256 `11bb15d39bb7e685b074e2ec7ed6f27bd87370e141aa6f62b67c338c21514b0f` matches Stage5A. No missing population checkpoints recovered or regenerated. Prior recovery-search receipt remains in `outputs/stage5b_a/prior/`; only the historical exported teacher is available, not verified original .pt bytes or a five-member population.

## Implementation and conditioning audit

For column vectors the native gate is `[[c,s],[-conj(s),c]]`, with real c. Row batches implement the transpose action, then D and orthonormal IFFT. Thus `W=F^H D L_last ... L_first`; adjoint reverses layers, conjugates D and uses FFT. Independent NumPy dense multiplication agrees to numerical precision. Native actions, materialization, unitarity, phase signs, ordering and adjoints pass. Original identity nesting was independently covered by unchanged Stage5A tests: appended zero-theta layers preserve the incumbent exactly, including step-zero selection. No incorrect ordering, complex autograd convention or loss-square interpretation explains the prior failure.

The original gate uses `c=cos(theta)`, `s=sin(theta) exp(i phi)`. This is a two-real-dimensional gate subfamily of SU(2), **not general three-dimensional SU(2)**. At theta=0, phi is unidentifiable and its derivative is zero; adjacent repeated scales at identity also create redundant first-order directions. Phase periodicity and multiple factorizations remain; nominal parameter count is only an upper bound on intrinsic dimension. Global phase and column phases are retained in the native matrix objective, with no post-hoc alignment used for fitting. Physical receiver global-phase invariance does not remove that fitted degree of freedom.

The new code stores `w=a+ib=theta exp(i phi)` in the inherited theta/phi fields, so prompt coordinates are `z=-conj(w)`. Coefficients are `c=cos(r)`, `s=w sin(r)/r`. This exactly represents the same gate family, including signed original theta; no new phase degree is added. Radius/phase periodicity, nonuniqueness, and tangential collapse at nonzero integer multiples of pi remain. It is regular at identity, not globally regular. Always use `from_polar` for declared initialization; the inherited bare constructor's random fields are not a practical Cartesian initializer. Compilation is disabled for this fitting-only subclass.

For q=a²+b²<1e-8, c and sinc use cubic polynomials in q; the other branch clamps q before sqrt to keep inactive-branch gradients finite. O(q^4) truncation is below double-precision significance at that cutoff. Exact-zero identity and finite gradients are tested. Tests compare gates at radii 0,1e-12,1e-5,0.4,pi,7 against a separate matrix exponential. The reviewer also checked both sides of the branch boundary and radius9 using SciPy expm.

144 central finite-difference checks pass at h=1e-6, maximum absolute discrepancy 1.78e-10. Acceptance: absolute error <2e-8 +2e-4|autograd|; relative discrepancy reported only for |gradient|>1e-7. Cases cover random shallow/deep N8 and N64, identity, near identity, nontrivial phases and prespecified local/cold four-pass checkpoints. Small absolute errors near zero are not reported as large meaningful relative-gradient failures. Raw checks: `gradient_audit.json`.

N8 one-pass generic Jacobians have rank32/32 for both charts at singular-value threshold1e-7, but extreme singular values are approximately 1.414/0.00230 (polar) versus 1.453/0.883 (Cartesian). At identity, ranks are20/32 versus32/32. N8 four-pass generic ranks are64 with104 nominal parameters; near identity both show rank32 at this threshold. This separates parameter redundancy from the polar singularity and poor conditioning; it does not establish N64 generic rank. Dense Jacobians were constructed only at N8.

At N64 four-pass identity, normalized random parameter-direction JVP norms are1.480/0/1.000 for polar theta/phi/output phase, versus1.372/1.372/1.000 for Cartesian a/b/output phase. At random parameters the norms are1.466/0.211/1.000 versus1.365/1.364/1.000. These few directional probes demonstrate the identity defect and anisotropy, **not a full condition-number or rank estimate**. Raw results: `geometry.json`.

## Controlled optimization experiment

The [frozen protocol](stage5b_a_protocol.json) declares N64, passes1/2/4, ascending/descending, three independent generation seeds per configuration:18 exactly representable targets. Each target uses the original generation law theta=.15 Normal, phi uniform, output phase zero. This deliberately preserves Stage5A continuity; it does not qualify all large-angle or arbitrary-phase targets.

Each target has two common near-truth starts, formed by independent polar perturbations of0.001 and0.03 to all parameters, and one practical cold start from the unchanged original procedure. Initial matrix error is recorded; perturbation scale is never treated as reconstruction error. B converts each entire initial matrix from A, with numerical agreement. Thus A/B compare coordinates rather than different starts. Generation and fit seed namespaces are distinct. The cold model reads only the target to initialize output phase from diag(F target); it never copies truth factors. The exact planted factors are algebraic references, not optimization trials. No capacity conclusion uses their zero error.

A calls the original `fit_matrix` without edits. B calls the same solver using Cartesian fields:250 Adam iterations at0.035, followed by up to200 L-BFGS iterations, history30, strong Wolfe, original tolerances. Both retain minimum native complex squared Frobenius/N across **all evaluated states**, including step zero and line-search trials. Each closure performs one objective and one backward; all closure calls, including final evaluation, are counted. Nominal schedule is matched, not exact evaluations or runtime. Equal-common-prefix diagnostics reproduce the same local/cold success counts, ruling out extra B closure work as the explanation of its qualification improvement on this suite.

The principal test is strictly relative error<1e-4 (loss<1e-8). A local12/36, practical0/18; B local36/36, practical12/18. B's worst local error is3.09e-5, versus A1.48e-3. Four-pass cold B errors remain approximately0.015–0.022. No cases were dropped; no numerical failures occurred. One cold start per target does not estimate within-target restart sensitivity; variation across three generation/fit seeds is reported, with no broad reliability inference.

## Single solver intervention and diagnosis

The [solver amendment](stage5b_a_solver_amendment.json) was frozen after A/B completion, before C execution. C uses Cartesian L-BFGS directly, hard-capped at500 objective/gradient evaluations, tolerance_grad1e-13 and tolerance_change1e-19, on the same54 targets/starts. This is one bundled intervention: removing Adam plus changing stopping tolerances. It cannot identify which component causes any improvement. No new targets, topology, initialization oracle or restarts were introduced.

C passes36/36 local cases and14/18 cold cases, with worst cold error0.01430; all runs respect500 closures. The four failures remain, and the conditional Learned pilot stays closed.

The motivation was local success together with nonzero gradients in failed deep cold fits at the iteration budget. For example, ascending four-pass seed0 cold B ends with gradient norm9.34e-4, not a demonstrated stationary minimum. The experiment therefore tests direct refinement at comparable total evaluations. It does not prove convergence would follow from unlimited work. Failed recovery cannot mean insufficient capacity because these exact targets were generated by this same model family.

The evidence identifies a concrete coordinate-conditioning defect and demonstrates that removing it improves numerical behavior. Remaining explanations include weak deep-product directions, initialization basins and finite refinement budgets. Neither local minima nor irreducible approximation floors have been established. A further rescue sweep was not authorized by this bounded protocol and was not run.

## Learned pilot and capacity readiness

The [conditional pilot protocol](stage5b_a_pilot_protocol.json) froze12 four-pass Learned/Haar/OFDM fits, but practical qualification failed, so none ran. No new Learned-versus-Haar compressibility comparison is supported. Stage5A's attained errors0.987802/0.710912/0.383131 remain historical optimizer-dependent approximations. Their Fourier concentration evidence concerns one teacher and is not a population result. Local recovery alone cannot establish intrinsic capacity for a historical matrix outside the generating law.

## Computational accounting and reproduction

CPU float64/complex128, one PyTorch thread, existing `pytorch` conda environment. Dense fitting cost is distinct from factor deployment or full receiver work. Each closure materializes the entire64×64 matrix and differentiates the complete objective. Counts include line searches. Aggregate fitting seconds exclude input generation, auditing, serialization and tests; comparisons are local runtime observations, not speed claims. End-to-end shell timings were not separately instrumented. All raw per-fit fitting times and norm trajectories remain saved; see the results ledger.

```bash
/Users/saksh/miniforge3/envs/pytorch/bin/python -m pytest tests/test_stage5b_a.py tests/test_stage5a.py -q
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5b_a
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5b_a --arm C
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5b_a_audit
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5b_a_verify
/Users/saksh/miniforge3/envs/pytorch/bin/python -m scripts.stage5b_a_report
```

Runner refuses existing AB/C directories; fresh reproduction requires a separate checkout/output root, preserving this run. Main runner/source snapshots and per-file hashes identify executed uncommitted changes relative to the recorded Git revision. Final manifest hashes raw data, checkpoints and sources. Full suite:155 pass, two failures from the same missing historical Stage1A/1A.2 checkpoint files as Stage5A; no failures hidden or disabled. Focused checks:23 pass. Independent artifact verification reloads selected checkpoints, recomputes every achieved error, selection and success count, checks seed separation and parameter count, and compares saved truth against targets. Reviewer uses separate NumPy/SciPy implementations, preventing production code from being its sole numerical validator.

## Stage 5B-B recommendation and receiver positioning

Ranked next experiments:

1. **Target-derived initialization at fixed four-pass topology**, retaining the qualified Cartesian local fitter and this synthetic gate. Recommend one bounded constructive/progressive initializer with no planted inputs, explicit target/seed coverage and matched closure accounting. Practical deep fitting is the demonstrated unresolved bottleneck. Target-derived initialization is a testable next hypothesis; current evidence does not separate starting basins, weak directions and finite refinement work. Do not infer that merely increasing iterations will solve it.
2. **A mathematically more tractable factor representation**, if the bounded initializer fails. Require guaranteed constructive fitting or a demonstrably better-conditioned synthesis procedure; count any changed capacity explicitly.
3. **Frequency-local topology**, motivated by the historical Fourier interactions, only after practical positive controls pass. Current failures cannot justify a capacity-driven topology change.
4. **Direct structured waveform design for a separately specified finite-alphabet full-receiver objective**, a separate research direction. Historical teacher compression is useful as a bounded mathematical object, but its limited-channel training objective and weaker selected full-receiver communication points limit the scientific value of an indefinite compression campaign.
5. **Termination of this family**, justified if the next bounded practical fitting test fails at useful cost. Current evidence supports stopping open-ended fitting, not declaring all structured waveforms impossible.

Recommend option1 for one bounded Stage5B-B investigation, then an explicit stop/reconsideration gate. Do not pursue larger historical compression populations before qualifying practical fitting. Longer-term, direct structured design under a receiver-aware finite-alphabet objective is more aligned with the full-receiver program than faithfully reproducing this single historical teacher. Full-LMMSE mean MSE is square-unitary invariant and is unsuitable for such retraining; Q concentration and historical Top-K BER were not optimization gates here. No parameter-count-to-receiver-savings inference is made. Neither proposed direction was implemented.

## Independent review and resolution

A separate review agent inspected code, protocol, traces and final artifacts read-only; [review resolution](stage5b_a_review_resolution.md). It used independent NumPy dense products and SciPy expm rather than production pair_action/coefficients as its sole validator. It audited108 unique A/B and54 C fits, verified all54 C inputs equal A/B bitwise, and independently reconstructed all six four-pass cold C checkpoints (saved errors agree within2e-15). It confirmed capacity equivalence, seed separation, no planted leakage, actual budget disclosure and success counts. Final verdict: no material correctness or interpretation blocker; practical fitter remains unqualified. No unresolved disagreement. Its requested wording changes clarify C counts and the hypothesis status of initialization.

B/C common-evaluation prefixes, local: B 24/36, C 36/36 pass. B/C common-evaluation prefixes, cold: B 6/18, C 13/18 pass. Full endpoints remain the predeclared primary results; this prefix audit changes no run or selection.
