# Research Council — what to do after Stage 5B-A

## Chairman’s recommendation

**Revise the initializer-first recommendation. Run one bounded comparison of sustained refinement versus restart allocation before engineering an initializer.** Keep the representation, objective and Cartesian fitter fixed. The current evidence does not identify initialization as the dominant cause of the four-pass failures.

This council produced new algebraic results and a read-only diagnostic, not a new successful fitter. No Stage 5B-B fitting, waveform training, architecture search, BER simulation or remote computation was performed.

## What changed our assessment

### The original budget ends during useful progress

Every four-pass C cold fit used all 500 objective/gradient evaluations. The two successful targets first crossed the strict threshold at **456 and 412 one-based closure counts**. The four failures were still improving. Thus stopping at 500 is a live competing explanation; a flat-looking plot is not a demonstrated approximation floor or stationary minimum.

At the four failures, squared-loss reductions over the final 100 evaluations were approximately 76.6%, 21.0%, 27.4% and 16.4%. These differ substantially. There is no evidence that all four share one limiting mechanism.

Trace source: `outputs/stage5b_a/C/*_p4_*_cold_C_trace.json`; extracted receipt: [endpoint progress](../outputs/stage5b_a/council/endpoint_progress.json).

### A mathematically exact gate update exists

The native gate family is

\[
G(c,s)=\begin{bmatrix}c&s\\-\bar s&c\end{bmatrix},\qquad c\in\mathbb R,\quad c^2+|s|^2=1.
\]

It has two real degrees of freedom. Hold every other factor fixed and strip them from the target using their adjoints. For a scheduled pair, let the resulting 2×2 target block be H. Define

\[
\alpha=\Re(H_{11}+H_{22}),\quad
\beta=H_{12}-\overline{H_{21}},\quad
\rho=\sqrt{\alpha^2+|\beta|^2}.
\]

Then the exact conditional optimum is

\[
c=\alpha/\rho,\qquad s=\beta/\rho.
\]

Keep the incumbent if rho=0. The derivation maximizes `alpha*c + Re(conj(s)*beta)` on the unit sphere. All disjoint gates within a layer separate; output phases likewise have exact conditional updates.

This is a genuine available mathematical solution to each restricted subproblem. It preserves topology and capacity. It does **not** solve the full factorization globally. Exact block sweeps would produce nonincreasing, bounded objective values; that does not guarantee exact recovery or convergence of the factor iterates. No sweep solver was implemented or tested.

### The saved failures have little isolated layer improvement

We froze a read-only calculation on all six saved four-pass C cold checkpoints: 144 layer environments plus six output-phase checks. It reconstructs the selected matrices independently using NumPy, computes the best possible decrease from changing one layer while freezing all others, and applies no changes.

| Saved checkpoint | Current relative error | Best one-layer decrease, % of current squared loss |
|---|---:|---:|
| Ascending seed 0, failed | 0.00328111 | 0.294428% |
| Ascending seed 1, passed | 0.0000362729 | 0.439505% |
| Ascending seed 2, failed | 0.01430465 | 0.0167349% |
| Descending seed 0, failed | 0.00735813 | 0.00351236% |
| Descending seed 1, passed | 0.0000368798 | 0.170061% |
| Descending seed 2, failed | 0.01229914 | 0.00334336% |

These percentages concern **squared loss**, not relative matrix error or captured energy. They are independent conditional opportunities, not additive full-sweep gains. Successful checkpoints have some larger relative opportunities than failures; this is not a failure classifier.

First Principles and Contrarian independently reproduced the failure percentages with a cancellation-resistant identity:

\[
\Delta_\ell=\frac1N\sum_k\rho_k\left\|q_k/\rho_k-x_k\right\|^2,
\quad q=(\alpha,\Re\beta,\Im\beta),\ x=(c,\Re s,\Im s).
\]

Failure results agree with direct score subtraction to relative error below 2e-8. Minor cancellation at already successful tiny residuals does not affect interpretation. The table uses stable independently verified values. Source and original receipt: [read-only calculation](../outputs/stage5b_a/council/conditional_opportunity.py), [raw opportunities](../outputs/stage5b_a/council/conditional_opportunity.json).

The finding demotes native exact sweeps as the immediate intervention. It does not bound their cumulative improvement: a changed layer would alter later environments. Nor does it establish that Gauss–Newton will work. It identifies why exactness of each isolated update is not sufficient evidence for prioritizing a sweep campaign.

### Deep products retain an identity degeneracy after the coordinate fix

At all-identity mixers, repeated passes use the same XOR edges. A first-order change on an edge has the same matrix effect in every pass. There are `N log2(N)/2` distinct edges, two independent real off-diagonal directions per edge, plus N diagonal phase directions. Their distinct supports establish the exact identity tangent rank:

\[
\operatorname{rank}J_{\mathrm{identity}}=N\log_2N+N.
\]

At N=64 this is **448**, independent of pass count. The four-pass model has 1,600 coordinates, so **1,152 parameter directions cancel to first order at identity**.

This is an algebraic identity-point result, not a measured generic N64 rank. Away from identity these directions can become identifiable; their singular values and coupling to the actual residual remain unmeasured. Unlike the N8 case, N64 dimension counting does not force generic redundancy: 1,600 is below `dim U(64)=4,096`.

## Concrete next experiment: two capped practical policies

This is a proposed Stage 5B-B protocol, **not executed here**.

- **Targets:** all six existing four-pass synthetic targets, including both successes. This is diagnostic reuse, not an independent generalization test.
- **Persistence policy P:** one fresh C run from each original cold input, at most 2,000 objective/gradient evaluations.
- **Restart policy R:** four fresh runs per target, at most 500 evaluations each. First run uses the original cold input; three additional seeds are frozen before execution. Use the original polar cold-start distribution, target-diagonal phase initialization and explicit Cartesian conversion. Never use the inherited bare Cartesian constructor as that initializer.
- **Settings:** identical native objective, precision, C tolerances, L-BFGS history and selection rules. The threshold remains relative error <1e-4. This tests numerical qualification; it is not an established accuracy requirement for useful communication performance. It is report-only during a run: retain the planned trace after first crossing so later behavior and work to first success remain auditable.
- **Work:** maximum 24,000 new objective/backward evaluations across both fresh policies and all six targets. Charge every attempted restart and line search; no historical-prefix deduction. Equal caps are not equal actual work or equal time. Log actual counts, runtime and initialization costs. Distinguish observed work to first threshold crossing from the full campaign cost, which includes all planned attempts after success.
- **Time:** 30 fitting seconds per target/policy, cumulatively across R’s four runs. Check before each closure; finish any running closure and record overrun. Unstarted R runs are `not run: policy time cap`, not numerical failures.
- **Stopping:** retain natural optimizer termination, numerical exceptions and fixed caps. No transfer of unused allowances, forced duplicate evaluations, new starts or adaptive budget extension.
- **Replay:** compare fresh P’s available per-evaluation loss and gradient-norm prefix with the saved C trace, using tolerances frozen before execution. Full first-500 replay is checked when reached; otherwise label full replay unverified. Missing entries are not a mismatch. A genuine discrepancy is a reproducibility result, not evidence of a larger-budget effect.
- **Qualification:** a policy qualifies on this reused suite only if all six attained errors pass. Report individual outcomes even when neither policy qualifies; “neither qualifies” does not mean zero successful targets.

The saved best checkpoints do not contain L-BFGS history or necessarily an accepted endpoint. Fitting again from them is a **warm restart**, not exact continuation. Future resume instrumentation must save accepted model/optimizer state without silently restructuring the algorithm or changing its verified prefix.

| Result | Supported decision | Remaining limit |
|---|---|---|
| P qualifies, R does not | Sustained refinement is useful under this ceiling | Does not prove budget is the sole cause |
| R qualifies, P does not | Start diversity is useful under this ceiling | Does not establish distinct basins or a constructive initializer |
| Both qualify | Compare actual work to threshold | Causal dominance remains unresolved |
| Neither qualifies | Stop L-BFGS escalation at this ceiling | No capacity-impossibility conclusion |

The 2,000-evaluation and 30-second limits are affordability choices, not derived sufficient budgets. No choice of best result may hide attempted starts or capped cases.

## Ranked alternatives if the control is inadequate

1. **Damped matrix-free Gauss–Newton/trust-region diagnostic.** Realify the 64×64 complex residual into 8,192 real components with 1,600 real parameters. A damped least-squares step coordinates layers; JVPs/VJPs avoid a dense N64 Jacobian. Count inner products, rejected trials, objective evaluations and time. Test actual accepted nonlinear reduction, not predicted reduction alone. This is the strongest solver fallback, not demonstrated recovery.
2. **Specified target-derived initializer**, particularly if R outperforms P. State the construction, preserve the restricted family, prohibit planted inputs and include construction cost. “Progressive” alone is not an algorithm or a guarantee.
3. **Native exact block sweeps**, still valid but lower priority given the observed isolated opportunities. Their cumulative gains remain unknown.
4. **Different tractable representation or termination**, if bounded practical fitting remains unsatisfactory. Neither outcome proves all structured unitary families fail.

These are ranked future decisions, not an automatically authorized sequence of rescue campaigns. The proposed P/R experiment ends after its declared cells. Any fallback requires its own frozen method, work ceiling and stopping rule.

## Literature checked and applicability limits

- [SciPy least_squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) supports callable LinearOperator Jacobians, LSMR-based trust-region subproblems and regularization for deficient Jacobians. This establishes available numerical machinery, not effectiveness here. Verify the installed version before implementation.
- [Kotil et al., Riemannian quantum circuit optimization](https://arxiv.org/abs/2212.07556) supplies primary precedent for trust-region optimization of unitary circuits. Its Hamiltonian-simulation problem is different; its accuracy gains are not predictions for this factorization.
- [Zheng, Riccietti and Gribonval, Efficient Identification of Butterfly Sparse Matrix Factorizations](https://arxiv.org/abs/2110.01230) provides hierarchical identification results for its butterfly support family. No transfer theorem was established for our repeated restricted unitary matching network. It does not justify claiming a constructive initializer is already solved.
- [Dao et al., Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations](https://proceedings.mlr.press/v97/dao19a.html) demonstrates the broader sparse-product approach. It does not qualify this exact representation or its optimizer.

A broader U(2)-gate formulation with phase-pushing canonicalization was raised as a possible equivalent-family optimizer. It adds implementation and verification machinery and was excluded from the primary recommendation. No U(2) fitting or architecture change occurred.

## Council process, disagreement and confidence

Five actual fresh members: First Principles, Contrarian, Expansionist, Outsider and Executor; Chairman synthesized. Three active child slots required batching. All independent openings preceded sharing; members then engaged attributed peer claims and revised their positions. Shared workspace and common model/instructions mean these are independent opening assessments, not statistically independent scientific replications.

A material priority difference remains: First Principles prefers a bounded matrix-free coupled residual diagnostic **before** the practical policy comparison. The Chairman chooses P/R first because it requires less new solver machinery and directly addresses the observed cutoff. This is an implementation-cost judgment, not a proof that budget dominates geometry.

The council changed the earlier initializer-first recommendation. It also demoted exact sweeps after inspecting their actual available conditional improvements. Confidence is strong in the algebra and checked receipts; moderate in the practical ordering; low in any prediction that a particular untested solver will recover all targets.

Final audit status: **all five completed; no unresolved material correctness objection.** Editorial corrections applied: trace values versus parameter states, cumulative R time cap, original start distribution, all-six gate, objective convergence versus factor convergence, no automatic fallback chain.

Deliberation stopped after all final audits because the remaining scientific disagreement needs a new prospectively bounded experiment, not more committee discussion. Stage5A/5B-A recovery results, historical teacher evidence and preservation boundaries remain unchanged.
