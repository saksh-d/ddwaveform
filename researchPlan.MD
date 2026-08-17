# Forward Path: Final Project Plan for Communication-Driven Delay–Doppler Waveform Discovery

## 1. Project objective

The goal of the forward path is to discover a waveform basis—and ultimately a scalable waveform family—that is intrinsically well suited to doubly dispersive delay–Doppler channels.

The guiding principle is:

[
\boxed{
\text{Communication performance determines what constitutes a good waveform.}
}
]

We will **not** prescribe in advance that the effective channel matrix should be diagonal, banded, monomial, sparse, or resemble AFDM/OTFS.

Instead, the causal direction is

[
\boxed{
\text{DD channel distribution}
\rightarrow
\text{communication objective}
\rightarrow
W
\rightarrow
A=W^HHW
\rightarrow
\text{interpretation}
\rightarrow
\text{waveform law}.
}
]

The effective operator (A) remains central to the research, but it is an **induced object**, not the desired training target.

The eventual research output should not be one stored learned matrix. The desired endpoint is a mathematical family

[
\boxed{
W_N(\theta)
}
]

with a small parameter vector (\theta), deterministic generation for arbitrary (N), and ideally an (O(N\log N))-class implementation.

---

# 2. The precise initial research question

The first question is deliberately narrow:

> **Does there exist a single fixed unitary waveform basis that represents a distribution of delay–Doppler channels more usefully than OFDM, OTFS, and AFDM when the receiver has a limited representation budget?**

The phrase **limited representation budget** is critical.

If the receiver is allowed complete knowledge of the effective channel and unrestricted full LMMSE processing, every square unitary waveform is fundamentally equivalent.

Therefore we are not asking whether a learned unitary waveform can create more physical channel information.

It cannot.

We are asking:

[
\boxed{
\text{Can it organize the same physical information into coordinates that are substantially easier to exploit?}
}
]

That is the actual waveform-design problem.

---

# 3. Initial system model

Let

[
s\in\mathbb C^N
]

be the vector of transmitted information symbols, normalized such that

[
\mathbb E[ss^H]=I_N.
]

Initially:

[
\boxed{
N=64,
\qquad
\text{QPSK}.
}
]

The learned synthesis transform is

[
W\in\mathbb C^{N\times N}.
]

For the first research program we restrict it to

[
\boxed{
W^HW=I_N.
}
]

The transmitted block is

[
\boxed{
x=Ws.
}
]

Because (W) is unitary,

[
|x|_2^2=|s|_2^2.
]

Thus the learner cannot obtain a performance gain by increasing transmit energy.

The physical channel produces

[
\boxed{
y=HWs+n
}
]

with

[
n\sim\mathcal{CN}(0,\sigma^2I_N).
]

The waveform-domain analysis transform is

[
\boxed{
V=W^H.
}
]

Therefore

[
u=W^Hy
======

W^HHWs+W^Hn.
]

Define

[
\boxed{
A(H,W)=W^HHW.
}
]

Since (W) is unitary,

[
n'=W^Hn
\sim
\mathcal{CN}(0,\sigma^2I).
]

The equivalent symbol-domain system is therefore

[
\boxed{
u=As+n'.
}
]

This is the core model used throughout the first stages.

---

# 4. Delay–Doppler channel model

The physical channel is a (P)-path doubly dispersive operator

[
\boxed{
H
=

\sum_{p=1}^{P}
h_p
\Delta(\nu_p)
\Pi(\tau_p).
}
]

Here:

* (h_p) is the complex path coefficient;
* (\tau_p) is the path delay;
* (\nu_p) is the path Doppler shift.

Initially,

[
\boxed{P=3.}
]

Path gains are normalized such that

[
\boxed{
\sum_{p=1}^{P}
\mathbb E|h_p|^2=1.
}
]

For an integer circular delay,

[
(\Pi^\ell x)[n]
===============

x[(n-\ell)\bmod N].
]

A fixed Doppler convention is used throughout the entire repository, for example

[
\Delta(\nu)
===========

\operatorname{diag}
\left(
e^{-j2\pi\nu n/N}
\right)_{n=0}^{N-1}.
]

Fractional delay must not be implemented by rounding. A proper circular fractional-shift operator can be constructed as

[
\boxed{
\Pi(\tau)
=========

F^H
\operatorname{diag}
\left(
e^{-j2\pi k\tau/N}
\right)
F.
}
]

The channel implementation must support:

1. integer delay and Doppler;
2. fractional Doppler;
3. fractional delay;
4. simultaneous off-grid delay and Doppler.

Integer channels are useful for debugging, but final claims must be established on genuinely off-grid DD channels.

---

# 5. SNR convention

One SNR convention must be used everywhere.

With

[
\mathbb E|s_k|^2=1
]

and unitary (W), define symbol SNR as

[
\boxed{
\rho=\frac{E_s}{N_0}=\frac{1}{\sigma^2}.
}
]

Hence

[
\sigma^2=10^{-\rho_{\rm dB}/10}.
]

If results are ever reported using (E_b/N_0), conversion must be made explicitly. We will never silently mix SNR, (E_s/N_0), and (E_b/N_0).

The initial waveform-training distribution is

[
\boxed{
\rho_{\rm dB}
\sim
\mathcal U[12,20]\ {\rm dB}.
}
]

This interval contains the principal moderate/high-SNR operating region where DD interference structure matters while preventing optimization from being dominated by very low-SNR AWGN behavior.

---

# 6. Why unrestricted full LMMSE cannot train a unitary waveform

For a unitary transform,

[
A=W^HHW.
]

If the receiver knows the full (A), the full LMMSE error covariance is

[
C_e
===

\left(
I+\rho A^HA
\right)^{-1}.
]

Since

[
A^HA
====

W^HH^HHW,
]

we have

[
C_e
===

W^H
\left(
I+\rho H^HH
\right)^{-1}
W.
]

Therefore

[
\boxed{
\operatorname{tr}(C_e)
======================

\operatorname{tr}
\left[
\left(
I+\rho H^HH
\right)^{-1}
\right].
}
]

The result is independent of (W).

Consequently:

[
\boxed{
\text{full-information LMMSE gives no useful gradient for unitary waveform discovery.}
}
]

This is a mathematical invariance, not an implementation problem.

The research leverage therefore comes from imposing a finite receiver representation budget.

---

# 7. Finite receiver representation

Let

[
A=W^HHW.
]

The receiver is permitted to retain only a limited number (K) of effective interactions per output dimension.

Define

[
\boxed{
\widetilde A_K=T_K(A),
}
]

where (T_K) retains the (K) largest-magnitude entries of each row of (A) and sets all remaining entries to zero.

Initially, this is an **oracle top-(K) representation**.

That qualification is important.

Determining the strongest entries requires access to the complete (A). Therefore early experiments cannot claim that only (K) operations are required in a practical receiver.

The early question is instead:

[
\boxed{
\text{Can the waveform concentrate communication-relevant channel action into very few interactions?}
}
]

Practical support predictability and acquisition are addressed later.

---

# 8. Correct mismatched finite-budget receiver

The physical observation remains

[
\boxed{
u=As+n'.
}
]

The receiver does **not** get to replace the actual channel with (\widetilde A_K).

It merely designs its equalizer using the restricted model.

The finite-budget LMMSE equalizer is

[
\boxed{
G_K
===

\left(
\widetilde A_K^H\widetilde A_K
+
\sigma^2I
\right)^{-1}
\widetilde A_K^H.
}
]

The recovered symbols are

[
\boxed{
\hat s=G_Ku.
}
]

Hence

[
\hat s
======

G_KAs+G_Kn'.
]

This mismatch between the physical (A) and the receiver model (\widetilde A_K) is exactly what allows different waveform representations to matter.

---

# 9. Primary differentiable communication loss

The error is

[
e=\hat s-s.
]

Therefore

[
e
=

(G_KA-I)s
+
G_Kn'.
]

Using

[
\mathbb E[ss^H]=I
]

and

[
\mathbb E[n'n'^H]=\sigma^2I,
]

the exact expected normalized symbol MSE is

[
\boxed{
\mathcal L_{\rm comm}
=====================

\frac{1}{N}
\left[
|G_KA-I|_F^2
+
\sigma^2|G_K|_F^2
\right].
}
]

The terms have direct communication interpretations.

### Residual symbol distortion

[
|G_KA-I|_F^2
]

measures residual inter-symbol coupling and gain/phase distortion after the constrained receiver.

### Noise amplification

[
\sigma^2|G_K|_F^2
]

measures how strongly the receiver amplifies AWGN.

The first waveform-learning experiments use

[
\boxed{
\mathcal L_{\rm total}
======================

\mathcal L_{\rm comm}.
}
]

No structural operator loss is included.

---

# 10. What we explicitly do not optimize initially

The initial learner receives no direct reward for:

* sparsity of (A);
* diagonality;
* monomiality;
* bandedness;
* path-distance metrics;
* support overlap;
* entropy;
* a manually defined SINR score;
* resemblance to AFDM;
* resemblance to OTFS.

These remain post-training diagnostics.

The desired causal order is

[
\boxed{
\text{communication improvement}
\rightarrow
\text{successful }W
\rightarrow
\text{study }A
\rightarrow
\text{discover why}.
}
]

This avoids repeating the failure mode in which an (A)-metric improves without producing better communication.

---

# 11. Unitary optimization

The initial learned (W) is a dense complex matrix with essentially full unitary freedom.

After each optimizer update, project it back onto the unitary group.

If

[
W_{\rm raw}=U\Sigma V^H
]

is its singular-value decomposition, set

[
\boxed{
W\leftarrow UV^H.
}
]

This polar projection enforces

[
W^HW=I
]

to numerical precision.

We do not initially use a weak unitarity penalty because allowing temporary ill-conditioning introduces unnecessary ambiguity into the experiment.

The dense (W) is a **discovery instrument**.

It is not yet the final waveform family.

---

# 12. Training, validation, and test separation

Training must target a **distribution** of channels rather than a fixed channel bank.

### Training

Every minibatch generates new channels:

[
H_b\sim p_{\mathcal H}.
]

Training channels are therefore effectively unlimited.

### Validation

A frozen independent validation bank is generated once before training.

It is used for:

* checkpoint selection;
* early stopping;
* hyperparameter decisions;
* baseline parameter tuning.

### Test

A second frozen channel bank is generated and locked before training.

It is never used for:

* optimization;
* checkpoint selection;
* architecture choice;
* AFDM parameter tuning;
* hyperparameter selection.

It is used only for final evaluation.

This separation applies equally to Learned, AFDM, OTFS, OFDM, and all later parameterized waveform families.

---

# 13. Baselines

The required waveform set is

[
\boxed{
\text{TIME},
\quad
\text{OFDM},
\quad
\text{OTFS},
\quad
\text{AFDM},
\quad
\text{Learned}.
}
]

TIME is

[
W_{\rm TIME}=I.
]

All methods receive exactly the same:

* information symbols;
* DD channel realization;
* AWGN realization;
* modulation;
* transmit energy;
* block dimension (N);
* receiver budget (K);
* equalizer construction.

For every baseline,

[
A_b=W_b^HHW_b
]

and then exactly the same (T_K), (G_K), and hard symbol detector are used.

Any configurable baseline parameter must be selected using training/validation data only.

---

# 14. Mandatory full-information sanity check

When no effective-channel coefficients are discarded,

[
K=N,
]

all square unitary waveforms must recover the same full-LMMSE performance within numerical/Monte-Carlo tolerance.

Thus

[
\boxed{
\mathrm{MSE}*{\rm TIME}
\approx
\mathrm{MSE}*{\rm OFDM}
\approx
\mathrm{MSE}*{\rm OTFS}
\approx
\mathrm{MSE}*{\rm AFDM}
\approx
\mathrm{MSE}_{\rm Learned}.
}
]

The same should hold asymptotically for BER under the identical full-LMMSE receiver.

If it does not, there is an implementation, normalization, or convention error.

This invariant remains permanently in the test suite.

---

# 15. Stage 0 — Scientific harness and invariants

## Purpose

Before learning anything, prove that the experimental environment itself is correct.

Stage 0 should be intentionally boring.

## Stage 0A — Channel/operator validation

Verify:

* integer circular delays;
* fractional circular delays;
* Doppler phase convention;
* composition of multipath DD channels;
* path-energy normalization;
* AWGN scaling;
* deterministic random seeds.

Check known identities numerically.

## Stage 0B — Waveform validation

For every unitary waveform verify

[
W^HW\approx I.
]

Verify

[
|Ws|^2=|s|^2.
]

For

[
H=I
]

verify

[
W^HIW=I.
]

## Stage 0C — Receiver validation

Verify the mismatched receiver formula against direct Monte-Carlo symbol/noise simulation.

The analytic loss

[
\frac1N
\left[
|G_KA-I|_F^2
+
\sigma^2|G_K|_F^2
\right]
]

must match empirical averaged symbol MSE.

## Stage 0D — Full-LMMSE invariance

For random channels verify that all unitary transforms give the same full-information LMMSE MSE.

## Stage 0E — Classical baseline curves

Produce BER/NMSE curves for TIME, OFDM, OTFS and AFDM before introducing Learned.

## Stage 0F — Channel-specific oracle diagnostic

Optionally retain a fully channel-specific dense unitary oracle

[
W_H^\star
]

as an **upper-bound headroom diagnostic only**.

This asks:

> If the waveform could change freely for every individual channel, how much representation improvement is theoretically available?

It is not a candidate deployable waveform.

### Stage-0 exit condition

No waveform learning begins until every invariant passes.

---

# 16. Stage 1A — Fixed-budget universal waveform existence test

This is the first actual waveform-discovery experiment.

## Question

[
\boxed{
\text{At one stringent finite receiver budget, does a better universal unitary }W\text{ exist?}
}
]

For the initial (P=3) channel,

[
\boxed{
K_0=P=3.
}
]

The interpretation is approximately one retained effective interaction per physical path.

This is an intentionally severe receiver restriction.

## Training distribution

[
H\sim p_{\mathcal H_{\rm DD}},
]

[
\boxed{
K=3
}
]

throughout training,

and

[
\boxed{
\rho_{\rm dB}\sim U[12,20].
}
]

Thus

[
\boxed{
W^\star_{K=3}
=============

\arg\min_{W^HW=I}
\mathbb E_{
H,\rho
}
[
\mathcal L_{\rm comm}(W,H,\rho,K=3)
].
}
]

## Initializations

Train independently from:

[
I,
]

[
W_{\rm OFDM},
]

[
W_{\rm OTFS},
]

[
W_{\rm AFDM},
]

and several random Haar-unitary seeds.

## What Stage 1A is looking for

A genuine communication advantage at the exact operating point around which the waveform was designed.

The primary comparison is held-out BER at

[
K=3.
]

The waveform must outperform the strongest classical baseline on previously unseen DD channels.

## Interpretation

Stage 1A is an **existence test**.

Its claim is deliberately limited:

> A fixed learned unitary waveform can outperform the classical bases when all methods are constrained to (K=3) effective interactions per row.

It does not yet establish receiver-budget robustness.

### Stage-1A success gate

Require:

* held-out BER advantage over the strongest baseline;
* statistically meaningful paired confidence interval;
* consistent improvement across independent training seeds;
* no violation of full-LMMSE equivalence;
* no hidden normalization advantage.

If Stage 1A fails while the channel-specific oracle shows strong headroom, the likely conclusion is:

[
\boxed{
\text{better channel-dependent coordinates exist, but one fixed universal transform may be insufficient.}
}
]

---

# 17. Stage 1B — Multi-budget universal waveform discovery

Stage 1B addresses the most important qualification of Stage 1A.

A waveform optimized specifically for (K=3) may simply specialize to (K=3).

Therefore we train a second waveform that must work well under several receiver budgets.

## Question

[
\boxed{
\text{Can one fixed waveform remain communication-efficient across several receiver complexities?}
}
]

For (P=3),

[
\boxed{
\mathcal K_{\rm train}
======================

# {P,2P,3P}

{3,6,9}.
}
]

These have a useful physical interpretation:

* (K=3): approximately one retained interaction per path;
* (K=6): approximately two per path;
* (K=9): approximately three per path.

## Training objective

Use the same DD channel and SNR draw to evaluate all three budgets.

Define

[
\boxed{
\mathcal L_{\rm multiK}
=======================

\frac13
\sum_{K\in{3,6,9}}
\mathcal L_{\rm comm}(W,H,\rho,K).
}
]

Then

[
\boxed{
W^\star_{\rm multiK}
====================

\arg\min_{W^HW=I}
\mathbb E_{
H,,
\rho_{\rm dB}\sim U[12,20]
}
[
\mathcal L_{\rm multiK}
].
}
]

We do not initially introduce hand-tuned weights among the three budgets.

Equal weighting avoids deciding beforehand that one budget is more important.

## What Stage 1B is looking for

We want to know whether one representation is intrinsically efficient rather than merely tuned to one truncation point.

### Particularly strong outcome

If

[
W^\star_{\rm multiK}
]

nearly matches the separately optimized (K=3) waveform at (K=3), while also remaining strong at (K=6) and (K=9), then we have evidence for a genuinely receiver-budget-efficient basis.

### Interesting negative outcome

If (W^\star_{K=3}) is excellent at (K=3) but the multi-(K) waveform cannot simultaneously perform well at (K=3,6,9), that suggests:

[
\boxed{
\text{the optimal waveform representation may itself depend on receiver complexity.}
}
]

That would become an important later waveform-family question rather than an experimental failure.

---

# 18. Hard top-(K) training details

Hard top-(K) selection is piecewise differentiable.

Gradients naturally flow through selected entries but not through the discrete support decision.

We begin with the simplest implementation:

[
\boxed{
\text{exact hard top-}K.
}
]

If support switching causes unstable optimization, only then test:

* straight-through top-(K);
* soft top-(K);
* differentiable sparse masking.

However:

[
\boxed{
\text{all final reported results use the real hard top-}K\text{ receiver.}
}
]

A training relaxation must never silently become the evaluation receiver.

---

# 19. Stage 2 — Frozen communication validation

Once Stage 1A and Stage 1B finish, freeze the learned transforms.

No test-time adaptation.

No retraining.

No channel-specific optimization.

Stage 2 determines whether the learned result is genuinely useful.

---

## 19.1 BER versus receiver budget

Evaluate

[
K=1,2,3,\ldots,N
]

or an appropriately dense subset including

[
1,2,3,4,6,9,12,16,24,32,N.
]

Produce the key figure

[
\boxed{
\mathrm{BER}\text{ versus }K.
}
]

### Stage-1A waveform interpretation

For

[
W^\star_{K=3},
]

only (K=3) was seen during training.

Therefore its sweep is explicitly labeled

[
\boxed{
\text{receiver-budget generalization}.
}
]

### Stage-1B waveform interpretation

For

[
W^\star_{\rm multiK},
]

the training budgets were

[
3,6,9.
]

Its sweep tests both:

* robustness inside the training budget range;
* generalization outside it.

This is the waveform used for any broader claim that Learned reaches the full-information limit using fewer effective interactions.

---

# 20. Stage 2 SNR evaluation

Training uses only

[
12\leq\rho_{\rm dB}\leq20.
]

Stage 2 evaluates a much wider range, for example

[
\boxed{
0,\ 5,\ 10,\ 12,\ 15,\ 18,\ 20,\ 25\ {\rm dB}
}
]

with a denser grid if necessary.

Results between 12 and 20 dB are **in-distribution SNR evaluation**.

Results outside that range are explicitly **SNR generalization**.

This distinction must appear in plots and discussion.

---

# 21. Stage 2 channel-distribution generalization

The learned waveform must be tested against controlled changes in the DD environment.

Stress tests should include:

### Path count

Train initially with

[
P=3
]

and evaluate other values such as

[
P=1,2,4,5,6.
]

### Delay support

Test narrower and wider delay spreads than training.

### Doppler support

Test narrower and wider Doppler spreads.

### Fractional offsets

Increase off-grid severity.

### Path-power profiles

Test different path-power distributions.

### Boundary extrapolation

Test channels slightly beyond the trained support.

The purpose is to determine whether Learned captured a DD-relevant representation or merely a narrow training distribution.

---

# 22. Stage 2 BER remains the final judge

The analytic expected MSE is used because it supplies a clean differentiable gradient.

But final success is determined by actual symbol decisions.

Generate QPSK blocks and compute

[
\boxed{
\mathrm{BER}.
}
]

Also report

[
\boxed{
\mathrm{NMSE/EVM}.
}
]

Where possible, every waveform uses the identical:

* information bits;
* channel draw;
* AWGN draw.

This gives paired statistical comparisons.

For the major Learned-versus-baseline claims, report confidence intervals for

[
\Delta\mathrm{BER}
==================

## \mathrm{BER}_{\rm Learned}

\mathrm{BER}_{\rm baseline}.
]

---

# 23. Training-surrogate validation

A communication loss still needs empirical validation.

We therefore compare the training objective against held-out BER.

If

[
\mathcal L_{\rm comm}
]

improves substantially but BER does not, the response is **not** to return to a structural (A)-metric.

Instead, the communication surrogate itself must be improved.

Possible next losses include:

* differentiable QPSK likelihood;
* soft minimum-distance classification;
* cross-entropy over constellation points;
* GMI/rate surrogates.

The principle remains:

[
\boxed{
\text{replace an inadequate communication surrogate with a better communication surrogate.}
}
]

We do not replace it with an aesthetically attractive operator objective.

---

# 24. Stage 2 practical-support reality check

Top-(K) is initially oracle support.

Therefore a second question must be answered:

> Are the important interactions predictable enough that a real receiver could know where to look?

Analyze whether the selected support has:

* a fixed pattern;
* a deterministic shift law;
* locality;
* banded structure;
* a low-dimensional dependence on ((\tau,\nu));
* cheap pilot-based estimability.

If Learned wins only when given exact oracle support from the complete (A), we have established representation headroom but **not yet** practical receiver complexity reduction.

Any practical-complexity claim waits until this issue is resolved.

---

# 25. Stage 2 success criterion

The forward path proceeds to structural waveform discovery only if Learned demonstrates a real communication advantage.

The strongest outcome would be:

[
\boxed{
\begin{aligned}
&\text{Learned beats AFDM/OTFS/OFDM at matched }K,\
&\text{the gain survives unseen DD channels},\
&\text{the gain survives a useful SNR range},\
&\text{the result reproduces across training seeds},\
&\text{and Learned approaches the full-information bound at lower }K.
\end{aligned}
}
]

Only after this do we ask what the learned waveform looks like.

---

# 26. Stage 3 — Structural forensics

Stage 3 is deliberately downstream of communication validation.

Its question is:

[
\boxed{
\text{What did communication-driven optimization discover?}
}
]

---

## 26.1 Examine (W)

For every basis column (w_k), inspect:

### Time-domain magnitude

[
|w_k[n]|.
]

### Phase

[
\angle w_k[n].
]

### Unwrapped phase evolution

Look for polynomial/chirp relationships.

### Instantaneous frequency

Study discrete derivatives of phase.

### Frequency spectrum

[
|\mathcal F{w_k}|.
]

### Column relationships

Determine whether columns are generated by:

* cyclic shifts;
* frequency shifts;
* chirp shifts;
* permutation laws;
* common prototype atoms.

---

# 27. Stage 3 atomic DD operator analysis

For an elementary DD shift define

[
\boxed{
B_W(\tau,\nu)
=============

W^H
\Delta(\nu)
\Pi(\tau)
W.
}
]

Study:

* support;
* top-(K) retained energy;
* support trajectory versus delay;
* support trajectory versus Doppler;
* magnitude relationships;
* phase relationships;
* permutation-like actions;
* block actions;
* bandedness;
* chirp structure.

Compare all of these against OFDM, OTFS and AFDM.

---

# 28. Stage 3 composite-channel analysis

The actual multipath operator is

[
A_W(H)
======

\sum_p
h_p B_W(\tau_p,\nu_p).
]

Do not infer composite behavior only from individual paths.

Explicitly analyze how path contributions combine.

This is particularly important because earlier evidence suggests useful representations may exploit

[
\boxed{
\text{constructive accumulation in useful entries}
}
]

and

[
\boxed{
\text{destructive cancellation in unimportant entries}.
}
]

That phenomenon may be more important than making every atomic path individually sparse.

---

# 29. Stage 3 output

Structural forensics should produce concrete hypotheses such as:

* quadratic/chirp phase;
* higher-order polynomial phase;
* repeated sparse mixers;
* FFT/diagonal decomposition;
* deterministic DD support law;
* block-circulant structure;
* butterfly structure;
* common prototype pulse;
* low-dimensional parameter dependence.

We do not force one of these conclusions.

If the successful dense transform remains genuinely high dimensional, that is itself an important result.

---

# 30. Stage 4 — Discover a scalable structured waveform family

The dense learned matrix is an upper bound.

The next object must be

[
\boxed{
W_N(\theta)
}
]

with

[
\dim(\theta)\ll N^2.
]

A generic structured form is

[
\boxed{
W_N(\theta)
===========

T_L(\theta_L)
T_{L-1}(\theta_{L-1})
\cdots
T_1(\theta_1).
}
]

Possible stages include:

* FFT/IFFT;
* diagonal phase operators;
* chirp multipliers;
* permutations;
* sparse (2\times2) unitary rotations;
* butterfly stages;
* short filters.

The architecture should be informed by Stage 3.

---

# 31. Structured family is trained on communications, not matrix reconstruction

We do not define success by

[
|W_N(\theta)-W_{\rm dense}|_F.
]

Two matrices can be numerically close yet differ significantly in communication behavior.

Instead, train

[
\boxed{
\theta^\star
============

\arg\min_\theta
\mathbb E_{H,\rho,K}
[
\mathcal L_{\rm comm}(W_N(\theta),H,\rho,K)
].
}
]

Then compare

[
\boxed{
\text{dense Learned upper bound}
}
]

against

[
\boxed{
\text{structured Learned family}.
}
]

We care about how much **communication gain** the compact family retains.

---

# 32. Stage 4 requirements

A structured waveform becomes scientifically interesting only if it provides:

1. strong BER performance;
2. far fewer parameters than dense (W);
3. deterministic construction for arbitrary (N);
4. no stored (N\times N) codebook;
5. substantially subquadratic application complexity;
6. preferably approximately (O(N\log N)).

The final family should eventually resemble classical transforms in spirit:

[
\boxed{
\text{few parameters}
+
\text{mathematical construction}
+
\text{fast algorithm}.
}
]

---

# 33. Stage 5 — Fixed waveform versus channel-class-informed family

The optimal answer may not be one universal transform.

If channel-specific oracle experiments continue to show much greater headroom than a fixed waveform, investigate a controlled adaptive family.

The desired form is not

[
W(H)
]

with (O(N^2)) channel-dependent parameters.

Instead define a compact DD descriptor

[
\boxed{
q=
[
\tau_{\max},
\nu_{\max},
P,
\sigma_\tau,
\sigma_\nu,
\ldots
].
}
]

Then

[
\theta=f_\psi(q)
]

and

[
\boxed{
W_N=W_N(f_\psi(q)).
}
]

The network is permitted to output only a small waveform-parameter vector.

It never outputs an (N\times N) matrix.

This keeps the problem in waveform-family design rather than turning it into arbitrary channel-dependent precoding.

---

# 34. Desired analytical endpoint

Train the parameter mapper over many DD regimes.

Then inspect

[
q\rightarrow\theta^\star.
]

If regular relationships emerge, replace the learned mapper by an analytical law:

[
\boxed{
\theta
======

f(
\tau_{\max},
\nu_{\max},
P,
\sigma_\tau,
\sigma_\nu,\ldots
).
}
]

The ultimate waveform would then require:

* no online gradient descent;
* no dense matrix storage;
* no waveform codebook.

A handful of channel-class descriptors would generate the waveform parameters.

---

# 35. Optional later branch — nonorthogonal (W)

Nonorthogonality is not rejected.

It is intentionally deferred.

We first exhaust the clean unitary problem because it provides:

* exact power preservation;
* white post-analysis noise;
* no receiver conditioning pathology;
* only one learned transform;
* strong mathematical invariants.

If unitary learning reaches a clear ceiling, then permit

[
\boxed{
W^HW\neq I.
}
]

At that point we must explicitly control:

[
\operatorname{tr}(W^HW),
]

singular-value spread,

[
\kappa(W),
]

and receiver noise amplification.

If an independent receive transform (V) is learned,

[
A=VHW
]

and the transformed noise covariance becomes

[
\boxed{
C_n'
====

\sigma^2VV^H.
}
]

All communication losses must account for that covariance.

No result derived under unitary white-noise assumptions can be reused blindly.

---

# 36. Optional later branch — redundancy / rectangular synthesis

Another expansion is

[
W\in\mathbb C^{M\times N},
\qquad M>N.
]

This introduces genuine waveform-shaping redundancy.

Unlike a square unitary coordinate change, a rectangular transform can modify the geometry of the transmitted signal subspace.

This is potentially powerful, but it changes:

* rate;
* sample count;
* bandwidth;
* receiver observations;
* fairness conditions.

Therefore it is a separate research stage, not part of the initial waveform experiment.

---

# 37. Stage 6 — Physicalize the waveform

The early stages deliberately isolate discrete-time DD representation.

A complete physical waveform requires an extended model.

Conceptually,

[
\boxed{
s
\rightarrow
W_N(\theta)
\rightarrow
\text{interpolation/pulse shaping}
\rightarrow
x(t)
\rightarrow
H_c
\rightarrow
\text{receive filter}
\rightarrow
\text{sampling}
\rightarrow
\text{detector}.
}
]

Now introduce:

* transmit pulse shape;
* receive pulse shape;
* oversampling;
* cyclic prefix or guard interval;
* occupied bandwidth;
* PSD;
* OOB emissions;
* ACLR;
* PAPR;
* DAC/ADC sampling rate;
* receiver observation count.

These issues are necessary for a physically fair comparison of learned waveforms with OTFS and AFDM.

---

# 38. Physical fairness

Ultimately the comparison must control physical resources, not merely matrix dimension.

Report:

[
N_{\rm information\ symbols},
]

block duration,

occupied bandwidth,

sampling rate,

pulse shape,

CP/guard overhead,

transmit energy,

ACLR/OOB,

PAPR,

and receiver complexity.

Potential fairness modes include:

### Equal mathematical dimensions

Useful for fundamental operator analysis.

### Equal physical front end

Same bandwidth, duration, sampling rate, and energy.

### Equal spectral requirement

Require every waveform to satisfy the same ACLR/spectral-mask constraint.

Receiver oversampling such as

[
Q=1,2,4
]

must be available to every waveform, not only Learned or OTFS.

---

# 39. Stage 7 — Scaling

Once we possess a genuine rule

[
W_N(\theta),
]

test whether it scales.

Evaluate

[
\boxed{
64
\rightarrow
256
\rightarrow
1024
\rightarrow
4096
\rightarrow
8192
}
]

parallel transmitted waveform dimensions.

For every (N), report:

* BER;
* transform runtime;
* receiver runtime;
* parameter count;
* storage;
* memory;
* application complexity;
* PAPR;
* spectrum;
* physical-resource usage.

The family must be generated by its mathematical rule.

We do not store a separately learned dense matrix for every (N).

---

# 40. Final stage hierarchy

The complete project is:

## Stage 0 — Harness

**Question:** Is the simulation mathematically correct?

**Output:** validated channel, baselines, receiver and invariants.

---

## Stage 1A — Fixed-(K) existence

[
K=3,
\qquad
\rho_{\rm dB}\sim U[12,20].
]

**Question:** Can a universal unitary Learned waveform beat the strongest classical waveform at one stringent finite receiver budget?

---

## Stage 1B — Multi-(K) robustness

[
K\in{3,6,9},
\qquad
\rho_{\rm dB}\sim U[12,20].
]

**Question:** Can one learned representation remain efficient across several receiver budgets?

---

## Stage 2 — Frozen communication validation

Sweep:

[
K,
\qquad
\mathrm{SNR},
\qquad
\text{DD conditions}.
]

**Question:** Does the communication advantage survive held-out channels and operating conditions?

Primary judge:

[
\boxed{\mathrm{BER}.}
]

---

## Stage 3 — Structural forensics

Study

[
W
]

and

[
A=W^HHW.
]

**Question:** What structure did communication-driven optimization discover?

---

## Stage 4 — Compact waveform family

Construct

[
W_N(\theta).
]

**Question:** Can most of the dense learned communication gain be reproduced using a small, scalable mathematical transform?

---

## Stage 5 — Channel-class-informed family

Construct

[
W_N(f(q)).
]

**Question:** Can a few DD descriptors determine the appropriate waveform parameters?

---

## Optional branch — Nonorthogonal/redundant waveform

Open only if unitary learning reaches a scientifically meaningful ceiling.

---

## Stage 6 — Physical waveform

Introduce pulse shaping, sampling, bandwidth, ACLR, PAPR and realistic front-end constraints.

**Question:** Does the discovered waveform remain advantageous as an actual transmit waveform?

---

## Stage 7 — Scaling

Evaluate large (N).

**Question:** Is this truly a scalable waveform family rather than a small-matrix learning result?

---

# 41. Decision tree

The project is intentionally falsifiable.

### If Stage 1A succeeds

A better fixed representation exists at the target receiver budget.

Proceed to Stage 1B.

---

### If Stage 1A fails and the channel-specific oracle also shows little headroom

There is little evidence that the chosen channel/receiver regime contains meaningful waveform-design opportunity.

Reconsider the physical problem rather than forcing learning.

---

### If Stage 1A fails but the channel-specific oracle strongly wins

Better representations exist, but a universal fixed (W) may not.

Proceed toward a low-dimensional channel-class-dependent family.

---

### If Stage 1A succeeds but Stage 1B fails

The optimal waveform may depend on receiver complexity.

This becomes a waveform-family dimension:

[
W_N(\theta;K).
]

---

### If MSE improves but BER does not

The differentiable communications surrogate is inadequate.

Improve the communication loss.

Do **not** substitute an arbitrary (A)-structure loss.

---

### If Learned wins only under oracle top-(K)

The waveform possesses useful representation geometry, but practical support acquisition remains unsolved.

Do not make low-complexity receiver claims until that is addressed.

---

### If dense Learned wins but structured families cannot preserve the gain

A useful high-dimensional transform exists, but an analytic scalable waveform law has not yet been discovered.

That distinction must be reported accurately.

---

### If the unitary family reaches a hard ceiling

Only then investigate nonorthogonality or redundancy.

---

# 42. What the project will not do

We will not:

* design (A) first;
* invent target effective-channel matrices;
* optimize structural metrics and call them communication success;
* make Learned imitate AFDM or OTFS;
* give Learned a stronger receiver than the baselines;
* reuse test channels during waveform selection;
* call oracle top-(K) free receiver complexity;
* attach a large neural decoder simply to manufacture BER gains;
* call one dense (64\times64) matrix a waveform family;
* introduce every possible degree of freedom simultaneously;
* claim physical waveform superiority before bandwidth, sampling and spectral fairness are evaluated.

---

# 43. Central scientific hypothesis

The forward path tests the following hypothesis:

[
\boxed{
\begin{aligned}
&\text{There exists a coordinate system for doubly dispersive channels}\
&\text{that preserves the same underlying physical information as the}\
&\text{classical unitary transforms but concentrates that information}\
&\text{into a representation that can be exploited using substantially}\
&\text{fewer effective receiver interactions.}
\end{aligned}
}
]

If such a transform exists, communication-driven learning should be capable of finding it without us prescribing its effective-channel geometry.

---

# 44. Ultimate research hypothesis

The deeper hypothesis is:

[
\boxed{
\begin{aligned}
&\text{The successful learned transform will contain enough recurring}\
&\text{structure that its communication advantage can eventually be}\
&\text{distilled into a compact mathematical waveform family }W_N(\theta),\
&\text{possibly parameterized by a small number of DD channel descriptors.}
\end{aligned}
}
]

The dense learned matrix is therefore not the destination.

It is the experimental instrument used to expose the waveform physics.

---

# 45. Final governing principle

Throughout the entire project:

[
\boxed{
\textbf{BER and symbol recoverability decide whether the waveform is good.}
}
]

[
\boxed{
\textbf{(A=W^HHW) tells us what the waveform did to the channel representation.}
}
]

[
\boxed{
\textbf{Structural properties are explanations only after communications success.}
}
]

The final causal program is therefore:

[
\boxed{
\begin{aligned}
\text{DD physics}
&\rightarrow
\text{communication objective}
\
&\rightarrow
\text{dense waveform discovery}
\
&\rightarrow
\text{held-out BER proof}
\
&\rightarrow
\text{operator forensics}
\
&\rightarrow
\text{compact transform family}
\
&\rightarrow
\text{channel-class law}
\
&\rightarrow
\text{physical waveform}
\
&\rightarrow
\text{large-scale implementation}.
\end{aligned}
}
]

That is the forward-path project.
