# Pub-1E frozen protocol

Frozen before new production evaluation on 2026-09-05. Machine-readable authority: `configs/stage_pub1e.json`.

## Question and boundary

Test where the existing N=64 Learned unitary helps under limited effective-channel representation, whether the effect survives stronger linear receivers, and how fixed support behaves under prior shift. No training, checkpoint selection, channel estimation, coding, MIMO, pulse shaping, hardware work, or Pub-1F execution.

The model remains `x=Ws`, `y=Hx+n`, and `z=W^H y=As+n'`, with `A=W^H H W`. `H=sum_p h_p Delta(nu_p) Pi(tau_p)`: negative Doppler sign; delay first; each realization normalized to `||H||_F^2/N=1`. Simulated observations always use full `A`. Restricted matrices affect receiver construction only.

## Frozen inputs and controls

- Learned: Stage-1A.2 `tail_from_ofdm_best.pt`, required SHA-256 `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`.
- Primary baselines: OTFS, validation-tuned AFDM, OFDM. Identity is retained in machine-readable numerical checks.
- AFDM named curves: historical `2Nc1=1`, canonical `3`, fractional guard `5`. Selection considers guarded odd slopes 3/5/7/9 using validation-only mean analytic MSE; smallest multiplier breaks exact ties.
- Every waveform gets its own realized row-top-K or its own separately constructed fixed statistical support. Retained entries use actual realized coefficients.

## Banks, pairing, and precision

All bank seeds and sizes are frozen in the JSON. AFDM validation, receiver validation, nominal-support construction, residual-statistics construction, exploratory mapping, confirmatory evaluation, and shifted-prior construction/evaluation are disjoint. Within a condition, all waveforms share physical channels, bit tensors, and physical-domain noise. Controlled fractionality shares integer anchors, gains, bits, and noise across epsilon values through deterministic reseeding; only declared offsets change.

Primary confirmatory conditions use 512 channels and 2,048 bits/channel/modulation. The 6x6 map uses 128 channels/cell and is exploratory. BER is total errors/total bits. Uncertainty is a paired percentile bootstrap over independent channels. Map intervals are pointwise. Raw zero errors remain zero; log-ratio plots use the declared Jeffreys display estimate and mark comparisons with fewer than five errors on either side.

## Receiver ladder

1. Historical: row-top-K `A_tilde`, `G=(A_tilde^H A_tilde + sigma2 I)^-1 A_tilde^H`.
2. Tuned regularization: same equation with validation-selected multiplier on `sigma2`, candidates 0.25/0.5/1/2/4.
3. Residual-aware approximation: let `E=A-A_tilde`. A separate bank estimates `M_E=E[E]` and `R_E=E[(E-M_E)(E-M_E)^H]` for each waveform/condition/K. Online, `B=A_tilde+M_E` and `G=B^H(BB^H+R_E+sigma2 I)^-1`. This includes mean cross terms through `B`, but neglects dependence of residual statistics on the realized retained values/support. It is an approximation, not the conditional-optimal LMMSE. Online input remains `A_tilde`; extra stored information is one dense mean and receive covariance per waveform/condition/K.
4. Full-information: direct LMMSE from complete `A`, the unlimited-representation reference.

At K=N, `E=0`; receivers 1 and 3 must coincide. Direct complex128 solves avoid iterative-convergence confounding.

## Experiments

- E1: budgets K=8/16/24/32/48/64, nominal DD-v1, anchored integer, and anchored joint epsilon=0.5; all receiver families; selected SNR curves on the frozen grid 12/15/18/20/25 dB.
- E2: epsilon-delay by epsilon-Doppler 0.0:0.1:0.5, N=64/K=24/18 dB, 16-QAM primary and QPSK secondary; historical receiver. K=16/32 are companion conditions if completed, never silently pooled.
- E3: nominal fixed, separately recalibrated fixed, and oracle row-top-K supports under delay factors 1/1.5/2, Doppler factors 1/1.5/2, and delay-correlated path-power exponents 0/0.75/1.5. Historical receiver first.
- E4: locked favorable joint (0.5,0.5), boundary joint (0.2,0.2), unfavorable integer (0,0), plus nominal full-information. These labels are historical/predeclared, not selected from Pub-1E test outcomes.

## Protocol differences from historical stages

| Item | Earlier publication protocol | Pub-1E |
|---|---|---|
| Receiver | Historical truncated LMMSE; full reference | Adds validation-tuned regularization and disclosed residual-aware approximation |
| Fractionality | Mostly matched-axis severity | Full separate 6x6 delay/Doppler map |
| Fixed support | Nominal or regime-informed | Explicit nominal-prior mismatch versus recalibration versus oracle |
| AFDM | Historical slope 1 or Pub-1A candidates 3/5 | Preserves named 1/3/5; tunes only within guarded 3/5/7/9 |
| Uncertainty | Paired bootstrap in selected studies | Channel-level paired bootstrap and per-channel counts throughout |
| Low errors | Study-specific ratio fallbacks | Raw counts plus Jeffreys display-only ratio and precision marks |

Equal representation is primary: same K and receiver family. Equal detector budget is deferred to Pub-1F; no waveform-specific K/iteration optimization is used for the headline claim.
