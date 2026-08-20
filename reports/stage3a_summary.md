# Stage 3A — Formal DD Coupling Law and Q-Concentration Principle

## Executive verdict

**Outcome A — Coupling/concentration principle supported.** The raw law passed every controlled case within the predetermined 3-SE plus Q-sampling tolerance; normalized Q-to-C rank correlation remained at least 0.994032. Matched-budget eta_Q cleanly separated all OFDM-start/authoritative checkpoints from all Haar checkpoints (0.9870 versus at most 0.8580), while Learned eta_24=0.998136 tracked 0.998090 realized retained energy.

No waveform was trained, tuned, or modified. Stage 2E was neither imported nor used.

## 1. Model and unnormalized law

With the existing atomic operator $B_W(\tau,\nu)=W^H\Delta(\nu)\Pi(\tau)W$, define $b_{ij}(\tau,\nu)=[B_W(\tau,\nu)]_{ij}$. Then $A_{ij}^{raw}=\sum_p h_p b_{ij,p}$ and

$$\mathbb E|A_{ij}^{raw}|^2=\sum_{p,q}\mathbb E[h_ph_q^*b_{ij,p}b_{ij,q}^*].$$

Independent zero-mean gains, independent of path locations, give $\mathbb E[h_ph_q^*]=0$ for $p\ne q$. Thus only diagonal terms remain:

$$C_{ij}^{raw}=\sum_p \sigma_p^2\int |w_i^H\Delta(\nu)\Pi(\tau)w_j|^2\,d\mu_p(\tau,\nu).$$

For iid locations $\mu_p=\mu$ and $h_p\sim\mathcal{CN}(0,1/P)$, this is exactly $C_{ij}^{raw}=Q_{ij}$. If gains have nonzero cross-covariance or are correlated with locations, cross terms need not vanish and this equality does not follow.

## 2. Q as discrete DD cross-ambiguity

$Q_{ij}=\iint p_{\tau,\nu}(\tau,\nu)|w_i^H\Delta(\nu)\Pi(\tau)w_j|^2d\tau d\nu$ is the channel-distribution-weighted second moment of the finite-dimensional pairwise DD cross-ambiguity. It is not claimed to be a continuous-time ambiguity function. The bridge is $W\rightarrow B_W\rightarrow Q\rightarrow C$.

## 3. Per-realization normalization

The simulator uses $\alpha(H_{raw})=\sqrt{N}/\|H_{raw}\|_F$, so $C_{ij}^{norm}=\mathbb E[\alpha^2|A_{ij}^{raw}|^2]$. No independence between $\alpha$ and an entry is assumed. Normalization fixes $\sum_{ij}C_{ij}^{norm}=N$ exactly but can change scale, ordering, and support.

| waveform | Q/raw rho | Q/norm rho | raw->norm scale | raw/norm rel. Frobenius | Q/norm top-24 overlap | exact fixed-Q capture from Cnorm |
|---|---:|---:|---:|---:|---:|---:|
| Learned | 0.999967 | 0.999964 | 0.999094 | 0.0191 | 1.0000 | 0.998102 |
| OFDM | 0.998623 | 0.998422 | 1.020509 | 0.0080 | 0.9857 | 0.990735 |
| OTFS | 0.995508 | 0.994032 | 0.973977 | 0.0244 | 1.0000 | 0.939999 |
| AFDM | 0.999691 | 0.999780 | 0.983243 | 0.0141 | 0.9967 | 0.982999 |

## 4. Numerical theorem verification

| case | waveform | expected scale | relative error | relative MC SE | within 3SE + Q tolerance |
|---|---|---:|---:|---:|---|
| A_P1_RAW | Learned | 1.000 | 0.04408 | 0.02723 | True |
| B_P3_RAW | Learned | 1.000 | 0.01915 | 0.02199 | True |
| B_P3_RAW | OFDM | 1.000 | 0.01595 | 0.01960 | True |
| B_P3_RAW | OTFS | 1.000 | 0.02439 | 0.03150 | True |
| B_P3_RAW | AFDM | 1.000 | 0.01227 | 0.02567 | True |
| D_UNEQUAL_RAW | Learned | 1.500 | 0.01649 | 0.02215 | True |

Case A is P=1 raw; Case B is P=3 raw with variance 1/3 per path; Case C uses the same Case-B bank after exact project normalization; Case D uses unequal variances (0.9, 0.4, 0.2). P variation is limited to this required theorem check, not waveform exploration.

## 5. Q concentration and exact support link

For row-sorted $q_{i,(m)}$, $r_i^Q(K)=\sum_{m\le K}q_{i,(m)}/\sum_jQ_{ij}$ and $\eta_K^Q=\sum_i\sum_{m\le K}q_{i,(m)}/\sum_{ij}Q_{ij}$. Effective support is row IPR, $(\sum_jQ_{ij})^2/\sum_jQ_{ij}^2$; normalized entropy is also reported.

For a fixed mask $S$, normalized unitarity gives $\|A\|_F^2=N$ per realization, hence $\mathbb E[\mathrm{retained}(S)]=N^{-1}\sum_{(i,j)\in S}C_{ij}^{norm}$ exactly. For $S=S^Q$ and $C^{norm}\approx Q$ with both total masses $N$, this is approximately $\eta_K^Q$. The C statement is exact; the Q replacement is empirical.

| waveform | eta24 | row p05 | effective support | K90 | K95 | K99 | K99.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned | 0.998136 | 0.992160 | 8.93 | 15 | 17 | 21 | 23 |
| OFDM | 0.990948 | 0.990948 | 2.48 | 3 | 6 | 23 | 36 |
| OTFS | 0.941752 | 0.941752 | 12.13 | 17 | 27 | 44 | 48 |
| AFDM | 0.983470 | 0.979648 | 6.24 | 7 | 10 | 34 | 47 |

## 6. Frozen learned checkpoints

| waveform | K | start | validation CVaR | Q/C rho | etaK | row p05 | eff. support | fixed retained | QPSK fixed BER | 16-QAM fixed BER |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned | 24 | OFDM (Stage-1A.2) | 0.16288 | 0.999818 | 0.998136 | 0.992160 | 8.93 | 0.998090 | 0.00111961 | 0.0248499 |
| K16_OFDM | 16 | OFDM | 0.23233 | 0.999911 | 0.986976 | 0.967009 | 3.93 | 0.986515 | 0.00389194 | 0.0413809 |
| K16_HAAR_1 | 16 | HAAR_1 | 0.55617 | 0.999727 | 0.858023 | 0.766315 | 12.85 | 0.858456 | 0.0670376 | 0.191014 |
| K16_HAAR_2 | 16 | HAAR_2 | 0.92465 | 0.999387 | 0.759505 | 0.657155 | 18.84 | 0.760218 | 0.135456 | 0.264681 |
| K24_OFDM | 24 | OFDM | 0.18731 | 0.999912 | 0.996538 | 0.989293 | 4.65 | 0.996429 | 0.00186443 | 0.0270424 |
| K24_HAAR_1 | 24 | HAAR_1 | 0.58543 | 0.999258 | 0.810180 | 0.728457 | 21.80 | 0.809790 | 0.0988016 | 0.229485 |
| K24_HAAR_2 | 24 | HAAR_2 | 0.66526 | 0.999016 | 0.782275 | 0.698727 | 23.98 | 0.781265 | 0.11876 | 0.249338 |
| K32_OFDM | 32 | OFDM | 0.15389 | 0.999912 | 0.996857 | 0.989525 | 9.04 | 0.996765 | 0.0010128 | 0.0261908 |
| K32_HAAR_1 | 32 | HAAR_1 | 0.35054 | 0.999070 | 0.855648 | 0.781634 | 27.86 | 0.854767 | 0.069767 | 0.198132 |
| K32_HAAR_2 | 32 | HAAR_2 | 0.35401 | 0.999056 | 0.852949 | 0.778187 | 28.16 | 0.852620 | 0.0721169 | 0.201159 |

Exploratory across only ten frozen checkpoints: eta_vs_validation_objective_spearman=-0.952, eta_vs_fixed_retained_spearman=1.000, eta_vs_qpsk_fixed_ber_spearman=-0.988, eta_vs_qam16_fixed_ber_spearman=-1.000. These correlations are descriptive, not universal evidence.

## 7. Interpretation gate

1. **When is Q=C exact?** Raw channels; zero-mean mutually uncorrelated gains independent of locations; iid DD locations; and total gain variance one (specifically 1/P per path).
2. **What does normalization change?** It couples the scalar alpha to every entry, fixes total energy exactly, and modestly changes scale/structure/support as quantified above; it does not invalidate Q here.
3. **Good versus bad checkpoints?** The frozen OFDM-start/authoritative solutions occupy a visibly more concentrated regime than Haar solutions at matched K.
4. **Fixed-support energy?** Exactly determined by masked Cnorm mass; Q concentration predicts it closely because Q and Cnorm agree strongly.
5. **Finite-K communication?** Plausibly tracked across this small set, but BER also depends on coefficient geometry, conditioning, noise, and constellation; concentration alone is not a BER theorem.
6. **Candidate Stage-3 property.** Strong row concentration of channel-weighted Q at the receiver budget, equivalently large etaK with a robust row-tail constraint. This is a property, not yet an optimized loss.
7. **Unresolved.** Sufficiency beyond this channel class/checkpoint set, differentiable surrogates, row-tail tradeoffs, conditioning, modulation robustness, and behavior under direct optimization remain open.

## 8. Evidence classification

- **Proven/derived:** raw weighted Q-to-C law under stated assumptions; exact normalized masked-C retained-energy identity.
- **Numerically established:** raw theorem within Monte Carlo uncertainty; actual normalization distortion and preserved Q/C correspondence.
- **Empirically suggested:** Q concentration separates these good/weak frozen solutions and tracks finite-K BER.

Stage 3A stops here. No Stage 3B optimization was performed.
