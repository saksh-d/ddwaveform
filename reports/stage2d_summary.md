# Stage 2D — DD Coupling Law and Oracle-Support Removal

Outcome **A** under preregistered quantitative gates. This is diagnostic evidence only; no waveform was trained or selected.

## Protocol

- DD-v1 unchanged: N=64, P=3, continuous delay [0.0,4.0] and Doppler [-0.75,0.75].
- Q: 2048 deterministic scrambled Sobol single paths (seed 11062); C: 512 held-out channels (seed 11061).
- Empirical support prior: separate 512-channel construction bank (seed 11060); final receiver/BER: untouched 512-channel bank (seed 11063).
- Correlations exclude self-coupling diagonals. Fixed masks never inspect a realized test operator; retained coefficients remain the true realized coefficients.
- Full principal BER uses at least 1,048,576 paired bits per point; all-checkpoint 18 dB checks use the same paired-bit target.
- No training, fine-tuning, waveform selection, channel-spread variation, P variation, structural loss, or Stage 3 action.

## S/Q/C agreement

| waveform | comparison | Spearman | row mean | top-24 overlap | scaled Frobenius discrepancy |
|---|---|---:|---:|---:|---:|
| OFDM | S->Q | 0.9994 | 0.9993 | 0.9811 | 0.7192 |
| OFDM | Q->C | 0.9969 | 0.9984 | 0.9766 | 0.0288 |
| OFDM | S->C | 0.9969 | 0.9985 | 0.9785 | 0.7187 |
| OTFS | S->Q | 0.6730 | 0.6756 | 0.7500 | 0.7449 |
| OTFS | Q->C | 0.9901 | 0.9902 | 1.0000 | 0.0656 |
| OTFS | S->C | 0.6712 | 0.6740 | 0.7500 | 0.7449 |
| AFDM | S->Q | -0.0212 | -0.0213 | 0.4382 | 0.9562 |
| AFDM | Q->C | 0.9986 | 0.9970 | 0.9863 | 0.0563 |
| AFDM | S->C | -0.0202 | -0.0200 | 0.4355 | 0.9555 |
| Learned | S->Q | 0.9345 | 0.8981 | 0.9329 | 0.5876 |
| Learned | Q->C | 0.9998 | 0.9992 | 0.9993 | 0.0369 |
| Learned | S->C | 0.9348 | 0.8985 | 0.9323 | 0.5920 |

## K=24 support removal at 18 dB

| waveform | rule | recall | Jaccard | retained energy | analytic MSE | limiting distortion | condition p95 | stream CVaR 0.95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OFDM | Oracle | 1.0000 | 1.0000 | 0.99151 | 0.08143 | 229.50739 | 3083.21 | 0.26147 |
| OFDM | Q | 0.8484 | 0.7539 | 0.99041 | 0.08569 | 116144.73721 | 6349.10 | 0.26899 |
| OFDM | S | 0.8484 | 0.7541 | 0.99040 | 0.08576 | 200.42719 | 2148.89 | 0.27003 |
| OFDM | Centroid | 0.8486 | 0.7544 | 0.99040 | 0.08572 | 16632.49408 | 5770.12 | 0.26886 |
| OFDM | Prior | 0.8486 | 0.7543 | 0.99040 | 0.08574 | 784.74743 | 3685.91 | 0.26976 |
| OTFS | Oracle | 1.0000 | 1.0000 | 0.97609 | 0.10668 | 14.69127 | 969.24 | 0.14469 |
| OTFS | Q | 0.7155 | 0.5628 | 0.93827 | 0.22251 | 14614.32685 | 1202.88 | 0.29525 |
| OTFS | S | 0.6574 | 0.4945 | 0.91779 | 0.33297 | 21078.01360 | 4157.61 | 0.42749 |
| OTFS | Centroid | 0.5457 | 0.3833 | 0.75655 | 1.18775 | 151.21106 | 310.43 | 4.51200 |
| OTFS | Prior | 0.7155 | 0.5628 | 0.93827 | 0.22251 | 14614.32685 | 1202.88 | 0.29525 |
| AFDM | Oracle | 1.0000 | 1.0000 | 0.98606 | 0.09048 | 10.82167 | 664.57 | 0.18177 |
| AFDM | Q | 0.8069 | 0.6832 | 0.98261 | 0.10931 | 101.87505 | 1274.58 | 0.22709 |
| AFDM | S | 0.4565 | 0.2974 | 0.79672 | 0.69349 | 1815.49232 | 768.56 | 1.24161 |
| AFDM | Centroid | 0.7906 | 0.6610 | 0.98173 | 0.11269 | 5683.58926 | 3750.45 | 0.23456 |
| AFDM | Prior | 0.8070 | 0.6833 | 0.98260 | 0.10944 | 52.19868 | 1239.72 | 0.22770 |
| Learned | Oracle | 1.0000 | 1.0000 | 0.99854 | 0.04980 | 2.73338 | 1678.22 | 0.12769 |
| Learned | Q | 0.9635 | 0.9328 | 0.99809 | 0.05065 | 2.19086 | 1507.85 | 0.13030 |
| Learned | S | 0.9231 | 0.8734 | 0.99033 | 0.07421 | 14.60987 | 1205.75 | 0.26963 |
| Learned | Centroid | 0.7442 | 0.6167 | 0.87057 | 0.51322 | 55.05722 | 269.25 | 1.66027 |
| Learned | Prior | 0.9637 | 0.9331 | 0.99807 | 0.05069 | 2.90275 | 1744.69 | 0.13036 |

## Matched-budget selected Learned support test at 18 dB

| K | rule | retained energy | analytic MSE | limiting distortion | QPSK BER | 16-QAM BER |
|---:|---|---:|---:|---:|---:|---:|
| 16 | Oracle | 0.98985 | 0.06751 | 7.90800 | 0.003099442 | 0.03611374 |
| 16 | Q | 0.98651 | 0.07564 | 5.80209 | 0.003898621 | 0.04180717 |
| 16 | S | 0.98231 | 0.08717 | 11.56480 | 0.005291939 | 0.04857445 |
| 16 | Centroid | 0.88806 | 0.40271 | 304.91990 | 0.0571661 | 0.1508932 |
| 16 | Prior | 0.98647 | 0.07574 | 6.95222 | 0.003932953 | 0.04187202 |
| 24 | Oracle | 0.99854 | 0.04980 | 2.73338 | 0.001045227 | 0.02425671 |
| 24 | Q | 0.99809 | 0.05065 | 2.19086 | 0.001119614 | 0.02484989 |
| 24 | S | 0.99033 | 0.07421 | 14.60987 | 0.003615379 | 0.03931999 |
| 24 | Centroid | 0.87057 | 0.51322 | 55.05722 | 0.06751347 | 0.1843786 |
| 24 | Prior | 0.99807 | 0.05069 | 2.90275 | 0.001107216 | 0.02484608 |
| 32 | Oracle | 0.99798 | 0.05099 | 2.31505 | 0.0009717941 | 0.0246563 |
| 32 | Q | 0.99677 | 0.05336 | 1.61017 | 0.001055717 | 0.02636528 |
| 32 | S | 0.99275 | 0.06261 | 2.99738 | 0.00159359 | 0.03304958 |
| 32 | Centroid | 0.94725 | 0.22266 | 8.07038 | 0.0239687 | 0.1044674 |
| 32 | Prior | 0.99674 | 0.05340 | 5.80018 | 0.001049042 | 0.02644157 |

## Authoritative Learned BER

| modulation | SNR dB | support | BER | errors / bits |
|---|---:|---|---:|---:|
| QPSK | 12 | Oracle | 0.0093212128 | 9774 / 1048576 |
| QPSK | 12 | Q | 0.009513855 | 9976 / 1048576 |
| QPSK | 12 | S | 0.011926651 | 12506 / 1048576 |
| QPSK | 12 | Centroid | 0.060173035 | 63096 / 1048576 |
| QPSK | 12 | Prior | 0.0095205307 | 9983 / 1048576 |
| QPSK | 15 | Oracle | 0.0030517578 | 3200 / 1048576 |
| QPSK | 15 | Q | 0.0031328201 | 3285 / 1048576 |
| QPSK | 15 | S | 0.0055427551 | 5812 / 1048576 |
| QPSK | 15 | Centroid | 0.059943199 | 62855 / 1048576 |
| QPSK | 15 | Prior | 0.0031480789 | 3301 / 1048576 |
| QPSK | 18 | Oracle | 0.0010452271 | 1096 / 1048576 |
| QPSK | 18 | Q | 0.0011196136 | 1174 / 1048576 |
| QPSK | 18 | S | 0.0036153793 | 3791 / 1048576 |
| QPSK | 18 | Centroid | 0.067513466 | 70793 / 1048576 |
| QPSK | 18 | Prior | 0.0011072159 | 1161 / 1048576 |
| QPSK | 20 | Oracle | 0.00060844421 | 638 / 1048576 |
| QPSK | 20 | Q | 0.00065326691 | 685 / 1048576 |
| QPSK | 20 | S | 0.0034646988 | 3633 / 1048576 |
| QPSK | 20 | Centroid | 0.075487137 | 79154 / 1048576 |
| QPSK | 20 | Prior | 0.00064468384 | 676 / 1048576 |
| 16QAM | 12 | Oracle | 0.088097572 | 92377 / 1048576 |
| 16QAM | 12 | Q | 0.088505745 | 92805 / 1048576 |
| 16QAM | 12 | S | 0.095207214 | 99832 / 1048576 |
| 16QAM | 12 | Centroid | 0.18161488 | 190437 / 1048576 |
| 16QAM | 12 | Prior | 0.088438034 | 92734 / 1048576 |
| 16QAM | 15 | Oracle | 0.04846096 | 50815 / 1048576 |
| 16QAM | 15 | Q | 0.048859596 | 51233 / 1048576 |
| 16QAM | 15 | S | 0.059756279 | 62659 / 1048576 |
| 16QAM | 15 | Centroid | 0.17682838 | 185418 / 1048576 |
| 16QAM | 15 | Prior | 0.048913956 | 51290 / 1048576 |
| 16QAM | 18 | Oracle | 0.024256706 | 25435 / 1048576 |
| 16QAM | 18 | Q | 0.024849892 | 26057 / 1048576 |
| 16QAM | 18 | S | 0.039319992 | 41230 / 1048576 |
| 16QAM | 18 | Centroid | 0.18437862 | 193335 / 1048576 |
| 16QAM | 18 | Prior | 0.024846077 | 26053 / 1048576 |
| 16QAM | 20 | Oracle | 0.015584946 | 16342 / 1048576 |
| 16QAM | 20 | Q | 0.016119957 | 16903 / 1048576 |
| 16QAM | 20 | S | 0.032326698 | 33897 / 1048576 |
| 16QAM | 20 | Centroid | 0.19078827 | 200056 / 1048576 |
| 16QAM | 20 | Prior | 0.016171455 | 16957 / 1048576 |

## All Stage-2C checkpoints at 18 dB

| checkpoint | K | init | val. CVaR | Q->C | fixed-Q energy | QPSK oracle/fixed | 16-QAM oracle/fixed |
|---|---:|---|---:|---:|---:|---:|---:|
| K16_OFDM | 16 | OFDM | 0.232328 | 1.000 | 0.9865 | 0.0030632/0.0038919 | 0.036177/0.041381 |
| K16_HAAR_1 | 16 | HAAR_1 | 0.556167 | 1.000 | 0.8585 | 0.032915/0.067038 | 0.14071/0.19101 |
| K16_HAAR_2 | 16 | HAAR_2 | 0.924653 | 0.999 | 0.7602 | 0.07967/0.13546 | 0.20733/0.26468 |
| K24_OFDM | 24 | OFDM | 0.187307 | 1.000 | 0.9964 | 0.0016966/0.0018644 | 0.025324/0.027042 |
| K24_HAAR_1 | 24 | HAAR_1 | 0.585429 | 0.999 | 0.8098 | 0.045181/0.098802 | 0.16554/0.22948 |
| K24_HAAR_2 | 24 | HAAR_2 | 0.665261 | 0.999 | 0.7813 | 0.055766/0.11876 | 0.18031/0.24934 |
| K32_OFDM | 32 | OFDM | 0.153886 | 1.000 | 0.9968 | 0.00089359/0.0010128 | 0.0246/0.026191 |
| K32_HAAR_1 | 32 | HAAR_1 | 0.350538 | 0.999 | 0.8548 | 0.021462/0.069767 | 0.12566/0.19813 |
| K32_HAAR_2 | 32 | HAAR_2 | 0.354015 | 0.999 | 0.8526 | 0.021758/0.072117 | 0.127/0.20116 |

## Checkpoint hashes

- Authoritative Stage-1A.2: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`
- `K16_OFDM`: `a91d4fb2066d35a7b8e13ffbf869801544e63e8328a4318b04f5dab7ad49a115`
- `K16_HAAR_1`: `51c09dec0f98aba6bed7e0ea9cf9e6e2da01e288b05954368152d4795e0f4404`
- `K16_HAAR_2`: `b2bd1d2e8a6f649b46c4533319d1f0e50bfbd15078b1a55fa3827a4833d21e81`
- `K24_OFDM`: `73a401b6034c04863628684ae20be9a90c3ee866e1204445c205ab70027a8fa6`
- `K24_HAAR_1`: `f4e754cb3f74adfddb14f385a446b030d47cceaa09b8aa1d8008ad34f323e1e8`
- `K24_HAAR_2`: `68740e40d2f624f135d44b8bf648d892110d35e3234a8115a233f17be7d34ecb`
- `K32_OFDM`: `d8a04b5ce658e488021ef3c2f007556cea094a92320c31b4b7d05d0e23cd4e90`
- `K32_HAAR_1`: `3ffc33447951f8af98efefb4a0f196d2080f83be54ebcfc321a8b1b0aca7b38b`
- `K32_HAAR_2`: `9c18adc453b61370d35d61fdaf675a7a8417d196258a6d87ad70340f1599235c`

## Required figures

- `fig01_qpsk_ber_vs_snr_k24.png`, `fig02_16qam_ber_vs_snr_k24.png`: frozen communications context.
- `fig03_learned_s_q_c.png`, `fig04_q_vs_c_principal.png`, `fig05_coupling_vs_spectral_separation.png`, `fig06_coupling_vs_spectral_overlap.png`: coupling-law evidence.
- `fig07_support_retained_energy.png`, `fig08_support_qpsk_ber.png`, `fig09_support_16qam_ber.png`, `fig10_support_prior_q_oracle.png`: oracle-removal evidence.
- `fig11_checkpoint_communication_locality.png`: exploratory checkpoint cross-check.

## Preservation and runtime

- Authoritative checkpoint unitarity residual: `2.233e-14`.
- All Stage-2C checkpoint hashes and all 88 Stage-2A/2B/2C artifacts remained byte-identical: `True`.
- Maximum Stage-2C checkpoint unitarity residual: `2.291e-14`.
- Q half-sample maximum relative discrepancy: `3.021e-04`.
- Runtime: `97.5` s.

## Interpretation

Learned Q predicts C with off-diagonal Spearman 1.000 and fixed-Q support retains 99.809% mean energy. At 18 dB its QPSK BER is 0.00111961, 7.1% above oracle but 78.3% below the best classical fixed-Q result (AFDM). Outcome A is therefore supported: DD locality is a stable operator law and replaces realized top-K with a small BER cost for the authoritative solution. The all-checkpoint table also shows that high Q-to-C correlation alone is insufficient: weaker Haar checkpoints have diffuse fixed support and large BER penalties.

STOP — no channel-spread/P variation, waveform-family fitting, or Stage 3 work was performed.
