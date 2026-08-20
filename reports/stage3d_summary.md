# Stage 3D — Causal Completion of the DD Design Principle

## Result: Outcome C

The tested adverse tail remains descriptive but is insufficient as a causal design objective: aggregate error never improved out of sample and communication changes were small and inconsistent.

## Frozen analytic intervention

- Exact mismatched fixed-Q LMMSE geometry at 18 dB; K=24, N=64, P=3.
- Tail is the Stage-3C two-level upper stream/channel CVaR with beta=0.95.
- Per-start Q floor uses epsilon_Q=0.00025; Arm-B lambdas are (0.25, 1.0) after frozen per-start normalization.
- 14 analytic checkpoints entered paired communication evaluation; none was selected using BER.

## Central 18 dB fixed-Q results

| Checkpoint | Start | M change | T change | eta change | QPSK BER change | 16-QAM BER change |
|---|---|---:|---:|---:|---:|---:|
| GEOM_MEAN_FROM_Q_MEAN_FROM_OFDM | Q_MEAN_FROM_OFDM | +0.5% | +0.1% | -1.48e-04 | +0.5% | +0.8% |
| GEOM_PARETO_1_FROM_Q_MEAN_FROM_OFDM | Q_MEAN_FROM_OFDM | +0.5% | -0.2% | -1.39e-04 | +1.2% | +0.9% |
| GEOM_PARETO_2_FROM_Q_MEAN_FROM_OFDM | Q_MEAN_FROM_OFDM | +0.7% | +0.2% | -1.86e-04 | +0.2% | +0.7% |
| GEOM_MEAN_FROM_Q_MEAN_FROM_HAAR_1 | Q_MEAN_FROM_HAAR_1 | +0.5% | -0.6% | -1.49e-04 | -1.4% | +0.2% |
| GEOM_PARETO_1_FROM_Q_MEAN_FROM_HAAR_1 | Q_MEAN_FROM_HAAR_1 | +0.6% | +0.1% | -1.66e-04 | -0.4% | +0.6% |
| GEOM_MEAN_FROM_Q_MEAN_FROM_HAAR_2 | Q_MEAN_FROM_HAAR_2 | +0.4% | -0.0% | -1.28e-04 | +0.8% | +0.3% |
| GEOM_PARETO_1_FROM_Q_MEAN_FROM_HAAR_2 | Q_MEAN_FROM_HAAR_2 | +0.4% | -0.0% | -1.39e-04 | -0.4% | +0.6% |
| GEOM_PARETO_2_FROM_Q_MEAN_FROM_HAAR_2 | Q_MEAN_FROM_HAAR_2 | +0.7% | +0.0% | -2.26e-04 | +0.2% | +0.9% |
| GEOM_MEAN_FROM_Q_ROBUST_FROM_OFDM | Q_ROBUST_FROM_OFDM | +0.5% | +0.2% | -1.30e-04 | +0.3% | +0.5% |
| GEOM_PARETO_1_FROM_Q_ROBUST_FROM_OFDM | Q_ROBUST_FROM_OFDM | +0.6% | -0.0% | -1.63e-04 | +0.0% | +0.7% |
| GEOM_MEAN_FROM_Q_ROBUST_FROM_HAAR_1 | Q_ROBUST_FROM_HAAR_1 | +0.4% | +0.3% | -1.22e-04 | +1.0% | +0.1% |
| GEOM_PARETO_1_FROM_Q_ROBUST_FROM_HAAR_1 | Q_ROBUST_FROM_HAAR_1 | +0.4% | +0.1% | -1.25e-04 | +0.6% | +0.1% |
| GEOM_MEAN_FROM_Q_ROBUST_FROM_HAAR_2 | Q_ROBUST_FROM_HAAR_2 | +0.4% | +0.0% | -1.27e-04 | +0.7% | +0.5% |
| GEOM_PARETO_1_FROM_Q_ROBUST_FROM_HAAR_2 | Q_ROBUST_FROM_HAAR_2 | +0.4% | +0.1% | -1.32e-04 | +1.2% | +0.7% |

## Interpretation

- Held-out M changed +0.38% to +0.74%; no intervention improved it.
- Held-out T changed -0.58% to +0.31%; the best reduction was not material or consistent.
- Fixed-Q QPSK changed -1.41% to +1.20%; 16-QAM changed +0.11% to +0.89% and never improved.
- Lowest analytically frozen QPSK BER: `GEOM_PARETO_1_FROM_Q_ROBUST_FROM_OFDM`; this label is descriptive and caused no reselection.
- Best simultaneous relative-change control: `GEOM_MEAN_FROM_Q_MEAN_FROM_HAAR_1`; again, no post-hoc checkpoint choice was made.
- Central comparisons are same-start pre/post interventions; Learned and classical bases remain frozen controls.
- Full, oracle-K, and fixed-Q BER plus fixed-vs-oracle penalties are available for both modulations at 12/15/18/20 dB in the machine-readable communication and causal tables.
- STOP. No Stage 3E, from-scratch synthesis, receiver change, or waveform-law distillation was performed.

## Preservation

- Authoritative Learned hash: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`.
- All six Stage-3B hashes were verified; all 221 prior artifacts remained byte-identical.
- IDENTITY is retained in numerical outputs and excluded from figures. Stage 2E was not used.
