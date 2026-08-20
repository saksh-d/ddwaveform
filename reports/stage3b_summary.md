# Stage 3B — Direct Q-Concentration Optimization and Causal Validation

## Result: Outcome A

Both Q-only arms rescued both Haar starts through representation, analytic-MSE, and two-modulation BER outcomes.

## Causal interventions at 18 dB

| Arm/start | val eta initial→selected | held-out eta Δ | retained Δ | analytic MSE Δ | QPSK fixed BER Δ | 16-QAM fixed BER Δ |
|---|---:|---:|---:|---:|---:|---:|
| mean/OFDM | 0.990947→0.999777 | +0.008830 | +0.009157 | -0.038317 | -53.1% | -52.6% |
| mean/HAAR_1 | 0.537585→0.999658 | +0.461756 | +0.460219 | -2.260602 | -99.0% | -93.8% |
| mean/HAAR_2 | 0.537064→0.999487 | +0.462101 | +0.460552 | -2.260207 | -98.9% | -93.7% |
| robust/OFDM | 0.990947→0.999884 | +0.008937 | +0.009274 | -0.038535 | -53.6% | -52.4% |
| robust/HAAR_1 | 0.537585→0.999599 | +0.461697 | +0.460157 | -2.260480 | -99.0% | -93.9% |
| robust/HAAR_2 | 0.537064→0.999742 | +0.462355 | +0.460818 | -2.260747 | -99.0% | -93.8% |

## Interpretation gate

1. **Held-out concentration:** yes. Every selected run increased held-out eta; Haar gains were about +0.462 and OFDM gains about +0.009.
2. **Weak-row protection:** consistent but modest. Robust-minus-mean held-out row-p05 changes were +0.000614, +0.000288, +0.001434 for OFDM/HAAR-1/HAAR-2.
3. **Fixed-support energy:** yes. Realized retained-energy gains tracked eta in the predicted direction for all six interventions.
4. **Analytic MSE:** yes. Every intervention reduced fixed-Q analytic MSE; Haar starts moved from about 2.31 to about 0.046 at 18 dB.
5. **QPSK BER:** yes versus every matched raw start, but not versus authoritative Learned: selected range 0.00235748-0.00268364, Learned 0.00109291 at 18 dB.
6. **16-QAM BER:** yes versus every matched raw start and slightly better than Learned: selected range 0.0214195-0.0221443, Learned 0.0229979 at 18 dB.
7. **Initialization robustness:** yes for the causal direction; all OFDM and Haar interventions improved concentration, representation, MSE, and both BERs.
8. **Haar rescue:** yes. Both raw Haar starts reached held-out eta above 0.99948 and fixed-Q BER near the OFDM-start Q solutions.
9. **Structure:** no classical/learned collapse. OFDM is the nearest tested reference, but aligned overlap is only 0.2176-0.4817; pairwise Stage-3B overlap spans 0.2570-0.4798 despite nearly identical Q concentration.
10. **Sufficiency:** supported as a causal primary design property for useful finite-K waveforms in this frozen setting, but not a complete communication objective: the QPSK gap to Learned shows coefficient/conditioning geometry can still matter after concentration saturates.

## Structural resemblance

- Q_MEAN_FROM_HAAR_1: closest to OFDM with aligned mean squared overlap 0.2289.
- Q_MEAN_FROM_HAAR_2: closest to OFDM with aligned mean squared overlap 0.2359.
- Q_MEAN_FROM_OFDM: closest to OFDM with aligned mean squared overlap 0.4804.
- Q_ROBUST_FROM_HAAR_1: closest to OFDM with aligned mean squared overlap 0.2176.
- Q_ROBUST_FROM_HAAR_2: closest to OFDM with aligned mean squared overlap 0.2275.
- Q_ROBUST_FROM_OFDM: closest to OFDM with aligned mean squared overlap 0.4817.

## Protocol and preservation

- Q training/validation/evaluation banks: 128/256/2048; training used K=24 only.
- Robust arm fixed before evaluation: lambda=0.4, weakest-row fraction=0.1.
- Communication bank: 512 normalized P=3 DD-v1 channels with paired symbols/noise and no BER-based selection.
- Training/evaluation/full-protocol runtimes: 363.7/67.9/431.7 s.
- Authoritative checkpoint SHA-256: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`.
- All 128 Stage-2/3A artifacts remained byte-identical: `True`.
- IDENTITY is retained in numerical outputs and excluded from figures.
- Stage 2E was not imported or used. No push, merge, receiver change, or Stage 3C work was performed.

## Mandatory completion audit before Stage 3C

- Learned warm-start mean/robust Q-only arms raised validation eta from 0.998135 to 0.999897/0.999846, selected at steps 575/975, with no nonfinite values and unitarity residuals below `2.38e-14`.
- On the paired 18 dB bank, both warm starts improved 16-QAM BER from 0.023248 to 0.021026/0.021190 but worsened QPSK BER from 0.001046 to 0.001144/0.001196. This independently reproduces the modulation reversal without BER selection.
- Every primary selected waveform has explicit finite-Q conditioning, Gram lower-tail, equalizer-noise-enhancement, and post-LMMSE reliability distributions. Median condition numbers are 10.65-10.84 and median noise enhancement is 2.03-2.04, so neither aggregate quantity separates the solutions; lower-tail reliability does.
- Q-bank-size saturation is stable: eta changes by at most `5.91e-6` from 128 to 2,048 held-out Sobol atoms. Final-quarter support flip fractions are at most `1.46e-4` on average and `4.88e-4` at any recorded step.
- Generic-permutation aligned similarity is only 0.057-0.108 for the six primary solutions. Their fixed-Q masks are not one common Stage-2B block structure: within-Learned-community support spans 0.339-0.627 versus 0.842 for Learned, with OFDM-start solutions more local than Haar-start solutions.
- Machine-readable completion evidence is isolated in `outputs/stage3b_completion/`; all pre-existing Stage-1 through Stage-3B artifacts remained byte-identical.
