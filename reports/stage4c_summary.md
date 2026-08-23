# Stage 4C — Depth-Scaling Structured Transform Search

## Result: Outcome E — no systematic depth law

No waveform/channel training, BER/Q/geometry fitting, BER-based selection, or Stage 2E dependency was used. Every fit was frozen before Q, generator, and communication evaluation.

## Frozen family and protocol

The sole family is $W_L=D_0F^HD_1F^H\cdots F^HD_L$, where $[D_\ell]_{nn}=\exp(i\sum_{p=1}^3 a_{\ell p}x_n^p)$ and $x_n$ is uniformly spaced on $[-1,1]$.
All constant polynomial terms and global phases are removed. Depth 1 is Stage-4A `FOURIER_POLY_D3` up to that exact redundancy. A fixed target-specific phase/permutation convention against `F^H` is established before fitting and restored for native-order evaluation; it is not fit-dependent.
Both arms use 400 Adam steps, 3 deterministic restarts, and validation-loss-only selection. Arm M uses aligned atom overlap; Arm B uses 64/128 independent fixed DD-v1 train/validation atoms. Evaluation uses 512 third-bank atoms.
Depth-1 continuity check on authoritative LEARNED_0: Arm-M aligned NMSE `1.110186` versus Stage-4A `1.107286` (absolute difference `0.002900`).

| depth | real parameters | coefficient bytes | unitary compression | IFFT count | phase multiplies |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 48 | 682.7x vs U(64) dimension | 1 | 2 |
| 2 | 9 | 72 | 455.1x vs U(64) dimension | 2 | 3 |
| 3 | 12 | 96 | 341.3x vs U(64) dimension | 3 | 4 |
| 4 | 15 | 120 | 273.1x vs U(64) dimension | 4 | 5 |

Application complexity is `O(L N log N)`; float64 coefficient storage is `O(L)`. A minimal target convention adds 64 permutation indices plus 64 phase angles (640 bytes with uint16/float64), reported separately and excluded from trainable parameters.

## Learned-population depth curves

| arm | L | params | aligned NMSE | overlap | d_B | 1-eta24 | Q corr | G_D/G_S | Full QPSK BER | Full 16-QAM BER | Fixed-Q QPSK BER | Fixed-Q 16-QAM BER | Haar d_B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M | 1 | 6 | 1.1527 | 0.1930 | 1.3135 | 0.009672 | 0.8019 | 9.975 | 0.00220299 | 0.0238152 | 0.00600624 | 0.0467329 | 1.7378 |
| M | 2 | 9 | 1.5634 | 0.0531 | 1.8533 | 0.012903 | 0.3729 | 10.476 | 0.000749588 | 0.0226097 | 0.00429153 | 0.0528212 | 1.8554 |
| M | 3 | 12 | 1.1346 | 0.2001 | 1.8586 | 0.022751 | 0.7945 | 11.132 | 0.00205421 | 0.0237236 | 0.00863075 | 0.0651798 | 1.8603 |
| M | 4 | 15 | 1.5306 | 0.0652 | 1.8454 | 0.016669 | 0.3771 | 9.888 | 0.000761986 | 0.0227213 | 0.00624943 | 0.0641689 | 1.8521 |
| B | 1 | 6 | 1.1522 | 0.1925 | 1.3121 | 0.009274 | 0.8020 | 10.132 | 0.00216866 | 0.0240688 | 0.00593853 | 0.0467958 | 1.7366 |
| B | 2 | 9 | 1.5259 | 0.0585 | 1.8257 | 0.020980 | 0.3808 | 10.473 | 0.00067234 | 0.0226383 | 0.00854397 | 0.0762091 | 1.8438 |
| B | 3 | 12 | 1.2169 | 0.1596 | 1.8307 | 0.035062 | 0.7835 | 10.185 | 0.00188541 | 0.0236034 | 0.0190678 | 0.103184 | 1.8332 |
| B | 4 | 15 | 1.5076 | 0.0650 | 1.8111 | 0.024369 | 0.3805 | 10.644 | 0.000750542 | 0.0225668 | 0.00950432 | 0.0804195 | 1.8313 |

## Interpretation gate

1. **Matrix/atom fidelity improves systematically:** M=No, B=No by L1-to-L4 endpoint.
2. **Complex B fidelity improves systematically:** M=No, B=No by endpoint.
3. **Operator fitting materially outperforms matrix fitting:** No. Best median d_B changes from 1.3135 to 1.3121 (0.11% lower), while the depth trajectory remains non-monotonic.
4. **Residual Q mass approaches the basin:** No. Best median is 0.009274, versus 0.00186–0.00202.
5. **Generator asymmetry approaches the population:** Only directionally for Arm M; best absolute median target gap is 0.268, without joint matrix/operator/Q convergence.
6. **Full BER approaches the population with depth:** M=Yes, B=Yes by the all-modulation/SNR log-gap endpoint.
7. **Finite-K BER approaches the population:** M=No, B=No by the combined Oracle/Fixed-Q endpoint.
8. **Consistent across five Learned targets:** basin-entry counts are {'M_L1': 0, 'M_L2': 0, 'M_L3': 0, 'M_L4': 0, 'B_L1': 0, 'B_L2': 0, 'B_L3': 0, 'B_L4': 0} (four of five required).
9. **Parameters reproducible:** No coherent deep clustering. Best mean pairwise RMS distance is 0.1845 at shallow depth and grows materially in deeper fits; full vectors/SD/CV are in `parameter_reproducibility.csv`.
10. **Haar controls materially worse:** No at L4; Haar/Learned median d_B ratios are M=1.004, B=1.011. Outcome D is still rejected because depth did not improve either population.
11. **Minimum depth entering the empirical basin:** none.
12. **Compact at best operator point:** 6 real parameters, 1 IFFT operation, 682.7x dimensional compression; compactness does not rescue the failed basin gate.

## Preservation

- All five Learned and all six Stage-3B checkpoint hashes verified; maximum fitted unitarity residual `1.464e-13`.
- All 302 prior artifacts remained byte-identical: `True`.
- Common paired communication bank: 512 channels, at least 1000000 bits per waveform/point, QPSK and 16-QAM at 12/15/18/20 dB, Full/Oracle-K24/Fixed-Q-K24.
- Stage 2E absent. No fit/restart/depth decision used Q, generator, MSE, BER, modulation, or receiver results.

STOP — do not extend depth or proceed to Stage 4D automatically.
