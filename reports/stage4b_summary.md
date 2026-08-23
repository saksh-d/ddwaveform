# Stage 4B — Learned-Solution Ensemble and Functional Invariants

## Result: Outcome A — reproducible structural basin

All four repetitions used the exact frozen Stage-1A.2 OFDM-start recipe. Only the declared online training channel/SNR seeds changed. Selection remained validation-only; every independently selected checkpoint entered every analysis.

## Stage-4A completion check

| waveform | 1-eta24 Q |
|---|---:|
| Learned | 0.0018641548 |
| FIT_FOURIER_POLY_D1 | 0.009221224 |
| FIT_FOURIER_POLY_D2 | 0.0090861295 |
| FIT_FOURIER_POLY_D3 | 0.0091154846 |
| FIT_CHIRP_FOURIER | 0.0090521592 |
| FIT_GABOR_TIGHT_PROTOTYPE | 0.27853542 |
| FIT_SHALLOW_2FFT | 0.016974514 |

FOURIER_POLY_D3 retains 4.890 times Learned's residual Q mass.

| modulation | receiver | Learned BER | FOURIER_POLY_D3 BER | gap |
|---|---|---:|---:|---:|
| QPSK | Full | 0.000962257 | 0.00190258 | +0.000940323 |
| QPSK | Oracle | 0.00110722 | 0.00514126 | +0.00403404 |
| QPSK | Fixed-Q | 0.00117493 | 0.00547981 | +0.00430489 |
| 16QAM | Full | 0.0224915 | 0.022687 | +0.000195503 |
| 16QAM | Oracle | 0.0243654 | 0.0423241 | +0.0179586 |
| 16QAM | Fixed-Q | 0.0250053 | 0.0453329 | +0.0203276 |

The FOURIER_POLY_D3 communication gap **remains under Full reception**; finite-K truncation is therefore not its sole source.

## Training provenance

| member | channel seed | SNR seed | selected step | validation mean MSE | validation CVaR95 | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| LEARNED_0 | 11036 | 11037 | 50 | 0.07367612 | 0.16287674 | `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc` |
| LEARNED_1 | 11170 | 11171 | 175 | 0.07377758 | 0.15453312 | `fbe49eb8002e7e7272c94e6a362210e8cba3599d317a6cfe53275e2da1f32eef` |
| LEARNED_2 | 11172 | 11173 | 100 | 0.07381668 | 0.15835782 | `ad8d4e379c0f29369bf49d1b745a038cbf5326233cdbb05406cde957d4bbe7e5` |
| LEARNED_3 | 11174 | 11175 | 175 | 0.07378588 | 0.15466726 | `a52e1eccd333216a1dc360cd1d74be3d44b92170e044b589f0b68a3da05e3b75` |
| LEARNED_4 | 11176 | 11177 | 175 | 0.07378409 | 0.15471292 | `cf57edc354959e84560987f65075d4396ce2c2c3d5be5f9efc642aefe9c977ae` |

## Common held-out population

| member | eta24 Q | row-p05 | K99 | 18 dB Full QPSK | 18 dB Fixed-Q QPSK | 18 dB Fixed-Q 16-QAM |
|---|---:|---:|---:|---:|---:|---:|
| LEARNED_0 | 0.99813576 | 0.99216120 | 21 | 0.00108528 | 0.00124836 | 0.0240688 |
| LEARNED_1 | 0.99799928 | 0.99116379 | 21 | 0.000957489 | 0.00113392 | 0.0244389 |
| LEARNED_2 | 0.99797906 | 0.99099690 | 22 | 0.00101376 | 0.00119114 | 0.0244045 |
| LEARNED_3 | 0.99797727 | 0.99151444 | 21 | 0.000946045 | 0.00112057 | 0.0243826 |
| LEARNED_4 | 0.99797805 | 0.99173063 | 21 | 0.00096035 | 0.00112629 | 0.0243034 |

At 18 dB fixed-Q, QPSK spans 0.00112057-0.00124836 (sample SD 5.49e-05); 16-QAM spans 0.0240688-0.0244389 (sample SD 0.000149). New seeds slightly improve QPSK and slightly worsen 16-QAM relative to member 0, reproducing a tight modulation tradeoff rather than one identical BER value. Paired channel intervals for all points are in `paired_ber_bootstrap.csv`.

## Matrix, Q, and complex-operator reproducibility

Learned-pair aligned squared overlap spans 0.8932-0.9950; aligned matrix NMSE spans 0.0051-0.1101. The maximum Learned-to-control overlap is 0.2084.
Learned-pair Q correlation spans 0.9747-0.9990, with top-24 support overlap 0.9590-0.9954; Learned-to-Stage-3B median Q correlation is 0.6999.
After column phase/permutation alignment, Learned-pair complex d_B spans 0.0076-0.1703, versus a Stage-3B median 1.4071. Complex coefficient coherence spans 0.9148-0.9962.

Here $d_B$ is the held-out atom mean of $\|B_a-B_b^{aligned}\|_F^2/\|B_a\|_F^2$; complex coherence uses the normalized complex inner product over all held-out atoms and coefficients. Native-order generator/atom diagnostics remain separate.

## Generator reproducibility

| member | G_S top24 DD energy | G_D top24 DD energy | G_D/G_S ratio |
|---|---:|---:|---:|
| LEARNED_0 | 0.074914 | 0.700527 | 9.351 |
| LEARNED_1 | 0.071209 | 0.685052 | 9.620 |
| LEARNED_2 | 0.074461 | 0.692961 | 9.306 |
| LEARNED_3 | 0.069968 | 0.685950 | 9.804 |
| LEARNED_4 | 0.070071 | 0.686579 | 9.798 |

The asymmetry is stable and population-specific in this control set: Learned G_D/G_S ratios span 9.306-9.804, above the Stage-3B range 2.781-5.875.

## Interpretation gate

1. **Yes.** Communication quality is reproducible as a tight modulation-tradeoff population; no repetition degrades both modulations at 18 dB fixed-Q, and full curves/paired intervals support the same conclusion.
2. **High overlap.** Aligned squared overlap is 0.893-0.995, cleanly separated from controls.
3. **Yes.** Q correlations are 0.975-0.999; eta24 is 0.997977-0.998136.
4. **Yes.** Complex responses remain close after legitimate alignment: d_B 0.008-0.170 versus Stage-3B median 1.407.
5. **Yes.** G_D-concentrated/G_S-diffuse holds for all five and its ratio is separated from Stage-3B: **True**.
6. **Yes.** The control-separated matrix/operator basin plus the stronger generator asymmetry are shared by communication-trained Learned solutions and absent from the Q-only population.
7. Communication equivalence corresponds most strongly to matrix and complex-B similarity; Q is also stable within Learned but is less discriminative because Stage-3B can saturate concentration without entering this basin.
8. **Yes.** Member 0 is the lowest-overlap edge of the same empirically separated basin, not an isolated lucky solution.
9. Best description: **one reproducible structural basin with small within-basin functional variation**, not a fuzzy manifold.
10. The ensemble is sufficient to justify reviewed Stage 4C depth-scaling fits to the shared basin. Multi-N is not required first, but Stage 4C is not started here.

## Preservation

- Frozen authoritative SHA-256: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`.
- 273 prior artifacts remained byte-identical.
- Final complete suite: 113 passed; five-cell notebook executed with zero errors; four figures visually inspected.
- Stage 2E absent; no BER selection, Q loss, geometry loss, compact-family fitting, or receiver/channel/model change.
- STOP.
