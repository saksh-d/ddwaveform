# Stage 0 Scientific Harness Summary

## Scientific configuration

- Dimension: `N=64`; paths: `P=3`; channel family: `DD-v1`.
- Delays: iid continuous `Uniform[0,4]` normalized delay-bin units; Dopplers: iid continuous `Uniform[-0.75,0.75]` normalized Doppler-bin units. Neither is rounded and near-collisions are allowed.
- Pre-normalization gains: iid `CN(0,1/P)`, i.e. each real component has variance `1/(2P)`. The assembled channel receives one common scale `H = H_tilde / sqrt(||H_tilde||_F^2/N)`; final gains are therefore not iid Gaussian.
- DFT: `F[k,n]=exp(-j2*pi*k*n/N)/sqrt(N)`. DD ordering is `Delta(nu) Pi(tau)`: delay first, then Doppler, with negative Doppler exponent.
- QPSK: `((1-2b0)+j(1-2b1))/sqrt(2)`, unit symbol energy. SNR is `E_s/N_0`; `sigma^2=10^(-SNR_dB/10)` and each AWGN real component has variance `sigma^2/2`.
- Top-K: stable descending magnitude independently per output row, with lower column index winning exact ties. `K` is retained interactions per row.
- Receiver: `G_K=(A_tilde_K^H A_tilde_K + sigma^2 I)^(-1) A_tilde_K^H`, applied to the actual observation `u=A s+n'`. The implementation uses `torch.linalg.solve`.
- Seeds: global `11000`, validation `11001`, Stage-0 evaluation `11002`, locked final test `11003`, random unitary `11004`.
- Bank sizes: validation `256`, Stage-0 evaluation `512`, final test definition `1024`. The final test bank was not generated or evaluated.

## Classical waveform definitions

- TIME: `W=I_64`.
- OFDM: `W=F_64^H`.
- OTFS: `W=F_8^H kron I_8`, with delay index varying fastest in flattened vectors.
- AFDM: `W=Lambda_c1^H F_64^H Lambda_c2^H`, where `Lambda_c[n,n]=exp(-j2*pi*c*n^2)`.
- AFDM `c2=sqrt(2)/(100N)=0.0002209708691207961`. Validation candidates were `c1=m/(2N)`, `m=1,...,8`, at `K=3` and 12/14/16/18/20 dB.

| m | c1 | mean analytic MSE |
| --- | --- | --- |
| 1 | 0.0078125 | 6.02568632e-01 |
| 2 | 0.015625 | 8.35593811e-01 |
| 3 | 0.0234375 | 9.52403026e-01 |
| 4 | 0.03125 | 9.99799048e-01 |
| 5 | 0.0390625 | 1.02604580e+00 |
| 6 | 0.046875 | 1.03744506e+00 |
| 7 | 0.0546875 | 1.03992384e+00 |
| 8 | 0.0625 | 1.01714269e+00 |

Selected and locked: `m=1`, `c1=0.0078125`; selection used only the frozen validation bank and mean exact analytic communication MSE.

## Validation results

- DFT unitarity Frobenius residual: `7.175e-14`.
- Maximum integer-delay sample residual: `3.915e-14`.
- Maximum fractional-delay unitarity residual: `1.353e-13`; composition residual: `7.205e-14`.
- Doppler unitarity residual: `5.320e-16`; composition residual: `1.487e-15`.
- Integer-bin DD commutation/order residual: `1.038e-13`.
- Multipath direct-sum residual: `7.116e-16`.
- Maximum DD-v1 normalization residual `| ||H||_F^2/N - 1 |`: `4.441e-16`.
- Same-seed exact reproduction: `True`; distinct seeds differ: `True`; sampled delay/Doppler fractional fractions: `1.000` / `1.000`.
- AWGN empirical variance relative error: `0.056%`.
- Maximum noiseless physical-chain/effective-operator element error: `4.021e-15`.

| Waveform | Unitarity Fro | Power rel. | H=I Fro |
| --- | --- | --- | --- |
| TIME | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| OFDM | 7.175e-14 | 1.995e-15 | 7.175e-14 |
| OTFS | 7.768e-15 | 2.217e-16 | 7.768e-15 |
| AFDM | 7.176e-14 | 1.109e-15 | 7.176e-14 |
| RANDOM | 7.882e-15 | 2.217e-16 | 7.882e-15 |

## Receiver validation

Direct Monte Carlo used 4,096 independent QPSK blocks per case at 15 dB through the physical chain. All relative discrepancies are below 3%.

| Waveform | K | SNR dB | Analytic MSE | Empirical MSE | Relative gap |
| --- | --- | --- | --- | --- | --- |
| TIME | 3 | 15 | 2.86966765e-01 | 2.86284587e-01 | 0.238% |
| TIME | 6 | 15 | 1.74556704e-01 | 1.73671010e-01 | 0.507% |
| TIME | 64 | 15 | 6.39994446e-02 | 6.42156080e-02 | 0.338% |
| RANDOM | 3 | 15 | 3.65627483e+00 | 3.65623597e+00 | 0.001% |
| RANDOM | 6 | 15 | 2.77648817e+00 | 2.77225870e+00 | 0.152% |
| RANDOM | 64 | 15 | 6.39994446e-02 | 6.41889792e-02 | 0.296% |

## Full-LMMSE invariant

Four independent DD-v1 channels were checked at 0, 12, 18, and 25 dB. Receiver-form `K=N` MSE was independently compared with `tr[(I+rho H^H H)^(-1)]/N`.

| Channel | SNR dB | Trace MSE | TIME | OFDM | OTFS | AFDM | RANDOM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 5.79983121e-01 | 5.79983121e-01 | 5.79983121e-01 | 5.79983121e-01 | 5.79983121e-01 | 5.79983121e-01 |
| 0 | 12 | 1.68188756e-01 | 1.68188756e-01 | 1.68188756e-01 | 1.68188756e-01 | 1.68188756e-01 | 1.68188756e-01 |
| 0 | 18 | 8.22010681e-02 | 8.22010681e-02 | 8.22010681e-02 | 8.22010681e-02 | 8.22010681e-02 | 8.22010681e-02 |
| 0 | 25 | 3.99389775e-02 | 3.99389775e-02 | 3.99389775e-02 | 3.99389775e-02 | 3.99389775e-02 | 3.99389775e-02 |
| 1 | 0 | 5.57078399e-01 | 5.57078399e-01 | 5.57078399e-01 | 5.57078399e-01 | 5.57078399e-01 | 5.57078399e-01 |
| 1 | 12 | 1.10108990e-01 | 1.10108990e-01 | 1.10108990e-01 | 1.10108990e-01 | 1.10108990e-01 | 1.10108990e-01 |
| 1 | 18 | 3.51562249e-02 | 3.51562249e-02 | 3.51562249e-02 | 3.51562249e-02 | 3.51562249e-02 | 3.51562249e-02 |
| 1 | 25 | 7.76123634e-03 | 7.76123634e-03 | 7.76123634e-03 | 7.76123634e-03 | 7.76123634e-03 | 7.76123634e-03 |
| 2 | 0 | 5.81139322e-01 | 5.81139322e-01 | 5.81139322e-01 | 5.81139322e-01 | 5.81139322e-01 | 5.81139322e-01 |
| 2 | 12 | 1.26839669e-01 | 1.26839669e-01 | 1.26839669e-01 | 1.26839669e-01 | 1.26839669e-01 | 1.26839669e-01 |
| 2 | 18 | 3.85655586e-02 | 3.85655586e-02 | 3.85655586e-02 | 3.85655586e-02 | 3.85655586e-02 | 3.85655586e-02 |
| 2 | 25 | 8.18972428e-03 | 8.18972428e-03 | 8.18972428e-03 | 8.18972428e-03 | 8.18972428e-03 | 8.18972428e-03 |
| 3 | 0 | 5.79488216e-01 | 5.79488216e-01 | 5.79488216e-01 | 5.79488216e-01 | 5.79488216e-01 | 5.79488216e-01 |
| 3 | 12 | 1.74950740e-01 | 1.74950740e-01 | 1.74950740e-01 | 1.74950740e-01 | 1.74950740e-01 | 1.74950740e-01 |
| 3 | 18 | 8.43373948e-02 | 8.43373948e-02 | 8.43373948e-02 | 8.43373948e-02 | 8.43373948e-02 | 8.43373948e-02 |
| 3 | 25 | 3.08971632e-02 | 3.08971632e-02 | 3.08971632e-02 | 3.08971632e-02 | 3.08971632e-02 | 3.08971632e-02 |

- Maximum receiver-versus-trace absolute discrepancy: `3.608e-16`.
- Maximum absolute cross-waveform discrepancy: `1.110e-16`.
- Maximum relative cross-waveform discrepancy: `7.895e-16`.
- BER equality was **not** used as a full-information invariant.

## Classical baseline results

These are diagnostic curves at `K=3`, 512 frozen Stage-0 evaluation channels, four QPSK blocks per channel, and paired bits/channels/physical AWGN across waveforms. They are not a significance claim.

| SNR dB | Waveform | BER | Empirical NMSE | Analytic MSE | Full bound |
| --- | --- | --- | --- | --- | --- |
| 0 | TIME | 2.035332e-01 | 5.932411e-01 | 5.958347e-01 | 5.598585e-01 |
| 0 | OFDM | 2.019424e-01 | 5.878975e-01 | 5.902780e-01 | 5.598585e-01 |
| 0 | OTFS | 2.430115e-01 | 6.784008e-01 | 6.788818e-01 | 5.598585e-01 |
| 0 | AFDM | 2.209892e-01 | 6.292989e-01 | 6.304742e-01 | 5.598585e-01 |
| 5 | TIME | 1.082764e-01 | 4.014486e-01 | 4.004708e-01 | 3.282087e-01 |
| 5 | OFDM | 1.056900e-01 | 3.903762e-01 | 3.903912e-01 | 3.282087e-01 |
| 5 | OTFS | 1.691704e-01 | 5.665521e-01 | 5.657688e-01 | 3.282087e-01 |
| 5 | AFDM | 1.353493e-01 | 4.697569e-01 | 4.678268e-01 | 3.282087e-01 |
| 10 | TIME | 5.721283e-02 | 3.057809e-01 | 3.050601e-01 | 1.633385e-01 |
| 10 | OFDM | 5.587769e-02 | 2.921597e-01 | 2.919822e-01 | 1.633385e-01 |
| 10 | OTFS | 1.397667e-01 | 6.300169e-01 | 6.314538e-01 | 1.633385e-01 |
| 10 | AFDM | 9.347916e-02 | 4.327412e-01 | 4.336167e-01 | 1.633385e-01 |
| 12 | TIME | 5.024338e-02 | 3.025758e-01 | 2.987044e-01 | 1.199085e-01 |
| 12 | OFDM | 4.870605e-02 | 2.884874e-01 | 2.858208e-01 | 1.199085e-01 |
| 12 | OTFS | 1.398849e-01 | 7.172023e-01 | 7.145623e-01 | 1.199085e-01 |
| 12 | AFDM | 9.003448e-02 | 4.619419e-01 | 4.606967e-01 | 1.199085e-01 |
| 15 | TIME | 4.732513e-02 | 3.202334e-01 | 3.157954e-01 | 7.422504e-02 |
| 15 | OFDM | 4.541397e-02 | 3.066311e-01 | 3.052683e-01 | 7.422504e-02 |
| 15 | OTFS | 1.460800e-01 | 8.954583e-01 | 8.884590e-01 | 7.422504e-02 |
| 15 | AFDM | 9.202194e-02 | 5.428153e-01 | 5.372237e-01 | 7.422504e-02 |
| 18 | TIME | 4.787064e-02 | 3.547214e-01 | 3.558323e-01 | 4.586677e-02 |
| 18 | OFDM | 4.711151e-02 | 3.476417e-01 | 3.514966e-01 | 4.586677e-02 |
| 18 | OTFS | 1.542816e-01 | 1.096011e+00 | 1.106990e+00 | 4.586677e-02 |
| 18 | AFDM | 9.677124e-02 | 6.428664e-01 | 6.506683e-01 | 4.586677e-02 |
| 20 | TIME | 5.016327e-02 | 3.834833e-01 | 3.914431e-01 | 3.359711e-02 |
| 20 | OFDM | 4.941559e-02 | 3.955717e-01 | 3.950660e-01 | 3.359711e-02 |
| 20 | OTFS | 1.611862e-01 | 1.276603e+00 | 1.272335e+00 | 3.359711e-02 |
| 20 | AFDM | 1.010857e-01 | 7.385581e-01 | 7.474662e-01 | 3.359711e-02 |
| 25 | TIME | 5.972672e-02 | 5.150591e-01 | 5.085572e-01 | 1.667597e-02 |
| 25 | OFDM | 5.965424e-02 | 5.632449e-01 | 5.633850e-01 | 1.667597e-02 |
| 25 | OTFS | 1.788330e-01 | 1.766038e+00 | 1.775544e+00 | 1.667597e-02 |
| 25 | AFDM | 1.208992e-01 | 1.106435e+00 | 1.103359e+00 | 1.667597e-02 |

Best observed BER baseline by SNR: 0 dB: OFDM, 5 dB: OFDM, 10 dB: OFDM, 12 dB: OFDM, 15 dB: OFDM, 18 dB: OFDM, 20 dB: OFDM, 25 dB: OFDM. No ordering was imposed.

Generated files:

- `outputs/stage0/ber_vs_snr_k3.png`
- `outputs/stage0/nmse_vs_snr_k3.png`
- `outputs/stage0/baseline_metrics.csv`
- `outputs/stage0/afdm_validation_scores.csv`
- `outputs/stage0/full_lmmse_invariance.csv`
- `outputs/stage0/stage0_diagnostics.json`

## Tests and software

- Python `3.10.17`; PyTorch `2.7.0`; NumPy `2.2.6`; Matplotlib `3.10.3`; pytest `9.0.2`.
- Test result: `23 passed in 0.58s` (23 tests passed).
- Stage command: `python -m scripts.stage0`.
- Runtime: `11.6` seconds.

## Git/repository hygiene

- `outputs/` is ignored and generated artifacts are not tracked.
- `reports/` is tracked; this report is the review artifact.
- `researchPlan.MD` was not modified.
- No final test-bank performance was generated or inspected.
- No Learned waveform, trainable dense transform, or Stage 0F oracle was implemented.

## Problems or anomalies

- The first local pytest invocation failed at collection because its entry point omitted the repository root from `sys.path`; `pytest.ini` now supplies the root and the unchanged numerical tests pass.
- For circular `Pi(ell)` and sampled fractional `Delta(nu)`, the globally scalar Weyl relation has a wrap-boundary phase term unless `nu` is an integer bin. The sign/order identity was therefore checked exactly at integer `nu=3`; the deliberately measured uncorrected fractional-`nu=0.43` residual was `3.904e+00`. The mandated DD-v1 operators were not changed or rounded.
- Stage-0 baseline ranking is reported as observed and was not used to alter the implementation.

## Final gate

STAGE 0 PASS

All mandatory Stage 0A-E checks passed, validation-only AFDM tuning was locked, diagnostic curves were generated, and the final test bank remained untouched.
