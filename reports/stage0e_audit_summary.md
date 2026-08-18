# Stage 0E Corrective Audit Summary

## A. Root-cause executive summary

This audit validates the baseline implementations but finds that the original `DD-v1 + K=3` configuration requires scientific review before Stage 1A.

1. **Why the original curves did not waterfall:** `K=3` is a mismatched receiver, not a full-CSI receiver. As noise regularization vanishes, it approaches `T_3(A)^\dagger A`, not the identity. Omitted interactions and severe/occasionally singular truncated models leave deterministic distortion; less regularization can therefore worsen BER.
2. **Why OFDM beat OTFS:** under DD-v1, OFDM retains `91.62%` mean top-3 operator energy versus OTFS `70.69%`. The row-level 10th percentile is `81.86%` versus `54.16%`. The current one-dimensional circular block channel with continuous delay is substantially more concentrated in OFDM coordinates.
3. **Why OFDM beat AFDM:** the tuned AFDM retains `82.11%` mean top-3 energy and has worse median truncated-model conditioning (`83.5`) than OFDM (`42.7`). Its K=3 receiver therefore sees both more omitted action and stronger inversion sensitivity.
4. **Was OTFS implemented correctly?** Yes. The independent DD-to-TF plus OFDM chain agrees sample-by-sample with `F_8^H kron I_8` to `4.08e-15`, and clean inverse reconstruction is accurate to `7.80e-15`. On-grid single paths have one significant entry per row; the three-path case has exactly three.
5. **Was AFDM implemented correctly?** Yes. The direct IDAFT sum agrees with the factored synthesis matrix to `7.04e-14`. All 20 tested integer delay/Doppler path offsets match the canonical support law.
6. **Was AFDM parameterized appropriately?** The previous `c1=1/128` is appropriate as the validation-selected **AFDM_K3_TUNED** baseline, but it is not a canonical support-rule choice. A guarded DD-v1 reference with `c1=5/128` is materially worse at K=3. Canonical integer-delay guarantees do not extend to continuous fractional delay.
7. **Is DD-v1 OFDM-friendly?** Yes, for this finite-representation experiment and these transforms. OFDM has the highest top-3 concentration for every audited Doppler range. The mild `nu_max=0.25` case is especially OFDM-friendly, but widening to `nu_max=4` does not reverse the ordering.
8. **Is K=3 too restrictive?** It is too restrictive for ordinary noise-limited communication behavior for every baseline. At 20 dB, BER only approaches full-information behavior around K=48 (and OTFS needs nearly K=64 at its very low full-CSI BER). K=3 remains valid only if explicitly interpreted as an extreme representation-stress test.
9. **Does fractional delay materially change ordering?** Yes. GRID-DD is exactly top-3 representable for every waveform. Fractional Doppler alone introduces leakage, but IDENTITY remains exact because integer delays preserve three time-domain shifts. Adding fractional delay reduces top-3 energy to OFDM `0.917`, AFDM `0.823`, and OTFS `0.709`, establishing fractional delay as a major compatibility change.
10. **Is a change required before Stage 1A?** No baseline code correction is required. A scientific decision is required: either retain Stage 1A as an explicitly extreme DD-v1/K=3 representation test, or broaden the design/evaluation gate to include larger K and/or controlled grid/fractional regimes. No learned waveform should be started until that choice is made.

Confidence is high. The direct transform equivalences, controlled grid-to-fractional transition, million-bit full-CSI runs, retained-energy measurements, and N=256 reference independently support the same explanation. The principal unresolved question is not software correctness; it is whether DD-v1/K=3 is the intended scientific target.

## B. Full-CSI results

Each waveform/SNR used 1,048,576 paired bit decisions over the frozen 512-channel Stage-0 evaluation bank. Every principal transform shows the required high-SNR decline.

| SNR dB | OFDM BER | OTFS BER | AFDM_K3_TUNED BER |
| --- | ---: | ---: | ---: |
| 0 | 1.886969e-1 | 1.876450e-1 | 1.876688e-1 |
| 5 | 8.005714e-2 | 7.631111e-2 | 7.741165e-2 |
| 10 | 2.058792e-2 | 1.338482e-2 | 1.576805e-2 |
| 12 | 1.102638e-2 | 4.936218e-3 | 6.772995e-3 |
| 15 | 4.594803e-3 | 8.487701e-4 | 1.818657e-3 |
| 18 | 1.955986e-3 | 9.155273e-5 | 5.645752e-4 |
| 20 | 1.297951e-3 | 2.288818e-5 | 3.156662e-4 |
| 25 | 5.798340e-4 | 4.768372e-6 | 1.049042e-4 |
| 30 | 3.948212e-4 | 0 observed errors | 5.912781e-5 |

The OTFS 30 dB point had zero errors in 1,048,576 decisions; it is recorded in CSV with the rule-of-three 95% empirical upper bound `BER < 2.861023e-6`, not interpreted as a literal zero probability. Full-information BER is not waveform-invariant for finite-alphabet QPSK, and no such equality was assumed.

The full-CSI result falsifies a physical-chain or general receiver failure. The original floor exists only after truncating the receiver model.

## C. Finite-budget results

Each entry is `BER / empirical NMSE / analytic MSE`. Empirical and analytic MSE track closely throughout.

### 15 dB

| K | IDENTITY | OFDM | OTFS | AFDM_K3_TUNED |
| ---: | --- | --- | --- | --- |
| 1 | .142 / .794 / .795 | .115 / .607 / .606 | .215 / 1.49 / 1.49 | .183 / 1.07 / 1.07 |
| 2 | .0882 / .540 / .541 | .0730 / .457 / .454 | .175 / 1.23 / 1.23 | .139 / .850 / .849 |
| 3 | .0457 / .315 / .316 | .0452 / .305 / .305 | .146 / .890 / .888 | .0912 / .538 / .537 |
| 4 | .0289 / .235 / .237 | .0322 / .242 / .242 | .115 / .718 / .719 | .0649 / .403 / .401 |
| 6 | .0158 / .170 / .171 | .0207 / .183 / .182 | .0795 / .506 / .507 | .0368 / .274 / .273 |
| 9 | .00890 / .134 / .134 | .0145 / .145 / .144 | .0486 / .340 / .339 | .0202 / .196 / .196 |
| 12 | .00604 / .117 / .117 | .0119 / .127 / .126 | .0309 / .249 / .248 | .0136 / .162 / .161 |
| 16 | .00458 / .105 / .105 | .00985 / .113 / .112 | .0172 / .180 / .180 | .00884 / .135 / .135 |
| 24 | .00321 / .0915 / .0916 | .00729 / .0958 / .0955 | .00580 / .118 / .118 | .00456 / .104 / .104 |
| 32 | .00262 / .0837 / .0841 | .00599 / .0863 / .0857 | .00223 / .0926 / .0928 | .00302 / .0889 / .0889 |
| 48 | .00215 / .0768 / .0771 | .00477 / .0770 / .0768 | .000984 / .0776 / .0776 | .00203 / .0770 / .0771 |
| 64 | .00201 / .0739 / .0742 | .00454 / .0743 / .0742 | .000809 / .0742 / .0742 | .00186 / .0741 / .0742 |

### 20 dB

| K | IDENTITY | OFDM | OTFS | AFDM_K3_TUNED |
| ---: | --- | --- | --- | --- |
| 1 | .137 / .833 / .831 | .108 / .625 / .627 | .210 / 1.63 / 1.63 | .178 / 1.13 / 1.14 |
| 2 | .0978 / .761 / .761 | .0805 / .616 / .625 | .193 / 1.96 / 1.95 | .153 / 1.24 / 1.26 |
| 3 | .0499 / .388 / .391 | .0501 / .392 / .395 | .162 / 1.27 / 1.27 | .102 / .731 / .747 |
| 4 | .0313 / .277 / .282 | .0372 / .307 / .308 | .133 / 1.07 / 1.06 | .0741 / .540 / .558 |
| 6 | .0169 / .179 / .183 | .0240 / .217 / .218 | .0927 / .711 / .713 | .0451 / .357 / .365 |
| 9 | .00933 / .127 / .127 | .0163 / .160 / .161 | .0567 / .437 / .438 | .0262 / .242 / .245 |
| 12 | .00550 / .101 / .102 | .0125 / .131 / .131 | .0349 / .292 / .292 | .0172 / .187 / .188 |
| 16 | .00321 / .0819 / .0825 | .00941 / .106 / .106 | .0169 / .187 / .188 | .00972 / .139 / .140 |
| 24 | .00137 / .0604 / .0608 | .00500 / .0735 / .0739 | .00426 / .0970 / .0978 | .00286 / .0819 / .0828 |
| 32 | .000763 / .0482 / .0483 | .00265 / .0534 / .0540 | .000633 / .0599 / .0608 | .000965 / .0556 / .0559 |
| 48 | .000378 / .0378 / .0376 | .00134 / .0373 / .0376 | 3.81e-5 / .0382 / .0384 | .000313 / .0375 / .0377 |
| 64 | .000355 / .0337 / .0336 | .00121 / .0336 / .0336 | 1.14e-5 / .0336 / .0336 | .000275 / .0334 / .0336 |

K=3 is far from full-information behavior for all four transforms. Around K=48, IDENTITY, OFDM, and AFDM are near their K=64 BER; OTFS is already the best BER by K=32 but needs K=64 to resolve its much lower full-CSI error rate.

## D. Representation concentration

Mean DD-v1 retained energy:

| K | IDENTITY | OFDM | OTFS | AFDM_K3_TUNED |
| ---: | ---: | ---: | ---: | ---: |
| 1 | .591 | .674 | .416 | .501 |
| 2 | .815 | .863 | .605 | .716 |
| 3 | .904 | .916 | .707 | .821 |
| 4 | .939 | .939 | .769 | .875 |
| 6 | .964 | .961 | .838 | .925 |
| 9 | .977 | .975 | .891 | .955 |
| 12 | .983 | .982 | .922 | .968 |
| 16 | .988 | .987 | .949 | .977 |
| 24 | .992 | .992 | .977 | .987 |
| 32 | .995 | .995 | .990 | .992 |
| 48 | .998 | .999 | .998 | .998 |
| 64 | 1.000 | 1.000 | 1.000 | 1.000 |

At K=3, channel-level mean/median/10th/90th percentiles are:

| Waveform | Mean | Median | P10 | P90 |
| --- | ---: | ---: | ---: | ---: |
| IDENTITY | .9043 | .9094 | .8131 | .9791 |
| OFDM | .9162 | .9150 | .8676 | .9714 |
| OTFS | .7069 | .7120 | .5420 | .8692 |
| AFDM_K3_TUNED | .8211 | .8222 | .7278 | .9083 |

Across every row and channel at K=3:

| Waveform | Mean | Median | P05 | P10 | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDENTITY | .9017 | .9160 | .7590 | .7950 | .5648 |
| OFDM | .9039 | .9132 | .7880 | .8186 | .3467 |
| OTFS | .7067 | .7105 | .5014 | .5416 | .3064 |
| AFDM_K3_TUNED | .8161 | .8303 | .6369 | .6806 | .3680 |

The retained/omitted partition residual is at double-precision scale, retained energy is nondecreasing in K, and K=N equals one.

### High-SNR mismatch mechanism

| Waveform | Mean omitted energy | Median cond(T3) | P90 cond(T3) | Singular fraction | Median limiting distortion | P90 limiting distortion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDENTITY | .0957 | 26.2 | 3.90e3 | 0 | 1.03 | 5.32e3 |
| OFDM | .0838 | 42.7 | 6.70e3 | .00586 | 1.81 | 2.74e4 |
| OTFS | .2931 | 29.8 | 7.27e2 | 0 | 5.82 | 2.06e3 |
| AFDM_K3_TUNED | .1789 | 83.5 | 5.58e3 | .00195 | 15.95 | 4.26e4 |

`||T_3(A)^\dagger A-I||_F^2/N` is heavy-tailed because a small number of truncated matrices are nearly singular; medians and percentiles are therefore more informative than means. These results explain the high-SNR rise: regularization initially protects against both omitted interactions and ill-conditioned inversion, then disappears.

## E. OTFS audit

The independent transmitter uses an `8 x 8` DD matrix `X`,

`Y = F_8 X F_8^H`, then `S = F_8^H Y = X F_8^H`,

and serializes `S` column-major, so the first (delay/subcarrier) index varies fastest. Because the unitary DFT matrix is symmetric, this serialization is exactly multiplication by `F_8^H kron I_8`.

- Two-stage versus Kronecker maximum error: `4.076e-15`.
- Clean modulation/demodulation reconstruction error: `7.796e-15`.
- Identity: one significant entry per row, top-1 energy one.
- One integer delay, one integer Doppler, and one joint integer path: one significant entry per row, top-1 energy one.
- Three distinct on-grid DD paths: exactly three significant entries per row and top-3 energy one.
- The integer-delay cases can have two dominant circular offsets because the one-dimensional serialization wraps across 8-sample column boundaries; numerical sparsity remains exact.

No OTFS source correction was required. Its poor DD-v1/K=3 result is not a basic modulation-matrix error. It reflects leakage under the continuous-delay/continuous-Doppler one-dimensional circular channel framing. This Stage-0 transform is still an algebraic unitary OTFS/DD comparison, not a physically framed multicarrier OTFS system with per-slot CP and pulses.

## F. AFDM audit

The audited analysis transform is `A_DAFT = Lambda_c2 F Lambda_c1`; synthesis is `Lambda_c1^H F^H Lambda_c2^H`. The independently evaluated IDAFT equation matches the factored matrix to `7.041e-14`.

- `c2 = sqrt(2)/(100N) = 2.2097086912e-4` at N=64.
- **AFDM_K3_TUNED:** `c1=1/128=0.0078125`, selected solely on validation analytic K=3 MSE.
- Canonical integer-support rule: `c1=(2 alpha_max+1)/(2N)`.
- GRID-DD audit with `alpha_max=2`: `c1=5/128=0.0390625`.
- Guarded DD-v1 reference: `alpha_max=ceil(0.75)=1`, one extra leakage bin, `c1=5/128`. This is a declared reference, not a fractional-delay guarantee.
- N=256 reference with `alpha_max=2`: `c1=5/512`.

For all 20 combinations `delay in {0,1,2,4}` and `Doppler in {-2,-1,0,1,2}`, every row's observed canonical support offset equals

`(2*N*c1*delay + Doppler) mod N`,

and the minimum dominant magnitude is one to numerical precision. Distinct supported paths therefore separate as prescribed on the integer grid.

On DD-v1, the guarded canonical variant is inferior to AFDM_K3_TUNED:

| Variant | eta3 | BER K3, 15 dB | BER K3, 20 dB | Analytic MSE K3, 20 dB | Median cond(T3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AFDM_K3_TUNED | .8211 | .09136 | .10204 | .74747 | 83.5 |
| AFDM_CANONICAL_GUARDED | .7027 | .14644 | .16455 | 1.30858 | 46.1 |

The tuned variant remains the fair AFDM baseline for the project's exact K=3 objective. The canonical variant is retained as a support-law reference, not substituted merely because it is canonical.

## G. Channel compatibility

At 20 dB (`BER K3 / BER full / mean eta3`):

| Regime | IDENTITY | OFDM | OTFS | AFDM_K3_TUNED |
| --- | --- | --- | --- | --- |
| GRID-DD | .00190 / .00190 / 1.000 | .00204 / .00204 / 1.000 | .000519 / .000519 / 1.000 | .00111 / .00111 / 1.000 |
| FRAC-DOPPLER | .000542 / .000542 / 1.000 | .05850 / .00164 / .908 | .11440 / 9.16e-5 / .815 | .08013 / .000282 / .866 |
| DD-v1 | .04907 / .000397 / .907 | .05102 / .00130 / .917 | .16491 / 7.63e-6 / .709 | .10207 / .000374 / .823 |

GRID-DD proves that three paths are exactly representable by K=3 in all four bases for this controlled family. Fractional Doppler spreads transformed-domain representations but leaves IDENTITY exact because delays remain integer. Fractional delay then spreads IDENTITY and further degrades OTFS/AFDM concentration; OFDM becomes the most concentrated of the principal transforms.

## H. Classical reference experiment

Configuration: `N=256`, `16 x 16` OTFS, `P=3`, distinct integer delays `0..2`, `nu=2 cos(theta)`, raw iid `CN(0,1/P)` gains without per-realization normalization, QPSK, paired noise/bits, perfect full LMMSE, and canonical AFDM `c1=5/512`. Each point uses 1,048,576 bits.

| SNR dB | OFDM | OTFS | AFDM_CANONICAL |
| --- | ---: | ---: | ---: |
| 0 | .20816 | .20690 | .20693 |
| 5 | .10288 | .10206 | .10198 |
| 10 | .03306 | .03015 | .03045 |
| 15 | .006678 | .004354 | .004547 |
| 20 | .001033 | .000299 | .000324 |
| 25 | .000176 | 4.77e-6 | 5.72e-6 |
| 30 | 3.81e-5 | 0 observed | 0 observed |

The 30 dB OTFS/AFDM points each have the rule-of-three bound `BER < 2.861e-6`. The expected qualitative classical behavior is recovered: all three waterfall, and OTFS/AFDM outperform OFDM at moderate/high SNR. This is a sanity experiment, not a numerical reproduction of a published figure.

## I. Doppler-severity audit

Continuous delays remain `Uniform[0,4]`; only `nu_max` changes. Entries are `mean eta3 / K3 BER / full BER` at 20 dB.

| nu_max | OFDM | OTFS | AFDM_K3_TUNED |
| ---: | --- | --- | --- |
| .25 | .972 / .0130 / .00216 | .848 / .0725 / 3.81e-5 | .875 / .0675 / .000183 |
| .75 | .916 / .0491 / .00141 | .713 / .1578 / 0 observed | .820 / .1019 / .000313 |
| 1.5 | .917 / .0447 / .00110 | .712 / .1590 / 4.58e-5 | .826 / .1004 / .000465 |
| 2.0 | .906 / .0523 / .000824 | .707 / .1621 / 9.92e-5 | .816 / .1088 / .000847 |
| 4.0 | .882 / .0675 / .000694 | .702 / .1655 / .000198 | .795 / .1235 / .000771 |

DD-v1's original Doppler range is OFDM-friendly, but this is not only a mild-Doppler artifact: OFDM remains the most concentrated and best K=3 baseline through `nu_max=4`. Full-CSI BER remains noise-limited in every range, again localizing the problem to representation truncation.

## J. Changes made

- `src/waveforms.py`: renamed the control to IDENTITY; added explicit AFDM_K3_TUNED and AFDM_CANONICAL constructors. Core matrix formulas are unchanged.
- `src/config.py`: made the tuned and guarded canonical AFDM roles explicit while preserving backward-compatible Stage-0 constants.
- `src/audit.py`: added independent OTFS/AFDM references, retained-energy diagnostics, support rules, and controlled audit channel families.
- `scripts/stage0.py`: future user-facing labels use IDENTITY. The historical Stage-0 report and outputs were deliberately not regenerated.
- `scripts/stage0e_audit.py`: added paired full/K sweeps, structural checks, compatibility/severity experiments, N=256 reference, CSV output, and figures.
- `tests/test_core.py`: changed the control label to IDENTITY.
- `tests/test_stage0e_audit.py`: added nine audit test functions covering all requested invariants.
- `README.md`: clarified IDENTITY's control role.
- `HISTORY.md`: recorded concise audit iterations and findings.

`src/channel.py`, `src/receiver.py`, and `src/metrics.py` were not modified. No validated Stage 0A-D numerical logic was refactored.

## K. Tests

Final command: `pytest -q`

Result: `31 passed`.

The audit experiment command was `python -m scripts.stage0e_audit`; final runtime was approximately 61.5 seconds in the PyTorch conda environment.

## L. Git hygiene and artifacts

- `outputs/` remains ignored; no generated audit file is tracked.
- All new generated artifacts are under `outputs/stage0e_audit/`.
- The original `outputs/stage0/` artifacts and `reports/stage0_summary.md` were preserved.
- `researchPlan.MD` was not modified.
- The final test seed/bank was not generated, loaded, or evaluated.
- No learned waveform or Stage 1 code was implemented.

Principal figures:

- `outputs/stage0e_audit/ber_full_vs_k3.png`
- `outputs/stage0e_audit/ber_vs_K_15dB.png`
- `outputs/stage0e_audit/ber_vs_K_20dB.png`
- `outputs/stage0e_audit/mse_vs_K_15dB.png`
- `outputs/stage0e_audit/mse_vs_K_20dB.png`
- `outputs/stage0e_audit/retained_energy_vs_K.png`
- `outputs/stage0e_audit/classical_reference_full_lmmse.png`
- `outputs/stage0e_audit/doppler_severity_ber_k3.png`
- `outputs/stage0e_audit/doppler_severity_retained_energy.png`

Structural heatmaps are under `outputs/stage0e_audit/otfs_sanity/`, `afdm_sanity/`, and `channel_regimes/`. Complete numerical tables are stored alongside them as CSV/JSON.

## Final decision

The classical implementations are validated. The original result is caused by an extreme mismatched K=3 receiver applied to a continuous-delay DD-v1 framing that strongly favors OFDM concentration and lies outside canonical AFDM integer-delay guarantees. Human review must decide whether that narrow stress test remains the intended Stage 1A target.

STAGE 0E AUDIT PASSED — CONFIGURATION REQUIRES SCIENTIFIC REVIEW
