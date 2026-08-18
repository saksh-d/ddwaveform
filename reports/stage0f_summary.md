# Stage 0F — channel-specific dense-unitary oracle diagnostic

## A. Scope, question, and verdict

Stage 0F asks whether the validated DD-v1 channel family contains representation headroom beyond the fixed OFDM, OTFS, and validation-tuned AFDM bases at practical hard-support budgets. It is a diagnostic only: every physical channel is allowed its own dense unitary matrix, so the result is neither a deployable waveform nor Stage 1.

The answer is yes. At K=16, 24, and 32, the channel-bootstrap 95% confidence interval for the mean paired oracle-minus-best-principal analytic MSE is entirely below zero. Paired BER also improves at both tested SNRs. The correct name is **best-found channel-specific unitary oracle**: projected Adam with six starts does not certify a global optimum.

K=16 has the largest measurable headroom. K=24 is the recommended discovery regime because it retains substantial headroom while the strongest classical BER falls normally with SNR; K=16's classical receivers remain dominated by a high-SNR truncation-mismatch effect.

## B. Locked protocol and AFDM calibration

- CPU complex128 throughout oracle construction and evaluation.
- 24 independent DD-v1 calibration channels, seed 11005.
- Exact row-wise hard top-K supports with K in {16, 24, 32}.
- Objective averaged over 12, 15, 18, and 20 dB; an additional multi-K oracle averages all three K values and all four SNRs.
- Six deterministic starts per channel: IDENTITY, OFDM, OTFS, tuned AFDM, and two Haar unitaries.
- 300 projected-Adam updates per start: learning rate 0.03 for steps 1–150, 0.01 for 151–250, and 0.003 for 251–300. Every update is mapped to the nearest unitary by an SVD polar projection.
- Best-seen, rather than final, iterate used. Every restart records initial, best, and final objective, best step, unitarity residual, and non-finite status.

AFDM was tuned only on validation channels. With c2 fixed at 0.0002209708691, every objective selected m=1, hence c1=1/128.

| m | K=16 | K=24 | K=32 | Multi-K |
|---:|---:|---:|---:|---:|
| 1 | **0.146162** | **0.107047** | **0.088115** | **0.113774** |
| 2 | 0.174724 | 0.118736 | 0.092902 | 0.128787 |
| 3 | 0.203904 | 0.132691 | 0.100228 | 0.145608 |
| 4 | 0.217334 | 0.134739 | 0.100662 | 0.150911 |
| 5 | 0.237976 | 0.148640 | 0.108236 | 0.164951 |
| 6 | 0.252593 | 0.158439 | 0.112827 | 0.174620 |
| 7 | 0.249756 | 0.155559 | 0.111793 | 0.172369 |
| 8 | 0.229319 | 0.143015 | 0.105603 | 0.159312 |

## C. Optimization reliability

Across the 576 primary restart runs, 537 (93.2%) improved on their initialization. The median best step was 257.5, with range 0–300. The final iterate was worse than the stored best iterate in 426 runs, consistent with hard-support switching and confirming that best-seen retention is necessary. No run encountered a non-finite value. The maximum final unitarity residual was 7.18e-14.

Winner counts across the 96 channel/objective combinations were AFDM 41, OFDM 33, IDENTITY 19, and Haar-1 3; OTFS and Haar-2 did not win. Classical informed starts therefore matter, but no single initialization is sufficient.

## D. Analytic headroom

The principal baseline is the best of OFDM, OTFS, and tuned AFDM on each channel. Pooled means average the four locked design SNRs within each channel before averaging channels. Negative delta is better. Confidence intervals use 10,000 paired channel bootstrap resamples with seed 11008.

| K | Best principal MSE | Oracle MSE | Delta | 95% CI for delta | Mean relative gain | Mean gap closed |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.122537 | 0.077910 | -0.044626 | [-0.061819, -0.030102] | 31.3% | 73.2% |
| 24 | 0.096304 | 0.074782 | -0.021521 | [-0.029410, -0.014442] | 20.4% | 69.5% |
| 32 | 0.082689 | 0.073127 | -0.009561 | [-0.012953, -0.006372] | 11.2% | 61.6% |

All 24 channels improve over their best principal baseline at every K. Fractions exceeding relative-gain thresholds of 5% / 10% / 20% are 87.5% / 79.2% / 66.7% at K=16, 75.0% / 70.8% / 50.0% at K=24, and 66.7% / 45.8% / 20.8% at K=32.

IDENTITY beats all principal waveforms on 16/24, 14/24, and 13/24 pooled channel cases for K=16, 24, and 32. Using the stronger post hoc baseline `min(IDENTITY, best principal)`, the oracle still improves 20/24, 19/24, and 18/24 channels, with mean relative gains 18.6%, 12.7%, and 7.8%. Thus the conclusion is not solely an artifact of excluding the untransformed control, although the preregistered inferential comparison remains against the principal classical set.

At 15 and 20 dB respectively, mean oracle-minus-principal deltas are:

| K | 15 dB delta (95% CI) | 20 dB delta (95% CI) |
|---:|---:|---:|
| 16 | -0.031893 [-0.042319, -0.022387] | -0.073322 [-0.106300, -0.046395] |
| 24 | -0.015412 [-0.020086, -0.010915] | -0.033200 [-0.045911, -0.021785] |
| 32 | -0.007817 [-0.010560, -0.005242] | -0.013281 [-0.018054, -0.008887] |

## E. Paired BER transfer

Each K/SNR/waveform point uses 1,001,472 paired bits: identical channels, bits, and time-domain noise across waveforms. The oracle matrices are fixed before BER evaluation. The strongest classical entry in each row is marked with an asterisk.

| K | SNR | OFDM | OTFS | AFDM | Oracle | Multi-K oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15 | **0.010265** * | 0.018281 | 0.010304 | 0.001914 | 0.002229 |
| 16 | 20 | **0.012365** * | 0.019310 | 0.013916 | 0.000288 | 0.000412 |
| 24 | 15 | 0.007342 | 0.006781 | **0.004949** * | 0.001290 | 0.001723 |
| 24 | 20 | 0.006395 | 0.004943 | **0.003546** * | 0.000139 | 0.000215 |
| 32 | 15 | 0.006209 | **0.002956** * | 0.003032 | 0.001294 | 0.001623 |
| 32 | 20 | 0.003355 | 0.001146 | **0.000943** * | 0.000147 | 0.000207 |

Relative to the best classical aggregate BER, oracle reductions are 81.4% and 97.7% at K=16, 73.9% and 96.1% at K=24, and 56.2% and 84.4% at K=32 for 15 and 20 dB. Analytic gains therefore transfer decisively to BER. K=16's best classical BER rises from 0.010265 at 15 dB to 0.012365 at 20 dB, while its oracle BER waterfalls; this is evidence of severe classical model mismatch and the reason not to recommend K=16 as the primary discovery budget.

## F. Representation mechanism

Retained energy is averaged across channels. Condition number and high-SNR limiting distortion are channel medians.

| K | Entity | Retained energy | Condition number | Limiting distortion |
|---:|---|---:|---:|---:|
| 16 | Best principal | 0.984231 | 22.4446 | 0.398779 |
| 16 | Oracle | 0.991735 | 15.1985 | 0.014709 |
| 24 | Best principal | 0.990104 | 18.0340 | 0.083246 |
| 24 | Oracle | 0.995417 | 16.6910 | 0.009090 |
| 32 | Best principal | 0.994029 | 15.4398 | 0.029938 |
| 32 | Oracle | 0.997395 | 16.1255 | 0.005176 |

The oracle consistently retains more operator energy and, more importantly, sharply reduces limiting mismatch distortion. Conditioning improves at K=16 and K=24 but is slightly worse at K=32. The mechanism is therefore primarily better support concentration and a better truncated operator, not universal conditioning improvement.

## G. Single-K versus multi-K oracle

The ratio is the multi-K objective evaluated at K divided by the independently best-found single-K objective.

| K | Mean | Median | 10th–90th percentile | Range |
|---:|---:|---:|---:|---:|
| 16 | 1.0461 | 1.0429 | 1.0000–1.0971 | 1.0000–1.1224 |
| 24 | 1.0047 | 1.0000 | 0.9910–1.0217 | 0.9877–1.0317 |
| 32 | 0.9993 | 1.0000 | 0.9878–1.0091 | 0.9781–1.0207 |

One channel-specific basis nearly spans K=24 and K=32, with a modest mean penalty at K=16. Ratios below one do not violate an optimum bound: they show that the multi-K run occasionally found a better local basin than the non-certified single-K run.

## H. Negative and mechanism controls

Controls use seed 11006; random oracle starts use seed 11007.

- **GRID-DD, K=3, 8 channels:** IDENTITY is already exactly path-sparse. Oracle-minus-full MSE ranges from -2.78e-17 to +2.78e-17; no full-reference violation occurs. The optimizer correctly finds no physical headroom beyond full information.
- **Fractional Doppler, K=24, 8 channels:** oracle-minus-full ranges from -6.94e-17 to +5.55e-17, and IDENTITY wins all eight channels. With integer delays, fractional Doppler alone remains a time-domain row-sparse modulation, so K=24 already contains the full operator. This small control validates the mechanism but is not a general fractional-offset claim.

## I. Required artifacts

Generated, ignored artifacts are under `outputs/stage0f/`:

- `oracle_mse_vs_K_15dB.png`, `oracle_mse_vs_K_20dB.png`
- `oracle_ber_vs_K_15dB.png`, `oracle_ber_vs_K_20dB.png`
- `oracle_relative_gain_distribution.png`
- `oracle_retained_energy_vs_K.png`
- `oracle_gap_closure_vs_K.png`
- `multik_oracle_penalty.png`
- `convergence/k16.png`, `k24.png`, `k32.png`, and `multik.png`
- Per-channel analytic, BER, bootstrap, initialization, multi-K, control, representation, AFDM-tuning, checkpoint, and run-diagnostic records.

All required figures were rendered and visually inspected. Curves, legends, scales, and support labels are readable and unclipped.

## J. Scientific recommendation

1. **Where is headroom largest?** K=16: 31.3% mean relative analytic improvement, 73.2% mean classical-to-full gap closure, and the largest BER reduction.
2. **Which budget is most informative without pathological mismatch dominance?** K=24. It retains a 20.4% mean analytic improvement and 69.5% gap closure, while the best classical BER declines from 15 to 20 dB. K=32 is cleaner but leaves materially less discoverable headroom.
3. **What should any later Stage 1 target?** K=24 as the primary regime, K=16 as a stress test, and K=32 as a near-saturation control. A learned method should be compared both with the principal baselines and IDENTITY because the latter is unexpectedly strong on many DD-v1 channels.

## K. Reproducibility, boundaries, and status

The production run completed in 1735.5 s. All 37 deterministic tests pass. The validated Stage 0A-E channel, waveform, receiver, and metric implementations were not modified; Stage 0F adds only an oracle module, runner, tests, configuration constants, ignored outputs, and this report. `researchPlan.MD` is unchanged. No final test bank was opened, tuned on, or evaluated, and no Stage 1 model was implemented.

STAGE 0F COMPLETE — CHANNEL-SPECIFIC HEADROOM CONFIRMED
