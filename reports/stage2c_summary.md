# Stage 2C — Receiver-Budget Causality and Learned Subspace Reorganization

Outcome **C (qualified)**. Receiver-budget law is not supported as a stable 4-to-3-to-2 law. Group counts are initialization-sensitive, but the Haar runs have materially worse validation objectives, so this does not establish communication-equivalent symmetry solutions.

## Protocol

- N=64, P=3, DD-v1 unchanged; K train in [16, 24, 32].
- Loss: mean MSE + 0.40 [CVaR_0.95 - mean MSE]; 1,000 steps, batch 16, unchanged learning-rate schedule.
- Starts: OFDM plus Haar seeds 11012 and 11013; online channel/SNR seeds 11036/11037 shared across K.
- Checkpoint selection: lowest validation stream CVaR 0.95 within the per-start 2% mean-MSE gate; BER and structure excluded.
- Branch `cambridge`, target `origin/main`; no commit, push, merge, P variation, or Stage 3 action.
- Added `src/stage2c.py`, `scripts/stage2c.py`, `tests/test_stage2c.py`, `notebooks/03_stage2c_receiver_budget_causality.ipynb`, and this report; numerical artifacts remain ignored.

## Validation-selected runs

| K | start | step | validation mean MSE | validation CVaR 0.95 |
|---:|---|---:|---:|---:|
| 16 | OFDM | 1000 | 0.08957261 | 0.23232815 |
| 24 | OFDM | 1000 | 0.07505774 | 0.18730690 |
| 32 | OFDM | 1000 | 0.07468262 | 0.15388583 |

## Structural result

| K | start | predicted | discovered | sizes | within energy | within/cross | support within/across | rank corr. | ARI |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 16 | OFDM | 4 | 6 | 6;13;14;13;8;10 | 0.9094 | 10.03 | 0.839/0.122 | 0.958 | 0.273 |
| 16 | HAAR_1 | 4 | 2 | 30;34 | 0.9017 | 9.17 | 0.443/0.056 | 0.122 | 0.020 |
| 16 | HAAR_2 | 4 | 3 | 15;29;20 | 0.6675 | 2.01 | 0.434/0.148 | 0.137 | 0.159 |
| 24 | OFDM | 3 | 5 | 8;15;14;13;14 | 0.9255 | 12.41 | 0.928/0.230 | 0.970 | 0.922 |
| 24 | HAAR_1 | 3 | 2 | 31;33 | 0.7525 | 3.04 | 0.516/0.234 | 0.559 | 0.060 |
| 24 | HAAR_2 | 3 | 4 | 13;18;17;16 | 0.5256 | 1.11 | 0.607/0.296 | 0.583 | 0.210 |
| 32 | OFDM | 2 | 3 | 19;22;23 | 0.9105 | 10.17 | 0.953/0.271 | 0.968 | 0.555 |
| 32 | HAAR_1 | 2 | 2 | 35;29 | 0.7774 | 3.49 | 0.685/0.312 | 0.708 | 0.509 |
| 32 | HAAR_2 | 2 | 2 | 30;34 | 0.7867 | 3.69 | 0.691/0.307 | 0.560 | 0.343 |

## Matched-budget Learned BER

| K | modulation | SNR dB | BER |
|---:|---|---:|---:|
| 16 | QPSK | 12 | 0.013094902 |
| 16 | QPSK | 15 | 0.0056676865 |
| 16 | QPSK | 18 | 0.002948761 |
| 16 | QPSK | 20 | 0.0024337769 |
| 16 | 16QAM | 12 | 0.092395782 |
| 16 | 16QAM | 15 | 0.056120872 |
| 16 | 16QAM | 18 | 0.035552979 |
| 16 | 16QAM | 20 | 0.028291702 |
| 24 | QPSK | 12 | 0.010803223 |
| 24 | QPSK | 15 | 0.0038585663 |
| 24 | QPSK | 18 | 0.0017032623 |
| 24 | QPSK | 20 | 0.0011196136 |
| 24 | 16QAM | 12 | 0.08600235 |
| 24 | 16QAM | 15 | 0.047445297 |
| 24 | 16QAM | 18 | 0.025443077 |
| 24 | 16QAM | 20 | 0.016760826 |
| 32 | QPSK | 12 | 0.0088224411 |
| 32 | QPSK | 15 | 0.0027770996 |
| 32 | QPSK | 18 | 0.00091648102 |
| 32 | QPSK | 20 | 0.00054645538 |
| 32 | 16QAM | 12 | 0.088199615 |
| 32 | 16QAM | 15 | 0.048203468 |
| 32 | 16QAM | 18 | 0.024544716 |
| 32 | 16QAM | 20 | 0.015789032 |

## Same-seed retraining comparison at 18 dB

| K | modulation | retrained BER | frozen K=24 BER | relative change |
|---:|---|---:|---:|---:|
| 16 | QPSK | 0.002948761 | 0.0056247711 | -47.6% |
| 16 | 16QAM | 0.035552979 | 0.061632156 | -42.3% |
| 24 | QPSK | 0.0017032623 | 0.0010967255 | +55.3% |
| 24 | 16QAM | 0.025443077 | 0.023892403 | +6.5% |
| 32 | QPSK | 0.00091648102 | 0.00098991394 | -7.4% |
| 32 | 16QAM | 0.024544716 | 0.022621155 | +8.5% |

## Cross-budget analytic MSE at 18 dB

| K train | K eval | MSE | retained energy | limiting distortion median |
|---:|---:|---:|---:|---:|
| 16 | 8 | 0.16755193 | 0.951516 | 0.340386 |
| 16 | 12 | 0.08965071 | 0.980273 | 0.098705 |
| 16 | 16 | 0.06644784 | 0.990170 | 0.042179 |
| 16 | 20 | 0.05784572 | 0.994153 | 0.023456 |
| 16 | 24 | 0.05352914 | 0.996269 | 0.014211 |
| 16 | 28 | 0.05099711 | 0.997572 | 0.008915 |
| 16 | 32 | 0.04938734 | 0.998424 | 0.005759 |
| 16 | 40 | 0.04760921 | 0.999384 | 0.002174 |
| 16 | 48 | 0.04683005 | 0.999806 | 0.000699 |
| 16 | 64 | 0.04647122 | 1.000000 | 0.000000 |
| 24 | 8 | 0.20439745 | 0.936488 | 0.513383 |
| 24 | 12 | 0.09910151 | 0.976298 | 0.125577 |
| 24 | 16 | 0.06725207 | 0.989783 | 0.044261 |
| 24 | 20 | 0.05545460 | 0.995302 | 0.018031 |
| 24 | 24 | 0.05065953 | 0.997742 | 0.008306 |
| 24 | 28 | 0.04870255 | 0.998783 | 0.004458 |
| 24 | 32 | 0.04780509 | 0.999270 | 0.002599 |
| 24 | 40 | 0.04698650 | 0.999719 | 0.000996 |
| 24 | 48 | 0.04664484 | 0.999906 | 0.000345 |
| 24 | 64 | 0.04647122 | 1.000000 | 0.000000 |
| 32 | 8 | 0.64607406 | 0.813118 | 2.861944 |
| 32 | 12 | 0.30010554 | 0.903332 | 0.844995 |
| 32 | 16 | 0.15481434 | 0.952432 | 0.276811 |
| 32 | 20 | 0.09232123 | 0.978049 | 0.104093 |
| 32 | 24 | 0.06600991 | 0.990186 | 0.041964 |
| 32 | 28 | 0.05492095 | 0.995657 | 0.017742 |
| 32 | 32 | 0.05024381 | 0.998043 | 0.007795 |
| 32 | 40 | 0.04746358 | 0.999479 | 0.001934 |
| 32 | 48 | 0.04675445 | 0.999851 | 0.000540 |
| 32 | 64 | 0.04647122 | 1.000000 | 0.000000 |

## Checkpoint hashes

- `k16_ofdm_best.pt`: `a91d4fb2066d35a7b8e13ffbf869801544e63e8328a4318b04f5dab7ad49a115`
- `k16_haar_1_best.pt`: `51c09dec0f98aba6bed7e0ea9cf9e6e2da01e288b05954368152d4795e0f4404`
- `k16_haar_2_best.pt`: `b2bd1d2e8a6f649b46c4533319d1f0e50bfbd15078b1a55fa3827a4833d21e81`
- `k24_ofdm_best.pt`: `73a401b6034c04863628684ae20be9a90c3ee866e1204445c205ab70027a8fa6`
- `k24_haar_1_best.pt`: `f4e754cb3f74adfddb14f385a446b030d47cceaa09b8aa1d8008ad34f323e1e8`
- `k24_haar_2_best.pt`: `68740e40d2f624f135d44b8bf648d892110d35e3234a8115a233f17be7d34ecb`
- `k32_ofdm_best.pt`: `d8a04b5ce658e488021ef3c2f007556cea094a92320c31b4b7d05d0e23cd4e90`
- `k32_haar_1_best.pt`: `3ffc33447951f8af98efefb4a0f196d2080f83be54ebcfc321a8b1b0aca7b38b`
- `k32_haar_2_best.pt`: `9c18adc453b61370d35d61fdaf675a7a8417d196258a6d87ad70340f1599235c`

## Preservation and runtime

- Frozen checkpoint SHA-256: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`.
- Stage-2A/2B artifacts unchanged: `True`.
- Training runtime: 371.1 s; final evaluation/runtime: 160.8 s; full protocol: 531.9 s.
- Final complete suite: 76 passed in 2.18 s; notebook executed top-to-bottom with zero error outputs; `git diff --check` passed.

Scientific interpretation: the group-count comparison is observational evidence from unconstrained communication training only. No physical-path, N, DD-distribution, or structural-loss variable changed. The 47.6% QPSK and 42.3% 16-QAM K=16 BER reductions show matched-budget repair, while the non-predicted and start-sensitive partitions reject a simple causal `ceil(64/K)` law in this protocol. The weaker Haar objectives prevent interpreting their partitions as communication-equivalent symmetry solutions. STOP — no P variation, waveform-family fitting, or Stage 3 was performed.
