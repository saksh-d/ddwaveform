# Pub-1F continuation — Qualified replication, matched quality, and scaling

## Decision
**Reproducible representation advantage with arithmetic/storage savings only.** The original resource gate remains **FAIL** (`passed=False`) and was not edited. This post-gate continuation was separately authorized, with 16-QAM primary and QPSK retained as a mandatory secondary outcome.

The same-K representation rule is **SUPPORTED**: 5/5 selected tail checkpoints beat validation-chosen OFDM on fresh joint-(0.5,0.5), K=24, 18 dB 16-QAM data. All 5/5 paired 95% channel-bootstrap intervals exclude zero in Learned's favor. The computational-saving rule is **NOT SUPPORTED**; selected recurring detector costs tie rather than beat classical alternatives.

## N=64 training replication and objective ablation
Every tail arm passes the seed-relative 2% mean gate (5/5) and improves validation stream CVaR_0.95 (5/5), so the frozen tail-contribution rule is **SUPPORTED**. That is not a BER-improvement claim: tail continuation has significantly higher primary 16-QAM BER than its matched mean-only arm in 4/5 seeds; the remaining seed interval includes zero.

| seed | parent mean MSE | mean CVaR | tail CVaR | tail BER | tail - OFDM 95% CI | tail - mean 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.071294 | 0.171346 | 0.139150 | 0.023963 | [-0.008991, -0.007497] | [-0.000070, 0.000886] |
| 1 | 0.068180 | 0.176488 | 0.139263 | 0.024025 | [-0.008920, -0.007429] | [0.000625, 0.001589] |
| 2 | 0.071654 | 0.181204 | 0.144319 | 0.024026 | [-0.008967, -0.007382] | [0.000215, 0.001265] |
| 3 | 0.074135 | 0.194279 | 0.163931 | 0.024352 | [-0.008559, -0.007111] | [0.000628, 0.001597] |
| 4 | 0.069399 | 0.173482 | 0.142232 | 0.024156 | [-0.008813, -0.007283] | [0.000649, 0.001683] |

Tail-seed primary BER range is 0.023963–0.024352, median 0.024026; OFDM is 0.032200. The historical checkpoint remains separate from these five OFDM-initialized replications.

## Regime and modulation controls

| condition | modulation | validation-chosen classical | tail wins / 5 | paired CIs below zero / 5 |
|---|---|---|---:|---:|
| joint_0.5 | 16QAM | OFDM (0.0321999) | 5 | 5 |
| joint_0.5 | QPSK | AFDM (0.00149441) | 5 | 4 |
| joint_0.2_probe | 16QAM | OFDM (0.0298891) | 5 | 5 |
| joint_0.2_probe | QPSK | AFDM (0.00111103) | 1 | 0 |
| integer | 16QAM | AFDM (0.0250883) | 0 | 0 |
| integer | QPSK | AFDM (0.000221252) | 0 | 0 |
| nominal_dd_v1 | 16QAM | OFDM (0.0319281) | 5 | 5 |
| nominal_dd_v1 | QPSK | AFDM (0.00155926) | 5 | 5 |

The joint-(0.2,0.2) row is a probe, not an inferred boundary. Integer-grid and nominal full-information results are explanatory controls, not claim gates. QPSK is favorable for 4/5 fresh tail seeds by paired interval at joint 0.5, while one seed is inconclusive; this does not rewrite the original historical-checkpoint QPSK gate failure.

## Matched-quality cost
At N=64, 150/152 declared validation cases attain a target and 150/150 selected cases re-attain it on fresh confirmation. Across the scaled historical-checkpoint study, 46/48 attain on validation and 46/46 confirm. The missing cases are retained as validation failures; there is no test reselection.

All selected implementations are precomputed dense equalizer applications. Their recurring detector ledgers are N² complex MACs: 4,096, 16,384, and 65,536 at N=64, 128, and 256. Thus Learned sometimes reaches a target at smaller K, but K does not reduce this dense application's recurring arithmetic. No common-target detector or runtime saving over every classical comparator is established.

| N | detector MAC/block | MAC/symbol | MAC/16-QAM bit | MAC/QPSK bit |
|---:|---:|---:|---:|---:|
| 64 | 4096 | 64 | 16 | 32 |
| 128 | 16384 | 128 | 32 | 64 |
| 256 | 65536 | 256 | 64 | 128 |

Relative-to-own-full degradation and absolute target attainment are both saved per selected row. They are not interchangeable. Complete A formation, receive transforms, oracle packing, dense equalizer preparation/storage, channel estimation, and RF are outside the detector-MAC number; the report therefore makes no total-receiver claim.

## Dimension scaling under the Pub-1B law

The exact normalized law is `tau_max=4N/64`, `nu_max=.75N/64`, P=3. It changes physical parameters with N and is not a fixed-bandwidth, fixed-delay, fixed-Doppler, or fixed-coherence experiment. N=128/256 Learned results reuse verified historical single checkpoints and are not training replications.

| N | Learned K/full 16-QAM BER | best classical K BER | Learned own-full degradation | tight iterations | sparse/precomputed latency | fixed-64 nonconvergence |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.024197 / 0.022526 | OFDM 0.042595 | +0.001671 | 61.0 | 2277.1 / 3.167 us (713x) | 8/32 ref; 115/384 all |
| 128 | 0.028515 / 0.025928 | OFDM 0.036015 | +0.002586 | 82.5 | 4030.2 / 10.739 us (377x) | 10/16 ref; 124/192 all |
| 256 | 0.032448 / 0.031372 | OFDM 0.034492 | +0.001076 | 108.0 | 11144.6 / 40.635 us (271x) | 7/8 ref; 85/96 all |

No measured sparse/precomputed crossover is observed (`False`). Sparse-CG becomes less relatively slow as N grows, but remains 269–718x slower at rho=0.375. Tight-reference median iterations rise from 61.0 to 108.0; fixed-64 nonconvergence is retained and grows sharply by proportion. Profiling's largest self-time operators are N=64: aten::mul, N=128: aten::index, N=256: aten::index; this is attribution evidence, not a causal hardware generalization.

The fixed-64 sparse ledger is `C(N,K,t)=NK+N+t(2NK+6N)`. At fixed rho it remains O(rho N²). Sparse complex128 values plus int64 indices use about 24NK bytes versus 16N² bytes for dense coefficients; at rho=0.375 this is 56.25% of dense coefficient storage before the dense waveform matrix and any residual statistics. N=512/1024 arithmetic/storage rows are analytical extrapolations only. N=64/128/256 timings are measured, unprofiled final repeats; production scaling runtime was 626.2 s.

## Claim-to-evidence

| claim | evidence | status |
|---|---|---|
| Same-K fractional 16-QAM representation | five fresh seeds; validation-chosen classical receiver/config; paired channel bootstrap | SUPPORTED |
| Tail objective improves its declared tail statistic | paired continuation streams and validation mean/CVaR | SUPPORTED |
| Tail objective improves BER over mean continuation | fresh paired BER; 4/5 significantly worse | NOT SUPPORTED |
| Common-target Learned detector saving | validation-selected, fresh-confirmed N=64/128/256 rows | NOT SUPPORTED |
| Measured sparse runtime saving | isolated N=64/128/256 CPU timing | NOT SUPPORTED |
| N=128/256 representation examples | verified historical single checkpoints | SUPPORTED AS SINGLE-CHECKPOINT EVIDENCE |
| N=128/256 training reproducibility | no new training | NOT TESTED |

## Reuse, completeness, and recommendation
Exact N=128/N=256 checkpoints and historical ledgers avoid 2974.2 s of recorded training. All supports, residual statistics, selection banks, and confirmation banks required by the continuation were regenerated under `pub1f-cont-v1`; smoke states and old test rows were not promoted.

Pub-1F is complete for the amended scope: five-seed training, objective ablation, fresh communication confirmation, paired uncertainty, matched targets, measured N=64/128/256 cost/scaling, analytical N=512/1024 references, figures, provenance, and preservation checks are present. Optimized sparse kernels, direct retained-entry construction, and a structured fast Learned transform remain genuinely new Pub-1G interventions.

Recommendation: retain **Reproducible representation advantage with arithmetic/storage savings only**. The positive result is a scoped fractional-channel, restricted-interaction representation result—not universal waveform or receiver efficiency superiority.
