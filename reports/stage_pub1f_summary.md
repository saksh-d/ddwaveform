# Pub-1F — Training reproducibility and receiver-complexity qualification

Run kind: **production**.

## Decision: historical-checkpoint early gate failed

The predeclared gate is **FAIL**. This is a receiver-qualification result, not training-reproducibility evidence. Production training was therefore not launched.

Fresh banks used 256 validation, 1024 residual-statistics, and 512 confirmatory channels with 2048 bits/channel/modulation. Validation selected configurations; the 512-channel test bank only confirmed them.

### Favorable joint-(0.5,0.5) paired gate

| modulation | comparison | Learned BER | classical BER | difference | paired 95% CI |
|---|---|---:|---:|---:|---:|
| 16QAM | Learned - OTFS | 0.0228615 | 0.0927305 | -0.0698690 | [-0.0719719, -0.0677326] |
| 16QAM | Learned - OFDM | 0.0228615 | 0.0325747 | -0.0097132 | [-0.0104532, -0.0089579] |
| 16QAM | Learned - AFDM | 0.0228615 | 0.0415506 | -0.0186892 | [-0.0197821, -0.0175991] |
| QPSK | Learned - OTFS | 0.0012035 | 0.0063257 | -0.0051222 | [-0.0056505, -0.0046272] |
| QPSK | Learned - OFDM | 0.0012035 | 0.0041933 | -0.0029898 | [-0.0034733, -0.0025177] |
| QPSK | Learned - AFDM | 0.0012035 | 0.0014610 | -0.0002575 | [-0.0005427, 0.0000248] |

All 16-QAM comparisons and QPSK comparisons against OTFS/OFDM support Learned. QPSK versus operational AFDM slope 1 is lower in point estimate but inconclusive because the interval includes zero; this single mandatory failure stops the gate.

### Operational selected configurations and test BER

| condition | modulation | Learned | OTFS | OFDM | AFDM | AFDM validation choice |
|---|---|---:|---:|---:|---:|---|
| joint_0.5 | 16QAM | 0.0228615 | 0.0927305 | 0.0325747 | 0.0415506 | slope 1, residual_aware |
| joint_0.5 | QPSK | 0.0012035 | 0.0063257 | 0.0041933 | 0.0014610 | slope 1, residual_aware |
| joint_0.2_probe | 16QAM | 0.0269718 | 0.0460100 | 0.0305786 | 0.0356817 | slope 1, residual_aware |
| joint_0.2_probe | QPSK | 0.0012751 | 0.0010796 | 0.0027399 | 0.0012140 | slope 1, residual_aware |
| integer | 16QAM | 0.0274591 | 0.0233040 | 0.0248699 | 0.0235882 | slope 9, historical |
| integer | QPSK | 0.0012627 | 0.0002947 | 0.0020952 | 0.0002613 | slope 7, historical |
| nominal_dd_v1 | 16QAM | 0.0232477 | 0.0541058 | 0.0303688 | 0.0368681 | slope 1, residual_aware |
| nominal_dd_v1 | QPSK | 0.0009880 | 0.0019693 | 0.0028591 | 0.0012150 | slope 3, residual_aware |

The intermediate (0.2,0.2) point is a probe, not a crossover boundary. Learned loses significantly to OTFS in probe QPSK, is inconclusive against AFDM there, and loses to OTFS/AFDM on the integer grid. Favorable and unfavorable regimes therefore persist. Fresh nominal residual-aware QPSK shows a small supported Learned-AFDM difference, unlike Pub-1E's inconclusive nominal result; this is a new bank/configuration result, not a reinterpretation of Pub-1E.

Residual-aware results use waveform/condition/K/oracle-support-specific independent omitted means and dense receive covariances. They neglect conditioning on realized retained values/support, are not conditional-optimal, and require dense stored information and direct solves.

## Training reproducibility and objective ablation

Not executed. The frozen early-gate failure prohibited the five-seed 2,000+1,000+1,000-step production workload. Smoke exercised all five seeds for two steps only; it is correctness evidence and has no scientific meaning.

Replication remains scientifically useful if the goal is to estimate optimization reproducibility despite an inconclusive operational gate. Proceeding requires an explicit decision to override the frozen cost-benefit gate; it must not be reframed as evidence that Learned won.

## Complexity: arithmetic, storage, latency, and amortization

For 64 executed CG iterations, the audited detector model is `NK + N + t(2NK + 6N)`. K=8 costs 90,688 complex-equivalent MACs versus 553,024 at K=64, an analytical reduction of 83.6%. Sparse complex128/int64 values+indices occupy 12,288 versus 98,304 bytes, a 87.5% reduction. At K=24 the representation stores 36,864 bytes versus 65,536 dense-coefficient bytes. Residual awareness adds 131,072 bytes for one dense complex mean and covariance before other state.

Measured prepared-detector median latency was 2003.3 us at K=8 and 2572.2 us at K=64, only a 22.1% reduction. Dense direct solve over a matrix containing zeros was about 101.3 us at K=24, while explicit sparse matrix-free CG was 2065.4 us. Thus fewer modeled operations did not produce a faster implementation.

The fixed 64-step sparse workload had 98 nonconverged channel/configuration cases out of 768; maximum 64-step relative solution error was 4.427e-06. Every implementation first passed a separate tight solve gate; maximum tight relative error was 4.855e-12. Residual and BER remain separate criteria.

Receive transform, complete effective-channel formation, oracle/fixed packing, direct equalizer setup, recurring application, and B=1/10/100/1000 amortization are saved separately. Because the implementation forms complete A first, it demonstrates no effective-channel construction saving. Oracle support requires coefficient access; fixed support saves position selection but still updates retained values. Precomputed dense G has dense application/storage even when A is sparse.

The measured study covers receiver processing from an available channel matrix and resident vectors. It excludes channel estimation, RF acquisition, coding, transmitter work, plotting, and BER bookkeeping. No total-transceiver or energy claim is made. The B sweep is hypothetical, not a physical coherence-time claim.

## Matched quality and claim boundary

The predeclared BER targets remain frozen, but minimum validated cost `C*_w(b*)` is **not estimable**: the early gate stopped the validation-selected K/iteration production grid. No test-set lower envelope, interpolation, or post-failure reselection is substituted. Consequently there is no supported same-BER receiver-processing saving, no BER-versus-latency frontier, and no F4c/F4d claim.

At fixed N, sparse products scale O(NK); at fixed rho they remain O(rho N^2). The measurements establish only a K-dependent constant reduction at N=64, not a better asymptotic class. No N=128/256 training was launched.

## Explicit answers

1. Five-seed reproducibility: **not tested**, stopped by the gate.
2. Tail versus matched mean continuation: **not tested**.
3. Slope-1 AFDM with improved detection: 16-QAM advantage survives; favorable QPSK is **inconclusive**, so the joint gate fails.
4. OFDM: Learned is significantly better in the favorable joint condition for both modulations; OFDM remains reported everywhere.
5. Nominal residual-aware QPSK: on the fresh Pub-1F bank it is no longer inconclusive versus selected AFDM; the paired interval is below zero. Pub-1E's result remains historically inconclusive.
6. Regimes: favorable 16-QAM and unfavorable integer behavior persist; probe QPSK is mixed/unfavorable.
7. Arithmetic/storage: K=8 versus 64 reduces the declared 64-step MAC ledger by the percentage above and sparse row storage by 87.5%.
8. Runtime/throughput: sparse latency falls modestly with K but remains much slower than dense direct/precomputed paths on this CPU implementation; no throughput advantage is shown.
9. Iterations/overhead/preparation: nonconvergence and Python indexing dominate; full-A construction and transforms remain; amortization helps reusable dense setup most.
10. Minimum cost per target: **not established** because the mandatory operating grid did not run.
11. Full receiver-processing savings: **not demonstrated**.
12. Conference figures: current defensible figures are F3 receiver gate, F4a operation/storage, and F4b latency/amortization. F1/F2 and F4c/F4d require an authorized replication/operating-grid continuation.

## Evidence category and recommendation

**Narrow receiver/regime-dependent result.** The historical waveform retains a strong 16-QAM advantage and several QPSK advantages, but the predeclared slope-1-inclusive favorable-QPSK gate is inconclusive. Arithmetic representation benefits are real; runtime and matched-quality receiver-processing savings are not demonstrated. For a six-page paper now, center the corrected Pub-1E regime map plus this stronger-baseline limitation and implementation reality. Run five-seed replication only if reproducibility itself is worth the substantial compute despite the failed operational gate.

## Artifacts

- `data/baseline_validation.csv`: all validation candidates.
- `data/baseline_selection.csv`: deployable validation choices.
- `data/baseline_confirmatory_per_channel.csv` and `data/baseline_paired_intervals.csv`: raw test evidence.
- `data/complexity_ledger.csv`, `timing/receiver_benchmark.csv`, `timing/receiver_setup.csv`, and `data/amortization.csv`: operation, storage, convergence, latency, and setup evidence.
- `figures/F3_historical_checkpoint_receiver_gate.*`, `F4a_operations_storage.*`, and `F4b_latency_amortization.*`: current primary figures.
