# Pub-1F continuation protocol amendment

Dated 2026-09-06 19:18 EDT. Machine-readable authority: `configs/stage_pub1f_continuation.json`.

## Original gate and outcome

The original frozen resource gate required statistically supported Learned superiority over every validation-selected classical alternative in favorable joint-(0.5,0.5) 16-QAM and QPSK. It failed only for favorable QPSK versus operational slope-1 residual-aware AFDM: Learned-AFDM BER difference `-2.5749e-4`, paired 95% CI `[-5.4266e-4, 2.4796e-5]`. `outputs/stage_pub1f/data/early_gate.json` remains unchanged and failed. This amendment does not claim that gate passed or that its original objective was different.

## Explicit continuation authorization

The 2026-09-06 user clarification explicitly authorizes five-seed N=64 replication, matched objective ablation, matched-quality receiver-cost evaluation, and N=64/128/256 measured scaling with verified existing checkpoints. The authorization source has SHA-256 `52f883ff...8c61`.

The primary scientific domain is the conjunction of fractional delay-Doppler propagation and restricted effective-channel interactions. The primary modulation is 16-QAM. QPSK remains mandatory and may be inconclusive or unfavorable. Integer-grid and full-information results remain explanatory controls, not pass/fail gates. A same-K representation claim and a matched-quality computational claim are evaluated independently; runtime failure does not erase a supported representation result.

The representation replication rule is frozen before continuation production: at least four of five selected tail checkpoints must have lower joint-(0.5,0.5) 16-QAM BER than the strongest operational classical baseline at K=24/18 dB, and the median seed difference must be negative. Every seed and channel interval remains visible. Tail contribution requires at least four of five tail arms to satisfy the mean gate and improve validation CVaR_0.95 over the matched mean continuation; BER effects remain separately reported and may be mixed. A computational saving requires fresh target confirmation at lower cost than each classical comparator at the same accounting level. A measured runtime crossover requires sparse receiver-processing latency below an equivalent valid dense/precomputed path.

## Why continuation is useful

The historical waveform retained supported favorable 16-QAM gains against improved slope-1-inclusive baselines, but that does not establish training reproducibility. The objective ablation tests whether tail refinement contributes beyond matched extra mean-MSE steps. A validation-selected cost study remains informative even without universal fixed-K superiority. Existing N=64 timing cannot determine scaling behavior at N=128/256.

These reasons are informed by observed Pub-1E and original Pub-1F results. All continuation-selected communication hypotheses therefore use fresh validation and confirmation banks under namespace `pub1f-cont-v1`; old test rows remain historical and are never pooled into fresh estimates.

## Frozen execution

N=64 training retains five independent OFDM starts, K=24, 2,000 parent mean-MSE steps, and paired fresh 1,000-step mean-only/tail continuations. Both arms clone the same validation-selected parent waveform and fresh Adam state. Tail beta/lambda are 0.95/0.40; the eligibility threshold is the seed's selected-parent validation mean MSE times 1.02. Selected checkpoints and fixed endpoints are both evaluated. No test BER selects a checkpoint or seed.

Fresh N=64 evaluation separates SNR, K, condition, receiver, support policy, training seed, and channel uncertainty. Every new waveform owns its oracle/fixed support and matching residual statistics. The full A generates every observation. The intermediate joint-(0.2,0.2) point remains a probe, not a discovered boundary.

Matched-quality selection uses validation only at the frozen 18 dB targets: 16-QAM BER 0.05/0.03 and QPSK BER 0.005/0.002. Fresh confirmation either attains or fails the target; no test reselection or unsupported interpolation is permitted. Relative-to-own-full degradation is reported separately and never substitutes for a common-target comparison.

N=128 and N=256 use verified historical single checkpoints, not replications and not new training. Their channel law is exactly Pub-1B's normalized scaling: `tau_max=4N/64`, `nu_max=.75N/64`, P=3, with delay-fastest OTFS grids 16x8 and 16x16. This changes physical channel parameters with N and is not a fixed-bandwidth/coherence experiment. Historical BER is attributed to Pub-1B and structural evidence to Pub-1C; fresh stronger-baseline confirmation remains separately labeled.

Fresh scaling configuration validation uses 64/64/32 channels at N=64/128/256; locked confirmation uses 128/128/64 channels and 2,048 bits per channel/modulation. These dimension-aware sizes are frozen before scaling evaluation and are not pooled with the 512-channel N=64 continuation replication.

Complexity reports dense direct, precomputed dense equalizer, explicit sparse matrix-free CG, and dense matrix-free CG where supported. Fixed 64-iteration work is separated from tolerance-based convergence with caps 256/512/1024 at N=64/128/256. Timing uses resident complex128 CPU data, one thread, restored global settings, separate setup/recurring regions, and unprofiled final repeats. B=1/10/100/1000 is hypothetical reuse. Complete-A construction means no channel-formation saving. Channel estimation, RF, coding, MIMO, and energy remain outside scope.

The old runner's production training path keeps its original failed-gate stop. Continuation runs only through a separate explicitly authorized runner and config.
