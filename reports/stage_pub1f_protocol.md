# Pub-1F frozen protocol

Frozen before Pub-1F production. Machine-readable authority: `configs/stage_pub1f.json`.

## Question and failure rules

Test whether a declared five-seed OFDM-initialized learning procedure reproduces Pub-1E's qualified limited-representation advantage; whether tail-sensitive continuation improves beyond matched additional mean-MSE training; and whether any advantage reduces detector or full declared receiver-processing cost. Learned is not presumed superior. A claim fails when paired confirmatory evidence is absent, seed behavior is inconsistent, a target is unattained, or savings disappear outside detector-only accounting.

The model is `x=Ws`, `y=Hx+n`, `z=W^H y=As+n'`, `A=W^H H W`. The receiver retains K coefficients in every row of A: full representation N^2, restricted representation NK, `rho=K/N`. All N observations remain available and all N symbols are estimated. Complete A always generates observations. This is not sample reduction, ADC reduction, or measured channel-estimation savings.

## Historical boundary and early gate

The frozen historical checkpoint is protected at SHA-256 `31e42c4...b2cc`. Baselines retain OTFS, OFDM, AFDM slopes 1/3/5, and historical guarded selection over 3/5/7/9. Operational validation additionally searches slopes 1/3/5/7/9. Odd slope 1 is mathematically valid; the old exclusion was an engineering guard criterion.

Before costly training, the historical checkpoint is tested at K=24 and 18 dB. The gate passes only when Learned has lower paired confirmatory BER than every validation-selected classical alternative in both QPSK and 16-QAM at joint (0.5,0.5), and both paired channel-bootstrap upper bounds are below zero. Integer, nominal, and full-information wins are not required. Gate failure stops expensive training but not independent correctness/complexity infrastructure.

Operational AFDM/receiver selection uses fresh validation BER separately by condition, modulation, SNR, K, and receiver family, tie-broken by analytic MSE then smaller slope. This assumes the operating labels/channel family are known; it never adapts to a realized test channel. Confirmatory BER never selects a configuration.

## Banks and pairing

Every random stream is derived from SHA-256 of namespace `pub1f-v1`, generator role, and semantic fields such as seed id, condition, SNR, modulation, K, and channel index. List positions are excluded. Training, checkpoint validation, receiver/classical validation, statistical support, residual-statistics, confirmatory, benchmark, symbol, noise, and bootstrap roles are separate. Common random numbers pair compared waveforms, K values, receivers, and continuation arms where the required tensor is shared.

Production sizes are 256 checkpoint-validation channels, 256 receiver/classical-validation channels, 1,024 support channels, 1,024 residual-statistics channels, 512 confirmatory channels, and 2,048 bits per channel/modulation. BER is aggregate errors/bits. Uncertainty is paired channel bootstrap. Five seeds are summarized by median and full range, separately from channel uncertainty.

## Training and ablation

N=64, K=24, five independent seed ids 0-4, OFDM initialization. Each parent uses 2,000 projected-Adam mean-MSE steps, batch 16, nominal DD-v1 channels, SNR uniform on [12,20] dB, and complex128/float64 CPU arithmetic. Parent learning rates are `1e-2/3e-3/1e-3` over steps 1-1000/1001-1600/1601-2000. Lowest validation mean MSE selects the parent.

Both 1,000-step continuations clone the exact selected parent waveform and start fresh Adam state. Their channel/SNR draws are identical and fresh relative to parent training. Learning rates are `3e-3/1e-3/3e-4` over steps 1-500/501-800/801-1000. Mean-only selects lowest validation mean MSE. Tail uses `mean + 0.40(CVaR_0.95-mean)`, where CVaR is across symbol streams within each channel, not channel-outage CVaR. Tail selects lowest validation CVaR_0.95 among checkpoints whose validation mean MSE is no more than 1.02 times that seed's selected parent mean MSE; ties use mean MSE then step. Validation-selected and fixed step-1000 endpoints are both retained. No seed/checkpoint is selected by test BER; no failed or unfavorable seed is replaced.

## Communication and receiver scope

Primary modulation is 16-QAM; QPSK is mandatory. Conditions are joint (0.5,0.5), intermediate joint probe (0.2,0.2), integer (0,0), nominal DD-v1, and nominal full information. The probe is not a discovered boundary. SNR is 12/15/18/20/25 dB; K is 8/16/24/32/48/64. Every selected seed is evaluated without condition-specific winner selection.

Each waveform/K constructs its own realization-wise row-top-K and fixed statistical support. Each residual-aware combination estimates its own omitted mean and receive covariance on an independent bank matching waveform, channel family, K, and support policy. Residual awareness is a moment approximation: it neglects conditioning on realized retained values/support, stores dense complex means/covariances, and has no conditional-optimality claim. Direct complex128 historical, tuned-ridge, residual-aware, and full-information solves are scientific authority.

## Complexity and matched quality

Accounting is separate for detector kernel; receiver processing including transforms and reusable preparation; and offline design/statistics. Channel estimation and RF acquisition are excluded. The historical `NK+N+t(2NK+6N)` complex-equivalent MAC model is audited, with `A_K^H z` recurring per vector and physical executed iterations distinguished from logical convergence. Dense-zero, explicit sparse, matrix-free iterative, and supported direct paths must agree numerically before timing.

Storage reports actual dtype/shape bytes for coefficients, indices, waveform, support, residual means/covariances, solver state, and peak temporaries. Setup includes forming A, support selection/lookup, packing, and receiver state. Amortization uses `C_receive_transform + C_recurring + C_reusable/B` for hypothetical B=1/10/100/1000. Primary validated 18 dB BER targets, chosen from Pub-1E evidence, are 16-QAM 0.05 and 0.03; QPSK 0.005 and 0.002. Validation chooses admissible K/iterations/receiver/classical slope; locked test either confirms or reports failure.

Primary benchmark is complex128 CPU, one thread, resident data, 20 warm-ups and 100 unprofiled repeats. Single-vector latency, distribution tails, batched throughput, and setup are separate. Structured classical transforms are stated analytically when no equivalent native timed implementation exists. Sparse lower operation count may be slower and will be reported as such.

Mandatory phases are prerequisites, protocol, baseline validation/confirmation and early gate, training, evaluation, complexity, figures, report, and manifest. N=128/256 historical benchmarking, hardware profiling, and energy are optional; no larger-N training is authorized.
