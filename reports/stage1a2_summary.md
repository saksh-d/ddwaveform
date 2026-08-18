# Stage 1A.2 — Tail-Robust Universal Unitary Waveform Training

## A. Executive verdict

Outcome `A`. Stream-tail training materially reduced the exact worst-stream MSE tail. QPSK repair at 12/15 dB: `True`. Mean-MSE 2% gate: `True`. 16-QAM remains meaningfully better than IDENTITY at every SNR: `True`, although it is significantly worse than frozen Stage 1A at 12/15 dB and its SER is significantly worse at all four SNRs. The matched-start HAAR-2 tail objective beat matched mean-only continuation in the target QPSK failure mode: `True`. Multiple-start tail improvement: `True`. The universal-transform hypothesis is strengthened, with a higher-order-modulation tradeoff.

## B. Tail calibration

All nine rows began from the exact frozen HAAR-2 tensor and shared online channels, continuous SNR draws, 400 steps, constant `3e-3` learning rate, and channel/SNR seeds `11034`/`11035`. The eight positive-lambda rows are the candidate grid; lambda zero is the required control. Selection used no BER.

| beta | lambda | step | mean MSE | relative | CVaR .90 | CVaR .95 | feasible | selected |
|---|---|---|---|---|---|---|---|---|
| 0.95 | 0.00 | 50 | 0.07374292 | +1.892% | 0.18172110 | 0.20583797 | True | False |
| 0.90 | 0.05 | 50 | 0.07377153 | +1.931% | 0.18083142 | 0.20460802 | True | False |
| 0.90 | 0.10 | 50 | 0.07375828 | +1.913% | 0.17997586 | 0.20341337 | True | False |
| 0.90 | 0.20 | 50 | 0.07374524 | +1.895% | 0.17858525 | 0.20143308 | True | False |
| 0.90 | 0.40 | 50 | 0.07378467 | +1.949% | 0.17651605 | 0.19854168 | True | False |
| 0.95 | 0.05 | 50 | 0.07376669 | +1.924% | 0.18050752 | 0.20410150 | True | False |
| 0.95 | 0.10 | 50 | 0.07373878 | +1.886% | 0.17946733 | 0.20260333 | True | False |
| 0.95 | 0.20 | 50 | 0.07369748 | +1.829% | 0.17765839 | 0.19998483 | True | False |
| 0.95 | 0.40 | 50 | 0.07380218 | +1.973% | 0.17525862 | 0.19672490 | True | True |

Selected `beta=0.95`, `lambda=0.40`.

## C. Production training

Three tail refinements started from HAAR-2, HAAR-1, and OFDM Stage-1A checkpoints. The matched continuation started from HAAR-2. All used 1,000 steps, batch 16, exact hard K=24, checkpoint cadence 25, and learning rates `3e-3` (1–500), `1e-3` (501–800), `3e-4` (801–1000). Channel/SNR seeds were `11036`/`11037`.

| run | step | mean MSE | CVaR .95 | tail improved | unitarity |
|---|---|---|---|---|---|
| TAIL_FROM_HAAR_2 | 50 | 0.07375148 | 0.19698087 | True | 2.50e-14 |
| TAIL_FROM_HAAR_1 | 25 | 0.07378233 | 0.21808523 | True | 2.44e-14 |
| TAIL_FROM_OFDM | 50 | 0.07367612 | 0.16287674 | True | 2.49e-14 |
| MEAN_CONTINUATION | 50 | 0.07377179 | 0.20586607 | True | 2.46e-14 |

Final validation-selected production waveform: `TAIL_FROM_OFDM`, step `50`. Selection used no BER.

## D. Exact communication metrics

`mean_exact_mse` and `mean_per_stream_mse` agree exactly in the evaluation CSV; the focused double-precision test independently verifies equality to the existing communication loss. Pooled held-out mean MSE is `0.071770` for Stage 1A.2 versus `0.070505` for frozen Stage 1A, a `1.79%` increase inside the 2% criterion. Per-SNR increases are `0.74%`, `1.50%`, `2.95%`, and `4.52%`; the preregistered feasibility rule is pooled over the frozen validation SNR/channel bank.

| SNR | waveform | mean MSE | CVaR .90 | CVaR .95 | p99 | mean max-stream |
|---|---|---|---|---|---|---|
| 12 | IDENTITY | 0.13419244 | 0.19631565 | 0.20424645 | 0.21125462 | 0.21993988 |
| 12 | OFDM | 0.13648226 | 0.28576023 | 0.30411468 | 0.31640549 | 0.32974897 |
| 12 | OTFS | 0.15442404 | 0.17542495 | 0.17801080 | 0.17999075 | 0.18198736 |
| 12 | AFDM | 0.14330709 | 0.20642612 | 0.21180613 | 0.21493906 | 0.21734278 |
| 12 | Stage-1A Learned | 0.12257752 | 0.27238005 | 0.30262906 | 0.32917901 | 0.36368413 |
| 12 | Mean-MSE continuation | 0.12353844 | 0.26669824 | 0.29443637 | 0.31821385 | 0.34797592 |
| 12 | Matched-start tail (HAAR-2) | 0.12352906 | 0.25744730 | 0.28258964 | 0.30403675 | 0.33134357 |
| 12 | Stage-1A.2 Learned | 0.12348702 | 0.22007748 | 0.23649287 | 0.25182592 | 0.27279674 |
| 15 | IDENTITY | 0.09262070 | 0.14924059 | 0.15835511 | 0.16634762 | 0.17619201 |
| 15 | OFDM | 0.09784185 | 0.23584989 | 0.25787981 | 0.27353808 | 0.29032155 |
| 15 | OTFS | 0.11793570 | 0.14253353 | 0.14579705 | 0.14831568 | 0.15075206 |
| 15 | AFDM | 0.10631731 | 0.17172455 | 0.17910846 | 0.18377834 | 0.18750901 |
| 15 | Stage-1A Learned | 0.07643394 | 0.19293928 | 0.21994657 | 0.24412145 | 0.27733575 |
| 15 | Mean-MSE continuation | 0.07764959 | 0.18976520 | 0.21470752 | 0.23669312 | 0.26486574 |
| 15 | Matched-start tail (HAAR-2) | 0.07763704 | 0.18268461 | 0.20510837 | 0.22481227 | 0.25046668 |
| 15 | Stage-1A.2 Learned | 0.07758136 | 0.15487114 | 0.16994179 | 0.18402298 | 0.20295369 |
| 18 | IDENTITY | 0.06924733 | 0.12595774 | 0.13678514 | 0.14607052 | 0.15763137 |
| 18 | OFDM | 0.07937283 | 0.22750388 | 0.25768177 | 0.28074516 | 0.30418160 |
| 18 | OTFS | 0.09972036 | 0.12963251 | 0.13380605 | 0.13697629 | 0.14012073 |
| 18 | AFDM | 0.08863862 | 0.16622835 | 0.17694916 | 0.18398025 | 0.18958744 |
| 18 | Stage-1A Learned | 0.04771321 | 0.13728081 | 0.16104610 | 0.18286647 | 0.21343444 |
| 18 | Mean-MSE continuation | 0.04919295 | 0.13643224 | 0.15859547 | 0.17838109 | 0.20457534 |
| 18 | Matched-start tail (HAAR-2) | 0.04917443 | 0.13124107 | 0.15105753 | 0.16877918 | 0.19268922 |
| 18 | Stage-1A.2 Learned | 0.04911952 | 0.11118810 | 0.12497156 | 0.13794368 | 0.15472848 |
| 20 | IDENTITY | 0.06080628 | 0.12127282 | 0.13369416 | 0.14408676 | 0.15737205 |
| 20 | OFDM | 0.07560328 | 0.24542672 | 0.28477986 | 0.31602290 | 0.34637210 |
| 20 | OTFS | 0.09493067 | 0.12978039 | 0.13476452 | 0.13858781 | 0.14233368 |
| 20 | AFDM | 0.08464705 | 0.17643615 | 0.19034426 | 0.19962056 | 0.20679620 |
| 20 | Stage-1A Learned | 0.03529699 | 0.11108461 | 0.13288871 | 0.15318901 | 0.18211665 |
| 20 | Mean-MSE continuation | 0.03695909 | 0.11165505 | 0.13213607 | 0.15059423 | 0.17551907 |
| 20 | Matched-start tail (HAAR-2) | 0.03693470 | 0.10740880 | 0.12573814 | 0.14221019 | 0.16509013 |
| 20 | Stage-1A.2 Learned | 0.03689084 | 0.09137633 | 0.10438064 | 0.11668146 | 0.13222364 |

Representation diagnostics:

| waveform | retained energy | median limiting distortion | median condition |
|---|---|---|---|
| IDENTITY | 0.99243868 | 0.04934315 | 14.71849 |
| OFDM | 0.99190175 | 0.11238996 | 16.42991 |
| OTFS | 0.97793173 | 0.11084551 | 16.45287 |
| AFDM | 0.98652886 | 0.12978791 | 15.97643 |
| Stage-1A Learned | 0.99935325 | 0.00262300 | 12.70800 |
| Mean-MSE continuation | 0.99852776 | 0.00568790 | 12.85371 |
| Matched-start tail (HAAR-2) | 0.99852948 | 0.00564023 | 12.81746 |
| Stage-1A.2 Learned | 0.99856221 | 0.00517614 | 12.90872 |

## E. QPSK end-to-end BER/SER

| SNR | waveform | BER | SER | bit errors/bits | symbol errors/symbols |
|---|---|---|---|---|---|
| 12 | IDENTITY | 9.16576385e-03 | 1.80072784e-02 | 9611/1048576 | 9441/524288 |
| 12 | OFDM | 1.44405365e-02 | 2.77843475e-02 | 15142/1048576 | 14567/524288 |
| 12 | OTFS | 1.17540359e-02 | 2.32639313e-02 | 12325/1048576 | 12197/524288 |
| 12 | AFDM | 1.04713440e-02 | 2.06184387e-02 | 10980/1048576 | 10810/524288 |
| 12 | Stage-1A Learned | 1.14603043e-02 | 2.21405029e-02 | 12017/1048576 | 11608/524288 |
| 12 | Mean-MSE continuation | 1.11560822e-02 | 2.15759277e-02 | 11698/1048576 | 11312/524288 |
| 12 | Matched-start tail (HAAR-2) | 1.08213425e-02 | 2.09903717e-02 | 11347/1048576 | 11005/524288 |
| 12 | Stage-1A.2 Learned | 9.21154022e-03 | 1.80797577e-02 | 9659/1048576 | 9479/524288 |
| 15 | IDENTITY | 3.49521637e-03 | 6.87408447e-03 | 3665/1048576 | 3604/524288 |
| 15 | OFDM | 7.44915009e-03 | 1.43737793e-02 | 7811/1048576 | 7536/524288 |
| 15 | OTFS | 5.55038452e-03 | 1.10130310e-02 | 5820/1048576 | 5774/524288 |
| 15 | AFDM | 4.59671021e-03 | 9.08279419e-03 | 4820/1048576 | 4762/524288 |
| 15 | Stage-1A Learned | 4.34875488e-03 | 8.44383240e-03 | 4560/1048576 | 4427/524288 |
| 15 | Mean-MSE continuation | 4.21428680e-03 | 8.20541382e-03 | 4419/1048576 | 4302/524288 |
| 15 | Matched-start tail (HAAR-2) | 3.93772125e-03 | 7.66754150e-03 | 4129/1048576 | 4020/524288 |
| 15 | Stage-1A.2 Learned | 3.15093994e-03 | 6.19888306e-03 | 3304/1048576 | 3250/524288 |
| 18 | IDENTITY | 1.76620483e-03 | 3.46374512e-03 | 1852/1048576 | 1816/524288 |
| 18 | OFDM | 5.24234772e-03 | 1.01108551e-02 | 5497/1048576 | 5301/524288 |
| 18 | OTFS | 4.09984589e-03 | 8.11576843e-03 | 4299/1048576 | 4255/524288 |
| 18 | AFDM | 2.89726257e-03 | 5.71823120e-03 | 3038/1048576 | 2998/524288 |
| 18 | Stage-1A Learned | 1.94549561e-03 | 3.79371643e-03 | 2040/1048576 | 1989/524288 |
| 18 | Mean-MSE continuation | 1.93500519e-03 | 3.74794006e-03 | 2029/1048576 | 1965/524288 |
| 18 | Matched-start tail (HAAR-2) | 1.74617767e-03 | 3.41987610e-03 | 1831/1048576 | 1793/524288 |
| 18 | Stage-1A.2 Learned | 1.09481812e-03 | 2.15339661e-03 | 1148/1048576 | 1129/524288 |
| 20 | IDENTITY | 1.41239166e-03 | 2.79235840e-03 | 1481/1048576 | 1464/524288 |
| 20 | OFDM | 5.23662567e-03 | 1.00574493e-02 | 5491/1048576 | 5273/524288 |
| 20 | OTFS | 3.81374359e-03 | 7.54165649e-03 | 3999/1048576 | 3954/524288 |
| 20 | AFDM | 3.01265717e-03 | 5.92231750e-03 | 3159/1048576 | 3105/524288 |
| 20 | Stage-1A Learned | 1.20639801e-03 | 2.33650208e-03 | 1265/1048576 | 1225/524288 |
| 20 | Mean-MSE continuation | 1.14631653e-03 | 2.22969055e-03 | 1202/1048576 | 1169/524288 |
| 20 | Matched-start tail (HAAR-2) | 1.06716156e-03 | 2.08473206e-03 | 1119/1048576 | 1093/524288 |
| 20 | Stage-1A.2 Learned | 6.39915466e-04 | 1.26457214e-03 | 671/1048576 | 663/524288 |

IDENTITY is included numerically and excluded from all figures.

## F. 16-QAM end-to-end BER/SER

| SNR | waveform | BER | SER | bit errors/bits | symbol errors/symbols |
|---|---|---|---|---|---|
| 12 | IDENTITY | 9.68475342e-02 | 3.41514587e-01 | 101552/1048576 | 89526/262144 |
| 12 | OFDM | 9.25178528e-02 | 3.16497803e-01 | 97012/1048576 | 82968/262144 |
| 12 | OTFS | 1.10800743e-01 | 3.87935638e-01 | 116183/1048576 | 101695/262144 |
| 12 | AFDM | 1.03208542e-01 | 3.61251831e-01 | 108222/1048576 | 94700/262144 |
| 12 | Stage-1A Learned | 8.38546753e-02 | 2.89627075e-01 | 87928/1048576 | 75924/262144 |
| 12 | Mean-MSE continuation | 8.48474503e-02 | 2.93682098e-01 | 88969/1048576 | 76987/262144 |
| 12 | Matched-start tail (HAAR-2) | 8.53433609e-02 | 2.96516418e-01 | 89489/1048576 | 77730/262144 |
| 12 | Stage-1A.2 Learned | 8.69503021e-02 | 3.04580688e-01 | 91174/1048576 | 79844/262144 |
| 15 | IDENTITY | 5.95731735e-02 | 2.16835022e-01 | 62467/1048576 | 56842/262144 |
| 15 | OFDM | 5.97972870e-02 | 2.09335327e-01 | 62702/1048576 | 54876/262144 |
| 15 | OTFS | 7.85732269e-02 | 2.82188416e-01 | 82390/1048576 | 73974/262144 |
| 15 | AFDM | 6.93788528e-02 | 2.50717163e-01 | 72749/1048576 | 65724/262144 |
| 15 | Stage-1A Learned | 4.50506210e-02 | 1.59858704e-01 | 47239/1048576 | 41906/262144 |
| 15 | Mean-MSE continuation | 4.61683273e-02 | 1.64310455e-01 | 48411/1048576 | 43073/262144 |
| 15 | Matched-start tail (HAAR-2) | 4.64935303e-02 | 1.66217804e-01 | 48752/1048576 | 43573/262144 |
| 15 | Stage-1A.2 Learned | 4.77323532e-02 | 1.72973633e-01 | 50051/1048576 | 45344/262144 |
| 18 | IDENTITY | 3.76462936e-02 | 1.39427185e-01 | 39475/1048576 | 36550/262144 |
| 18 | OFDM | 4.25939560e-02 | 1.49723053e-01 | 44663/1048576 | 39249/262144 |
| 18 | OTFS | 5.94224930e-02 | 2.15721130e-01 | 62309/1048576 | 56550/262144 |
| 18 | AFDM | 5.15470505e-02 | 1.88362122e-01 | 54051/1048576 | 49378/262144 |
| 18 | Stage-1A Learned | 2.32019424e-02 | 8.35685730e-02 | 24329/1048576 | 21907/262144 |
| 18 | Mean-MSE continuation | 2.40726471e-02 | 8.68072510e-02 | 25242/1048576 | 22756/262144 |
| 18 | Matched-start tail (HAAR-2) | 2.42595673e-02 | 8.78105164e-02 | 25438/1048576 | 23019/262144 |
| 18 | Stage-1A.2 Learned | 2.37102509e-02 | 8.73603821e-02 | 24862/1048576 | 22901/262144 |
| 20 | IDENTITY | 2.97555923e-02 | 1.09741211e-01 | 31201/1048576 | 28768/262144 |
| 20 | OFDM | 3.72743607e-02 | 1.30241394e-01 | 39085/1048576 | 34142/262144 |
| 20 | OTFS | 5.35631180e-02 | 1.93313599e-01 | 56165/1048576 | 50676/262144 |
| 20 | AFDM | 4.51564789e-02 | 1.64726257e-01 | 47350/1048576 | 43182/262144 |
| 20 | Stage-1A Learned | 1.54161453e-02 | 5.54733276e-02 | 16165/1048576 | 14542/262144 |
| 20 | Mean-MSE continuation | 1.62258148e-02 | 5.87081909e-02 | 17014/1048576 | 15390/262144 |
| 20 | Matched-start tail (HAAR-2) | 1.61647797e-02 | 5.86509705e-02 | 16950/1048576 | 15375/262144 |
| 20 | Stage-1A.2 Learned | 1.57928467e-02 | 5.85823059e-02 | 16560/1048576 | 15357/262144 |

## G. Causal ablation

The required frozen Stage 1A, matched mean-MSE continuation, and selected tail-robust refinement appear in every numerical table. `Matched-start tail (HAAR-2)` is an additional controlled ablation: it shares initialization, online samples, schedule, steps, and validation rule with mean-only continuation. Its target-failure superiority is `True`. The independently selected final Stage-1A.2 waveform starts from the Stage-1A OFDM checkpoint, so its larger absolute gain is not attributed to the objective alone.

## H. Statistical inference

Channel-level paired bootstrap uses 10,000 resamples and seed `11033`.

| domain | modulation | SNR | metric | comparison | delta | 95% CI |
|---|---|---|---|---|---|---|
| analytic |  | 12 | mean_exact_mse | Stage-1A.2 Learned minus Stage-1A Learned | +9.09497573e-04 | [+8.91384397e-04, +9.27805429e-04] |
| analytic |  | 12 | cvar_stream_095 | Stage-1A.2 Learned minus Stage-1A Learned | -6.61361904e-02 | [-6.95312988e-02, -6.28035429e-02] |
| analytic |  | 12 | mean_exact_mse | Stage-1A.2 Learned minus Mean-MSE continuation | -5.14178442e-05 | [-6.86308415e-05, -3.37963110e-05] |
| analytic |  | 12 | cvar_stream_095 | Stage-1A.2 Learned minus Mean-MSE continuation | -5.79434949e-02 | [-6.10755194e-02, -5.47889687e-02] |
| analytic |  | 12 | mean_exact_mse | Stage-1A.2 Learned minus IDENTITY | -1.07054250e-02 | [-1.13021103e-02, -1.01199283e-02] |
| analytic |  | 12 | cvar_stream_095 | Stage-1A.2 Learned minus IDENTITY | +3.22464206e-02 | [+2.71164006e-02, +3.75373459e-02] |
| analytic |  | 15 | mean_exact_mse | Stage-1A.2 Learned minus Stage-1A Learned | +1.14742063e-03 | [+1.12199829e-03, +1.17352920e-03] |
| analytic |  | 15 | cvar_stream_095 | Stage-1A.2 Learned minus Stage-1A Learned | -5.00047823e-02 | [-5.30466741e-02, -4.70653592e-02] |
| analytic |  | 15 | mean_exact_mse | Stage-1A.2 Learned minus Mean-MSE continuation | -6.82262658e-05 | [-9.20588272e-05, -4.34306288e-05] |
| analytic |  | 15 | cvar_stream_095 | Stage-1A.2 Learned minus Mean-MSE continuation | -4.47657306e-02 | [-4.75466418e-02, -4.20003956e-02] |
| analytic |  | 15 | mean_exact_mse | Stage-1A.2 Learned minus IDENTITY | -1.50393387e-02 | [-1.59318106e-02, -1.41621227e-02] |
| analytic |  | 15 | cvar_stream_095 | Stage-1A.2 Learned minus IDENTITY | +1.15866771e-02 | [+6.92949137e-03, +1.62158579e-02] |
| analytic |  | 18 | mean_exact_mse | Stage-1A.2 Learned minus Stage-1A Learned | +1.40630432e-03 | [+1.36962488e-03, +1.44473289e-03] |
| analytic |  | 18 | cvar_stream_095 | Stage-1A.2 Learned minus Stage-1A Learned | -3.60745426e-02 | [-3.87387961e-02, -3.34254698e-02] |
| analytic |  | 18 | mean_exact_mse | Stage-1A.2 Learned minus Mean-MSE continuation | -7.34321452e-05 | [-1.06965393e-04, -4.08634555e-05] |
| analytic |  | 18 | cvar_stream_095 | Stage-1A.2 Learned minus Mean-MSE continuation | -3.36239045e-02 | [-3.61707442e-02, -3.10999598e-02] |
| analytic |  | 18 | mean_exact_mse | Stage-1A.2 Learned minus IDENTITY | -2.01278153e-02 | [-2.14599199e-02, -1.88363948e-02] |
| analytic |  | 18 | cvar_stream_095 | Stage-1A.2 Learned minus IDENTITY | -1.18135756e-02 | [-1.70561342e-02, -6.71340164e-03] |
| analytic |  | 20 | mean_exact_mse | Stage-1A.2 Learned minus Stage-1A Learned | +1.59384618e-03 | [+1.54537482e-03, +1.64292411e-03] |
| analytic |  | 20 | cvar_stream_095 | Stage-1A.2 Learned minus Stage-1A Learned | -2.85080748e-02 | [-3.10096319e-02, -2.60791807e-02] |
| analytic |  | 20 | mean_exact_mse | Stage-1A.2 Learned minus Mean-MSE continuation | -6.82451438e-05 | [-1.08995265e-04, -2.72608897e-05] |
| analytic |  | 20 | cvar_stream_095 | Stage-1A.2 Learned minus Mean-MSE continuation | -2.77554271e-02 | [-3.01175027e-02, -2.54519779e-02] |
| analytic |  | 20 | mean_exact_mse | Stage-1A.2 Learned minus IDENTITY | -2.39154437e-02 | [-2.56041576e-02, -2.23035534e-02] |
| analytic |  | 20 | cvar_stream_095 | Stage-1A.2 Learned minus IDENTITY | -2.93135188e-02 | [-3.54363295e-02, -2.33068438e-02] |
| modulation | QPSK | 12 | ber | Stage-1A.2 Learned minus Stage-1A Learned | -2.24876404e-03 | [-2.56156921e-03, -1.93786621e-03] |
| modulation | QPSK | 12 | ser | Stage-1A.2 Learned minus Stage-1A Learned | -4.06074524e-03 | [-4.65583801e-03, -3.48467827e-03] |
| modulation | QPSK | 12 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | -1.94454193e-03 | [-2.25925446e-03, -1.63936615e-03] |
| modulation | QPSK | 12 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | -3.49617004e-03 | [-4.08177376e-03, -2.89535522e-03] |
| modulation | QPSK | 12 | ber | Stage-1A.2 Learned minus IDENTITY | +4.57763672e-05 | [-3.73840332e-04, +4.78744507e-04] |
| modulation | QPSK | 12 | ser | Stage-1A.2 Learned minus IDENTITY | +7.24792480e-05 | [-7.55310059e-04, +8.77380371e-04] |
| modulation | QPSK | 12 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -3.34739685e-04 | [-4.56809998e-04, -2.10738182e-04] |
| modulation | QPSK | 12 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -5.85556030e-04 | [-8.35418701e-04, -3.45230103e-04] |
| modulation | QPSK | 15 | ber | Stage-1A.2 Learned minus Stage-1A Learned | -1.19781494e-03 | [-1.40190125e-03, -9.98497009e-04] |
| modulation | QPSK | 15 | ser | Stage-1A.2 Learned minus Stage-1A Learned | -2.24494934e-03 | [-2.63023376e-03, -1.85961723e-03] |
| modulation | QPSK | 15 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | -1.06334686e-03 | [-1.26457214e-03, -8.63075256e-04] |
| modulation | QPSK | 15 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | -2.00653076e-03 | [-2.40325928e-03, -1.61552429e-03] |
| modulation | QPSK | 15 | ber | Stage-1A.2 Learned minus IDENTITY | -3.44276428e-04 | [-6.57081604e-04, -3.62396240e-05] |
| modulation | QPSK | 15 | ser | Stage-1A.2 Learned minus IDENTITY | -6.75201416e-04 | [-1.27220154e-03, -7.81536102e-05] |
| modulation | QPSK | 15 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -2.76565552e-04 | [-3.65257263e-04, -1.87873840e-04] |
| modulation | QPSK | 15 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -5.37872314e-04 | [-7.15255737e-04, -3.64303589e-04] |
| modulation | QPSK | 18 | ber | Stage-1A.2 Learned minus Stage-1A Learned | -8.50677490e-04 | [-9.92774963e-04, -7.11441040e-04] |
| modulation | QPSK | 18 | ser | Stage-1A.2 Learned minus Stage-1A Learned | -1.64031982e-03 | [-1.91116333e-03, -1.37901306e-03] |
| modulation | QPSK | 18 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | -8.40187073e-04 | [-9.83238220e-04, -6.98089600e-04] |
| modulation | QPSK | 18 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | -1.59454346e-03 | [-1.87110901e-03, -1.31988525e-03] |
| modulation | QPSK | 18 | ber | Stage-1A.2 Learned minus IDENTITY | -6.71386719e-04 | [-8.85009766e-04, -4.63485718e-04] |
| modulation | QPSK | 18 | ser | Stage-1A.2 Learned minus IDENTITY | -1.31034851e-03 | [-1.71279907e-03, -9.11712646e-04] |
| modulation | QPSK | 18 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -1.88827515e-04 | [-2.53677368e-04, -1.24931335e-04] |
| modulation | QPSK | 18 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -3.28063965e-04 | [-4.53948975e-04, -2.02178955e-04] |
| modulation | QPSK | 20 | ber | Stage-1A.2 Learned minus Stage-1A Learned | -5.66482544e-04 | [-6.75201416e-04, -4.62532043e-04] |
| modulation | QPSK | 20 | ser | Stage-1A.2 Learned minus Stage-1A Learned | -1.07192993e-03 | [-1.27792358e-03, -8.65936279e-04] |
| modulation | QPSK | 20 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | -5.06401062e-04 | [-6.16073608e-04, -4.03404236e-04] |
| modulation | QPSK | 20 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | -9.65118408e-04 | [-1.17301941e-03, -7.57217407e-04] |
| modulation | QPSK | 20 | ber | Stage-1A.2 Learned minus IDENTITY | -7.72476196e-04 | [-9.67025757e-04, -5.89370728e-04] |
| modulation | QPSK | 20 | ser | Stage-1A.2 Learned minus IDENTITY | -1.52778625e-03 | [-1.89781189e-03, -1.15776062e-03] |
| modulation | QPSK | 20 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -7.91549683e-05 | [-1.27792358e-04, -3.14712524e-05] |
| modulation | QPSK | 20 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -1.44958496e-04 | [-2.40325928e-04, -4.76837158e-05] |
| modulation | 16QAM | 12 | ber | Stage-1A.2 Learned minus Stage-1A Learned | +3.09562683e-03 | [+2.38225460e-03, +3.83570194e-03] |
| modulation | 16QAM | 12 | ser | Stage-1A.2 Learned minus Stage-1A Learned | +1.49536133e-02 | [+1.25807762e-02, +1.73340797e-02] |
| modulation | 16QAM | 12 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | +2.10285187e-03 | [+1.39999390e-03, +2.81429291e-03] |
| modulation | 16QAM | 12 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | +1.08985901e-02 | [+8.50286484e-03, +1.33094788e-02] |
| modulation | 16QAM | 12 | ber | Stage-1A.2 Learned minus IDENTITY | -9.89723206e-03 | [-1.08842850e-02, -8.91489983e-03] |
| modulation | 16QAM | 12 | ser | Stage-1A.2 Learned minus IDENTITY | -3.69338989e-02 | [-4.04320717e-02, -3.34472656e-02] |
| modulation | 16QAM | 12 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | +4.95910645e-04 | [+2.03132629e-04, +7.82966614e-04] |
| modulation | 16QAM | 12 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | +2.83432007e-03 | [+1.82342529e-03, +3.82232666e-03] |
| modulation | 16QAM | 15 | ber | Stage-1A.2 Learned minus Stage-1A Learned | +2.68173218e-03 | [+2.01890469e-03, +3.34074497e-03] |
| modulation | 16QAM | 15 | ser | Stage-1A.2 Learned minus Stage-1A Learned | +1.31149292e-02 | [+1.09214783e-02, +1.52893066e-02] |
| modulation | 16QAM | 15 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | +1.56402588e-03 | [+9.25064087e-04, +2.21159458e-03] |
| modulation | 16QAM | 15 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | +8.66317749e-03 | [+6.50024414e-03, +1.08489990e-02] |
| modulation | 16QAM | 15 | ber | Stage-1A.2 Learned minus IDENTITY | -1.18408203e-02 | [-1.29528761e-02, -1.07250214e-02] |
| modulation | 16QAM | 15 | ser | Stage-1A.2 Learned minus IDENTITY | -4.38613892e-02 | [-4.77944374e-02, -3.98863792e-02] |
| modulation | 16QAM | 15 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | +3.25202942e-04 | [+6.57796860e-05, +5.85556030e-04] |
| modulation | 16QAM | 15 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | +1.90734863e-03 | [+9.95635986e-04, +2.84957886e-03] |
| modulation | 16QAM | 18 | ber | Stage-1A.2 Learned minus Stage-1A Learned | +5.08308411e-04 | [-3.53336334e-05, +1.06048584e-03] |
| modulation | 16QAM | 18 | ser | Stage-1A.2 Learned minus Stage-1A Learned | +3.79180908e-03 | [+1.91116333e-03, +5.68017960e-03] |
| modulation | 16QAM | 18 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | -3.62396240e-04 | [-9.10782814e-04, +1.84059143e-04] |
| modulation | 16QAM | 18 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | +5.53131104e-04 | [-1.32369995e-03, +2.46429443e-03] |
| modulation | 16QAM | 18 | ber | Stage-1A.2 Learned minus IDENTITY | -1.39360428e-02 | [-1.51768208e-02, -1.27228975e-02] |
| modulation | 16QAM | 18 | ser | Stage-1A.2 Learned minus IDENTITY | -5.20668030e-02 | [-5.63507080e-02, -4.78056908e-02] |
| modulation | 16QAM | 18 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | +1.86920166e-04 | [-1.62124634e-05, +3.88145447e-04] |
| modulation | 16QAM | 18 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | +1.00326538e-03 | [+2.63214111e-04, +1.74331665e-03] |
| modulation | 16QAM | 20 | ber | Stage-1A.2 Learned minus Stage-1A Learned | +3.76701355e-04 | [-1.07765198e-04, +8.58330727e-04] |
| modulation | 16QAM | 20 | ser | Stage-1A.2 Learned minus Stage-1A Learned | +3.10897827e-03 | [+1.36184692e-03, +4.82177734e-03] |
| modulation | 16QAM | 20 | ber | Stage-1A.2 Learned minus Mean-MSE continuation | -4.32968140e-04 | [-9.27925110e-04, +7.43865967e-05] |
| modulation | 16QAM | 20 | ser | Stage-1A.2 Learned minus Mean-MSE continuation | -1.25885010e-04 | [-1.85785294e-03, +1.57175064e-03] |
| modulation | 16QAM | 20 | ber | Stage-1A.2 Learned minus IDENTITY | -1.39627457e-02 | [-1.53512955e-02, -1.26151323e-02] |
| modulation | 16QAM | 20 | ser | Stage-1A.2 Learned minus IDENTITY | -5.11589050e-02 | [-5.61295509e-02, -4.63446617e-02] |
| modulation | 16QAM | 20 | ber | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -6.10351562e-05 | [-2.44140625e-04, +1.21140480e-04] |
| modulation | 16QAM | 20 | ser | Matched-start tail (HAAR-2) minus Mean-MSE continuation | -5.72204590e-05 | [-7.24792480e-04, +6.25610352e-04] |

## I. Full-information invariant

Maximum per-channel waveform-versus-independent-trace discrepancy: `1.443e-15`. Maximum cross-waveform aggregate discrepancy: `1.249e-16`. BER equality was not asserted.

## J. Interpretation

Mean-MSE representation geometry preserved: `True`. Worst-stream sacrifice reduced: `True`. Translation to low-SNR QPSK: `True`. Existing high-SNR behavior retained: `True`. The tail term trades some frozen 16-QAM performance for a large QPSK repair, but retains substantial 16-QAM superiority over IDENTITY. A constellation-aware objective is not required to fix this QPSK failure, though later work may seek a better Pareto balance.

## K. Files, tests, and git boundaries

- Added `src/tail_learning.py`, `scripts/stage1a2.py`, `tests/test_stage1a2.py`, and this report; extended `src/config.py` and `HISTORY.md`.
- Generated calibration, production, analytic, modulation, bootstrap, representation, diagnostics, seven figures, and checkpoints under ignored `outputs/stage1a2/`.
- Evaluation bank: 1024 channels, seed `11030`; symbol/noise seeds `11031`/`11032`.
- Phase runtimes: calibration `126.9` s, production `157.4` s, final evaluation/report `79.5` s; total `363.8` s.
- Final deterministic verification: `62 passed in 1.92 s`; generated outputs are ignored and no output is tracked.
- Generated outputs remain ignored; no final project test bank was touched; `researchPlan.MD` was unchanged.

STAGE 1A.2 COMPLETE — OUTCOME A: TAIL-ROBUST UNIVERSAL WAVEFORM CONFIRMED
