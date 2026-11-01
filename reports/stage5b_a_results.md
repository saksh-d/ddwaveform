
## Complete recovery ledger

Errors below are relative Frobenius errors, not squared losses. Each cell: best error / objective-and-gradient evaluations. Every fit retained; numerical failures: 0.

| Arm | Start | Pass / cases | Initial error range | Final error range | Total evaluations | Fitting seconds |
|---|---|---:|---:|---:|---:|---:|
| A | 0.001 | 6/18 | 0.00255–0.00512 | 3.65e-07–0.00148 | 7092 | 21.82 |
| A | 0.03 | 6/18 | 0.0759–0.161 | 1.37e-07–0.00093 | 7272 | 21.71 |
| A | cold | 0/18 | 0.462–0.954 | 0.000141–0.0374 | 7943 | 22.61 |
| B | 0.001 | 18/18 | 0.00255–0.00512 | 1.78e-07–3.09e-05 | 6140 | 17.00 |
| B | 0.03 | 18/18 | 0.0759–0.161 | 1.78e-07–2.66e-05 | 6024 | 16.97 |
| B | cold | 12/18 | 0.462–0.954 | 7.09e-07–0.0219 | 6551 | 17.96 |
| C | 0.001 | 18/18 | 0.00255–0.00512 | 1.68e-10–1.66e-05 | 6131 | 19.66 |
| C | 0.03 | 18/18 | 0.0759–0.161 | 5.19e-11–2.35e-05 | 6157 | 19.25 |
| C | cold | 14/18 | 0.462–0.954 | 1.26e-10–0.0143 | 6164 | 19.52 |

Each row below denotes one common target/start. `s0/s1/s2` are generation indices, **not restarts of one target**. `0.001/0.03` are polar perturbation scales; `cold` uses only the target diagonal for phase initialization. Exact seeds appear in raw results.

| Configuration / start | Initial error | A error / evals | B error / evals | C error / evals |
|---|---:|---:|---:|---:|
| ascending_p1_s0_0.001 | 0.00279719 | 1.00002e-06 / 252 | 4.17225e-07 / 252 | 2.47075e-10 / 23 |
| ascending_p1_s0_0.03 | 0.0796933 | 1.80482e-07 / 252 | 8.54054e-07 / 254 | 3.34509e-10 / 29 |
| ascending_p1_s0_cold | 0.511579 | 0.000173424 / 254 | 7.66306e-07 / 252 | 1.28091e-10 / 26 |
| ascending_p1_s1_0.001 | 0.00280673 | 1.12792e-06 / 252 | 1.88018e-07 / 252 | 4.0911e-10 / 20 |
| ascending_p1_s1_0.03 | 0.0836228 | 1.47024e-07 / 252 | 1.79664e-07 / 252 | 6.73383e-11 / 23 |
| ascending_p1_s1_cold | 0.499773 | 0.000141235 / 501 | 7.22255e-07 / 252 | 1.33393e-10 / 28 |
| ascending_p1_s2_0.001 | 0.00254884 | 3.13232e-06 / 279 | 1.84527e-07 / 252 | 1.67503e-10 / 23 |
| ascending_p1_s2_0.03 | 0.078116 | 1.44782e-07 / 252 | 1.78466e-07 / 252 | 2.9601e-10 / 30 |
| ascending_p1_s2_cold | 0.461803 | 0.000175179 / 501 | 7.09405e-07 / 252 | 2.1191e-10 / 26 |
| ascending_p2_s0_0.001 | 0.00355424 | 0.00054704 / 463 | 6.62817e-06 / 289 | 4.03904e-09 / 500 |
| ascending_p2_s0_0.03 | 0.122151 | 0.000466176 / 501 | 1.4557e-06 / 262 | 6.99851e-07 / 500 |
| ascending_p2_s0_cold | 0.674928 | 0.00762936 / 455 | 2.61732e-06 / 388 | 1.36422e-08 / 500 |
| ascending_p2_s1_0.001 | 0.00352786 | 0.000826736 / 502 | 1.2286e-06 / 258 | 9.50335e-07 / 500 |
| ascending_p2_s1_0.03 | 0.10883 | 0.000501223 / 501 | 4.61001e-07 / 252 | 5.14568e-07 / 500 |
| ascending_p2_s1_cold | 0.70555 | 0.00945958 / 454 | 1.27744e-06 / 375 | 8.15005e-07 / 500 |
| ascending_p2_s2_0.001 | 0.00340403 | 0.000534973 / 461 | 3.95957e-06 / 281 | 6.46989e-09 / 500 |
| ascending_p2_s2_0.03 | 0.108357 | 0.000432865 / 503 | 6.73147e-07 / 263 | 5.33602e-08 / 500 |
| ascending_p2_s2_cold | 0.688959 | 0.00731946 / 453 | 1.62054e-06 / 360 | 1.31822e-08 / 500 |
| ascending_p4_s0_0.001 | 0.00498187 | 0.00145866 / 455 | 2.36641e-05 / 501 | 7.95906e-06 / 500 |
| ascending_p4_s0_0.03 | 0.153289 | 0.000923454 / 458 | 2.66013e-05 / 501 | 2.20499e-05 / 500 |
| ascending_p4_s0_cold | 0.881516 | 0.0310876 / 453 | 0.0152542 / 454 | 0.00328111 / 500 |
| ascending_p4_s1_0.001 | 0.00511965 | 0.00139456 / 457 | 2.40181e-05 / 502 | 1.14256e-05 / 500 |
| ascending_p4_s1_0.03 | 0.150693 | 0.0008685 / 456 | 1.94805e-05 / 470 | 2.22238e-05 / 500 |
| ascending_p4_s1_cold | 0.954051 | 0.0349317 / 455 | 0.0157833 / 454 | 3.62729e-05 / 500 |
| ascending_p4_s2_0.001 | 0.00504084 | 0.00148228 / 454 | 2.82143e-05 / 501 | 7.22829e-06 / 500 |
| ascending_p4_s2_0.03 | 0.161235 | 0.000899058 / 460 | 2.16739e-05 / 475 | 2.35485e-05 / 500 |
| ascending_p4_s2_cold | 0.898673 | 0.0335667 / 456 | 0.0218617 / 455 | 0.0143047 / 500 |
| descending_p1_s0_0.001 | 0.0025915 | 6.84884e-07 / 252 | 1.77563e-07 / 252 | 2.5683e-10 / 23 |
| descending_p1_s0_0.03 | 0.0789397 | 1.44792e-07 / 252 | 1.80803e-07 / 252 | 6.60001e-11 / 28 |
| descending_p1_s0_cold | 0.499045 | 0.000401534 / 501 | 7.72948e-07 / 252 | 3.34011e-10 / 30 |
| descending_p1_s1_0.001 | 0.00279011 | 1.19388e-06 / 252 | 1.88429e-07 / 252 | 2.73368e-10 / 22 |
| descending_p1_s1_0.03 | 0.0759432 | 1.15613e-06 / 252 | 1.83386e-07 / 252 | 6.46813e-11 / 24 |
| descending_p1_s1_cold | 0.532842 | 0.000364193 / 473 | 7.72324e-07 / 252 | 1.58116e-10 / 28 |
| descending_p1_s2_0.001 | 0.00257056 | 3.65071e-07 / 252 | 1.89371e-07 / 252 | 2.14305e-10 / 20 |
| descending_p1_s2_0.03 | 0.0791248 | 1.36785e-07 / 252 | 1.83003e-07 / 252 | 5.18996e-11 / 23 |
| descending_p1_s2_cold | 0.512194 | 0.00031768 / 254 | 7.86723e-07 / 252 | 1.26076e-10 / 26 |
| descending_p2_s0_0.001 | 0.00352146 | 0.000675795 / 459 | 1.96801e-06 / 279 | 5.96891e-07 / 500 |
| descending_p2_s0_0.03 | 0.108759 | 0.00038645 / 501 | 3.2643e-07 / 252 | 1.21849e-07 / 500 |
| descending_p2_s0_cold | 0.696345 | 0.00766767 / 457 | 2.38776e-06 / 502 | 8.75904e-07 / 500 |
| descending_p2_s1_0.001 | 0.00383898 | 0.000436878 / 457 | 9.36734e-07 / 266 | 6.31907e-08 / 500 |
| descending_p2_s1_0.03 | 0.107235 | 0.000344394 / 502 | 6.23817e-07 / 253 | 3.14375e-07 / 500 |
| descending_p2_s1_cold | 0.734203 | 0.00745493 / 456 | 1.79664e-06 / 333 | 2.26572e-07 / 500 |
| descending_p2_s2_0.001 | 0.00353906 | 0.000503005 / 473 | 7.16739e-07 / 274 | 2.20699e-07 / 500 |
| descending_p2_s2_0.03 | 0.109129 | 0.000485148 / 501 | 1.58014e-06 / 278 | 2.44153e-07 / 500 |
| descending_p2_s2_cold | 0.677553 | 0.00839276 / 457 | 1.68832e-06 / 350 | 1.00592e-06 / 500 |
| descending_p4_s0_0.001 | 0.00491145 | 0.00142214 / 458 | 1.46065e-05 / 474 | 1.6557e-05 / 500 |
| descending_p4_s0_0.03 | 0.154033 | 0.000929605 / 457 | 1.75533e-05 / 502 | 1.77083e-05 / 500 |
| descending_p4_s0_cold | 0.874366 | 0.0366441 / 455 | 0.0204412 / 454 | 0.00735813 / 500 |
| descending_p4_s1_0.001 | 0.00497982 | 0.00133325 / 456 | 1.88342e-05 / 501 | 1.07836e-05 / 500 |
| descending_p4_s1_0.03 | 0.137945 | 0.000735392 / 458 | 2.4356e-05 / 501 | 1.47945e-05 / 500 |
| descending_p4_s1_cold | 0.892448 | 0.0373552 / 455 | 0.0189811 / 458 | 3.68798e-05 / 500 |
| descending_p4_s2_0.001 | 0.00484485 | 0.00140496 / 458 | 3.08695e-05 / 502 | 1.2132e-05 / 500 |
| descending_p4_s2_0.03 | 0.148587 | 0.000837103 / 462 | 2.31917e-05 / 501 | 1.58683e-05 / 500 |
| descending_p4_s2_cold | 0.927201 | 0.0324017 / 453 | 0.0170897 / 456 | 0.0122991 / 500 |

## Equal-evaluation prefix check

A/B at the smaller of each pair's recorded closure counts, local: A 12/36, B 36/36 pass. This is a diagnostic trace-prefix comparison, not equal wall time.

A/B at the smaller of each pair's recorded closure counts, cold: A 0/18, B 12/18 pass. This is a diagnostic trace-prefix comparison, not equal wall time.


## Representative gradient evolution

Ascending four-pass seed0, near perturbation0.001 and cold, fixed independently of outcomes. Norms are Euclidean parameter-gradient norms and depend on coordinates.

| Start | Arm | Initial norm | Maximum norm | Last evaluated norm |
|---|---|---:|---:|---:|
| 0.001 | A | 0.002574 | 0.1517 | 3.483e-05 |
| 0.001 | B | 0.003108 | 0.2739 | 7.185e-07 |
| 0.001 | C | 0.003108 | 0.003108 | 2.513e-07 |
| cold | A | 0.3344 | 0.3344 | 0.0003734 |
| cold | B | 0.4371 | 0.4371 | 0.0009336 |
| cold | C | 0.4371 | 0.4371 | 0.0002582 |

## Algebraic planted references

18 distinct targets; exact saved polar factors reproduce targets with zero discrepancy. Cartesian conversion maximum relative error 2.12e-16. These are algebraic references, never counted as optimization successes. Raw receipt repeats each reference across start categories.
