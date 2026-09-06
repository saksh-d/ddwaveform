# Pub-1E — Frozen-waveform qualification and operating-regime mapping

## Recommendation: Proceed with a narrower claim

Classification: under the historical K=24 receiver, the limited-representation claim is supported for joint/off-grid and nominal DD-v1 conditions, but not on the integer grid or with full information. Stronger receivers repair classical baselines and reduce the advantage. Proceed only with this qualified claim; do not retrain or start Pub-1F automatically.

All numbers below are production results from 512-channel confirmatory banks unless explicitly labeled exploratory. Learned was frozen; training steps were zero.

## E1 — Receiver sensitivity

At K=24 and 18 dB:

| condition | modulation | receiver | Learned | AFDM | OFDM | Learned-AFDM paired 95% CI |
|---|---|---|---:|---:|---:|---:|
| integer | 16QAM | historical | 0.0282497 | 0.0245819 | 0.0257301 | [0.00292871, 0.00438595] |
| integer | 16QAM | tuned | 0.0282497 | 0.0245819 | 0.0257301 | [0.00292778, 0.00439739] |
| integer | 16QAM | residual_aware | 0.0283194 | 0.0245819 | 0.0257301 | [0.00298402, 0.00448418] |
| integer | QPSK | historical | 0.00122929 | 0.000245094 | 0.00234604 | [0.000827789, 0.00114632] |
| integer | QPSK | tuned | 0.00122929 | 0.000245094 | 0.00234604 | [0.000828743, 0.00114632] |
| integer | QPSK | residual_aware | 0.00120831 | 0.000245094 | 0.00234604 | [0.000812531, 0.00112727] |
| joint_0.5 | 16QAM | historical | 0.0220203 | 0.116496 | 0.0596609 | [-0.0964357, -0.0925531] |
| joint_0.5 | 16QAM | tuned | 0.0220203 | 0.0916281 | 0.0470695 | [-0.0714483, -0.0677089] |
| joint_0.5 | 16QAM | residual_aware | 0.0221119 | 0.0664129 | 0.032258 | [-0.0459013, -0.0426683] |
| joint_0.5 | QPSK | historical | 0.0011158 | 0.0157261 | 0.00959396 | [-0.0156288, -0.0136394] |
| joint_0.5 | QPSK | tuned | 0.0011158 | 0.00681782 | 0.00437164 | [-0.00629044, -0.00514793] |
| joint_0.5 | QPSK | residual_aware | 0.00109386 | 0.00222778 | 0.00369263 | [-0.00135138, -0.000923157] |
| nominal_dd_v1 | 16QAM | historical | 0.0240173 | 0.0715513 | 0.0440149 | [-0.0499716, -0.0451565] |
| nominal_dd_v1 | 16QAM | tuned | 0.0240173 | 0.0658083 | 0.0420923 | [-0.0438252, -0.0397262] |
| nominal_dd_v1 | 16QAM | residual_aware | 0.0241547 | 0.0531864 | 0.0311689 | [-0.0305061, -0.0275879] |
| nominal_dd_v1 | QPSK | historical | 0.00115299 | 0.00460052 | 0.0052557 | [-0.00395873, -0.0029726] |
| nominal_dd_v1 | QPSK | tuned | 0.00115299 | 0.00298595 | 0.00414658 | [-0.00219631, -0.00150394] |
| nominal_dd_v1 | QPSK | residual_aware | 0.00114441 | 0.00129795 | 0.00278091 | [-0.00033474, 2.76566e-05] |

Validation-selected scales vary across waveform/condition/K in `[1.0, 2.0, 4.0]`. The all-waveform companion figure shows that receiver improvement is not a Learned-only intervention.

Residual covariance materially repairs classical baselines. In nominal DD-v1/QPSK, residual-aware Learned-AFDM has a CI containing zero: inconclusive, not evidence of equality. Joint-fractionality gains survive but shrink; integer-grid losses persist. Full-information comparisons use a different declared reference condition.

Residual statistics are waveform/condition/K-specific means and dense receive covariances estimated on an independent bank using the same oracle row-top-K rule later used by E1. They omit conditioning on realized retained values/support and remain moment-matched approximations. E3 fixed-support policies were evaluated only with the historical receiver, so no residual-statistics/support mismatch is present. Equal K does not imply equal stored information or compute: residual-aware adds two dense 64x64 complex matrices per bank entry and a direct dense solve.

## E2 — Exploratory two-dimensional fractionality map

Descriptive lower-BER cell counts only: 16-QAM Learned is lower in 33/36 cells versus AFDM, 25/36 versus OTFS, and 24/36 versus OFDM; QPSK counts are 28/36, 17/36, and 36/36.

The map suggests joint fractionality is favorable, while integer and isolated-axis losses remain. It does not estimate a decision boundary. Pointwise intervals do not support a map-wide significance claim. Predeclared K=16/32 companion maps were not run and must not be inferred from K=24.

## E3 — Fixed-support prior mismatch

For Learned 16-QAM at maximum delay shift, nominal/recalibrated/oracle BER is 0.04168/0.03833/0.03316; at maximum Doppler shift it is 0.03698/0.03383/0.02916. Recalibration recovers part, not all, of the loss. At delay factor 2, nominal support retains 99.13% energy yet BER is 25.7% worse than oracle. Retained energy is therefore not a sufficient receiver-quality statistic.

The power-profile shift was benign only for the declared one-factor experiment. These results do not establish robustness to joint shifts.

## E4 — Locked confirmatory envelope

The label `boundary_joint_0.2` is retained as a historical data key; scientifically it is the predeclared intermediate/probe point, not an empirically estimated boundary.

| condition | modulation | Learned | OTFS | AFDM | OFDM | Learned-AFDM paired 95% CI |
|---|---|---:|---:|---:|---:|---:|
| favorable_joint_0.5 | 16QAM | 0.0238132 | 0.136333 | 0.119285 | 0.0620184 | [-0.097291, -0.0935993] |
| favorable_joint_0.5 | QPSK | 0.00109482 | 0.0257511 | 0.0167961 | 0.0101089 | [-0.0167094, -0.0147018] |
| boundary_joint_0.2 | 16QAM | 0.0280066 | 0.0508938 | 0.0561829 | 0.0343037 | [-0.029687, -0.0267105] |
| boundary_joint_0.2 | QPSK | 0.00132084 | 0.00178051 | 0.00223446 | 0.00361252 | [-0.00119114, -0.000636101] |
| unfavorable_integer | 16QAM | 0.0303268 | 0.0261154 | 0.0259066 | 0.0270653 | [0.00360107, 0.005229] |
| unfavorable_integer | QPSK | 0.00145149 | 0.000429153 | 0.000293732 | 0.00277519 | [0.00098896, 0.0013361] |
| full_information_nominal | 16QAM | 0.0227489 | 0.0205898 | 0.021039 | 0.0235491 | [0.00104804, 0.00237277] |
| full_information_nominal | QPSK | 0.000967979 | 0.000134468 | 0.000130653 | 0.00218105 | [0.000716209, 0.000967026] |

Named AFDM 16-QAM results at 18 dB:

| condition | selected | historical slope 1 | canonical slope 3 | guard slope 5 |
|---|---:|---:|---:|---:|
| favorable_joint_0.5 | 0.119285 | 0.0743694 | 0.119285 | 0.141567 |
| boundary_joint_0.2 | 0.0561829 | 0.0406094 | 0.0561829 | 0.0654478 |
| unfavorable_integer | 0.0259066 | 0.0267954 | 0.0259066 | 0.0258102 |
| full_information_nominal | 0.021039 | 0.0219631 | 0.021039 | 0.0210304 |
Post hoc paired Learned-slope1 CI at favorable_joint_0.5: [-0.0517311, -0.0494213].
Post hoc paired Learned-slope1 CI at boundary_joint_0.2: [-0.0136757, -0.0115556].
Post hoc paired Learned-slope1 CI at unfavorable_integer: [0.00273037, 0.00433159].
Post hoc paired Learned-slope1 CI at full_information_nominal: [0.000173521, 0.00139904].

The favorable and intermediate/probe results use the historical receiver; integer reverses and full-information nominal favors classical baselines. Nominal full-information Learned/classical-best 16-QAM BER ratio is 1.105. Aggregate LMMSE MSE is unitary-invariant under full information, but finite-constellation BER need not be.

AFDM guard selection searched odd slopes 3/5/7/9; slope 1 was a named historical comparator, not a candidate. Odd slopes are mathematically unitary here; the lower bound was an engineering guard against delay-Doppler overlap, not a unitarity requirement. The frozen selection remains slope 3 with no test-set reselection, although slope 1 is stronger in some E4 regimes. Slope1 plus the improved receiver ladder was not tested. OFDM remains a primary baseline and often sits between Learned and the delay-Doppler baselines in the favorable historical-receiver regime.

## Established, unsupported, and limitations

Established: historical-receiver joint-fractionality gains; narrower nominal-DD-v1 evidence; receiver-driven shrinkage; fixed-support degradation and partial recalibration recovery; integer/full-information reversals.

Unsupported: universal superiority, empirical fractionality boundary, specialized classical receivers, equal information/storage/compute, joint prior shifts, coding/MIMO/pulse-shaped/RF behavior, or a new-training implication.

## Reproducibility and preservation

Waveforms share channels/bits/noise within each evaluated row set, and each waveform constructs its own support. E1 K=24 at 18 dB consumes SNR index 2 while other K values evaluate only 18 dB at index 0; cross-K curves are therefore not common-random-number paired. The E1 K=64 historical and full rows also must not be treated as paired duplicates. Some numeric seeds recur across different generator roles (bootstrap seed 18000 and E4 physical seed 18000); that is harmless cross-domain reuse, not leakage, but it invalidates a blanket claim that every numeric seed is unique. Exact derivations are in the protocol addendum.

No simulations were rerun for this correction. The active plots, post hoc slope-1 intervals, report, and handoffs were regenerated only from saved Pub-1E tables. The old mixed-waveform E1 budget rendering is retained under `outputs/stage_pub1e/superseded/`.

## Pub-1F boundary

Pub-1F may begin only after its own protocol is frozen, fresh independent banks are assigned by generator role, and training reproducibility/objective ablation/regime replication/compute qualification are specified. It must include per-waveform constructions, OFDM, broader AFDM validation including slope 1, and explicit support/residual rules. Pub-1F was not executed.
