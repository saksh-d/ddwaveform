# Stage 2E — Classical Baseline Recalibration and Complete Frozen-Waveform Forensics

## Protocol and provenance

- Frozen checkpoint: `31e42c4da541a913c9b738eeb6dc96451981ad91ccd95005f3444fe520d5b2cc`; unitarity residual `2.233e-14`.
- AFDM validation used 256 validation-only DD-v1 channels and exact K=24 analytic MSE; no BER, test channel, or Learned result entered selection.
- Untouched nominal test bank: 512 channels; headline points use 1048576 paired bits per waveform.
- Learned was loaded once through the exact hash gate and was never trained, selected, or modified.

## AFDM and OTFS audit

DD-v1 uses normalized Doppler bins. With `nu=alpha+a`, `a in [-1/2,1/2)`, endpoints +/-0.75 require `alpha_max=1`. Published integer canonical slope is `2Nc1=3`; the declared one-bin fractional guard gives canonical `2Nc1=5`. The constrained validation sweep over integer slopes 1..11 selected `2Nc1=1`, so `c1=0.0078125`. `c2=sqrt(2)/(100N)` remains the small irrational principal value.
Primary AFDM model/rule source: Bemani, Ksairi, and Kountouris, *Affine Frequency Division Multiplexing for Next Generation Wireless Communications*, https://www.eurecom.edu/publication/6896/download/comsys-publi-6896.pdf. The repository sign convention is audited independently; no literature ordering is assumed.
Canonical integer path-location gate: `True`. OTFS independent-reference modulation error `3.233e-15` and clean inverse error `5.095e-15`.

## Principal DD-v1 communication

QPSK BER at 18 dB:

| Receiver | IDENTITY | OFDM | OTFS | AFDM canonical | AFDM tuned | Learned |
|---|---:|---:|---:|---:|---:|---:|
| fixed-Q K=24 | 0.00195122 | 0.00592804 | 0.0204487 | 0.0239182 | 0.0051899 | 0.0012598 |
| full K=64 | 0.000793457 | 0.00210381 | 0.000116348 | 0.000102997 | 0.000551224 | 0.00107193 |

Recalibrated AFDM changes its prior Stage-2 fixed-Q 18 dB BER by `+0.0%`; validation reselected the identical waveform. Full-information MSE remains unitary invariant, while finite-alphabet BER need not be waveform invariant.

## Interpretation gates

### Classical-baseline gate

- No: historical `2Nc1=1` was not numerically inappropriate for K=24, because the independent K=24 validation sweep reselected it. Carrying it forward without this audit was scientifically under-justified. Current principal AFDM is `c1=1/(2N)=0.0078125`, `c2=sqrt(2)/(100N)`.
- Corrected canonical AFDM reproduces exact separated grid-DD diagonals. Fractional Doppler/delay spreads them continuously, as predicted by the sampled operator model.
- Learned-vs-AFDM fixed-Q conclusion at 18 dB does not reverse: Learned `0.0012598`, AFDM `0.0051899`.
- OTFS is correct for the existing algebraic 8x8, N=64, column-major unitary transform; this stage does not substitute physical CP/pulse-shaped OTFS.

### Communication gate

- Nominal QPSK: at fixed-Q K=24 and K=32, Learned beats OFDM/OTFS/AFDM at every 0–30 dB point; at K=16 it beats only OTFS. With full information it beats OFDM throughout, loses to OTFS throughout, and loses to AFDM from 5–30 dB.
- Nominal 16-QAM: at fixed-Q K=24/K=32, Learned beats all three from 10–30 dB but loses to OFDM at 0/5 dB; at K=16 it beats only OTFS. With full information it beats OTFS/AFDM through 15 dB, then beats only OFDM from 18–30 dB.
- At 18 dB K=24, fixed-Q costs Learned `4.7%` QPSK and `1.8%` 16-QAM BER versus oracle; fixed-Q is the headline receiver.
- Stress at 18 dB with regime-informed Q: Learned QPSK beats all three classical waveforms for fractional-Doppler, DD-v1, wider-Doppler, wider-delay, and more-path regimes; it loses to OTFS and AFDM on GRID-DD and fractional-delay. Learned 16-QAM beats all three except on GRID-DD, where it loses to all.
- Relative to the stress DD-v1 bank, Learned QPSK BER rises `101.8%` for wider Doppler, `170.4%` for wider delay, and `38.0%` for more paths. Corresponding 16-QAM increases are `52.4%`, `77.1%`, and `30.0%`.
- `fixedOriginalK24` versus `fixedRegimeK24` separates waveform generalization from support adaptation; AFDM parameter adaptation is explicitly labeled `AFDM adapted`.

### Physical-forensics gate

- Learned atom RMS bandwidth spans `0.05224`–`0.1698`; spectral entropy spans `0.7407`–`0.8685`. Magnitude CV spans `0.2934`–`0.7475`.
- The all-64 overlay/heatmaps show centroid-ordered frequency populations with non-identical bandwidths and wrap-around structure, not one translated prototype or a single chirp law.
- Strongest pairwise rank association with Q is `cross_ambiguity_peak` (`rho=0.985`); controlled linear R2 is `0.911`. Cross-ambiguity distinguishes objectively high/median/low-Q pairs.
- Evidence favors the induced DD coupling geometry over a simple exact atom/filterbank/chirp parameterization.

### Practicality gate

- QPSK 4x-PAPR p99: OFDM `9.80` dB, OTFS `9.20` dB, AFDM `9.79` dB, Learned `9.65` dB.
- Dense discrete-time spectra and occupied-energy summaries are diagnostic only. Without pulse shaping, DAC/front-end, or a spectral mask, no RF ACLR claim is made.

## Established

- Frozen Learned communication, structure, spectra, time behavior, STFT, ambiguity, DD concentration, and PAPR under one consistent baseline set.
- Validation-selected AFDM and canonical AFDM are distinct named roles; fixed-Q is the primary practical receiver.

## Corrected from prior stages

- The principal AFDM is no longer silently inherited from K=3. K=24 validation retained slope `1`, so the correction is provenance and role separation, not a waveform change. Historical artifacts were not overwritten.

## Suggested

- Carry PAPR/spectral occupancy and Q concentration as explicit constraints only in a separately authorized future training stage.
- Treat pulse-shaped/CP-OTFS and continuous-time AFDM as a later physicalization study.

## Still unknown

- Whether a compact constrained waveform family can reproduce Learned's DD coupling without its atom irregularity.
- RF spectral-mask behavior, amplifier sensitivity, synchronization/channel-estimation cost, and performance with nonlinear or nonorthogonal receivers.

STOP — no waveform training, channel-family retraining, waveform-family fitting, pulse shaping, nonorthogonal transform, neural receiver, or Stage 3 work was performed.
