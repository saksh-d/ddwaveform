# Completed supplementary studies

All paths below relative to provisioned `local/artifacts`. The compact package
reads these saved results; it offers no new training, factorization or discovery
campaign. Archived source under `sources/wms` is evidence, never runtime imports.

| Study | Inputs/evidence | Established finding and limit |
|---|---|---|
| W–A/orthogonality/frequency | `supplementary/orthogonality_bridge/{summary.json,diagnostics.npz,all_2016_basis_pairs.csv,per_channel_bridge.csv,REPORT.md}` | Complex spectral reconstruction; columns orthogonal although overlapping spectral powers. Phase-only DFT-bin control changes Doppler/DD coupling while preserving bin powers. Pure delay commutes with that phase control. Does not preserve interpolated spectra or establish a new waveform. |
| Frequency groups | `supplementary/block_frequency/{association.json,spectral_data.npz,per_column.csv,group_profiles.csv,REPORT.md}` | Frozen DD groups correlate strongly with broad frequency regions; leakage/outlier columns retained. Group projector profiles survive within-group rotations. Descriptive, not a unique exact analytical construction. |
| Training | `supplementary/main/{training_selected_summary.csv,training/*/*/trace.csv,training/*/*/selected.json,training_trial_ber_variability.csv}` |3complete OFDM-start trials:2000mean/1000mean continuation/1000tail/2000Q steps. All mean continuations select unchanged step0. Tail lowers stream tails; Q maximizes concentration without matching tail/BER behavior. Step0 trace training_loss is a validation placeholder, excluded from loss interpretations. No initialization-universality claim. |
| Interventions | `supplementary/main/{interventions.csv,intervention_bases.npz,projector_profile_comparison.npz,paired_differences.csv,communication/*.npz}` | Wexp(tS), five norm-matched within/across/random directions, t=.5 frozen for BER. Within-group block energy maxchange2.84e-14, projector profiles1.33e-15. Yet14/20 within-direction/modulation18dB fixed-Q BER contrasts exclude zero. Algebraic invariance does not imply finite-QAM invariance; intervals descriptive, multiplicity uncorrected. |
| Training/BER connection | `supplementary/main/{communication_summary.csv,paired_analytic_differences.csv,paired_differences.csv,communication/*.npz}` | Tail-vs-mean full-QPSK improvement at18dB in3/3trial intervals;16QAM tail-vs-mean intervals cross0. OTFS lower full18dB BER than tested Learned candidates. Training-seed spread separate from channel uncertainty. Full/realized24/fixed24 separate. |
| Structural impairment transfer | `supplementary/main/impairment_transfer.csv` | Frozen bases under identity/delay/Doppler/one DD path/multipath, strengths .25/.5/1;256elementary sweep points. No impairment-specific retraining or BER. Multipath same nominal bank repeated across strength labels. Not genuine CFO W/R transfer. |

Completed main status:3trials,12phases,50BER candidates,5760summary points,
1024channels and1,048,576bits/point. Source ledger2042.9s. These are archival
study facts, not new runs in this package. Structural gate correction occurred
before production structural/BER evaluation; `sources/wms/STRUCTURAL_AMENDMENT.md`
and both executed source closures retained. Main report in
`supplementary/main/REPORT.md` preserves protocol detail and unfavorable results.

Historical findings retained in [provenance](../docs/PROVENANCE.md): no ceil(N/K) block
law; concentration-only improvements lack one common group organization;
modulation matters beyond concentration; added geometry objective not
consistently improving; compact reconstruction unqualified. No research resumed.

N128/N256 active sampling excluded at snapshot time. No partial counts labeled
final; no waiting or interference with the active study. An owner can review its
eventual results separately before any later supplementary snapshot.
