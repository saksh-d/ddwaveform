# Delay-Doppler Waveform Discovery

This repository studies communication-driven waveform representations for doubly dispersive channels. [`researchPlan.MD`](researchPlan.MD) is the scientific roadmap; the current implementation covers the Stage 0A-E validation harness, classical baselines, and the Stage 0F channel-specific dense-unitary oracle diagnostic.

The identity matrix is labeled `IDENTITY` throughout as an untransformed time-sample control. The principal classical DD comparison is OFDM, OTFS, and AFDM.

Stage 2E recalibrates AFDM for the K=24 DD-v1 objective and runs the complete frozen-waveform communication/physical audit:

```bash
conda run -n pytorch python -m scripts.stage2e
```

Use `--smoke` only to verify the complete output path at reduced statistical resolution; smoke results are not scientific production results. Stage 2E loads the exact Stage-1A.2 checkpoint and performs no Learned-waveform training.

Stage Pub-1A independently qualifies that frozen N=64 waveform. Run the AFDM gate first, then any resumable phase:

```bash
conda run -n pytorch python -m scripts.stage_pub1a --phase afdm-audit
conda run -n pytorch python -m scripts.stage_pub1a --phase communication
conda run -n pytorch python -m scripts.stage_pub1a --phase k-sweep
conda run -n pytorch python -m scripts.stage_pub1a --phase stress
conda run -n pytorch python -m scripts.stage_pub1a --phase operators
conda run -n pytorch python -m scripts.stage_pub1a --phase waveform-figures
conda run -n pytorch python -m scripts.stage_pub1a --phase practicality
conda run -n pytorch python -m scripts.stage_pub1a --phase figure-updates
conda run -n pytorch python -m scripts.stage_pub1a --phase report
```

`--phase all` runs every scientific phase. `figure-updates` refreshes publication layouts from saved tables/arrays without Monte Carlo. `--smoke` is path validation only. Pub-1A writes ignored data, PNG/PDF figures, and `report.md` under `outputs/stage_pub1a/`; it has no training interface.

F1/F2 use one 0 dB spectral-peak normalization per waveform, with Learned-only F1L/F2L supplements; absolute dense-DTFT arrays remain saved unchanged.

Stage Pub-1B tests actual sparse receiver compute and N=128/256 scaling on the independent publication lineage. Run the direct-equivalence gate before the compute grid, freeze scaling configuration before training, and respect the N=128 gate before N=256:

```bash
conda run -n pytorch python -m scripts.stage_pub1b --phase complexity-validate
conda run -n pytorch python -m scripts.stage_pub1b --phase complexity-run
conda run -n pytorch python -m scripts.stage_pub1b --phase complexity-plots
conda run -n pytorch python -m scripts.stage_pub1b --phase scaling-config
conda run -n pytorch python -m scripts.stage_pub1b --phase scaling-n128
conda run -n pytorch python -m scripts.stage_pub1b --phase scaling-n256
conda run -n pytorch python -m scripts.stage_pub1b --phase scaling-plots
conda run -n pytorch python -m scripts.stage_pub1b --phase report
```

`--phase all` runs the same sequence and skips N=256 unless the saved N=128 hard gate passed. `--smoke` validates output paths only. Re-plottable data, checkpoints, provenance, PNG/PDF figures, and `report.md` remain ignored under `outputs/stage_pub1b/`.

Stage Pub-1C completes the discrete-time Phase-1 story with a frozen, per-waveform oracle interaction budget:

```bash
conda run -n pytorch python -m scripts.stage_pub1c --phase provenance
conda run -n pytorch python -m scripts.stage_pub1c --phase fractionality
conda run -n pytorch python -m scripts.stage_pub1c --phase budget
conda run -n pytorch python -m scripts.stage_pub1c --phase constellations
conda run -n pytorch python -m scripts.stage_pub1c --phase constellations-15db
conda run -n pytorch python -m scripts.stage_pub1c --phase structure
conda run -n pytorch python -m scripts.stage_pub1c --phase scaling-structure
conda run -n pytorch python -m scripts.stage_pub1c --phase report
conda run -n pytorch python -m scripts.stage_pub1c --phase manifest
```

`--phase all` runs the ordered evaluation. Each waveform keeps its own realization-wise row-top-K interactions; K=24 means rho=0.375 only at N=64. The controlled sweep separates delay, Doppler, and joint fractional severity. Structural Part A thresholds one unchanged authoritative map; Part B compares full maps from separately trained K=16/24/32 tensors. No training, constellation alignment, pulse shaping, estimator, or Phase-2 physical-front-end change occurs. Outputs remain ignored under `outputs/stage_pub1c/`; the tracked summary is `reports/stage_pub1c_summary.md`.

`scaling-structure` extends P4 under `outputs/stage_pub1c/` using preserved Pub-1B N=128/N=256 checkpoints. It makes one fixed-waveform, fixed-ensemble-map threshold graph per N over the shared `K/N = 0.125, 0.1875, ..., 1` grid; it does not retrain a waveform or imply matching community counts across dimensions.

P4 retains both renderings in `outputs/stage_pub1c/figures/`: current `viridis` figures and `_magma_legacy` copies using the prior colorway.

`constellations-15db` adds the named-regime P7 diagnostic without rewriting P3. It selects one channel nearest the waveform-invariant full-information median from independent 256-channel GRID-DD, FRACTIONAL-DOPPLER, and DD-v1 banks, then compares raw 16-QAM Learned/OTFS/AFDM outputs using identical symbols and physical noise.

Pub-1E maps the frozen waveform's representation-limited operating regime and adds controlled receiver/prior-robustness evaluations:

```bash
conda run -n pytorch python -m scripts.stage_pub1e --phase provenance
conda run -n pytorch python -m scripts.stage_pub1e --phase validate
conda run -n pytorch python -m scripts.stage_pub1e --phase e1
conda run -n pytorch python -m scripts.stage_pub1e --phase e2
conda run -n pytorch python -m scripts.stage_pub1e --phase e3
conda run -n pytorch python -m scripts.stage_pub1e --phase e4
conda run -n pytorch python -m scripts.stage_pub1e --phase plots
conda run -n pytorch python -m scripts.stage_pub1e --phase report
conda run -n pytorch python -m scripts.stage_pub1e --phase manifest
```

The frozen human/machine protocols are `reports/stage_pub1e_protocol.md` and `configs/stage_pub1e.json`; the post-production interpretation/provenance corrections are in `reports/stage_pub1e_protocol_addendum.md` and `reports/stage_pub1e_correction_log.md`. `--smoke` checks integration only. Production phases are resumable and write ignored per-channel counts, selections, figures, timing, handoff, and manifests under `outputs/stage_pub1e/`. The residual-aware linear receiver is a disclosed moment approximation, not a conditional-optimal receiver. No training or Pub-1F execution is available in this runner.

Pub-1F first qualifies the historical waveform against slope-1-inclusive operational receivers, then obeys a frozen gate before five-seed training:

```bash
conda run -n pytorch python -m scripts.stage_pub1f --phase prerequisites
conda run -n pytorch python -m scripts.stage_pub1f --phase baseline
conda run -n pytorch python -m scripts.stage_pub1f --phase gate
conda run -n pytorch python -m scripts.stage_pub1f --phase training
conda run -n pytorch python -m scripts.stage_pub1f --phase complexity
conda run -n pytorch python -m scripts.stage_pub1f --phase setup
conda run -n pytorch python -m scripts.stage_pub1f --phase figures
conda run -n pytorch python -m scripts.stage_pub1f --phase report
conda run -n pytorch python -m scripts.stage_pub1f --phase manifest
```

The protocol is frozen in `configs/stage_pub1f.json` and `reports/stage_pub1f_protocol.md`. Production `training` refuses to run unless the saved production gate authorizes it. `--smoke` exercises the implementation only and never authorizes a scientific conclusion. Receiver timing restores the caller's thread setting and keeps detector, transform, reusable preparation, offline statistics, and channel-estimation exclusions separate.

An explicitly authorized post-gate continuation tests the narrower fractional-channel, retained-interaction claim without changing the failed original gate:

```bash
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase amendment
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase reuse
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase training
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase statistics
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase operating-validation
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase communication
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase intervals
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase matched-confirmation
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase scaling
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase scaling-matched
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase figures
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase report
conda run -n pytorch python -m scripts.stage_pub1f_continuation --phase manifest
```

Its frozen amendment is `reports/stage_pub1f_protocol_amendment.md`; machine-readable settings are in `configs/stage_pub1f_continuation.json`. The primary gate is five-seed same-K 16-QAM representation performance under joint fractional propagation. QPSK, integer-grid, and full-information rows remain mandatory controls. Matched-quality detector cost and measured runtime are separate claims. Smoke outputs are integration checks only; production artifacts remain ignored under `outputs/stage_pub1f_continuation/`.

The final ICC N=64 confirmation is an evaluation-only, fresh-bank run. It compares fixed support, realized Top-K, and full information; applies unchanged nominal fixed supports under four stress conditions; and evaluates the five selected Pub-1F continuation checkpoints with one common receiver protocol:

```bash
conda run -n pytorch python -m scripts.icc27_final_n64_confirm --smoke
conda run -n pytorch python -m scripts.icc27_final_n64_confirm
```

The frozen inputs are in `configs/icc27_final_n64_confirm.json`. Machine-readable production outputs are written under `outputs/icc27_final_n64_confirm/`; no waveform is retrained and no receiver choice uses test BER.

## Setup and run

```bash
conda activate pytorch
python -m pip install -r requirements.txt
pytest -q
python -m scripts.stage0
python -m scripts.stage0f
```

Stage 0 runs on CPU in double precision. Stage 0F uses complex128 projected optimization and checkpoints its long run. Generated CSV, JSON, figures, and checkpoints are written under `outputs/`; this directory is intentionally Git-ignored. Tracked scientific reports live under `reports/`.
