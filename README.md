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
conda run -n pytorch python -m scripts.stage_pub1c --phase report
conda run -n pytorch python -m scripts.stage_pub1c --phase manifest
```

`--phase all` runs the ordered evaluation. Each waveform keeps its own realization-wise row-top-K interactions; K=24 means rho=0.375 only at N=64. The controlled sweep separates delay, Doppler, and joint fractional severity. Structural Part A thresholds one unchanged authoritative map; Part B compares full maps from separately trained K=16/24/32 tensors. No training, constellation alignment, pulse shaping, estimator, or Phase-2 physical-front-end change occurs. Outputs remain ignored under `outputs/stage_pub1c/`; the tracked summary is `reports/stage_pub1c_summary.md`.

`constellations-15db` adds the named-regime P7 diagnostic without rewriting P3. It selects one channel nearest the waveform-invariant full-information median from independent 256-channel GRID-DD, FRACTIONAL-DOPPLER, and DD-v1 banks, then compares raw 16-QAM Learned/OTFS/AFDM outputs using identical symbols and physical noise.

## Setup and run

```bash
conda activate pytorch
python -m pip install -r requirements.txt
pytest -q
python -m scripts.stage0
python -m scripts.stage0f
```

Stage 0 runs on CPU in double precision. Stage 0F uses complex128 projected optimization and checkpoints its long run. Generated CSV, JSON, figures, and checkpoints are written under `outputs/`; this directory is intentionally Git-ignored. Tracked scientific reports live under `reports/`.
