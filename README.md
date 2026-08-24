# Delay-Doppler Waveform Discovery

This repository studies communication-driven waveform representations for doubly dispersive channels. [`researchPlan.MD`](researchPlan.MD) is the scientific roadmap; the current implementation covers the Stage 0A-E validation harness, classical baselines, and the Stage 0F channel-specific dense-unitary oracle diagnostic.

The identity matrix is labeled `IDENTITY` throughout as an untransformed time-sample control. The principal classical DD comparison is OFDM, OTFS, and AFDM.

Stage 2E recalibrates AFDM for the K=24 DD-v1 objective and runs the complete frozen-waveform communication/physical audit:

```bash
conda run -n pytorch python -m scripts.stage2e
```

Use `--smoke` only to verify the complete output path at reduced statistical resolution; smoke results are not scientific production results. Stage 2E loads the exact Stage-1A.2 checkpoint and performs no Learned-waveform training.

## Setup and run

```bash
conda activate pytorch
python -m pip install -r requirements.txt
pytest -q
python -m scripts.stage0
python -m scripts.stage0f
```

Stage 0 runs on CPU in double precision. Stage 0F uses complex128 projected optimization and checkpoints its long run. Generated CSV, JSON, figures, and checkpoints are written under `outputs/`; this directory is intentionally Git-ignored. Tracked scientific reports live under `reports/`.
