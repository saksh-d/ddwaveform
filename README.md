# Delay-Doppler Waveform Discovery

This repository studies communication-driven waveform representations for doubly dispersive channels. [`researchPlan.MD`](researchPlan.MD) is the scientific roadmap; the current implementation covers the Stage 0A-E validation harness, classical baselines, and the Stage 0F channel-specific dense-unitary oracle diagnostic.

The identity matrix is labeled `IDENTITY` throughout as an untransformed time-sample control. The principal classical DD comparison is OFDM, OTFS, and AFDM.

## Setup and run

```bash
conda activate pytorch
python -m pip install -r requirements.txt
pytest -q
python -m scripts.stage0
python -m scripts.stage0f
```

Stage 0 runs on CPU in double precision. Stage 0F uses complex128 projected optimization and checkpoints its long run. Generated CSV, JSON, figures, and checkpoints are written under `outputs/`; this directory is intentionally Git-ignored. Tracked scientific reports live under `reports/`.
