# Configurable waveform notebook

```bash
conda activate pytorch
python -m pip install -r requirements.lock.txt
# Optional frontend, if needed: python -m pip install jupyterlab
```

Open [notebooks/results.ipynb](notebooks/results.ipynb), select `pytorch` kernel,
edit its **one CONFIGURATION cell**, restart/clear outputs, **Run All**.
Figures appear sequentially inline; no plotting-folder steps.

- **DD `load_existing`**: verified own W at N64/128/256, or your local DD checkpoint.
- **DD `train_new` + `fresh`**: actual bounded MPS/FP32 unitary concentration training,
  train-only support/groups, held-out validation selection, independent test BER.
  One selected Learned per run/N, compared fairly with OTFS/AFDM at the same K.
- **Residual CFO `load_existing` + `fresh`**: qualified W[64,45], R[45,64], A=RHW;
  new direct-R BER and distinct transmit/receive basis figures. CFO training,
  DD grouping/sparse receivers, size changes and archived CFO BER replay unsupported.
- **`saved_replay`**: exact nominal archived DD evidence only, using the preserved
  CPU double reference path. Changed W/channel/K/support requires fresh evaluation.

Evaluation knobs change a frozen W's test scenario; **TRAINING** knobs create a new
W. Default K/N=3/8 gives K24/48/96; no matrix resizing or per-SNR winner switching.
New trainer optimizes an multipath concentration surrogate, not BER. Learned need not win.
Fresh default: at most8channels/point; training20steps/120s. These are bounded
checks, not precise campaigns; counts/stopping/zero-error uncertainty stay visible.

Both sealed input bundles must be copied into this worktree's `local/`; see
[ARTIFACTS](docs/ARTIFACTS.md). Run All verifies/provisions them without source-path
or network dependencies. New training needs neither bundle. Fresh numerical work
requires MPS/FP32, CPU fallback disabled; plotting uses exported NumPy arrays.
Outputs/checkpoints stay ignored under `local/`, source notebook output-free.

[Controls, scientific scope and checkpoint use](docs/CONFIGURABLE.md) ·
[Validation and exact commands](docs/VALIDATION_CONFIGURABLE.md) ·
[Existing coupling/size provenance](docs/COUPLING.md)
