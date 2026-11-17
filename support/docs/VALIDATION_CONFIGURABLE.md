# CPU-only workflow validation

Current runtime uses CPU FP32 for fresh DD/CFO evaluation and DD training, with
one numerical thread. Archived DD replay preserves CPU float64/complex128.
No device selector or accelerator package path exists. All six pretrained input
files are tracked; optional full audit archives remain ignored.

```bash
python -m pytest support/tests -q
python support/scripts/qualify_notebook.py --out local/qualification
python support/scripts/audit_configurable.py --qualification local/qualification
```

Use a new output path for each qualification; saved checkpoints are never overwritten.
Full historical audit checks additionally accept --artifacts and --scaling-artifacts.
They skip when the optional archives are absent. Delivered-input tests do not skip.

Verification covers source/input hashes, reference operators and receivers, native
size/supports, reproducible seeds, train/validation/test separation, actual updates,
checkpoint reload, budget/zero-error handling and the distinct rectangular CFO model.
Notebook outputs retain all13DD/5CFO figures, complex/grouped spectra and overlays.

Current CPU-only qualification is in progress. Exact commands/results stay under
ignored local/cpu-only/. Older verification is retained under
support/internal/VALIDATION_BEFORE_CPU_ONLY.md and the timestamped HISTORY.md.
Linux/Windows execution is untested; no production precision/training claim.
