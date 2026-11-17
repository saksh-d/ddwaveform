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

Current frozen qualification:

- Full130tests pass8.13s; integrated target130pass9.05s. Clean clone95pass and
  35optional audit checks skip. Generic venv/requirements/Jupyter install succeeds.
- CPU-only17notebook cases:16successful and expected CFO-training rejection,
  192figures/188.61s. Replay/fresh/train cover N64/128/256; changed laws, support
  recalibration,20step default training and checkpoint reload also execute.
- Against prior CPU runs:140PNG bytes/973array fields exact, CSV exact except
  timing; selection/trace unchanged. CPU algebra/RNG and all input bytes unchanged.
- Genuine older checkpoint retains original weights/support/training provenance
  and evaluates24fresh BER rows on CPU. Current wheel has only CPU numerical code
  and all six inputs. Source/input/export300DPI/checkpoint audits pass.
- Active runtime/notebook/current docs contain no accelerator code or selector.
  Fresh DD/CFO/train all report CPU FP32; replay reports CPU reference arithmetic.

Exact commands/results stay under ignored local/cpu-only/. Older verification is retained under
support/internal/VALIDATION_BEFORE_CPU_ONLY.md and the timestamped HISTORY.md.
Linux/Windows execution is untested; no production precision/training claim.
