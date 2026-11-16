# Waveform Exploration

One research notebook for delay–Doppler waveforms and residual CFO.

```bash
python -m pip install -r requirements.lock.txt
# Optional frontend: python -m pip install jupyterlab
```

Open **[Waveform_Exploration.ipynb](Waveform_Exploration.ipynb)**, edit its one
configuration cell, restart, **Run All**. Figures appear inline; full results and
metadata stay under ignored `local/`.

A **bundle** is a supplied archive of previously learned matrices and saved results.
The notebook verifies/unpacks it automatically; you do not assemble individual files.

| Run | Input needed in a clean clone |
|---|---|
| Saved DD N64/128/256, or fresh tests of archived Learned | `local/advisor-scaling-artifacts.tar.gz` — one archive for all N |
| New DD training + fresh tests | **None**: `MODE="train_new"`, `EVIDENCE_MODE="fresh"` |
| Fresh tests of your newly trained checkpoint | Its matching local `.npz` + `.json` pair |
| Frozen CFO + fresh tests | `local/advisor-n64-artifacts.tar.gz` only |

An already provisioned input directory needs no archive at runtime. Both archives
are needed only for **both** archived-DD and CFO workflows. [Exact input details](support/docs/ARTIFACTS.md).
Fresh work requires **MPS/FP32**, without CPU fallback; replay keeps reference arithmetic.

Evaluation tests a frozen W; training creates W. One Learned per N/run, no resizing
or reuse of old BER after model changes. Surrogate fits and limited trials make no
performance guarantee. **CFO training is unsupported.**

`waveforms/` is the implementation; `support/` holds checks, notes and preserved history.
Bundles, outputs, checkpoints, executed notebooks, caches and build products are
**gitignored**; the source notebook stays output-free.
