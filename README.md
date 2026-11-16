# Waveform Exploration

One research notebook for delay–Doppler waveforms and residual CFO.

```bash
python -m pip install -r requirements.lock.txt
# Optional frontend: python -m pip install jupyterlab
```

Open **[Waveform_Exploration.ipynb](Waveform_Exploration.ipynb)**, edit its one
configuration cell, restart, **Run All**. Figures appear inline; complete results
and metadata are saved under ignored `local/`.

- **Existing DD results:** put the provided `advisor-scaling-artifacts.tar.gz` in `local/`.
- **Existing CFO:** put `advisor-n64-artifacts.tar.gz` in `local/`.
- **New DD training:** `MODE="train_new"`, `EVIDENCE_MODE="fresh"`; no bundle required.

Inputs are verified/provisioned automatically. A clean clone contains code, not
these bundles or checkpoints. Fresh runs require **MPS/FP32**, without CPU fallback;
archived replay retains its reference arithmetic. [Input details](support/docs/ARTIFACTS.md).

Evaluation tests a frozen W; training creates W. One Learned per N/run, no resizing
or reuse of old BER after model changes. The surrogate fit and bounded/limited-trial
results make no performance guarantee. **CFO training is unsupported.**

`waveforms/` contains the implementation. `support/` holds manifests, tests, utilities
and preserved research/history; [methods](support/docs/CONFIGURABLE.md) are there if needed.
Bundles, generated figures/arrays, executed notebooks, checkpoints, caches and build
artifacts stay **gitignored**. The single source notebook stays output-free in Git.
